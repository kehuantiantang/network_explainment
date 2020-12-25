# coding=utf-8
from torch.utils import model_zoo
from torch.utils.data import DataLoader
from torchvision.models import resnet18, resnet34, resnet50, vgg16
from torchvision.transforms import transforms
from tqdm import tqdm
import torch
import gc
import random
import numpy as np
import torch.nn as nn
import utils.img_utils as utils
from dataset import PascalVOC_Dataset
import os

from utils.gpu_utils import auto_select_gpu

os.environ['CUDA_VISIBLE_DEVICES'] = auto_select_gpu()

def train_model(model, device, optimizer, scheduler, train_loader, valid_loader, save_dir, model_num, epochs, log_file):
    """
    Train a deep neural network model

    Args:
        model: pytorch model object
        device: cuda or cpu
        optimizer: pytorch optimizer object
        scheduler: learning rate scheduler object that wraps the optimizer
        train_dataloader: training  images dataloader
        valid_dataloader: validation images dataloader
        save_dir: Location to save model weights, plots and log_file
        epochs: number of training epochs
        log_file: text file instance to record training and validation history

    Returns:
        Training history and Validation history (loss and average precision)
    """

    tr_loss, tr_map = [], []
    val_loss, val_map = [], []
    best_val_map = 0.0

    # Each epoch has a training and validation phase
    for epoch in range(epochs):
        log_file.write("Epoch {} >>".format(epoch+1))

        for phase in ['train', 'valid']:
            running_loss = 0.0
            running_ap = 0.0

            criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
            m = torch.nn.Sigmoid()

            if phase == 'train':
                model.train(True)  # Set model to training mode

                for data, target in tqdm(train_loader, desc='Epoch:%s'%epoch):
                    #print(data)
                    target = target.float()
                    data, target = data.to(device), target.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    output = model(data)

                    loss = criterion(output, target)

                    # Get metrics here
                    running_loss += loss # sum up batch loss
                    running_ap += utils.get_ap_score(torch.Tensor.cpu(target).detach().numpy(), torch.Tensor.cpu(m(output)).detach().numpy())

                    # Backpropagate the system the determine the gradients
                    loss.backward()

                    # Update the paramteres of the model
                    optimizer.step()

                    # clear variables
                    del data, target, output
                    gc.collect()
                    torch.cuda.empty_cache()

                    #print("loss = ", running_loss)

                num_samples = float(len(train_loader.dataset))
                tr_loss_ = running_loss.item()/num_samples
                tr_map_ = running_ap/num_samples

                print('train_loss: {:.4f}, train_avg_precision:{:.3f}'.format(
                    tr_loss_, tr_map_))

                log_file.write('train_loss: {:.4f}, train_avg_precision:{:.3f}, '.format(
                    tr_loss_, tr_map_))

                # Append the values to global arrays
                tr_loss.append(tr_loss_), tr_map.append(tr_map_)
                scheduler.step()

            else:
                model.eval()  # Set model to evaluate mode

                # torch.no_grad is for memory savings
                with torch.no_grad():
                    for data, target in tqdm(valid_loader):
                        target = target.float()
                        data, target = data.to(device), target.to(device)
                        output = model(data)

                        loss = criterion(output, target)

                        running_loss += loss # sum up batch loss
                        running_ap += utils.get_ap_score(torch.Tensor.cpu(target).detach().numpy(), torch.Tensor.cpu(m(output)).detach().numpy())

                        del data, target, output
                        gc.collect()
                        torch.cuda.empty_cache()

                    num_samples = float(len(valid_loader.dataset))
                    val_loss_ = running_loss.item()/num_samples
                    val_map_ = running_ap/num_samples

                    # Append the values to global arrays
                    val_loss.append(val_loss_), val_map.append(val_map_)

                    print('val_loss: {:.4f}, val_avg_precision:{:.3f}'.format(
                        val_loss_, val_map_))

                    log_file.write('val_loss: {:.4f}, val_avg_precision:{:.3f}\n'.format(
                        val_loss_, val_map_))

                    # Save model using val_acc
                    if val_map_ >= best_val_map:
                        best_val_map = val_map_
                        log_file.write("saving best weights...\n")
                        print("saving best weights...\n", val_map_)
                        torch.save(model.state_dict(), os.path.join(save_dir,"model-{}.pth".format(model_num)))

    return ([tr_loss, tr_map], [val_loss, val_map])



def test(model, device, test_loader, returnAllScores=False):
    """
    Evaluate a deep neural network model

    Args:
        model: pytorch model object
        device: cuda or cpu
        test_dataloader: test images dataloader
        returnAllScores: If true addtionally return all confidence scores and ground truth

    Returns:
        test loss and average precision. If returnAllScores = True, check Args
    """
    model.train(False)

    running_loss = 0
    running_ap = 0

    criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
    m = torch.nn.Sigmoid()

    if returnAllScores == True:
        all_scores = np.empty((0, 20), float)
        ground_scores = np.empty((0, 20), float)

    with torch.no_grad():
        for data, target in tqdm(test_loader):
            #print(data.size(), target.size())
            target = target.float()
            data, target = data.to(device), target.to(device)
            bs, ncrops, c, h, w = data.size()

            output = model(data.view(-1, c, h, w))
            output = output.view(bs, ncrops, -1).mean(1)

            loss = criterion(output, target)

            running_loss += loss # sum up batch loss
            running_ap += utils.get_ap_score(torch.Tensor.cpu(target).detach().numpy(), torch.Tensor.cpu(m(output)).detach().numpy())

            if returnAllScores == True:
                all_scores = np.append(all_scores, torch.Tensor.cpu(m(output)).detach().numpy() , axis=0)
                ground_scores = np.append(ground_scores, torch.Tensor.cpu(target).detach().numpy() , axis=0)

            del data, target, output
            gc.collect()
            torch.cuda.empty_cache()

    num_samples = float(len(test_loader.dataset))
    avg_test_loss = running_loss.item()/num_samples
    test_map = running_ap/num_samples

    print('test_loss: {:.4f}, test_avg_precision:{:.3f}'.format(
        avg_test_loss, test_map))


    if returnAllScores == False:
        return avg_test_loss, running_ap

    return avg_test_loss, running_ap, all_scores, ground_scores


def main(data_dir, model_name, num, lr, epochs, batch_size = 16, download_data = False, save_results=False):
    """
    Main function

    Args:
        data_dir: directory to download Pascal VOC data
        model_name: resnet18, resnet34 or resnet50
        num: model_num for file management purposes (can be any postive integer. Your results stored will have this number as suffix)
        lr: initial learning rate list [lr for resnet_backbone, lr for resnet_fc]
        epochs: number of training epochs
        batch_size: batch size. Default=16
        download_data: Boolean. If true will download the entire 2012 pascal VOC data as tar to the specified data_dir.
        Set this to True only the first time you run it, and then set to False. Default False
        save_results: Store results (boolean). Default False

    Returns:
        test-time loss and average precision

    Example way of running this function:
        if __name__ == '__main__':
            main('../data/', "resnet34", num=1, lr = [1.5e-4, 5e-2], epochs = 15, batch_size=16, download_data=False, save_results=True)
    """

    model_dir = os.path.join("./vector/img", model_name)
    os.makedirs(model_dir, exist_ok=True)

    model_urls = {
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    }

    model_collections_dict = {
        "resnet18": resnet18(),
        "resnet34": resnet34(),
        "resnet50": resnet50(),
        "vgg16": vgg16(),
    }

    # Initialize cuda parameters
    use_cuda = torch.cuda.is_available()
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cuda" if use_cuda else "cpu")

    print("Available device = ", device)
    model = model_collections_dict[model_name]
    model.load_state_dict(model_zoo.load_url(model_urls[model_name]))
    # model.avgpool = torch.nn.MaxPool2d((9, 9))
    classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 20),
    )
    model.classifier = classifier

    # model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
    #
    #
    # num_ftrs = model.fc.in_features
    # model.fc = torch.nn.Linear(num_ftrs, 20)
    model.to(device)

    optimizer = torch.optim.SGD([
        {'params': list(model.parameters())[:-1], 'lr': lr[0], 'momentum': 0.9},
        {'params': list(model.parameters())[-1], 'lr': lr[1], 'momentum': 0.9}
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 12, eta_min=0, last_epoch=-1)

    # Imagnet values
    mean=[0.457342265910642, 0.4387686270106377, 0.4073427106250871]
    std=[0.26753769276329037, 0.2638145880487105, 0.2776826934044154]

    #    mean=[0.485, 0.456, 0.406]
    #    std=[0.229, 0.224, 0.225]

    transformations = transforms.Compose([transforms.Resize((300, 300)),
                                          #                                      transforms.RandomChoice([
                                          #                                              transforms.CenterCrop(300),
                                          #                                              transforms.RandomResizedCrop(300, scale=(0.80, 1.0)),
                                          #                                              ]),
                                          transforms.RandomChoice([
                                              transforms.ColorJitter(brightness=(0.80, 1.20)),
                                              transforms.RandomGrayscale(p = 0.25)
                                          ]),
                                          transforms.RandomHorizontalFlip(p = 0.25),
                                          transforms.RandomRotation(25),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean = mean, std = std),
                                          ])

    transformations_valid = transforms.Compose([transforms.Resize(330),
                                                transforms.CenterCrop(300),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean = mean, std = std),
                                                ])

    # Create train dataloader
    dataset_train = PascalVOC_Dataset(data_dir,
                                      year='2009',
                                      image_set='train',
                                      download=download_data,
                                      transform=transformations,
                                      target_transform=utils.encode_labels)

    train_loader = DataLoader(dataset_train, batch_size=batch_size, num_workers=4, shuffle=True)

    # Create validation dataloader
    dataset_valid = PascalVOC_Dataset(data_dir,
                                      year='2009',
                                      image_set='val',
                                      download=download_data,
                                      transform=transformations_valid,
                                      target_transform=utils.encode_labels)

    valid_loader = DataLoader(dataset_valid, batch_size=batch_size, num_workers=4)

    # Load the best weights before testing
    weights_file_path =  os.path.join(model_dir, "model-{}.pth".format(num))
    if os.path.isfile(weights_file_path):
        print("Loading best weights")
        model.load_state_dict(torch.load(weights_file_path))


    log_file = open(os.path.join(model_dir, "log-{}.txt".format(num)), "w+")
    log_file.write("----------Experiment {} - {}-----------\n".format(num, model_name))
    log_file.write("transformations == {}\n".format(transformations.__str__()))
    trn_hist, val_hist = train_model(model, device, optimizer, scheduler, train_loader, valid_loader, model_dir, num, epochs, log_file)
    torch.cuda.empty_cache()

    utils.plot_history(trn_hist[0], val_hist[0], "Loss", os.path.join(model_dir, "loss-{}".format(num)))
    utils.plot_history(trn_hist[1], val_hist[1], "Accuracy", os.path.join(model_dir, "accuracy-{}".format(num)))
    log_file.close()

    #---------------Test your model here---------------------------------------
    # Load the best weights before testing
    print("Evaluating model on test set")
    print("Loading best weights")

    model.load_state_dict(torch.load(weights_file_path))
    transformations_test = transforms.Compose([transforms.Resize(330),
                                               transforms.FiveCrop(300),
                                               transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                               transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean = mean, std = std)(crop) for crop in crops])),
                                               ])


    dataset_test = PascalVOC_Dataset(data_dir,
                                     year='2019',
                                     image_set='val',
                                     download=download_data,
                                     transform=transformations_test,
                                     target_transform=utils.encode_labels)


    test_loader = DataLoader(dataset_test, batch_size=int(batch_size/5), num_workers=0, shuffle=False)

    if save_results:
        loss, ap, scores, gt = test(model, device, test_loader, returnAllScores=True)

        gt_path, scores_path, scores_with_gt_path = os.path.join(model_dir, "gt-{}.csv".format(num)), os.path.join(model_dir, "scores-{}.csv".format(num)), os.path.join(model_dir, "scores_wth_gt-{}.csv".format(num))

        utils.save_results(test_loader.dataset.images, gt, utils.object_categories, gt_path)
        utils.save_results(test_loader.dataset.images, scores, utils.object_categories, scores_path)
        utils.append_gt(gt_path, scores_path, scores_with_gt_path)

        utils.get_classification_accuracy(gt_path, scores_path, os.path.join(model_dir, "clf_vs_threshold-{}.png".format(num)))

        return loss, ap

    else:
        loss, ap= test(model, device, test_loader, returnAllScores=False)

        return loss, ap


if __name__ == '__main__':
   main('/home/your_dir/dataset/na_experiment', "vgg16", num=1, lr = [1.5e-4, 5e-2], epochs = 50, batch_size=16,
        download_data=False,
        save_results=True)