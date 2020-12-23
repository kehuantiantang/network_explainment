# coding=utf-8
import torchvision
from PIL import Image
from skimage.segmentation import mark_boundaries
from torchvision.models import resnet18, vgg16
from torchvision.transforms import transforms
import numpy as np
from LRP._core.LRP import LRP
from LRP._core.lrp_rules import Alpha1_Beta0_Rule
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2.cv2
from LRP1.innvestigator import InnvestigateModel
from utils.img_utils import toconv, imgnet_classes, newlayer, heatmap, pascal_classes

device = torch.device("cpu")
def text_lrp():
    pass


# def img_lrp(img_path, model_path):
#     mean=[0.457342265910642, 0.4387686270106377, 0.4073427106250871]
#     std=[0.26753769276329037, 0.2638145880487105, 0.2776826934044154]
#     trans = transforms.Compose([transforms.Resize(330),
#                         transforms.CenterCrop(300),
#                         transforms.ToTensor(),
#                         transforms.Normalize(mean = mean, std = std),
#                         ])
#     img = Image.open(img_path).convert('RGB')
#     img_array = np.array(img)
#     input = torch.unsqueeze(trans(img), 0).to(device)
#
#     model = vgg16()
#     # model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
#     # model.avgpool = torch.nn.MaxPool2d((10, 10), stride=1)
#     # num_ftrs = model.fc.in_features
#     # model.fc = torch.nn.Linear(num_ftrs, 20)
#
#     model.avgpool = torch.nn.MaxPool2d((9, 9))
#     classifier = nn.Sequential(
#         nn.Linear(512 * 1 * 1, 4096),
#         nn.ReLU(True),
#         nn.Dropout(),
#         nn.Linear(4096, 4096),
#         nn.ReLU(True),
#         nn.Dropout(),
#         nn.Linear(4096, 20),
#     )
#     model.classifier = classifier
#     model.load_state_dict(torch.load(model_path))
#     model.to(device)
#
#
#     # # Convert to innvestigate model
#     inn_model = InnvestigateModel(model, lrp_exponent=2,
#                                   method="e-rule",
#                                   beta=.5)
#     model_prediction, heatmap = inn_model.innvestigate(in_tensor=input)
#     heatmap = heatmap.permute(0, 2, 3, 1).numpy()[0]
#     print(model_prediction)
#     print(heatmap.shape)
#     print(heatmap)
#
#     plt.figure()
#     plt.subplot(1, 2, 1)
#     plt.imshow(heatmap * 100, cmap=plt.cm.jet)
#     plt.colorbar()
#     plt.subplot(1, 2, 2)
#     plt.imshow(img_array)
#
#     plt.tight_layout()
#     plt.title('1')
#     plt.savefig('./test.jpg', dpi = 200)

    # rules = [Alpha1_Beta0_Rule()]
    # lrp = LRP(model, rules)
    # # input = torch.randn(3, 3, 32, 32)
    # mask = lrp.attribute(input, target=5)
    # mask[mask != 0] = 1
    # mask[mask == 0] = 2
    # print(mask)
    #
    # img_boundry1 = mark_boundaries(np.array(img)/255.0, mask)
    # plt.figure()
    # plt.imshow(img_boundry1)
    # plt.savefig('./test.jpg', dpi = 200)

# def img_lrp(img_path, model_path, target_index):
#     # img = np.array(cv2.imread(img_path))[...,::-1]/255.0
#     # mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(1,-1,1,1)
#     # std  = torch.Tensor([0.229, 0.224, 0.225]).reshape(1,-1,1,1)
#
#     mean=[0.457342265910642, 0.4387686270106377, 0.4073427106250871]
#     std=[0.26753769276329037, 0.2638145880487105, 0.2776826934044154]
#     trans = transforms.Compose([transforms.Resize(330),
#                                 transforms.CenterCrop(300),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize(mean = mean, std = std),
#                                 ])
#     img = Image.open(img_path).convert('RGB')
#     img_array = np.array(img)
#     X = torch.unsqueeze(trans(img), 0).to(device)
#
#
#
#     # X = (torch.FloatTensor(img[np.newaxis].transpose([0,3,1,2])*1) - mean) / std
#     model = torchvision.models.vgg16(pretrained=True)
#
#     model.eval()
#     layers = list(model._modules['features']) + toconv(list(model._modules['classifier']))
#     L = len(layers)
#     A = [X]+[None]*L
#     for l in range(L): A[l+1] = layers[l].forward(A[l])
#     scores = np.array(A[-1].data.view(-1))
#     ind = np.argsort(-scores)
#     for i in ind[:10]:
#         print('%20s (%3d): %6.3f' % (imgnet_classes[i][:20], i, scores[i]))
#
#     T = torch.FloatTensor((1.0*(np.arange(1000)==483).reshape([1,1000,1,1])))
#     R = [None]*L + [(A[-1]*T).data]
#
#     for l in range(1,L)[::-1]:
#         A[l] = (A[l].data).requires_grad_(True)
#
#         if isinstance(layers[l],torch.nn.MaxPool2d): layers[l] = torch.nn.AvgPool2d(2)
#
#         if isinstance(layers[l],torch.nn.Conv2d) or isinstance(layers[l],torch.nn.AvgPool2d):
#
#             if l <= 16:       rho = lambda p: p + 0.25*p.clamp(min=0); incr = lambda z: z+1e-9
#             if 17 <= l <= 30: rho = lambda p: p;                       incr = lambda z: z+1e-9+0.25*((z**2).mean()**.5).data
#             if l >= 31:       rho = lambda p: p;                       incr = lambda z: z+1e-9
#
#             z = incr(newlayer(layers[l],rho).forward(A[l]))  # step 1
#             s = (R[l+1]/z).data                                    # step 2
#             (z*s).sum().backward(); c = A[l].grad                  # step 3
#             R[l] = (A[l]*c).data                                   # step 4
#
#         else:
#
#             R[l] = R[l+1]
#
#     for i,l in enumerate([31,21,11,1]):
#         heatmap(np.array(R[l][0]).sum(axis=0),0.5*i+1.5,0.5*i+1.5)
#
#     A[0] = (A[0].data).requires_grad_(True)
#
#     lb = (A[0].data*0+(0-mean)/std).requires_grad_(True)
#     hb = (A[0].data*0+(1-mean)/std).requires_grad_(True)
#
#     z = layers[0].forward(A[0]) + 1e-9                                     # step 1 (a)
#     z -= newlayer(layers[0],lambda p: p.clamp(min=0)).forward(lb)    # step 1 (b)
#     z -= newlayer(layers[0],lambda p: p.clamp(max=0)).forward(hb)    # step 1 (c)
#     s = (R[1]/z).data                                                      # step 2
#     (z*s).sum().backward(); c,cp,cm = A[0].grad,lb.grad,hb.grad            # step 3
#     R[0] = (A[0]*c+lb*cp+hb*cm).data                                       # step 4
#
#     heatmap(np.array(R[0][0]).sum(axis=0),3.5,3.5)

def img_lrp(img_path, model_path, target_index, nb_classes = 20):
    # img = np.array(cv2.imread(img_path))[...,::-1]/255.0
    # mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(1,-1,1,1)
    # std  = torch.Tensor([0.229, 0.224, 0.225]).reshape(1,-1,1,1)

    mean=[0.457342265910642, 0.4387686270106377, 0.4073427106250871]
    std=[0.26753769276329037, 0.2638145880487105, 0.2776826934044154]
    trans = transforms.Compose([transforms.Resize(330),
                        transforms.CenterCrop(300),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = mean, std = std),
                        ])
    img = Image.open(img_path).convert('RGB')
    img_array = np.array(img)
    X = torch.unsqueeze(trans(img), 0).to(device)



    # X = (torch.FloatTensor(img[np.newaxis].transpose([0,3,1,2])*1) - mean) / std
    model = torchvision.models.vgg16(pretrained=True)
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
    model.load_state_dict(torch.load(model_path))

    model.eval()
    layers = list(model._modules['features']) + toconv(list(model._modules['classifier']))
    L = len(layers)
    A = [X]+[None]*L
    for l in range(L): A[l+1] = layers[l].forward(A[l])
    scores = np.array(A[-1].data.view(-1))
    ind = np.argsort(-scores)
    for i in ind[:10]:
        if i >= nb_classes: continue;
        print('%20s (%3d): %6.3f' % (pascal_classes[i][:20], i, scores[i]))

    T = torch.FloatTensor((1.0*(np.arange(nb_classes)==target_index).reshape([1,nb_classes,1,1])))
    R = [None]*L + [(A[-1]*T).data]

    for l in range(1,L)[::-1]:
        A[l] = (A[l].data).requires_grad_(True)

        if isinstance(layers[l],torch.nn.MaxPool2d): layers[l] = torch.nn.AvgPool2d(2)

        if isinstance(layers[l],torch.nn.Conv2d) or isinstance(layers[l],torch.nn.AvgPool2d):

            if l <= 16:       rho = lambda p: p + 0.25*p.clamp(min=0); incr = lambda z: z+1e-9
            if 17 <= l <= 30: rho = lambda p: p;                       incr = lambda z: z+1e-9+0.25*((z**2).mean()**.5).data
            if l >= 31:       rho = lambda p: p;                       incr = lambda z: z+1e-9

            z = incr(newlayer(layers[l],rho).forward(A[l]))  # step 1
            s = (R[l+1]/z).data                                    # step 2
            (z*s).sum().backward(); c = A[l].grad                  # step 3
            R[l] = (A[l]*c).data                                   # step 4

        else:

            R[l] = R[l+1]

    for i,l in enumerate([31,21,11,1]):
        print(i, l)
        heatmap(np.array(R[l][0]).sum(axis=0),0.5*i+1.5,0.5*i+1.5)

    A[0] = (A[0].data).requires_grad_(True)


    mean = torch.Tensor(mean).reshape(1,-1,1,1)
    std  = torch.Tensor(std).reshape(1,-1,1,1)

    lb = (A[0].data*0+(0-mean)/std).requires_grad_(True)
    hb = (A[0].data*0+(1-mean)/std).requires_grad_(True)

    z = layers[0].forward(A[0]) + 1e-9                                     # step 1 (a)
    z -= newlayer(layers[0],lambda p: p.clamp(min=0)).forward(lb)    # step 1 (b)
    z -= newlayer(layers[0],lambda p: p.clamp(max=0)).forward(hb)    # step 1 (c)
    s = (R[1]/z).data                                                      # step 2
    (z*s).sum().backward(); c,cp,cm = A[0].grad,lb.grad,hb.grad            # step 3
    R[0] = (A[0]*c+lb*cp+hb*cm).data                                       # step 4

    heatmap(np.array(R[0][0]).sum(axis=0),3.5,3.5)

if __name__ == '__main__':
    img_lrp('/home/khtt/code/LCFCN/datasets/download/pascal/VOCdevkit/VOC2007/JPEGImages/006223.jpg',
            '/home/khtt/code/network_explainment/vector/img/vgg16/model-1.pth', 6)