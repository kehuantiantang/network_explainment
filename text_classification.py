# coding=utf-8

import argparse
import pprint
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from NBOW import NBOW, NBOW_CONV
from dataset import IMDB_PKL, IMDB_Raw, glove_vector, IMDB_Seq
from utils.gpu_utils import auto_select_gpu
from utils.misc import AverageMeter, accuracy
import os.path as osp

import os
os.environ['CUDA_VISIBLE_DEVICES'] = auto_select_gpu()
torch.manual_seed(1)


def build_model(args):
    # model = NBOW(args.max_seq_length, args.embedding_dim)

    model = NBOW_CONV(max_seq_length = args.max_seq_length, embedding_dim = args.embedding_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model


def build_dataset(args):
    # fc
    # embedding_dict = glove_vector()
    # train_dataset, val_dataset = IMDB_Raw(vocab_search_path= args.root_path, txt_path= args.train_path,
    #                                       is_train=True, embedding_dict = embedding_dict), \
    #                              IMDB_Raw(vocab_search_path= args.root_path, txt_path= args.test_path, is_train =
    #                              False, embedding_dict = embedding_dict)
    # train_loader, val_loader = DataLoader(train_dataset, args.batch_size, shuffle= True, num_workers =
    # args.num_workers), DataLoader(val_dataset, args.batch_size, shuffle= False, num_workers =
    # args.num_workers)
    # args.max_seq_length = train_dataset.vocab_length


    # conv
    if not osp.exists('./vector/train_seq_context.pkl'):
        embedding_dict = glove_vector()
    else:
        embedding_dict = {}
    train_dataset, val_dataset = IMDB_Seq(vocab_search_path=args.root_path, txt_path=args.train_path, is_train=True,
                                          embedding_dict=embedding_dict, max_seq_length=args.max_seq_length,
                                          embedding_dim = args.embedding_dim, vector_save_path =
                                          './vector/train_seq.pkl'), IMDB_Seq(
        vocab_search_path=args.root_path, txt_path=args.test_path, is_train=False, embedding_dict=embedding_dict,
        max_seq_length=args.max_seq_length, embedding_dim = args.embedding_dim, vector_save_path =
        './vector/test_seq.pkl')

    train_loader, val_loader = DataLoader(train_dataset, args.batch_size, shuffle= True, num_workers =
    args.num_workers), DataLoader(val_dataset, args.batch_size, shuffle= False, num_workers =
    args.num_workers)

    return train_loader, val_loader


def eval(model, dataloader, loss_fun, args):
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()


    sum_targets, sum_outputs, sum_outputs_score = [], [], []


    dtqdm = tqdm(enumerate(dataloader), total = len(dataloader))
    for batch_index, (x, y) in dtqdm:
        x, y = x.float().cuda(), y.long().cuda()

        output = model(x)
        loss = loss_fun(output, y)

        prec1 = accuracy(output.data, y.data, nb_classes = args.nb_classes)[0]
        top1.update(prec1.data.cpu().numpy(), x.size(0))
        losses.update(loss.data.cpu().numpy(), x.size(0))


        sum_targets.extend(y.data.cpu().numpy().flatten())

        if args.nb_classes == 1:
            output =  output.squeeze(dim=-1)
            sum_outputs.extend((output.data.cpu().numpy() > 0.5).astype(np.float32))

        else:
            sum_outputs.extend(np.argmax(output.data.cpu().numpy(), -1).flatten())

        sum_outputs_score.extend(F.softmax(output.data, dim = -1).cpu().numpy()[:, 0])


    print(classification_report(sum_targets, sum_outputs))

    b = np.array(sum_outputs_score)
    b[b >= 0.5] = 1
    b[b < 0.5] = 0
    acc = accuracy_score(sum_targets, b)
    print(1 - acc)

    return 1 - acc


def train(model, trainloader, valloader, optimizer, loss_fun, args):
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    test_top_acc = -1

    for epoch in range(args.epochs):
        dtqdm = tqdm(enumerate(trainloader), total = len(trainloader))
        for batch_index, (x, y) in dtqdm:
            x, y = x.float().cuda(), y.long().cuda()
            output = model(x)
            loss = loss_fun(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prec1 = accuracy(output.data, y.data, nb_classes = args.nb_classes)[0]
            losses.update(loss.data.cpu().numpy(), x.size(0))
            top1.update(prec1.data.cpu().numpy(), x.size(0))


            info = 'Epoch:{epoch} Loss:{loss:.8f}, Acc:{top1:.4f}'.format(epoch = epoch, loss=losses.avg,
                                                                          top1=top1.avg)
            dtqdm.set_description(info)

        if epoch % args.interval_val == 0:
            current_accuracy = eval(model, valloader, loss_fun, args)
            if current_accuracy > test_top_acc:
                current_accuracy = test_top_acc
                torch.save({'model':model, 'best_accuracy': test_top_acc}, osp.join(args.checkpoint_path,
                                                                                    'best_seq.pt'))



parser = argparse.ArgumentParser(description='PyTorch detection')
parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--batch-size', default=100, type=int)
parser.add_argument('--nb_classes', default=2, type=int)

parser.add_argument('--interval-val', default=5, type=int)
parser.add_argument('--checkpoint-path', default="./vector", type=str, help = 'word2vector vector size')

# dataset
# parser.add_argument('--root-path', default='/home/khtt/dataset/na_experiment/aclImdb/train', type=str)
parser.add_argument('--root-path', default='/home/khtt/code/network_explainment/train_vocab.txt', type=str)
parser.add_argument('--train-path', default='/home/khtt/dataset/na_experiment/aclImdb/train', type=str)
parser.add_argument('--test-path', default='/home/khtt/dataset/na_experiment/aclImdb/test', type=str)

# model config
parser.add_argument('--max-seq-length', default=120, type=int, help = 'vocabulary or sequence length')
parser.add_argument('--embedding-dim', default=300, type=int, help = 'word2vector vector size')


parser.add_argument('--num-workers', default=0, type=int)

args = parser.parse_args()
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(vars(args))

trainloader, valloader = build_dataset(args)
model = build_model(args)
# loss_function = torch.nn.MSELoss()
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

train(model, trainloader, valloader, optimizer, loss_function, args)