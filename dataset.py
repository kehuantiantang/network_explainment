# coding=utf-8
from sklearn import datasets
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.utils import load_pkl, Vocab, MakeReview, Text2Vector
import numpy as np
import os

def glove_vector(path = "/home/khtt/dataset/na_experiment/glove.42B.300d.txt"):
    embedding_dict = {}
    with open(path, 'r', encoding="utf-8") as f:
        for line in tqdm(f, desc = "Glove vector:", total=1917494):
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embedding_dict[word] = vector

    return embedding_dict

class IMDB(Dataset):

    def __init__(self, **kwargs):
        self.context = self.get_context()
        self.y = self.get_y()
        print('X, Y Shape', self.context.shape, len(self.y))

    def get_context(self):
        raise NotImplementedError()

    def get_y(self):
        y = []
        for i in range(len(self.context)):
            if i > len(self.context) // 2:
                y.append(1.0)
            else:
                y.append(0.0)
        return y

    def __getitem__(self, index):
        x = self.context[index]
        y = self.y[index]
        return x, y

    def __len__(self):
        return len(self.context)


class IMDB_PKL(IMDB):
    def __init__(self, pkl_path):
        self.pkl_path = pkl_path
        super(IMDB_PKL, self).__init__()

    def get_context(self):
        context = load_pkl(self.pkl_path)
        print("Load data %s"%self.pkl_path)
        return context

class IMDB_Raw(IMDB):
    def __init__(self, vocab_search_path, embedding_dict, txt_path = '/home/khtt/dataset/na_experiment/aclImdb/%s',
                 is_train = True):

        self.txt_path = txt_path
        self.is_train = is_train
        self.vocab_search_path = vocab_search_path
        self.embedding_dict = embedding_dict
        print("Make dataset %s"%self.txt_path)
        super(IMDB_Raw, self).__init__()


    def get_context(self):
        if not os.path.isfile(self.vocab_search_path):
            v = Vocab(path = self.vocab_search_path)
        else:
            print("Load vocabulary file from %s" %self.vocab_search_path)
        mr = MakeReview(vocab_path = './train_vocab.txt', txt_path =
        self.txt_path, is_train = self.is_train)
        docs = mr.get_docs()
        self.vocab_length = mr.vocab_length

        t2v = Text2Vector('./train_vocab.txt', docs, save_path=None, embedding_dict = self.embedding_dict)
        context = t2v.word2vector()
        return context


if __name__ == '__main__':
    pass