# coding=utf-8
# import nltk
from os import listdir
import numpy as np
from string import punctuation
# nltk.download('stopwords')
# from keras.datasets import imdb
from nltk.corpus import stopwords
import string
from collections import Counter
import os.path as osp
# load doc into memory
from scipy import spatial
from tqdm import tqdm
import os
from fuzzywuzzy import process, fuzz
import pickle

def save_pkl(fname, data):
    with open(fname, "wb") as f:
        pickle.dump(data, f, protocol=4)

def load_pkl(fname):
    with open(fname, "rb") as f:
        return pickle.load(f, encoding='utf8')



class Vocab(object):
    '''clean document and make it to vocabulary'''

    def __init__(self, path):
        # define vocab
        vocab = Counter()
        # add all docs to voca

        for p in os.listdir(path):
            p = osp.join(path, p)
            if osp.isdir(p):
                self.process_docs(p, vocab)

        # print the size of the vocab
        print('Before remove %s'%len(vocab))
        self.vocab = self.remove_few_occurrence(vocab)
        print('After remove %s'%len(self.vocab))

        self.vocab_length = len(self.vocab)
        self.save_list( self.vocab, 'train_vocab.txt')
        print('Make vocabulary finish')


    def load_doc(self, filename):
        # open the file as read only
        file = open(filename, 'r')
        # read all text
        text = file.read()
        # close the file
        file.close()
        return text

    # turn a doc into clean tokens
    def clean_doc(self, doc):
        # split into tokens by white space
        tokens = doc.replace("<br />", " ").replace('-', ' ').replace("'s", '').split()
        tokens = [word.lower() for word in tokens]

        # remove punctuation from each token
        table = str.maketrans('', '', string.punctuation)
        tokens = [w.translate(table) for w in tokens]
        # remove remaining tokens that are not alphabetic
        tokens = [word for word in tokens if word.isalpha()]
        # filter out stop words
        stop_words = set(stopwords.words('english') + ['im', 'isnt', 'others'])
        tokens = [w for w in tokens if not w in stop_words]
        # filter out short tokens
        tokens = [word for word in tokens if len(word) > 1]

        return tokens


    # load doc and add to vocab
    def add_doc_to_vocab(self, filename, vocab):
        # load doc
        doc = self.load_doc(filename)
        # clean doc
        tokens = self.clean_doc(doc)
        # update counts
        vocab.update(tokens)

    # load all docs in a directory
    def process_docs(self, directory, vocab):
        # walk through all files in the folder
        for filename in tqdm(listdir(directory), desc=directory):
            # if filename.startswith('cv9'):
            #     continue

            # create the full path of the file to open
            path = osp.join(directory, filename)
            # add doc to vocab
            self.add_doc_to_vocab(path, vocab)


    def remove_few_occurrence(self, vocab, min_occurane = 5):
        # tokens = {k:c for k,c in vocab.items() if c >= min_occurane}
        # return Counter(tokens)
        tokens = sorted(set([k for k,c in vocab.items() if c >= min_occurane]))
        return tokens



    def save_list(self, lines, filename):
        # convert lines to a single blob of text
        data = '\n'.join(lines)
        # open file
        file = open(filename, 'w')
        # write text
        file.write(data)
        # close file
        file.close()



class MakeReview(object):
    '''
    all the txt to document list
    '''
    def __init__(self, vocab_path, txt_path, is_train):
        # load the vocabulary
        vocab = self.load_doc(vocab_path)
        vocab = vocab.split()
        vocab = set(vocab)
        self.vocab_length = len(vocab)

        self.is_train = is_train
        # load all training reviews
        self.positive_lines = self.process_docs(osp.join(txt_path, 'pos'), vocab)
        self.negative_lines = self.process_docs(osp.join(txt_path, 'neg'), vocab)


        print('Doc length Pos Neg:', len(self.positive_lines), len(self.negative_lines))



    def get_docs(self):
        return self.positive_lines + self.negative_lines

    # load doc into memory
    def load_doc(self, filename):
        # open the file as read only
        file = open(filename, 'r')
        # read all text
        text = file.read()
        # close the file
        file.close()
        return text

    # turn a doc into clean tokens
    def clean_doc(self, doc):
        # split into tokens by white space
        tokens = doc.replace("<br />", " ").replace('-', ' ').replace("'s", '').split()
        tokens = [word.lower() for word in tokens]

        # remove punctuation from each token
        table = str.maketrans('', '', punctuation)
        tokens = [w.translate(table) for w in tokens]
        # remove remaining tokens that are not alphabetic
        tokens = [word for word in tokens if word.isalpha()]
        # filter out stop words
        stop_words = set(stopwords.words('english') + ['im', 'isnt', 'others'])
        tokens = [w for w in tokens if not w in stop_words]
        # filter out short tokens
        tokens = [word for word in tokens if len(word) > 1]

        # lower the tokens
        return tokens

    # load doc, clean and return line of tokens
    def doc_to_line(self, filename, vocab):
        # load the doc
        doc = self.load_doc(filename)
        # clean doc
        tokens = self.clean_doc(doc)
        # filter by vocab
        tokens = [w for w in tokens if w in vocab]
        return ' '.join(tokens)

    # load all docs in a directory
    def process_docs(self, directory, vocab):
        lines = list()
        # walk through all files in the folder
        for filename in listdir(directory):
            # skip any reviews in the test set
            # if self.is_train and filename.startswith('cv9'):
            #     continue
            # if not self.is_train and not filename.startswith('cv9'):
            #     continue
            # create the full path of the file to open
            path = osp.join(directory, filename)
            # load and clean the doc
            line = self.doc_to_line(path, vocab)
            # add to list
            lines.append(line)
        return lines


class Text2Vector(object):
    def __init__(self, vocal_path, docs, save_path, embedding_dict, vector_size = 1):
        self.text_dict = self.load_doc_dict(vocal_path)
        self.docs = docs
        self.embedding_dict = embedding_dict

        self.save_path = save_path
        self.vector_size = vector_size



    def load_doc_dict(self, filename):
        # open the file as read only
        file = open(filename, 'r')
        # read all text
        text = file.read()
        # close the file
        file.close()


        text_dict = {t:i for i, t in  enumerate(text.split('\n'))}
        return text_dict

    def find_closest_embeddings(self, embedding):
        return sorted(embedding.keys(), key=lambda word: spatial.distance.euclidean(self.embedding_dict[word],
                                                                                    embedding))

    def find_closest_word(self, doc, min_score = 60):
        unique_band = list(self.text_dict.keys())

        words = []
        for d in doc.split(" "):
            closed_word, score = process.extract(d, unique_band, scorer=fuzz.token_sort_ratio)[0]
            if score > min_score:
                words.append(score)
        return words

    def word2vector_seq(self):
        doc_lengths, doc  = [], []
        for doc_index, doc in tqdm(enumerate(self.docs), total=len(self.docs)):

            doc = self.find_closest_word(doc)
            for d in doc:
                index = self.text_dict[d]
                value  = self.embedding_dict[d][:self.vector_size]



        if self.save_path:
            os.makedirs(osp.split(self.save_path)[0], exist_ok=True)
            self.save(vector, self.save_path)
        return vector

    def word2vector(self):
        vector = np.zeros((len(self.docs), len(self.text_dict.keys())), dtype = np.float32)

        for doc_index, doc in tqdm(enumerate(self.docs), total=len(self.docs)):
            for d in doc.split(" "):
                index = self.text_dict[d]
                # just keep first feature
                try:
                    d = d.lower()
                    value  = self.embedding_dict[d][:self.vector_size]
                except Exception:
                    # print('%s is not exist in embedding dict' %d)
                    value = 0
                finally:
                    vector[doc_index][index] = value


        if self.save_path:
            os.makedirs(osp.split(self.save_path)[0], exist_ok=True)
            self.save(vector, self.save_path)
        return vector

    def save(self, vector, path):
        save_pkl(path, vector)



if __name__ == '__main__':
    from keras.preprocessing.text import Tokenizer
    # train

    v = Vocab(path = '/home/khtt/dataset/na_experiment/aclImdb', is_test = False)
    mr = MakeReview(vocab_path = './train_vocab.txt', txt_path =
    '/home/khtt/dataset/na_experiment/aclImdb/train', is_train = True)
    docs = mr.get_docs()
    #
    print('Get docs')
    t2v = Text2Vector('./train_vocab.txt', docs, './vector/train_vec.pkl')
    t2v.word2vector()


    # test
    mr = MakeReview(vocab_path = './train_vocab.txt', txt_path = '/home/khtt/dataset/na_experiment/aclImdb/train',
    is_train  = False)
    docs = mr.get_docs()
    print('Get docs')
    t2v = Text2Vector('./train_vocab.txt', docs, './vector/test_vec.pkl')
    t2v.word2vector()


    # tokenizer = Tokenizer(num_words=40000)
    # tokenizer = Tokenizer()
    # tokenizer.fit_on_texts(docs)
    # Xtrain = tokenizer.texts_to_matrix(docs, mode='freq')
    # #
    # print(Xtrain.shape)
    # save_pkl('./vector/train_fre.pkl', Xtrain)
    #
    #
    # mr = MakeReview(vocab_path = './train_vocab.txt', txt_path =
    # '/home/khtt/dataset/na_experiment/aclImdb/test', is_train = False)
    # docs = mr.get_docs()
    # Xtrain = tokenizer.texts_to_matrix(docs, mode='freq')
    # print(Xtrain.shape)
    # save_pkl('./vector/test_fre.pkl', Xtrain)


    #
    #
    # imdb.load_data(num_words=1000)
    # a = glove_vector()
    # print(a)

