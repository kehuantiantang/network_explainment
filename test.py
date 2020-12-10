# coding=utf-8
# from keras.models import Sequential
# from keras.layers import Dense
# import numpy as np
#
# from utils.utils import load_pkl
#
# Xtrain = load_pkl('/home/khtt/code/network_explainment/vector/train_vec.pkl')
# ytrain = np.array([0 for _ in range(len(Xtrain) // 2)] + [1 for _ in range(len(Xtrain) // 2)])
#
# Xtest = load_pkl('/home/khtt/code/network_explainment/vector/test_vec.pkl')
# ytest = np.array([0 for _ in range(len(Xtest) // 2)] + [1 for _ in range(len(Xtest) // 2)])
#
# n_words = Xtest.shape[1]
# # define network
# model = Sequential()
# model.add(Dense(50, input_shape=(n_words,), activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# # compile network
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# # fit network
# model.fit(Xtrain, ytrain, epochs=50, verbose=2, batch_size = 2000)
# # for i in range(50):
# #     model.train_on_batch(Xtrain, ytrain, epochs=50, verbose=2)
# # evaluate
# loss, acc = model.evaluate(Xtest, ytest, verbose=0, batch_size = 2000)
# print('Test Accuracy: %f' % (acc*100))



import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
word_to_ix = {'hello': 0, 'world': 1}
embeds = nn.Embedding(2, 5)
hello_idx = torch.LongTensor([[word_to_ix['hello'], word_to_ix['hello'], word_to_ix['hello'], word_to_ix['hello'], word_to_ix['hello'], word_to_ix['hello']]])
hello_idx = Variable(hello_idx)
hello_embed = embeds(hello_idx)
print(hello_embed)