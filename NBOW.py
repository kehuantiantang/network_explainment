# coding=utf-8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class NBOW(nn.Module):
    def __init__(self, max_features, embedding_dim):
        super(NBOW, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(max_features * embedding_dim, 50, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(50,2, bias=True),
        )

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.constant_(m.weight, 0.1)

    def forward(self, inputs):
        return self.model(inputs)


class NBOW_CONV(nn.Module):
    '''convolutional text classification'''
    def __init__(self, max_seq_length, embedding_dim, out_dim = 128, stride = 1):
        super(NBOW_CONV, self).__init__()

        # Parameters regarding text preprocessing
        self.seq_len = max_seq_length
        self.embedding_size = embedding_dim

        # Dropout definition
        # self.dropout = nn.Dropout(0.25)

        # CNN parameters definition
        # Kernel sizes
        self.kernel_1 = 1
        self.kernel_2 = 2
        self.kernel_3 = 3

        # Output size for each convolution
        self.out_dim = out_dim
        # Number of strides for each convolution
        self.stride = stride

        # Embedding layer definition

        # Convolution layers definition
        self.conv_1 = nn.Conv1d(self.seq_len, self.out_dim, self.kernel_1, self.stride)
        self.conv_2 = nn.Conv1d(self.seq_len, self.out_dim, self.kernel_2, self.stride)
        self.conv_3 = nn.Conv1d(self.seq_len, self.out_dim, self.kernel_3, self.stride)

        # Max pooling layers definition
        self.pool_1 = nn.MaxPool1d(self.kernel_1, self.stride)
        self.pool_2 = nn.MaxPool1d(self.kernel_2, self.stride)
        self.pool_3 = nn.MaxPool1d(self.kernel_3, self.stride)

        # Fully connected layer definition
        self.fc = nn.Linear(self.in_features_fc(), 2)
        # self.fc = nn.Sequential(nn.Linear(self.in_features_fc(), 50, bias=True), nn.Sigmoid(), nn.Linear(50, 2, bias=True))



    def in_features_fc(self):
        '''Calculates the number of output features after Convolution + Max pooling

        Convolved_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1
        Pooled_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1

        source: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        '''
        # Calcualte size of convolved/pooled features for convolution_1/max_pooling_1 features
        out_conv_1 = ((self.embedding_size - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
        out_conv_1 = math.floor(out_conv_1)
        out_pool_1 = ((out_conv_1 - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
        out_pool_1 = math.floor(out_pool_1)

        # Calcualte size of convolved/pooled features for convolution_2/max_pooling_2 features
        out_conv_2 = ((self.embedding_size - 1 * (self.kernel_2 - 1) - 1) / self.stride) + 1
        out_conv_2 = math.floor(out_conv_2)
        out_pool_2 = ((out_conv_2 - 1 * (self.kernel_2 - 1) - 1) / self.stride) + 1
        out_pool_2 = math.floor(out_pool_2)

        # Calcualte size of convolved/pooled features for convolution_3/max_pooling_3 features
        out_conv_3 = ((self.embedding_size - 1 * (self.kernel_3 - 1) - 1) / self.stride) + 1
        out_conv_3 = math.floor(out_conv_3)
        out_pool_3 = ((out_conv_3 - 1 * (self.kernel_3 - 1) - 1) / self.stride) + 1
        out_pool_3 = math.floor(out_pool_3)

        # Returns "flattened" vector (input for fully connected layer)
        return (out_pool_1 + out_pool_2 + out_pool_3) * self.out_dim

    def forward(self, inputs):

        # Convolution layer 1 is applied
        x1 = self.conv_1(inputs)
        x1 = F.relu(x1)
        x1 = self.pool_1(x1)

        # Convolution layer 2 is applied
        x2 = self.conv_2(inputs)
        x2 = F.relu(x2)
        x2 = self.pool_2(x2)

        # Convolution layer 3 is applied
        x3 = self.conv_3(inputs)
        x3 = F.relu(x3)
        x3 = self.pool_3(x3)


        # The output of each convolutional layer is concatenated into a unique vector
        union = torch.cat((x1, x2, x3), 2)
        union = union.reshape(union.size(0), -1)

        # The "flattened" vector is passed through a fully connected layer
        out = self.fc(union)

        return out


if __name__ == '__main__':
    a = NBOW_CONV(max_seq_length= 100, embedding_dim = 300)
    input = torch.rand(10, 100, 300)
    output = a(input)
    print(output.size())