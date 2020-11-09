from imgaug.augmenters.arithmetic import Dropout
from numpy.core.fromnumeric import squeeze
import torch.nn as nn


class BidirectionalLSTM(nn.Module):
    def __init__(self, in_channels, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(in_channels, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        x = self.embedding(t_rec)  # [T * b, nOut]
        x = x.view(T, b, -1)

        return x


class CNN(nn.Module):
    '''
        CNN+BiLstm做特征提取
    '''
    def __init__(self, in_channels, num_hidden, drop_rate=0.5):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),  # (N, 64, 32, 224)
            nn.ReLU(True),
            nn.MaxPool2d(2, 2), # (N, 64, 16, 112)
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2), # (N, 128, 8, 56)
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1), (0, 0)), # (N, 256, 4, 56)
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1), (0, 0)), # (N, 512, 2, 56)
            nn.ZeroPad2d((0, 1, 1, 1)),  #(left, right, up, down)
            nn.Conv2d(512, 512, 2, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d((3, 1), (1, 1), (0, 0)),  #(N, 512, 1, 56)
        )
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, num_hidden, num_hidden),
            BidirectionalLSTM(num_hidden, num_hidden, num_hidden)
        )
        self.dropout = nn.Dropout(drop_rate, inplace=True)

        self.apply(self._weights_init)

    def forward(self, x):
        x = self.cnn(x)
        #x = self.dropout(x)  # TODO do dropout in channels
        x = x.squeeze(2)
        x = x.permute(2, 0, 1)  # (w, N, c)

        # rnn features calculate
        x = self.rnn(x)  # seq * batch * n_classes// 25 × batchsize × 256（隐藏节点个数）
        
        return x
    
    def _weights_init(self, model):
        # Official init from torch repo.
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)