import torch.nn as nn


class CNN(nn.Module):
    '''
        CNN+BiLstm做特征提取
    '''
    def __init__(self, in_channels, num_hidden, dropout_rate=0.5):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False),  # (N, 64, 32, 224)
            nn.ReLU(True),
            nn.MaxPool2d(2, 2), # (N, 64, 16, 112)
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2), # (N, 128, 8, 56)
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1), (0, 0)), # (N, 256, 4, 56)
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1), (0, 0)), # (N, 512, 2, 56)
            nn.ZeroPad2d((0, 1, 0, 0)),  #(left, right, up, down)
            nn.Conv2d(512, 512, 2, 1, 0, bias=False), #(N, 512, 1, 56)
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )

        self.bi_lstm = nn.LSTM(512, num_hidden, bidirectional=True, num_layers=2)
        self.embedding = nn.Linear(num_hidden * 2, num_hidden)
        self.dropout = nn.Dropout(dropout_rate, inplace=True)

        self.apply(self._weights_init)

    def forward(self, x):
        x = self.cnn(x)  # (N, C, 1, w)
        x = x.squeeze(2)  # (N, C, w)
        x = x.permute((2, 0, 1))  # (w, N, C)
        # rnn features calculate
        x, _ = self.bi_lstm(x)  # (w, N, num_hidden*2)
        x = self.embedding(x)  # (w, N, num_hidden)
        x = self.dropout(x)

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