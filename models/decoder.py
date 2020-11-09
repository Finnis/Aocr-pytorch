import torch.nn as nn
import torch
import torch.nn.functional as F


class AttentionDecoder(nn.Module):
    """
        采用seq to seq模型，修改注意力权重的计算方式
    """
    def __init__(self, num_hidden, output_size, dropout_rate=0.1):
        super().__init__()
        self.num_hidden = num_hidden
        self.output_size = output_size

        self.embedding = nn.Embedding(output_size, num_hidden)
        self.attn_combine = nn.Linear(num_hidden * 2, num_hidden)
        self.dropout = nn.Dropout(dropout_rate)
        self.gru = nn.GRU(num_hidden, num_hidden)
        self.out = nn.Linear(num_hidden, output_size)

        # test
        self.vat = nn.Linear(num_hidden, 1)

        self.softmax = nn.Softmax(dim=2)
        self.log_softmax = nn.LogSoftmax(dim=1)

        self.apply(self._weights_init)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)  # 前一次的输出进行词嵌入
        embedded = self.dropout(embedded)

        # test
        batch_size = encoder_outputs.size(1)
        alpha = hidden + encoder_outputs  # (25, batchsize, 256)
        alpha = alpha.view(-1, alpha.size(-1))
        attn_weights = self.vat(torch.tanh(alpha))
        attn_weights = attn_weights.view(-1, 1, batch_size).permute((2, 1, 0))
        attn_weights = self.softmax(attn_weights)
        attn_applied = torch.matmul(attn_weights, encoder_outputs.permute((1, 0, 2)))  # 矩阵乘法，bmm（8×1×56，8×56×256）=8×1×256
        output = torch.cat([embedded, attn_applied.squeeze(1)], 1)  # 上一次的输出和attention feature，做一个线性+GRU
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.log_softmax(self.out(output.squeeze(0)))  # 最后输出一个概率
        
        return output, hidden, attn_weights

    def init_hidden(self, batch_size):
        result = torch.zeros(1, batch_size, self.num_hidden)

        return result
       
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