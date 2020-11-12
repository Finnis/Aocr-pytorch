import torch.nn as nn
import torch
import torch.nn.functional as F


class AttentionDecoder(nn.Module):
    """
        采用seq to seq模型，修改注意力权重的计算方式
    """
    def __init__(self, num_hidden, output_size, dropout_rate=0.5):
        super().__init__()
        self.num_hidden = num_hidden
        self.output_size = output_size

        self.embedding = nn.Embedding(output_size, num_hidden)
        self.attn_combine = nn.Linear(num_hidden * 2, num_hidden)
        self.dropout = nn.Dropout(dropout_rate)
        self.gru = nn.GRU(num_hidden, num_hidden)
        self.out = nn.Linear(num_hidden, output_size)

        self.vat = nn.Linear(num_hidden, 1)

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.relu = nn.ReLU(True)

        self.apply(self._weights_init)

    def forward(self, x, hidden, encoder_outputs):
        embedded = self.embedding(x)  # (N, num_hidden)
        embedded = self.dropout(embedded)

        alpha = hidden * encoder_outputs  # (w, N, num_hidden)
        attn_weights = self.vat(alpha)  #(w, N, 1)

        # alpha = hidden + encoder_outputs  # (w, N, num_hidden)
        # attn_weights = self.vat(torch.tanh(alpha))  #(w, N, 1)

        attn_weights = F.softmax(attn_weights, dim=0)
        attn_weights = attn_weights.permute([1, 2, 0])  # (N, 1, w)
        encoder_outputs = encoder_outputs.permute([1, 0, 2])  # (N, w, num_hidden) 
        attn_applied = torch.matmul(attn_weights, encoder_outputs)  # (N, 1, num_hidden)
        output = torch.cat([embedded, attn_applied.squeeze(1)], 1)  # (N, num_hidden*2)
        output = self.attn_combine(output).unsqueeze(0)  # (1, N, num_hidden)
        #output = self.relu(output)
        output, hidden = self.gru(output, hidden)  # (1, N, num_hidden)
        output = self.log_softmax(self.out(output.squeeze(0)))  # 最后输出一个概率
        
        return output, hidden, attn_weights
    
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