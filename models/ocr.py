import torch.nn as nn
import torch

from .decoder import AttentionDecoder
from .encoder import CNN


class Ocr(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = CNN(**config['encoder'])
        self.decoder = AttentionDecoder(**config['decoder'])
        
        self.encoder.apply(self._weights_init)
        self.decoder.apply(self._weights_init)
        
        self.local_rank = torch.distributed.get_rank()
        self.max_length = config['max_length']

    def forward(self, images, teach_forcing, labels=None):
        bsize = images.size(0)
        encoder_out = self.encoder(images)  # (57, 4, 256)
        
        decoder_hidden = self.decoder.init_hidden(bsize).cuda(self.local_rank)
        outputs = []
        if teach_forcing:
            for decoder_in in labels[:-1]:  # decoder_in (N,)
                decoder_out, decoder_hidden, decoder_attn = self.decoder(
                    decoder_in, decoder_hidden, encoder_out
                )
                outputs.append(decoder_out)  # (N, 39)
        else:
            decoder_in = torch.zeros(bsize).long().cuda(self.local_rank)
            if labels is not None:  # For training
                for _ in range(1, labels.size(0)):
                    decoder_out, decoder_hidden, decoder_attn = self.decoder(
                        decoder_in, decoder_hidden, encoder_out
                    )
                    decoder_in = torch.argmax(decoder_out, dim=1)
                    outputs.append(decoder_out)
            else:  # For inference
                for _ in range(self.max_length):
                    decoder_out, decoder_hidden, decoder_attn = self.decoder(
                        decoder_in, decoder_hidden, encoder_out
                    )
                    decoder_in = torch.argmax(decoder_out, dim=1)
                    pred = decoder_in.squeeze()  # scalar tensor
                    outputs.append(pred)
                    if pred == 1:  # EOS
                        break
        return outputs
    
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