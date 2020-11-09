#! /usr/bin/env python
import os
import torch
import yaml
import cv2
from tqdm import tqdm
import numpy as np
import argparse

from models.ocr import Ocr
from utils import logger, LabelConverter
from models.decoder import AttentionDecoder
from models.encoder import CNN


class Prediction(object):
    def __init__(self, args):
        config = yaml.load(open(args.config_file), Loader=yaml.FullLoader)
        torch.backends.cudnn.benchmark = args.use_benchmark

        self.encoder = CNN(**config['arch']['encoder'])
        self.decoder = AttentionDecoder(**config['arch']['decoder'])
        self.num_hidden = self.decoder.num_hidden
        
        if os.path.isfile(args.model_path):
            model_path = args.model_path
        else:
            model_path = os.path.join(args.model_path, sorted(os.listdir(args.model_path))[-1])
        ckpt = torch.load(model_path, map_location=torch.device('cpu'))
        self.encoder.load_state_dict(ckpt['encoder'])
        self.decoder.load_state_dict(ckpt['decoder'])
        logger.info(f'Model loaded from {model_path}')
        self.encoder = self.encoder.cuda()
        self.decoder = self.decoder.cuda()

        self.label2text = LabelConverter()
        self.max_h = config['dataset']['train']['img_size']['max_h']
        self.max_w = config['dataset']['train']['img_size']['max_w']
        self.max_length = args.max_pred_length
        
    def predict(self, imgs_path):
        imgs_list = sorted(os.listdir(imgs_path))
        logger.info(f'Find {len(imgs_list)} images in {imgs_path}')
        results = []
        for name in tqdm(imgs_list):
            text = self._predict(os.path.join(imgs_path, name))
            out = name + ' ' + ''.join(text) + '\n'
            results.append(out)
        with open('results.txt', 'w') as f:
            f.writelines(results)
    
    def _predict(self, img_path):
        im = cv2.imread(img_path, 0).astype(np.float32)
        im = self._preprocess_img(im)
        im = im.cuda()
        encoder_out = self.encoder(im)  # (56, 4, 256)

        pred_labels = []
        decoder_in = torch.zeros(1).long().cuda()
        decoder_hidden = torch.zeros(1, 1, self.num_hidden).cuda()
        for _ in range(self.max_length):
            decoder_out, decoder_hidden, decoder_attn = self.decoder(
                decoder_in, decoder_hidden, encoder_out
            )
            decoder_in = torch.argmax(decoder_out, dim=1)
            pred = decoder_in.squeeze()  # scalar tensor
            pred_labels.append(pred)
            if pred == 1:  # EOS
                break
 
        text = self.label2text.decode(pred_labels)

        return text

    def _preprocess_img(self, im):
        im = im / 127.5 - 1.

        h, w = im.shape[:2]
        if h < self.max_h:
            r = self.max_h - h
            p, q = r // 2, r % 2
            im = np.pad(im, [(p, p+q), (0, 0), (0, 0)], mode='constant', constant_values=0)
        elif h > self.max_h:
            w = int(self.max_h / h * w)
            im = cv2.resize(im, (w, self.max_h))
        
        if w > self.max_w:
            im = cv2.resize(im, (self.max_w, self.max_h))
        
        im = im[np.newaxis, np.newaxis, ...]
        im = torch.from_numpy(im)

        return im


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ocr')
    parser.add_argument('-c', '--config_file', default='config.yaml')
    parser.add_argument('--model_path', default='./outputs/checkpoints', type=str)
    parser.add_argument('--image_path', default='./datasets/images', type=str)
    parser.add_argument('--use_benchmark', action='store_false')
    parser.add_argument('--max_pred_length', default=16, type=int)
    args = parser.parse_args()

    pred = Prediction(args)
    if os.path.isfile(args.image_path):
        pred._predict(args.image_path)
    else:
        pred.predict(args.image_path)