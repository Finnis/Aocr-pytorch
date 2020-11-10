from torch.utils.data import Dataset
import cv2
import os
import numpy as np
import torch

from utils import LabelConverter, logger
from . import augs


class LatinDataset(Dataset):
    def __init__(self, list_file, num_channels=1, data_aug=[]):
        self.image_list, self.label_list = [], []
        data_root = os.path.dirname(list_file)
        with open(list_file) as f:
            lines = f.readlines()
        for line in lines:
            img, label = line.strip().split(' ')
            self.image_list.append(os.path.join(data_root, img))
            self.label_list.append(label)

        self.augs = [augs.elastic_transform, augs.shift_transform_cpp]

        if num_channels == 1:
            self.flag = 0
        elif num_channels == 3:
            self.flag = 1
        else:
            raise ValueError
        
        logger.info(f'Total {len(self.label_list)} images found.')

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        img = cv2.imread(self.image_list[index], self.flag).astype(np.float32)
        if not self.flag:
            img = img[..., np.newaxis]
        label = self.label_list[index]

        for aug in self.augs:
            img = aug(img)

        img = img / 127.5 - 1.
        return img, label


class Collate(object):
    def __init__(self, max_h=32, max_w=224, alphabet=None):
        self.max_h = max_h
        self.max_w = max_w
        self._convert_labels = LabelConverter(alphabet)

    def __call__(self, batch):
        images, labels = zip(*batch)
        images = self._resize_or_pad_imgs(images)
        images = torch.from_numpy(images)
        images = images.permute([0, 3, 1, 2])

        labels = self._convert_labels.encode(labels)
        labels = torch.from_numpy(labels)  # (max_length, N)
        
        return images, labels
    
    def _resize_or_pad_imgs(self, images):
        # Resize image
        max_w = 0
        imgs1 = []
        # Make sure image height == self.max_h
        for img in images:
            h, w = img.shape[:2]
            if h < self.max_h:
                r = self.max_h - h
                p, q = r // 2, r % 2
                img = np.pad(img, [(p, p+q), (0, 0), (0, 0)], mode='constant', constant_values=0)
            elif h > self.max_h:
                w = int(self.max_h / h * w)
                img = cv2.resize(img, (w, self.max_h))
            
            max_w = max(max_w, w)
            imgs1.append(img)
        
        # Make sure width of images are equal, and <= self.max_w
        max_w = min(max_w, self.max_w)
        images = []
        for img in imgs1:
            w = img.shape[1]
            if w < max_w:
                r = max_w - w
                img = np.pad(img, [(0, 0), (0, r), (0, 0)], mode='constant', constant_values=0)
            elif w > max_w:
                img = cv2.resize(img, (self.max_w, self.max_h))
            
            images.append(img)

        return np.stack(images, axis=0)
