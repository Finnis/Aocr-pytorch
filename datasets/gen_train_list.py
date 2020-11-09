#! /usr/bin/env python

import os

images_dir = 'images'

image_list = []
for fname in os.listdir(images_dir):
    label = fname.split('_')[1].replace('.jpg', '')
    line = images_dir + '/' + fname + ' ' + label + '\n'
    image_list.append(line)

with open('train_list.txt', 'w') as f:
    f.writelines(image_list)
