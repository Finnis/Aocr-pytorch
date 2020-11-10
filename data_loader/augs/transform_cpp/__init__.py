import example.example as ep
import random
import numpy as np
import cv2


def shift_transform_cpp(im):
    if random.random() < 0.3:
        return im
    d = random.randint(0, 10)
    di = random.choice([-2, -1, 1, 2])
    im = ep.shift_cpp(im, d, di)

    return im.astype(np.float32)


def sp_noise_cpp(im):
    im = ep.sp_noise_cpp(im, prob=0.002)
    return im.astype(np.float32)


def gaussian_noise_cpp(im):
    im = ep.gaussian_noise_cpp(im, u=0., v=0.5)

    return im[..., np.newaxis].astype(np.float32)


def elastic_transform_cpp(im):
    if random.random() < 0.5:
        return im
    im = ep.elastic_cpp(im, sigma=7., alpha=15., bNorm=True)

    return im[..., np.newaxis].astype(np.float32)


if __name__ == "__main__":
    import os
    for fn in os.listdir('../datasets/images'):
        print(fn)
        im = cv2.imread(os.path.join('../datasets/images', fn))[:, :, :1].astype(np.float32)
        # print(im.shape)
        #im = gaussian_noise(im)
        # print(im.shape)
        im = shift_transform_cpp(im)
        im = elastic_transform_cpp(im)
        
        #im = sp_noise(im)
        cv2.imwrite('after.jpg', im)
        input()

