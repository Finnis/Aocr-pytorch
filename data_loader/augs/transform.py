import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import cv2
import random


def elastic_transform(image):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    if random.random() < 0.2:
        return image
    alpha, sigma, random_state = 200, random.choice([4, 5, 6, 7]), np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    #dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    #print(x.shape)
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image.reshape(image.shape)


def shift_transform(im):
    if random.random() < 0.5:
        return im
    d, di = random.randint(0, 7), random.choice([-2, -1, 1, 2])
    if d <= 0:
        return im
    h, w, _ = im.shape
    r = ((w/2)**2 + d**2)/(2*d)
    if di == -1:
        s = lambda x: (np.sqrt(r**2 - (x-w/2)**2) + d - r).astype(int)
    elif di == 1:
        s = lambda x: (d-(np.sqrt(r**2 - (x-w/2)**2) + d - r)).astype(int)
    elif di == -2:
        s = lambda x: np.floor(d - d/w*x).astype(int)
    elif di == 2:
        s = lambda x: np.floor(d/w*x).astype(int)

    pad = np.zeros((d, w, 1), dtype=np.float32)
    im = np.concatenate((im, pad), 0)   
    for i in range(w):
        im[s(i):h+s(i), i] = im[:h, i]
        im[:s(i), i] = 0
    im = np.expand_dims(cv2.resize(im, (w, h)), axis=2)
    return im


def sp_noise(image):
    '''添加椒盐噪声
    prob:噪声比例 
    '''
    prob = 0.002
    output = np.zeros(image.shape, np.float32)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def gasuss_noise(image, mean=0, var=0.001):
    '''添加高斯噪声
    mean : 均值 
    var : 方差
    '''
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var**0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    
    return out.astype(np.float32)


if __name__ == "__main__":

    im = cv2.imread('/home/finnis/Workspace/Work/ocr/datasets/images/5296_WHSU5201750_201901110930464.jpg', 0)
    im = np.expand_dims(im, 2)
    for i in range(100):
        imx = elastic_transform(im)
        imx = shift_transform(imx)
        #imx = sp_noise(imx)
        cv2.imwrite(f'test.jpg', imx)
        input()
