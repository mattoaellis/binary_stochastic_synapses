import numpy as np
import scipy.io as spio
import torch
from torch import nn
from sklearn.model_selection import train_test_split

import os


def load_mnist(valid_size = None):
    dirpath = os.path.join(os.path.dirname(__file__), 'mnist')

    train_data = spio.loadmat( os.path.join( dirpath, 'train_data.mat'))
    test_data = spio.loadmat( os.path.join( dirpath, 'test_data.mat'))

    train_imgs   = train_data["x_train"]
    train_labels = train_data["x_train_labs"]
    test_imgs    = test_data["x_test"]
    test_labels  = test_data["x_test_labs"]

    if valid_size is not None:
        x_train, x_valid, y_train, y_valid = train_test_split( train_imgs, train_labels, test_size = valid_size, shuffle=True)
        return map( torch.tensor, (x_train, y_train, x_valid, y_valid, test_imgs, test_labels))
    else:
        return map( torch.tensor, ( train_imgs, train_labels, test_imgs, test_labels))


def load_mnist_test(test_size=None, stratify=False, seed=None):
    dirpath = os.path.join(os.path.dirname(__file__), 'mnist')

    test_data = spio.loadmat( os.path.join( dirpath, 'test_data.mat'))

    test_imgs    = test_data["x_test"]
    test_labels  = test_data["x_test_labs"]

    Ntest = len(test_imgs)
    if (test_size is not None) and (test_size < Ntest):
        fraction = test_size/Ntest
        x_rest, x_test, y_rest, y_test = train_test_split( test_imgs, test_labels,
                test_size = fraction,
                random_state = seed,
                shuffle = stratify,
                stratify = test_labels if stratify else None)
        return map( torch.tensor, (x_test, y_test))
    else:
        return map( torch.tensor, ( test_imgs, test_labels))

def downsample_mnist(x, N, scale=2):

    m = nn.MaxPool2d(scale)

    return m(x.reshape(-1, N, N)).reshape(-1, (N//scale)**2)



