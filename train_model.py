#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
import time

from stochastic_synapse import *
from models import *

def select_device(use_cuda=True, data=None):
    dev = torch.device("cuda:0") if (torch.cuda.is_available() and use_cuda) else torch.device("cpu")
    print('# Is CUDA available:', torch.cuda.is_available())
    print('# Using device: ', dev)

    # move all data to device
    if data is not None:
        for i in range(len(data)):
            data[i] = data[i].to(dev)
    return dev, data



def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()


def test_model( model, loss_func, test_dl, noise=None):
        model.eval()
        with torch.no_grad():
            loss = 0.0
            acc = 0
            for xb,yb in test_dl:
                pred = model(xb)
                if noise is not None:
                    pred = pred + torch.randn_like(pred)*noise
                loss += loss_func(pred, yb).item()
                acc += accuracy(pred, yb).item()
        return loss/len(test_dl), acc/len(test_dl)

def reward(out, yb):
    preds = torch.argmax(out, dim=1)
    return ((preds == yb).float())

def train(train_dl, model, loss_func, opt):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    for xb,yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        train_loss += loss.item()
        train_acc += accuracy(pred, yb).item()

        loss.backward()

        opt.step()
        opt.zero_grad()
    return train_loss/len(train_dl), train_acc/len(train_dl)

def train_model( model, opt, loss_func, epochs, train_dl, valid_dl, dev, test_dl=None, Npat=100, verbose=True):
    PATH="checkpoint.pt"
    patience = Npat
    old_valid_loss, old_valid_acc = test_model(model, loss_func, valid_dl)

    torch.save(model.state_dict(), PATH)
    tr = 0.01
    Lbar = 6.0

    res = []
    for epoch in range(epochs):

        train_loss, train_acc = train(train_dl, model, loss_func, opt)

        model.eval()
        with torch.no_grad():
            valid_loss, valid_acc = test_model(model, loss_func, valid_dl)
            test_loss, test_acc = test_model(model, loss_func, test_dl)

        Lbar = (1-tr)*Lbar + tr*train_loss

        if valid_loss > old_valid_loss:
            patience -= 1
        elif valid_loss < old_valid_loss:
            torch.save(model.state_dict(), PATH)
            patience = Npat
            old_valid_loss = valid_loss

        res.append( [epoch+1, train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc, old_valid_loss, patience] )
        if verbose:
            print('{:d}\t{:f}\t{:f}\t{:f}\t{:f}\t{:f}\t{:f}\t{:f}\t{:d}\t{:f}'.format(*res[epoch], Lbar))
        if patience == 0:
            break

    model.load_state_dict(torch.load(PATH))

    return np.array(res)


def run_model(model, _lr, epochs, train_dl, valid_dl, test_dl, dev, verbose=True):
    Npat = 20
    start = time.process_time()

    loss_func = F.cross_entropy
    opt = optim.Adam( model.parameters(), lr=_lr)

    train_err = train_model( model, opt, loss_func, epochs, train_dl, valid_dl, dev, test_dl, Npat, verbose)

    Ntest =16
    test_err = np.zeros((Ntest,2))
    for i in range(Ntest):
        test_err[i] = test_model(model, loss_func, test_dl)
    stop = time.process_time()
    print('Time taken: ', stop-start)
    return train_err, test_err
