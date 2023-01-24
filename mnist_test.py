#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


from data.load_mnist import load_mnist, downsample_mnist
from stochastic_synapse import *
from models import *
from train_model import *

def get_model(Nin, Nsynapse, device, mean_field=False, wire_mean=(0.0, 1.0, 1.0), wire_stdev=(0.0, 0.0, 0.0)):
    print(wire_mean)
    print(wire_stdev)
    Nout = 10
    print(Nsynapse)

    if Nsynapse == -1 :
        model = perceptron(Nin, Nout, bias=True)

    elif Nsynapse > 0 :
            if mean_field:
                model = Mnist_multi_stoch_MF_synapse(Nin, Nout, Nsynapse)
            else:
                model = Mnist_multi_stoch_synapse(Nin, Nout, Nsynapse, bias=True, wire_mean=wire_mean, wire_stdev=wire_stdev)

    return model.to(device)


x_train, y_train, x_valid, y_valid, x_test, y_test = load_mnist(0.2)

x_train = downsample_mnist(x_train, 28, scale=2)
x_valid = downsample_mnist(x_valid, 28, scale=2)
x_test = downsample_mnist(x_test, 28, scale=2)

Npix = int(np.sqrt(x_train.shape[-1]))
Nin = Npix*Npix

x_train = 1.0*(x_train.float() > 0.5)
x_valid = 1.0*(x_valid.float() > 0.5)
x_test =  1.0*(x_test.float() > 0.5)
y_train = y_train.squeeze(dim=1).long() -1
y_valid = y_valid.squeeze(dim=1).long() -1
y_test = y_test.squeeze(dim=1).long() -1

print(x_train.shape, y_train.shape)
print(x_valid.shape, y_valid.shape)
print(x_test.shape, y_test.shape)


Nout = 10


use_gpu = True
dev, [x_train, x_valid, x_test, y_train, y_valid, y_test] = select_device(use_gpu, [x_train, x_valid, x_test, y_train, y_valid, y_test])


bs = 64  # batch size


train_ds = TensorDataset((x_train), y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
valid_ds = TensorDataset( (x_valid), y_valid)
valid_dl = DataLoader( valid_ds, batch_size = bs*2)
test_ds = TensorDataset( (x_test), y_test)
test_dl = DataLoader( test_ds, batch_size = bs*2)


inval = int(sys.argv[1])
Nsy = 2**inval if inval >= 0 else inval
Nhid= int(sys.argv[2])
print(Nsy )


#lr = 0.02  # learning rate
lr = 0.02
epochs = 100
Nmodels = 5

wire_mean = np.array([4.63168, 2.72741, 0.97813])

for i in range(4):
    model = get_model(Nin, Nsy, device=dev, mean_field=False, wire_mean=wire_mean)
    print(model)


    #x0 = model.lin.x0.cpu().detach().numpy()
    #dx = model.lin.dx.cpu().detach().numpy()
    #a = model.lin.a.cpu().detach().numpy()
    #np.savez('Wire_params_start'+str(i), x0=x0, dx=dx, a=a)

    #model.load_state_dict(torch.load("checkpoint.pt"))
    train_err, test_err = run_model(model, lr, epochs, train_dl, valid_dl, test_dl, dev, reinforce=False)
    np.savetxt( str(Nsy)+'BSS_valid_err_'+str(i), train_err)
    np.savetxt( str(Nsy)+'BSS_test_err_'+str(i), test_err)
    print('Test: ', test_err.mean(axis=0))
    torch.save(model.state_dict(), str(Nsy)+'BSS_model_'+str(i)+'.pt')


    weights = model.lin.weight.cpu().detach().numpy()
    probs = model.lin.weight.cpu().detach().numpy()
    np.savez(str(Nsy)+'BSS_model_'+str(i), weights=weights, probs=probs)