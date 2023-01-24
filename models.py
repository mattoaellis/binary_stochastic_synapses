
from torch import nn
from stochastic_synapse import *


class perceptron(nn.Module):
    def __init__(self, Nin, Nout, bias=True):
        super().__init__()
        self.lin = nn.Linear(Nin, Nout, bias)

    def forward(self, x):
        return (self.lin(x))



class Mnist_multi_stoch_synapse(nn.Module):
    def __init__(self, Nin, Nout, Nsynapse, bias=None, wire_mean=(0.0, 1.0, 1.0), wire_stdev=(0.0, 0.0, 0.0)):
        super().__init__()
        self.lin = Stochastic_Linear_Synapse(Nin, Nout, Nsynapse, bias=bias, wire_mean=wire_mean, wire_stdev=wire_stdev)

    def set_stochastic(self, val):
        self.lin.set_stochastic(val)

    def forward(self, xb):
        return self.lin(xb)

    def mean_forward(self, xb):
        return self.lin.mean_forward(xb)

    def log_prob(self, xb, yb):
        return self.lin.log_prob(xb, yb)



class Mnist_multi_stoch_MF_synapse(nn.Module):
    def __init__(self, Nin, Nout, Nsynapse):
        super().__init__()
        self.lin = Stochastic_MF_Linear_Synapse(Nin, Nout, Nsynapse)

    def forward(self, xb):
        return self.lin(xb)
