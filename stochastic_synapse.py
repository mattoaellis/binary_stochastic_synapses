import numpy as np
import math
import torch
import torch.nn.functional as F
from torch import nn


class Stochastic_Synapse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, Nsynapses=1):
        prob = (input.unsqueeze(1) * torch.sigmoid(weight))
        #probs = prob.unsqueeze(-1).expand(-1,-1,-1,Nsynapses)
        m = torch.distributions.binomial.Binomial(Nsynapses, prob)
        J = m.sample()/Nsynapses
        #J = 1.0*(probs > torch.rand_like(probs)).sum(-1)/Nsynapses
        Yb = J.sum(axis=-1)

        if bias is not None:
            Yb += bias.unsqueeze(0).expand_as(Yb)
        ctx.save_for_backward(input, weight, bias, J, prob)
        return Yb

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, J, prob = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        grad_input = torch.zeros( (grad_output.shape[0], J.shape[-1]), device=grad_output.device)

        mu = prob.sum(axis=-1)
        sigma = (prob*(1.0 - prob)).sum(axis=-1)
        epsi = (J.sum(axis=-1) - mu) / sigma

        eps =  1 + ((J.sum(axis=-1) - mu)/sigma).unsqueeze(2) * (0.5 - prob)

        if ctx.needs_input_grad[0]:
            #grad_input = (grad_output.unsqueeze(2) * eps * torch.sigmoid(weight).unsqueeze(0)).sum(axis=1)
            grad_input = grad_output.mm( torch.sigmoid(weight) )
            grad_input += ((grad_output * epsi ).unsqueeze(2) * (0.5 - prob) * torch.sigmoid(weight).unsqueeze(0)).sum(axis=1)
            #for i,x in enumerate(grad_output):
            #    grad_input[i] = grad_output[i].unsqueeze(0).mm(J[i])
        if ctx.needs_input_grad[1]:
            dp = torch.sigmoid(weight)*(1.0 - torch.sigmoid(weight))
            #grad_weight = dp*grad_output.t().mm(input)
            grad_weight = dp * (grad_output.unsqueeze(2) *eps* input.unsqueeze(1)).sum(axis=0)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, None

def DW_prob(x, a, dx, x0):
    return 1.0 - a/(1.0 + torch.exp(dx*(x-x0)))

class Stochastic_Linear_Synapse(nn.Module):
    def __init__(self, input_features, output_features, Nsynapses = 1, bias=True, wire_mean=(0.0, 1.0, 1.0), wire_stdev=(0.0, 0.0, 0.0) ):
        super(Stochastic_Linear_Synapse, self).__init__()

        self.input_features = input_features
        self.output_features = output_features
        self.Nsynapses = Nsynapses

        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)

        self.weight.data.zero_()
        if bias is not None:
            self.bias.data.zero_()

        self.wire_mean = wire_mean
        self.wire_stdev = wire_stdev

        self.generate_dw_params()

        self.weight.data = self.x0 + self.weight.data/self.dx

    def set_Nsynapses(self, Nsynapses):
        self.Nsynapses = Nsynapses

    def generate_dw_params(self, sigma_scale=1.0):
        self.x0 = nn.Parameter(self.wire_mean[0] + sigma_scale*self.wire_stdev[0]*torch.randn_like(self.weight))
        self.x0.requires_grad_(False)
        self.dx = nn.Parameter(self.wire_mean[1] + sigma_scale*self.wire_stdev[1]*torch.randn_like(self.weight))
        self.dx.requires_grad_(False)
        self.a = nn.Parameter(self.wire_mean[2] + sigma_scale*self.wire_stdev[2]*torch.randn_like(self.weight))
        self.a.requires_grad_(False)
        self.d = nn.Parameter(1.0 - self.a)
        self.d.requires_grad_(False)


    def forward(self, input):
        prob = (input.unsqueeze(1) * (self.d + self.a*torch.sigmoid( self.dx*(self.weight - self.x0)))).clamp(0.0, 1.0)

        with torch.no_grad():
            m = torch.distributions.binomial.Binomial(self.Nsynapses, prob)
            J = m.sample()/self.Nsynapses
            Yb = J.sum(axis=-1)

        tiny = 1e-4
        mu = prob.sum(axis=-1)
        sigma = torch.sqrt( (prob*(1.0 - prob)).sum(axis=-1)/self.Nsynapses + tiny )
        with torch.no_grad():
            eps = (Yb - mu)/sigma

        res = mu + eps*sigma
        if self.bias is not None:
            res += self.bias.unsqueeze(0).expand_as(res)

        if np.any(np.isnan(self.weight.cpu().detach().numpy())):
            print("Weight nan")
            print(self.weight.cpu().detach().numpy())
            exit()
        if np.any(np.isnan(input.cpu().detach().numpy())):
            print(input.cpu().detach().numpy())
        if np.any(np.isnan(res.cpu().detach().numpy())):
            print(res.cpu().detach().numpy())
            inds = np.where( np.isnan(res.cpu().detach().numpy()))
            print(inds)
            print(Yb[inds])
            print(mu[inds])
            print(sigma)
            print(prob[inds])
            print(eps)
            exit()
        return res

    def log_prob(self, input, value):
        prob = (input.unsqueeze(1) * (self.d + self.a*torch.sigmoid( self.dx*(self.weight - self.x0)))).clamp(0.0, 1.0)
        mu = prob.sum(axis=-1)
        if self.bias is not None:
            mu += self.bias.unsqueeze(0).expand_as(mu)
        sigma = torch.sqrt( (prob*(1.0 - prob)).sum(axis=-1) )
        res =  -((value - mu) ** 2) / (2 * sigma**2) - sigma.log() - math.log(math.sqrt(2 * math.pi))
        return res.sum(axis=-1)

    def get_mean_var(self, input):
        prob = (input.unsqueeze(1) * (self.d + self.a*torch.sigmoid( self.dx*(self.weight - self.x0)))).clamp(0.0, 1.0)
        mu = prob.sum(axis=-1)
        if self.bias is not None:
            mu += self.bias.unsqueeze(0).expand_as(mu)
        sigma =  (prob*(1.0 - prob)).sum(axis=-1)
        return mu, sigma


    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, Nsynapses={}'.format(
            self.input_features, self.output_features, self.bias is not None, self.Nsynapses
        )



class Stochastic_MF_Synapse(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias=None, gI = 0.25):
        J = torch.sigmoid(weight)
        Yb = input.mm(J.T - gI)

        ctx.save_for_backward(input, weight, bias, J)
        if bias is not None:
            Yb += bias.unsqueeze(0).expand_as(Yb)
        return Yb

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, J = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(J)
        if ctx.needs_input_grad[1]:
            dp = torch.sigmoid(weight)*(1.0 - torch.sigmoid(weight))
            grad_weight = dp*grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias


class Stochastic_MF_Linear_Synapse(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(Stochastic_MF_Linear_Synapse, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)

        self.weight.data.zero_()
        if bias is not None:
            self.bias.data.zero_()

    def forward(self, input):
        return Stochastic_MF_Synapse.apply(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )

