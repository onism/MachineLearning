#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
implemented the NAF according to Neural Autoregressive Flows
@author: Junjie
"""


import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Module
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair
from torch.autograd import Variable
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import utils

delta = 1e-6
softplus_ = nn.Softplus()
softplus = lambda x: softplus_(x) + delta
sigmoid_ = nn.Sigmoid()
sigmoid = lambda x: sigmoid_(x) * (1-delta) + 0.5 * delta
sigmoid2 = lambda x: sigmoid(x) * 2.0
logsigmoid = lambda x: -softplus(-x)
logit = lambda x: torch.log
log = lambda x: torch.log(x*1e2)-np.log(1e2)
logit = lambda x: log(x) - log(1-x)
def softmax(x, dim=-1):
    e_x = torch.exp(x - x.max(dim=dim, keepdim=True)[0])
    out = e_x/e_x.sum(dim=dim, keepdim=True)
    return out

class BaseFlow(Module):

    def sample(self, n=1, context=None, **kwargs):
        dim = self.dim
        if isinstance(self.dim, int):
            dim = [dim,]

        spl = Variable(torch.FloatTensor(n,*dim).normal_())
        lgd = Variable(torch.from_numpy(np.zeros(n).astype('float32')))
        if context is None:
            context = Variable(torch.from_numpy(np.ones((n, self.context_dim)).astype('float32')))

        return self.forward((spl, lgd, context))

class SigmoidFlow(BaseFlow):
    """docstring for SigmoidFlow"""
    def __init__(self, num_ds_dim=4):
        super(SigmoidFlow, self).__init__()
        self.num_ds_dim = num_ds_dim
        self.act_a = lambda x: softplus(x)
        self.act_b = lambda x: x
        self.act_w = lambda x: softmax(x, dim=2)

    def forward(self, x, logdet, dsparams):
        ndim = self.num_ds_dim
        a = self.act_a(dsparams[:,:, 0 *ndim:1*ndim]) # d * 1
        b = self.act_b(dsparams[:,:, 1*ndim:2*ndim])# d * 1
        w = self.act_w(dsparams[:,:, 2*ndim:3*ndim])# d * 1
        # equation 8
        pre_sigm = a * x[:,:,None] + b
        sigm = torch.sigmoid(pre_sigm)
        x_pre = torch.sum(w*sigm, dim=2)
        x_pre_clipped = x_pre * (1- delta) + delta*0.5
        xnew = log(x_pre_clipped) - log(1-x_pre_clipped)

        logj = F.log_softmax(w, dim=2) + logsigmoid(pre_sigm) + logsigmoid(-pre_sigm) + log(a) # I do not understand
        logj = utils.log_sum_exp(logj,2).sum(2)
        logdet_ = logj + np.log(1-delta) - \
        (log(x_pre_clipped) + log(-x_pre_clipped+1))
        logdet = logdet_.sum(1) + logdet

        return xnew, logdet



class model_1d(object):
    """docstring for model_1d"""
    def __init__(self, target_energy):
        nd = 24
        self.sf = SigmoidFlow(num_ds_dim=nd)
        self.target_energy = target_energy
        self.params = Parameter(torch.FloatTensor(1,1,3*nd).normal_()) # a, b, w
        self.optim = optim.Adam([self.params,], lr=0.01, betas=(0.9,0.999))


    def sample(self, n):
        spl = Variable(torch.FloatTensor(n,1).normal_())
        lgd = Variable(torch.from_numpy(np.zeros(n).astype('float32')))
        h, logdet = self.sf.forward(spl, lgd, self.params)
        out = sigmoid_(h)*2.0
        logdet = logdet + logsigmoid(h) + logsigmoid(-h) + np.log(2.0)
        return out, logdet

    def train(self):
        total = 2000
        for it in range(total):
            self.optim.zero_grad()
            w = min(1.0, 0.2+it/float((total*0.80)))
            spl, logdet = self.sample(320)
            losses = - w * self.target_energy(spl) - logdet
            loss = losses.mean()
            loss.backward()
            self.optim.step()
            if ((it + 1) % 100) == 0:
                print 'Iteration: [%4d/%4d] loss: %.8f' % \
                    (it+1, total, loss.data[0])

class Sinewave(object):
    """docstring for Sinewave"""
    def __init__(self, a, f, phi):
        self.a = a
        self.f = f
        self.phi = phi

    def evaluate(self, t):
        return self.a * np.sin(2*np.pi*self.f*t + self.phi)

xx = np.linspace(0,2,1000)
fig = plt.figure()
ax = plt.subplot(111)
sw1 = Sinewave(1.0, 0.6, 0.0)
yy1 = sw1.evaluate(xx)
plt.plot(xx,yy1,':')

sw2 = Sinewave(1.0, 1.2, 0.0)
yy2 = sw2.evaluate(xx)
plt.plot(xx,yy2,'--')

sw3 = Sinewave(1.0, 1.8, 0.0)
yy3 = sw3.evaluate(xx)
plt.plot(xx,yy3)

a0 = 1.0
f0 = 0.6
b0 = 0.0

x0 = utils.varify(np.array([[0.0],[5/6.],[10/6.]]).astype('float32'))
y0 = torch.mul(torch.sin(x0*2.0*np.pi*f0+b0), a0)
plt.scatter(x0.data.numpy()[:,0],y0.data.numpy()[:,0], color=(0.8,0.4,0.4),
            marker='x',s=100)

plt.rc('font', family='serif')
plt.title(r'$y(t) = \sin(2\pi f t)$', fontsize=18)
leg = plt.legend(['$f$=0.6','$f$=1.2','$f$=1.8'],
                 loc=4, fontsize=20)


plt.xlabel('t', fontsize=15)
plt.ylabel('y(t)', fontsize=15)
plt.tight_layout()
plt.show()

# Define energy function
# inferring q(f|(xi,yi)_i=1^3)
zero = Variable(torch.FloatTensor(1).zero_())
def energy1(f):
    mu = torch.mul(torch.sin(x0.permute(1,0)*2.0*np.pi*f+b0), a0)
    return - ((mu-y0.permute(1,0))**2 * (1/0.25)).sum(1)
    ll = utils.log_normal(y0.permute(1,0),mu,zero).sum(1)
    return ll

# build and train
mdl = model_1d(energy1)
mdl.train()
n = 10000
# plot figure
fig = plt.figure()

ax = fig.add_subplot(111)
spl = mdl.sample(n)[0]
plt.hist(spl.data.numpy()[:,0],100)
plt.xlabel('f', fontsize=15)
ax.set_yticklabels([])
plt.title('$f\sim q(f)$', fontsize=18)
plt.grid()
#plt.savefig('sinewave_qf.pdf',format='pdf')

spl = spl.data.numpy()
mdl_density = gaussian_kde(spl[:,0],0.05)
xx = np.linspace(0,2,1000)

plt.plot(xx,300*mdl_density(xx),'r')
plt.tight_layout()
plt.legend(['kde', 'counts'], loc=2, fontsize=20)
plt.show()