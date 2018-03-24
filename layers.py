#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 16:55:52 2017

@author: Pablo Navarrete Michelini
"""
import numpy as np
import torch
from torch import nn
from torch.nn.functional import conv2d
from torch.optim import Optimizer
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from sympy.utilities.iterables import multiset_permutations


class nReLU(nn.Module):
    def __init__(self, features, norm, leak=0):
        super().__init__()
        self.leak = leak
        if norm is None:
            self.norm = Bias(features)
        else:
            self.norm = norm(features)
        if self.leak == 0:
            self.act = nn.ReLU(True)
        else:
            self.act = nn.LeakyReLU(negative_slope=self.leak, inplace=True)

        self.norm_name = str(self.norm).split('(')[0] + '(%d)' % features

    def forward(self, input):
        return self.act(self.norm(input))

    def __repr__(self):
        if self.leak == 0:
            s = ('{norm_name}, ReLU(True)')
        else:
            s = ('{norm_name}, LeakyReLU({leak}, True)')
        return s.format(name=self.__class__.__name__, **self.__dict__)


class IntUpscale(nn.Module):
    def __init__(self, nfeat, stride, mode='bicubic', param=4):
        assert isinstance(stride[0], int) and isinstance(stride[1], int)
        self.nfeat = nfeat
        assert stride[0] < 9, 'Currently limited by wrong border calculation'
        self.stride = stride
        self.mode = mode
        self.param = param
        super().__init__()

        if mode == 'bicubic':
            fh = np.asarray([
                kernel_cubic(self.stride[1], -(self.stride[1]-1.)/2.)
            ])
            fv = np.asarray([
                kernel_cubic(self.stride[0], -(self.stride[0]-1.)/2.)
            ])
        else:
            assert mode == 'lanczos'
            fh = np.asarray([
                kernel_lanczos(self.param, self.stride[1], -(self.stride[1]-1.)/2.)
            ])
            fv = np.asarray([
                kernel_lanczos(self.param, self.stride[0], -(self.stride[0]-1.)/2.)
            ])
        a = np.size(fh)
        npad = ((0, 0), (0, np.int(np.ceil(a/self.stride[1])*self.stride[1])-a))
        fh = np.pad(fh, pad_width=npad, mode='constant', constant_values=0)
        b = np.size(fv)
        npad = ((0, 0), (0, np.int(np.ceil(b/self.stride[0])*self.stride[0])-b))
        fv = np.pad(fv, pad_width=npad, mode='constant', constant_values=0)
        f2d = fh * fv.T

        groups = self.stride[0] * self.stride[1]
        f = np.zeros([
            groups*nfeat,
            nfeat,
            f2d.shape[0]//self.stride[0],
            f2d.shape[1]//self.stride[1],
        ])

        for k in range(nfeat):
            g = groups - 1
            for i in range(self.stride[0]):
                for j in range(self.stride[1]):
                    f[groups*k+g, k, :, :] = np.asarray(
                        f2d[i::self.stride[0], j::self.stride[1]]
                    )
                    g -= 1
        self.register_buffer('weight', torch.FloatTensor(np.asarray(f)))

        self.mx = Muxout(self.stride, pmode=None)

        self.border = np.zeros(4, dtype=np.int)
        self.border[0] = (a-1)//2
        self.border[1] = a - self.border[0] - 2 * (self.stride[1]//7+self.stride[1]//5+1)
        self.border[2] = (b-1)//2
        self.border[3] = b - self.border[2] - 2 * (self.stride[0]//7+self.stride[0]//5+1)

    def forward(self, x):
        y = conv2d(
            input=x,
            weight=Variable(self.weight),
            bias=None, stride=1, padding=0, dilation=1, groups=1
        )

        mux = self.mx(y)

        return mux

    def __repr__(self):
        s = ('{name}({nfeat}, stride={stride}), mode={mode}, param={param}, border={border})')
        return s.format(name=self.__class__.__name__, **self.__dict__)


def kernel_lanczos(a, zoom, phase, length=None):
    assert a > 0 and zoom > 0

    lower_bound = np.ceil(-a*zoom-phase)
    higher_bound = np.floor(a*zoom-phase)

    anchor = max(abs(lower_bound), abs(higher_bound))
    index = np.arange(-anchor+1, anchor+1)
    if length is not None:
        assert length >= 2*anchor
        anchor = np.ceil(length/2)
        index = np.arange(-anchor+1, length-anchor+1)

    pos = abs(index+phase) / zoom

    kernel = a * np.sin(np.pi*pos) * np.sin(np.pi*pos/a) / (np.pi**2 * pos*pos)
    kernel[pos > a] = 0
    kernel[pos == 0.] = 1

    kernel = kernel * zoom / np.sum(kernel)

    return kernel


def kernel_cubic(zoom, phase, length=None):
    assert zoom > 0

    lower_bound = np.ceil(-2*zoom-phase)
    higher_bound = np.floor(2*zoom-phase)

    anchor = max(abs(lower_bound), abs(higher_bound))
    index = np.arange(-anchor+1, anchor+1)
    if length is not None:
        assert length >= 2*anchor
        anchor = np.ceil(length/2)
        index = np.arange(-anchor+1, length-anchor+1)

    pos = abs(index+phase) / zoom

    kernel = np.zeros(np.size(pos))
    idx = (pos < 2)
    kernel[idx] = -0.5 * pos[idx]**3 + 2.5 * pos[idx]**2 - 4*pos[idx] + 2
    idx = (pos < 1)
    kernel[idx] = 1.5 * pos[idx]**3 - 2.5 * pos[idx]**2 + 1

    kernel = kernel * zoom / np.sum(kernel)

    return kernel


class LR_scheduler(object):
    def __init__(self, optimizer, mode='hyp', g_label=None, g_lr_factor=None,
                 init_lr=0.001, tau=100.):
        assert isinstance(optimizer, Optimizer)
        assert tau >= 1.
        self.optimizer = optimizer
        self.mode = mode
        self.init_lr = init_lr
        self.tau = tau
        self.g_label = g_label
        self.g_lr_factor = g_lr_factor
        self.last_step = 0

    def step(self, metrics, step=None):
        if step is None:
            self.last_step = self.last_step + 1
            step = self.last_step

        if self.mode == 'exp':
            new_lr = self.init_lr*np.exp(-(step-1.)/self.tau)
        elif self.mode == 'hyp':
            if step < self.tau:
                new_lr = self.init_lr
            else:
                new_lr = (self.init_lr*self.tau)/step
        elif self.mode == 'sqrt':
            new_lr = self.init_lr/np.sqrt((step+self.tau)/self.tau)
        else:
            assert False

        if self.g_label is None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
        else:
            for key in self.g_label:
                self.optimizer.param_groups[self.g_label[key]]['lr'] = \
                    new_lr * self.g_lr_factor[key]


def Cutborder(input, border, feature=None):
    """Cut borders considering that some borders can be zero (no border)
    """
    assert isinstance(border, tuple)
    if border == (0, 0, 0, 0):
        return input
    left, right, top, bottom = border
    assert top >= 0 and bottom >= 0 and \
        left >= 0 and right >= 0
    if np.all(np.asarray(border) == 0) and feature is None:
        return input
    if bottom == 0:
        bottom = None
    else:
        bottom = -bottom
    if right == 0:
        right = None
    else:
        right = -right
    if feature is None:
        return input[:, :, top:bottom, left:right]
    else:
        return input[:, feature:feature+1, top:bottom, left:right]


class MSE_pix(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return ((x-y)**2).mean(3).mean(2)

    def __repr__(self):
        s = '{name}'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class Charbonnier_pix(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, eps=1e-3):
        return torch.sqrt((x-y)**2+eps**2).mean(3).mean(2)

    def __repr__(self):
        s = '{name}'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class SSIM_pix(nn.Module):
    def __init__(self, maxv=1.):
        super().__init__()
        self.register_buffer('C1', torch.FloatTensor([(.01*maxv)**2]))
        self.register_buffer('C1', torch.FloatTensor([(.03*maxv)**2]))

    def forward(self, x, y):
        mux = x.mean(3).mean(2)
        muy = y.mean(3).mean(2)
        sigmax2 = (x**2).mean(3).mean(2) - (mux**2)
        sigmay2 = (y**2).mean(3).mean(2) - (muy**2)
        sigmaxy = (x*y).mean(3).mean(2) - mux*muy
        return (
            (2.*mux*muy + self.C1) * (2.*sigmaxy + self.C2) /
            (((mux**2) + (muy**2) + self.C1) * (sigmax2 + sigmay2 + self.C2))
        )

    def __repr__(self):
        s = '{name}'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class PSNR_RGB(nn.Module):
    def __init__(self, maxv=1., cut=None):
        super().__init__()
        assert cut is None or len(cut) == 4
        if cut is not None:
            assert np.all(np.asarray(cut) >= 0)
        self.cut = cut

        self.register_buffer('c1', torch.log(torch.FloatTensor([255./maxv])))
        self.register_buffer('c2', 20./torch.log(torch.FloatTensor([10.])))

    def forward(self, x, y):
        assert len(x.shape) == 4 and x.shape[1] == 3
        assert len(y.shape) == 4 and y.shape[1] == 3

        A = x.clamp(0., 1.)*Variable(self.c1)
        B = y.clamp(0., 1.)*Variable(self.c1)

        mse = ((A - B)**2).mean(3).mean(2)
        return Variable(self.c2) * torch.log(255./torch.sqrt(mse))

    def __repr__(self):
        s = ('{name}()')
        return s.format(name=self.__class__.__name__, **self.__dict__)


class Activ(nn.Module):
    def __init__(self, features, norm, leak=0, inplace=True):
        super().__init__()
        self.leak = leak
        self.inplace = inplace
        if norm is None or norm == 'selu' or norm == 'prelu':
            self.norm = Bias(features)
        else:
            self.norm = norm(features)
        if norm == 'selu':
            self.act = nn.SELU(inplace=self.inplace)
        elif norm == 'prelu':
            self.act = nn.PReLU(num_parameters=features)
        else:
            if self.leak == 0:
                self.act = nn.ReLU(inplace=self.inplace)
            else:
                self.act = nn.LeakyReLU(
                    negative_slope=self.leak,
                    inplace=self.inplace
                )

    def forward(self, input):
        return self.act(self.norm(input))

    def __repr__(self):
        return str(self.norm) + ', ' + str(self.act)


class ClassicUpscale(nn.Module):
    def __init__(self, nfeat, stride, mode='bicubic', param=4, train=False):
        assert stride[0] == stride[1]
        assert stride[0] == 2  # FIXME!
        assert isinstance(stride[0], int) and isinstance(stride[1], int)
        self.nfeat = nfeat
        self.stride = stride
        self.mode = mode
        self.param = param
        self._train = train
        super().__init__()

        if mode == 'bicubic':
            fh = np.asarray([
                kernel_cubic(self.stride[1], (self.stride[1]-1.)/2.)
            ])
            fv = np.asarray([
                kernel_cubic(self.stride[0], (self.stride[0]-1.)/2.)
            ])
        else:
            assert mode == 'lanczos'
            fh = np.asarray([
                kernel_lanczos(self.param, self.stride[1], (self.stride[1]-1.)/2.)
            ])
            fv = np.asarray([
                kernel_lanczos(self.param, self.stride[0], (self.stride[1]-1.)/2.)
            ])
        a = np.size(fh)
        npad = ((0, 0), (0, np.int(np.ceil(a/self.stride[1])*self.stride[1])-a))
        fh = np.pad(fh, pad_width=npad, mode='constant', constant_values=0)
        b = np.size(fv)
        npad = ((0, 0), (0, np.int(np.ceil(b/self.stride[0])*self.stride[0])-b))
        fv = np.pad(fv, pad_width=npad, mode='constant', constant_values=0)
        f2d = fh * fv.T

        groups = self.stride[0] * self.stride[1]
        f = np.zeros([
            groups*nfeat,
            nfeat,
            f2d.shape[0]//self.stride[0],
            f2d.shape[1]//self.stride[1],
        ])

        for k in range(nfeat):
            g = groups - 1
            for i in range(self.stride[0]):
                for j in range(self.stride[1]):
                    f[groups*k+g, k, :, :] = np.asarray(
                        f2d[i::self.stride[0], j::self.stride[1]]
                    )
                    g -= 1
        if self._train:
            self.weight = Parameter(torch.FloatTensor(np.asarray(f)))
        else:
            self.register_buffer(
                'weight', Variable(torch.FloatTensor(np.asarray(f)))
            )

        self._padding = max(
            int(np.ceil((f2d.shape[0]//self.stride[0]-1)/2)),
            int(np.ceil((f2d.shape[1]//self.stride[1]-1)/2))
        )

        self.mx = Muxout(self.stride, pmode=None)

    def forward(self, x, padding=True):
        pad = self._padding if padding else 0
        y = conv2d(
            input=x,
            weight=self.weight,
            bias=None, stride=1,
            padding=pad,
            dilation=1, groups=1
        )[:, :, 1:, 1:]

        mux = self.mx(y)

        return mux

    def __repr__(self):
        s = '{name}({nfeat}, stride={stride}), mode={mode}, param={param}, ' \
            'kernel_size=%dx%d, padding={_padding})' % (self.weight.shape[2], self.weight.shape[3])
        return s.format(name=self.__class__.__name__, **self.__dict__)


class Muxout(nn.Module):
    def __init__(self, stride, pmode='all', colors=1):
        assert stride[0] > 0 and stride[1] > 0, \
            'Muxout factors must be greater than 0 (%dx%d)' \
            % (stride[0], stride[1])
        self.stride = stride
        self.pmode = pmode
        self.colors = colors
        perm_v = [p for p in multiset_permutations(np.arange(self.stride[0]))]
        perm_h = [p for p in multiset_permutations(np.arange(self.stride[1]))]
        super().__init__()
        self.register_buffer('perm_v', torch.LongTensor(np.asarray(perm_v)))
        self.register_buffer('perm_h', torch.LongTensor(np.asarray(perm_h)))

    def forward(self, x):
        assert np.size(x.size()) == 4
        B, C, inH, inW = x.size()
        Sh, Sw = self.stride
        Gin = Sh * Sw
        assert C % Gin == 0, \
            'Channels=%d must be divisible by Gin=%d' % (C, Gin)

        Ng = C // (Gin*self.colors)
        group_shape = [B, Ng*self.colors, Sh*inH, Sw*inW]

        y = x.contiguous().view(B, Ng, self.colors, Sh, Sw, inH, inW)
        if self.pmode is None:
            mux = y.permute(0, 2, 1, 5, 3, 6, 4).contiguous().view(group_shape)
        else:
            mux = torch.cat(
                [y.index_select(3, i).index_select(4, j).
                 permute(0, 2, 1, 5, 3, 6, 4).contiguous().view(group_shape)
                 for i in Variable(self.perm_v)
                 for j in Variable(self.perm_h)],
                dim=1
            )

        return mux

    def __repr__(self):
        s = '{name}(stride={stride}, pmode={pmode})'
        if self.pmode == 'all':
            s = '{name}(stride={stride})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class MuxoutTranspose(nn.Module):
    def __init__(self, stride, norm=False, pmode='all'):
        assert stride[0] > 0 and stride[1] > 0, \
            'Muxout factors must be greater than 0 (%dx%d)' \
            % (stride[0], stride[1])
        self.stride = stride
        self.norm = norm

        B, Ng, Sh, Sw, H, W = 1, 1, stride[0], stride[1], 1, 1
        self.Gin = stride[0]*stride[1]
        probe = torch.\
            arange(0, B*Ng*self.Gin*H*W).\
            view(B, Ng*self.Gin, H, W).\
            type(torch.LongTensor)
        mx = Muxout((Sh, Sw), pmode=pmode)
        probe_mx = mx(Variable(probe)).data
        self.Gout = probe_mx.size()[1]
        iperm = probe_mx.\
            view(B, self.Gout, Ng, H, Sh, W, Sw).\
            permute(0, 1, 2, 4, 6, 3, 5).contiguous().\
            view(B, self.Gout, Ng,  self.Gin, H, W).\
            squeeze(5).squeeze(4).squeeze(2).squeeze(0).\
            numpy().argsort()

        super().__init__()
        self.register_buffer('iperm', torch.LongTensor(iperm))

    def forward(self, x):
        assert np.size(x.size()) == 4
        B, channels, inH, inW = x.size()
        assert channels % self.Gout == 0, 'Channels=%d must be divisible by ' \
            'Gout=%d' % (channels, self.Gout)
        Ng = channels // self.Gout
        Sh, Sw = self.stride
        assert inH % Sh == 0, \
            'inH=%d must be divisible by Sh=%d' % (inH, Sh)
        assert inW % Sw == 0, \
            'inH=%d must be divisible by Sh=%d' % (inH, Sh)
        H, W = inH//Sh, inW//Sw

        y = x.contiguous().\
            view(B, self.Gout, Ng, H, Sh, W, Sw).\
            permute(0, 1, 2, 4, 6, 3, 5).contiguous().\
            view(B, self.Gout, Ng, self.Gin, H, W)

        tmux = torch.cat(
            [y[:, 0, g, :, :, :][:, self.iperm[0, :], :, :]
             for g in range(Ng)],
            dim=1
        )
        for c in range(1, self.Gout):
            tmux += torch.cat(
                [y[:, c, g, :, :, :][:, self.iperm[c, :], :, :]
                 for g in range(Ng)],
                dim=1
            )

        if self.norm:
            return tmux / self.Gout
        return tmux

    def __repr__(self):
        s = ('{name}(stride={stride}, {Gin}->{Gout})')
        return s.format(name=self.__class__.__name__, **self.__dict__)


class Bias(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.features = features
        self.bias = Parameter(torch.zeros(features))

    def forward(self, input):
        return input + self.bias.\
            unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(input)

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.features) + ')'


class L1WbLoss(nn.Module):
    def __init__(self, model):
        super().__init__()

        nweight = 0.
        nbias = 0.
        self.wlist = []
        self.blist = []
        for name, par in model.named_parameters():
            if name.endswith('weight'):
                self.wlist.append(par)
                nweight += 1.
            if name.endswith('bias'):
                self.blist.append(par)
                nbias += 1.
        assert nbias > 0. and nweight > 0.
        self.register_buffer('L1weight', torch.zeros([1]))
        self.register_buffer('L1bias', torch.zeros([1]))
        self.register_buffer(
            'c1', torch.FloatTensor(np.asarray([1./nweight]))
        )
        self.register_buffer(
            'c2', torch.FloatTensor(np.asarray([1./nbias]))
        )

    def forward(self):
        self.L1weight.zero_()
        self.L1bias.zero_()
        S1 = Variable(self.L1weight)
        S2 = Variable(self.L1bias)
        for par in self.wlist:
            S1 += torch.mean(torch.abs(par))
        for par in self.blist:
            S2 += torch.mean(torch.abs(par))
        return (S1[0]*self.c1[0]) / (S2[0]*self.c2[0] + 1e-6)


def weight_bias_init(net, mode, gain=1.):
    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
            if mode == 'dirac':
                torch.nn.init.dirac(m.weight)
            elif mode == 'dirac_normal':
                torch.nn.init.dirac(m.weight)
                m.weight.data += torch.zeros_like(m.weight).normal_(0, gain).data
            elif mode == 'xavier':
                torch.nn.init.xavier_normal(m.weight, gain=gain)
            elif mode == 'kaiming':
                torch.nn.init.kaiming_normal(m.weight)
            elif mode == 'normal':
                torch.nn.init.normal(m.weight, mean=0, std=gain)
            if m.bias is not None:
                m.bias.data.fill_(0)
        if isinstance(m, torch.nn.BatchNorm2d):
            m.bias.data.fill_(0)
            m.weight.data.fill_(1)
        if isinstance(m, Bias):
            m.bias.data.fill_(0)
