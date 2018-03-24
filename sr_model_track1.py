#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 15:59:11 2017

@author: pablo
"""
import torch
import numpy as np
from torch import nn
from collections import OrderedDict
from torch.autograd import Variable
from layers import Muxout, Cutborder, IntUpscale, nReLU


class Upscaler(nn.Module):
    def __init__(self,
                 model_id,
                 noise_samples=None,
                 noise_shape=None,
                 dtype='float32',
                 name='upscaler',
                 device=0,
                 str_tab='',
                 vlevel=0):
        self.model_id = model_id
        self.name = name
        self.str_tab = str_tab
        self.vlevel = vlevel
        self.device = -1
        self.dtype = dtype
        super().__init__()

        if self.model_id.endswith('_ms'):
            parse = self.model_id.split('_')
            self.factor = [2, 2]
            self._repeat = 1
            self.input_channels = int([s for s in parse if s.startswith('CH')][0][2:])
            self.lanczos = int([s for s in parse if s.startswith('LZ')][0][2:])
            FE =  int([s for s in parse if s.startswith('FE')][0][2:])
            leak =  float([s for s in parse if s.startswith('LEAK')][0][4:])
            probe_shape = (1, self.input_channels, 64, 64)
            bnorm = None
            if [s for s in parse if s.startswith('BN')][0] == 'BN1':
                bnorm = nn.BatchNorm2d
            self.net = {}
            self.net['Classic'] = IntUpscale(
                self.input_channels,
                (2, 2), mode='lanczos', param=self.lanczos
            )
            an_parse = [s for s in parse if s.startswith('Analysis')][0].split('#')
            an_k = int([s for s in an_parse if s.startswith('K')][0][1:])
            an_d = int([s for s in an_parse if s.startswith('D')][0][1:])
            an_x = int([s for s in an_parse if s.startswith('X')][0][1:])
            an_nlayers = int([s for s in an_parse if s.startswith('L')][0][1:])
            assert an_nlayers > 0
            self.net['Analysis'] = torch.nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(
                    self.input_channels, FE, an_k, dilation=an_d, bias=False)
                 ),
                ('act1', nReLU(FE, bnorm, leak))
            ]))
            k = 2
            for l in range(1, an_nlayers):
                for x in range(an_x-1):
                    self.net['Analysis'].add_module(
                        'conv%d' % k, nn.Conv2d(
                            FE, FE, 1, dilation=1, bias=False
                        )
                    )
                    self.net['Analysis'].add_module(
                        'act%d' % k, nReLU(FE, bnorm, leak)
                    )
                    k += 1
                self.net['Analysis'].add_module(
                    'conv%d' % k, nn.Conv2d(
                        FE, FE, an_k, dilation=an_d, bias=False
                    )
                )
                self.net['Analysis'].add_module(
                    'act%d' % k, nReLU(FE, bnorm, leak)
                )
                k += 1
            for x in range(an_x-1):
                self.net['Analysis'].add_module(
                    'conv%d' % k, nn.Conv2d(
                        FE, FE, 1, dilation=1, bias=False
                    )
                )
                self.net['Analysis'].add_module(
                    'act%d' % k, nReLU(FE, bnorm, leak)
                )
                k += 1
            up_parse = [s for s in parse if s.startswith('Upscaling')][0].split('#')
            up_k = int([s for s in up_parse if s.startswith('K')][0][1:])
            up_d = int([s for s in up_parse if s.startswith('D')][0][1:])
            up_x = int([s for s in up_parse if s.startswith('X')][0][1:])
            up_m = int([s for s in up_parse if s.startswith('M')][0][1:])
            up_nlayers = int([s for s in up_parse if s.startswith('L')][0][1:])
            assert up_nlayers > 1
            assert up_m < up_nlayers
            up_nlayers1 = up_m
            up_nlayers2 = up_nlayers - up_nlayers1
            self.net['Upscaling'] = torch.nn.Sequential()
            k = 1
            for l in range(up_nlayers1-1):
                self.net['Upscaling'].add_module(
                    'conv%d' % k, nn.Conv2d(
                        FE, FE, up_k, dilation=up_d, bias=False
                    )
                )
                self.net['Upscaling'].add_module(
                    'act%d' % k, nReLU(FE, bnorm, leak)
                )
                k += 1
                for x in range(up_x-1):
                    self.net['Upscaling'].add_module(
                        'conv%d' % k, nn.Conv2d(
                            FE, FE, 1, dilation=1, bias=False
                        )
                    )
                    self.net['Upscaling'].add_module(
                        'act%d' % k, nReLU(FE, bnorm, leak)
                    )
                    k += 1
            self.net['Upscaling'].add_module(
                'conv%d' % k, nn.Conv2d(
                    FE, FE, up_k, dilation=up_d, bias=False
                )
            )
            for x in range(up_x-1):
                self.net['Upscaling'].add_module(
                    'act%d' % k, nReLU(FE, bnorm, leak)
                )
                k += 1
                self.net['Upscaling'].add_module(
                    'conv%d' % k, nn.Conv2d(
                        FE, FE, 1, dilation=1, bias=False
                    )
                )
            self.net['Upscaling'].add_module(
                'mx', Muxout((2, 2))
            )
            self.net['Upscaling'].add_module(
                'act%d' % k, nReLU(FE, bnorm, leak)
            )
            k += 1
            for l in range(up_nlayers2):
                self.net['Upscaling'].add_module(
                    'conv%d' % k, nn.Conv2d(
                        FE, FE, up_k, dilation=up_d, bias=False
                    )
                )
                self.net['Upscaling'].add_module(
                    'act%d' % k, nReLU(FE, bnorm, leak)
                )
                k += 1
                for x in range(up_x-1):
                    self.net['Upscaling'].add_module(
                        'conv%d' % k, nn.Conv2d(
                            FE, FE, 1, dilation=1, bias=False
                        )
                    )
                    self.net['Upscaling'].add_module(
                        'act%d' % k, nReLU(FE, bnorm, leak)
                    )
                    k += 1
            syn_parse = [s for s in parse if s.startswith('Synthesis')][0].split('#')
            syn_k = int([s for s in syn_parse if s.startswith('K')][0][1:])
            syn_d = int([s for s in syn_parse if s.startswith('D')][0][1:])
            syn_x = int([s for s in syn_parse if s.startswith('X')][0][1:])
            syn_nlayers = int([s for s in syn_parse if s.startswith('L')][0][1:])
            assert syn_nlayers > 0
            k = 1
            if syn_nlayers > 1:
                self.net['Synthesis'] = torch.nn.Sequential(OrderedDict([
                    ('conv1', nn.Conv2d(
                        FE, FE, syn_k, dilation=syn_d, bias=False)
                     ),
                    ('act1', nReLU(FE, bnorm, leak))
                ]))
                k += 1
            else:
                self.net['Synthesis'] = torch.nn.Sequential()
            for l in range(1, syn_nlayers-1):
                for x in range(syn_x-1):
                    self.net['Synthesis'].add_module(
                        'conv%d' % k, nn.Conv2d(
                            FE, FE, 1, dilation=1, bias=False
                        )
                    )
                    self.net['Synthesis'].add_module(
                        'act%d' % k, nReLU(FE, bnorm, leak)
                    )
                    k += 1
                self.net['Synthesis'].add_module(
                    'conv%d' % k, nn.Conv2d(
                        FE, FE, syn_k, dilation=syn_d, bias=False
                    )
                )
                self.net['Synthesis'].add_module(
                    'act%d' % k, nReLU(FE, bnorm, leak)
                )
                k += 1
            for x in range(syn_x-1):
                self.net['Synthesis'].add_module(
                    'conv%d' % k, nn.Conv2d(
                        FE, FE, 1, dilation=1, bias=False
                    )
                )
                self.net['Synthesis'].add_module(
                    'act%d' % k, nReLU(FE, bnorm, leak)
                )
                k += 1
            self.net['Synthesis'].add_module(
                'conv%d' % k, nn.Conv2d(
                    FE, self.input_channels, syn_k, dilation=syn_d, bias=True
                )
            )
        else:
            assert False
        self.border_in = (0, 0, 0, 0)
        self.border_out = (0, 0, 0, 0)
        self.border_cut = (0, 0, 0, 0)

        probe = Variable(
            torch.FloatTensor(np.zeros(probe_shape)),
            requires_grad=False
        )

        if self.model_id.endswith('_ms'):
            out_an = self.net['Analysis'](probe)
            out_up = self.net['Upscaling'](out_an)
            out_syn = self.net['Synthesis'](out_up)
            out1_shape = self.net['Classic'](probe).shape
            out2_shape = out_syn.shape
            pad_h = out1_shape[2] - out2_shape[2]
            pad_v = out1_shape[3] - out2_shape[3]
            self._b1 = (
                (pad_h+1)//2, (pad_h+1)//2,
                (pad_v+1)//2, (pad_v+1)//2
            )
            assert np.all(np.asarray(self._b1) >= 0)

            out1_shape = self.net['Classic'](out_syn).shape
            out2_shape = self.net['Synthesis'](self.net['Upscaling'](out_up)).shape
            pad_h = out1_shape[2] - out2_shape[2]
            pad_v = out1_shape[3] - out2_shape[3]
            self._b2 = (
                (pad_h+1)//2, (pad_h+1)//2,
                (pad_v+1)//2, (pad_v+1)//2
            )
            assert np.all(np.asarray(self._b2) >= 0)

            for b in [self._b1, self._b2]:
                for v in b:
                    if v < 0 and self.vlevel > 0:
                        print(self.str_tab, self.name,
                              'probe:', probe.shape)
                        print(self.str_tab, self.name,
                              'out_an:', out_an.shape)
                        print(self.str_tab, self.name,
                              'out_up:', out_up.shape)
                        print(self.str_tab, self.name,
                              'out_syn:', out_syn.shape)
                        print(self.str_tab, self.name,
                              'out1:', out1_shape)
                        print(self.str_tab, self.name,
                              'out2:', out2_shape)
                        print(self.str_tab, self.name,
                              'b1:', self._b1)
                        print(self.str_tab, self.name,
                              'b2:', self._b2)
                        assert v >= 0

        out_size = self.forward(probe, pad=False).size()
        pad_v = probe.size()[2]*self.factor[0]-out_size[2]
        pad_h = probe.size()[3]*self.factor[1]-out_size[3]
        self.border_out = (
            pad_h//2, pad_h-pad_h//2,
            pad_v//2, pad_v-pad_v//2
        )
        assert np.all(np.asarray(self.border_out) >= 0)

        pad_h = pad_h//self.factor[1]
        pad_v = pad_v//self.factor[0]
        self.border_in = (
            (pad_h+1)//2, (pad_h+1)//2,
            (pad_v+1)//2, (pad_v+1)//2
        )
        assert np.all(np.asarray(self.border_in) >= 0)

        self.reflection = torch.nn.ReflectionPad2d(self.border_in)

        if noise_samples is not None:
            assert noise_samples > 0
            self.noise = nn.Parameter(torch.randn(
                noise_samples,
                self.input_channels-3,
                noise_shape[0],
                noise_shape[1]
            ))
            if self.vlevel > 0:
                print(self.str_tab, self.name,
                      '- noise_size', self.noise.size())

        outwithpad_size = self.forward(probe, pad=True).size()
        pad_v = max(outwithpad_size[2] - probe.size()[2]*self.factor[0], 0)
        pad_h = max(outwithpad_size[3] - probe.size()[3]*self.factor[1], 0)
        self.border_cut = (
            pad_h//2, pad_h-pad_h//2,
            pad_v//2, pad_v-pad_v//2
        )
        assert np.all(np.asarray(self.border_cut) >= 0)

        if self.model_id.endswith('_ms'):
            for name, layer in self.net.items():
                self.add_module(name, layer)

        self.stat_nparam = 0
        for name, par in self.named_parameters():
            if name.endswith('weight'):
                self.stat_nparam += np.prod(par.shape)
            if name.endswith('bias'):
                self.stat_nparam += np.prod(par.shape)

        s = '%dx%d' % (self.factor[0], self.factor[1])
        self._cached_border_in = {s: self.border_in}
        self._cached_border_out = {s: self.border_out}
        self._cached_border_cut = {s: self.border_cut}
        if self.vlevel > 0:
            print(self.str_tab, self.name,
                  '- Model', self.model_id)
            print(self.str_tab, self.name,
                  '- Factor set to %dx%d' % (self.factor[0], self.factor[1]))
            print(self)
            print(self.str_tab, self.name,
                  '- probe_shape', probe.data.size())
            print(self.str_tab, self.name,
                  '- out_size', out_size)
            print(self.str_tab, self.name,
                  '- border_out', self.border_out)
            print(self.str_tab, self.name,
                  '- border_in', self.border_in)
            print(self.str_tab, self.name,
                  '- border_cut', self.border_cut)
            print(self.str_tab, self.name,
                  '- # weight/bias: {:_}'.format(self.stat_nparam))

    def forward(self, x, pad=False):
        x_pad = x
        if pad:
            x_pad = self.reflection(x)

        if self.model_id.endswith('_ms'):
            out = Cutborder(self.net['Classic'](x_pad), self._b1)
            B = self.net['Analysis'](x_pad)
            for loop in range(self._repeat-1):
                B = self.net['Upscaling'](B)
                out = out + self.net['Synthesis'](B)
                out = Cutborder(self.net['Classic'](out), self._b2)
            B = self.net['Upscaling'](B)
            out = out + self.net['Synthesis'](B)
        else:
            out = self.net(x_pad)

        if pad:
            return Cutborder(out, self.border_cut)
        return out

    def all_out(self, x):
        assert self.model_id.endswith('_ms')
        x_pad = x

        ret = []
        out = Cutborder(self.net['Classic'](x_pad), self._b1)
        B = self.net['Analysis'](x_pad)
        for loop in range(self._repeat-1):
            B = self.net['Upscaling'](B)
            out = out + self.net['Synthesis'](B)
            ret.append(out)
            out = Cutborder(self.net['Classic'](out), self._b2)
        B = self.net['Upscaling'](B)
        out = out + self.net['Synthesis'](B)
        ret.append(out)

        return ret

    def cpu(self, **kwds):
        self.device = -1
        return super().cpu(**kwds)

    def cuda(self, gpu, **kwds):
        self.device = gpu
        return super().cuda(gpu, **kwds)

    def half(self, **kwds):
        self.dtype = 'float16'
        return super().half(**kwds)

    def float(self, **kwds):
        self.dtype = 'float32'
        return super().half(**kwds)

    def set_mu(self, mu):
        pass

    def set_factor(self, factor):
        s = '%dx%d' % (factor[0], factor[1])
        if s in self._cached_border_in:
            self._repeat = int(np.log2(factor[0]))
            self.factor = factor
            self.border_in = self._cached_border_in[s]
            self.border_out = self._cached_border_out[s]
            self.border_cut = self._cached_border_cut[s]
            self.reflection = torch.nn.ReflectionPad2d(self.border_in)
        else:
            assert len(factor) == 2
            assert np.log2(factor[0]) == np.floor(np.log2(factor[0]))
            assert np.log2(factor[1]) == np.floor(np.log2(factor[1]))
            if self.model_id.endswith('_ms'):
                assert self.factor[0] == self.factor[1]
                assert factor[0] == factor[1]
                self._repeat = int(np.log2(factor[0]))
                self.factor = factor

                self.border_in = (0, 0, 0, 0)
                self.border_out = (0, 0, 0, 0)
                self.border_cut = (0, 0, 0, 0)

                probe_shape = (1, self.input_channels, 64, 64)
                probe = Variable(
                    torch.FloatTensor(np.zeros(probe_shape)),
                    requires_grad=False
                )
                if self.device >= 0:
                    probe = probe.cuda(self.device)
                if self.dtype == 'float16':
                    probe = probe

                out_size = self.forward(probe, pad=False).size()
                pad_v = probe.size()[2]*self.factor[0]-out_size[2]
                pad_h = probe.size()[3]*self.factor[1]-out_size[3]
                self.border_out = (
                    pad_h//2, pad_h-pad_h//2,
                    pad_v//2, pad_v-pad_v//2
                )
                assert np.all(np.asarray(self.border_out) >= 0)

                pad_h = pad_h//self.factor[1]
                pad_v = pad_v//self.factor[0]
                self.border_in = (
                    (pad_h+1)//2, (pad_h+1)//2,
                    (pad_v+1)//2, (pad_v+1)//2
                )
                assert np.all(np.asarray(self.border_in) >= 0)

                self.reflection = torch.nn.ReflectionPad2d(self.border_in)

                outwithpad_size = self.forward(probe, pad=True).size()
                pad_v = max(outwithpad_size[2] - probe.size()[2]*self.factor[0], 0)
                pad_h = max(outwithpad_size[3] - probe.size()[3]*self.factor[1], 0)
                self.border_cut = (
                    pad_h//2, pad_h-pad_h//2,
                    pad_v//2, pad_v-pad_v//2
                )
                assert np.all(np.asarray(self.border_cut) >= 0)

                if self.vlevel > 0:
                    print(self.str_tab, self.name,
                          '- probe_shape', probe.data.size())
                    print(self.str_tab, self.name,
                          '- out_size', out_size)
                    print(self.str_tab, self.name,
                          '- border_out', self.border_out)
                    print(self.str_tab, self.name,
                          '- border_in', self.border_in)
                    print(self.str_tab, self.name,
                          '- border_cut', self.border_cut)
            else:
                assert factor[0] == self.factor[0]
                assert factor[1] == self.factor[1]
            self._cached_border_in[s] = self.border_in
            self._cached_border_out[s] = self.border_out
            self._cached_border_cut[s] = self.border_cut
