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
from layers import Muxout, MuxoutTranspose, Bias, Cutborder, ClassicUpscale, Activ


class Upscaler(nn.Module):
    def __init__(self,
                 model_id,
                 noise_samples=None,
                 noise_shape=None,
                 dtype='float32',
                 device=-1,
                 name='upscaler',
                 str_tab='',
                 vlevel=0):
        self.model_id = model_id
        self.name = name
        self.str_tab = str_tab
        self.vlevel = vlevel
        self.device = device
        self._debug = False
        self.dtype = dtype
        super().__init__()

        if self.model_id.endswith('_v3_ms'):
            parse = self.model_id.split('_')
            self.factor = [2, 2]
            self._levels = 1
            self._v3_mu = 2

            self.input_channels = int([s for s in parse if s.startswith('CH')][0][2:])
            probe_shape = (1, self.input_channels, 30, 30)
            FE = int([s for s in parse if s.startswith('FE')][0][2:])
            bnorm = None
            if len([s for s in parse if s.startswith('PRELU')]) > 0:
                assert len([s for s in parse if s.startswith('LEAK')]) == 0
                assert len([s for s in parse if s.startswith('BN')]) == 0 \
                    or [s for s in parse if s.startswith('BN')][0] == 'BN0', \
                    'Cannot use BN with PRELU'
                bnorm = 'prelu'
                leak = 0
            else:
                leak =  float([s for s in parse if s.startswith('LEAK')][0][4:])
                if [s for s in parse if s.startswith('BN')][0] == 'BN1':
                    bnorm = nn.BatchNorm2d
            muxout = True
            if len([s for s in parse if s.startswith('MX')]) > 0:
                muxout = ([s for s in parse if s.startswith('MX')][0] == 'MX1')

            self.net = {}

            if len([s for s in parse if s.startswith('BIC')]) > 0:
                self.net['Classic'] = ClassicUpscale(
                    self.input_channels,
                    (2, 2), mode='bicubic', train=True
                )
            else:
                lanczos = int([s for s in parse if s.startswith('LZ')][0][2:])
                self.net['Classic'] = ClassicUpscale(
                    self.input_channels,
                    (2, 2), mode='lanczos', param=lanczos, train=True
                )

            an_parse = [s for s in parse if s.startswith('Analysis')][0].split('#')[1:]
            an_k = int([s for s in an_parse if s.startswith('K')][0][1:])
            assert (an_k-1) % 2 == 0
            an_d = int([s for s in an_parse if s.startswith('D')][0][1:])
            an_nlayers = int([s for s in an_parse if s.startswith('L')][0][1:])
            assert an_nlayers > 0
            self.net['Analysis'] = torch.nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(
                    self.input_channels, FE, an_k, padding=(an_k-1)//2, dilation=an_d, bias=False)
                 ),
            ]))
            k = 2
            for l in range(an_nlayers-1):
                self.net['Analysis'].add_module(
                    'act%d' % (k-1), Activ(FE, bnorm, leak)
                )
                self.net['Analysis'].add_module(
                    'conv%d' % k, nn.Conv2d(
                        FE, FE, an_k, padding=(an_k-1)//2, dilation=an_d, bias=False
                    )
                )
                k += 1

            up_parse = [s for s in parse if s.startswith('Upscaling')][0].split('#')[1:]
            up_k = int([s for s in up_parse if s.startswith('K')][0][1:])
            assert (up_k-1) % 2 == 0
            up_d = int([s for s in up_parse if s.startswith('D')][0][1:])
            up_m = int([s for s in up_parse if s.startswith('M')][0][1:])
            up_nlayers = int([s for s in up_parse if s.startswith('L')][0][1:])
            assert up_nlayers >= 0
            assert up_m <= up_nlayers
            up_nlayers1 = up_m
            up_nlayers2 = up_nlayers - up_nlayers1
            self.net['Upscaling'] = torch.nn.Sequential()
            k = 0
            for l in range(up_nlayers1):
                self.net['Upscaling'].add_module(
                    'act%d' % k, Activ(FE, bnorm, leak)
                )
                k += 1
                self.net['Upscaling'].add_module(
                    'conv%d' % k, nn.Conv2d(
                        FE, FE, up_k, padding=(up_k-1)//2, dilation=up_d, bias=False
                    )
                )
            if muxout:
                self.net['Upscaling'].add_module(
                    'mx', Muxout((2, 2))
                )
            else:
                self.net['Upscaling'].add_module(
                    'act_up', Activ(FE, bnorm, leak)
                )
                self.net['Upscaling'].add_module(
                    'conv_up', nn.ConvTranspose2d(
                        FE, FE, up_k, stride=(2, 2), padding=(up_k-1)//2, output_padding=(up_k-1)//2, dilation=1, bias=False
                    )
                )
            if k < up_nlayers:
                self.net['Upscaling'].add_module(
                    'act%d' % k, Activ(FE, bnorm, leak)
                )
            k += 1
            for l in range(up_nlayers2):
                self.net['Upscaling'].add_module(
                    'conv%d' % k, nn.Conv2d(
                        FE, FE, up_k, padding=(up_k-1)//2, dilation=up_d, bias=False
                    )
                )
                self.net['Upscaling'].add_module(
                    'act%d' % k, Activ(FE, bnorm, leak)
                )
                k += 1

            down_parse = [s for s in parse if s.startswith('Downscaling')][0].split('#')[1:]
            down_k = int([s for s in down_parse if s.startswith('K')][0][1:])
            assert (down_k-1) % 2 == 0
            down_d = int([s for s in down_parse if s.startswith('D')][0][1:])
            down_m = int([s for s in down_parse if s.startswith('M')][0][1:])
            down_nlayers = int([s for s in down_parse if s.startswith('L')][0][1:])
            assert down_nlayers > 0
            assert down_m <= down_nlayers
            down_nlayers1 = down_m + 1
            down_nlayers2 = down_nlayers - down_nlayers1 + 1
            self.net['Downscaling'] = torch.nn.Sequential()
            k = 0
            self.net['Downscaling'].add_module(
                'act%d' % k, Activ(FE, bnorm, leak)
            )
            k += 1
            for l in range(down_nlayers1-1):
                self.net['Downscaling'].add_module(
                    'conv%d' % k, nn.ConvTranspose2d(
                        FE, FE, down_k, padding=(down_k-1)//2, dilation=down_d, bias=False
                    )
                )
                self.net['Downscaling'].add_module(
                    'act%d' % k, Activ(FE, bnorm, leak)
                )
                k += 1
            if muxout:
                self.net['Downscaling'].add_module(
                    'tmx', MuxoutTranspose((2, 2))
                )
            else:
                self.net['Downscaling'].add_module(
                    'conv_down', nn.Conv2d(
                        FE, FE, down_k, stride=(2, 2), padding=(down_k-1)//2, dilation=1, bias=False
                    )
                )
                self.net['Downscaling'].add_module(
                    'act_down', Activ(FE, bnorm, leak)
                )
            for l in range(down_nlayers2-1):
                self.net['Downscaling'].add_module(
                    'conv%d' % k, nn.ConvTranspose2d(
                        FE, FE, down_k, padding=(down_k-1)//2, dilation=down_d, bias=False
                    )
                )
                self.net['Downscaling'].add_module(
                    'act%d' % k, Activ(FE, bnorm, leak)
                )
                k += 1
            if down_nlayers2 > 0:
                self.net['Downscaling'].add_module(
                    'conv%d' % k, nn.ConvTranspose2d(
                        FE, FE, down_k, padding=(down_k-1)//2, dilation=down_d, bias=False
                    )
                )

            syn_parse = [s for s in parse if s.startswith('Synthesis')][0].split('#')[1:]
            syn_k = int([s for s in syn_parse if s.startswith('K')][0][1:])
            assert (syn_k-1) % 2 == 0
            syn_d = int([s for s in syn_parse if s.startswith('D')][0][1:])
            syn_nlayers = int([s for s in syn_parse if s.startswith('L')][0][1:])
            assert syn_nlayers > 0

            k = 0
            self.net['Synthesis'] = torch.nn.Sequential()
            self.net['Synthesis'].add_module(
                'act%d' % k, Activ(FE, bnorm, leak)
            )
            k += 1
            for l in range(syn_nlayers-1):
                self.net['Synthesis'].add_module(
                    'conv%d' % k, nn.Conv2d(
                        FE, FE, syn_k, padding=(syn_k-1)//2, dilation=syn_d, bias=False
                    )
                )
                self.net['Synthesis'].add_module(
                    'act%d' % k, Activ(FE, bnorm, leak)
                )
                k += 1
            self.net['Synthesis'].add_module(
                'conv%d' % k, nn.Conv2d(
                    FE, self.input_channels, syn_k, padding=(syn_k-1)//2, dilation=syn_d, bias=True
                )
            )
        else:
            assert False
        self.border_in = (0, 0, 0, 0)
        self.border_out = (0, 0, 0, 0)
        self.border_cut = (0, 0, 0, 0)
        self._cached_border_in = {}
        self._cached_border_out = {}
        self._cached_border_cut = {}

        probe = Variable(
            torch.FloatTensor(np.zeros(probe_shape)),
            requires_grad=False
        )
        if self.device >= 0:
            probe = probe.cuda(self.device)
            if self.model_id.endswith('_ms'):
                for mod in self.net:
                    self.net[mod].cuda(self.device)
            else:
                self.net.cuda(self.device)
        if self.dtype == 'float16':
            probe = probe.half()
            if self.model_id.endswith('_ms'):
                for mod in self.net:
                    self.net[mod].half()
            else:
                self.net.half()

        for name, layer in self.net.items():
            self.add_module(name, layer)

        self.set_factor(self.factor)

        self.stat_nparam = 0
        for name, par in self.named_parameters():
            if name.endswith('weight'):
                self.stat_nparam += np.prod(par.shape)
            if name.endswith('bias'):
                self.stat_nparam += np.prod(par.shape)

        if self.vlevel > 0:
            print(self.str_tab, self.name,
                  '- Model', self.model_id)
            print(self.str_tab, self.name,
                  '- Factor set to %dx%d' % (self.factor[0], self.factor[1]))
            print(self)
            print(self.str_tab, self.name,
                  '- border_out', self.border_out)
            print(self.str_tab, self.name,
                  '- border_in', self.border_in)
            print(self.str_tab, self.name,
                  '- border_cut', self.border_cut)
            print(self.str_tab, self.name,
                  '- # weight/bias: {:_}'.format(self.stat_nparam))

    def _v3_RecBP(self, hr, level, last_step=None,
                  loss_metric=None,
                  lr_out=None, lr_ref=None,
                  hr_out=None, hr_ref=None):
        step = last_step
        new_hr = hr
        new_hr_step = step
        if level > 0:
            for k in range(self._v3_mu):
                lr = self.net['Downscaling'](new_hr)
                if loss_metric is not None:
                    self._v3_loss_lr[level] += loss_metric(
                        self.net['Synthesis'](lr) + lr_out,
                        lr_ref
                    )
                    self._v3_loss_lr_count[level] += 1.

                if self._debug:
                    step += 1
                    self.pos[step] = (step, level-1)
                    self.color.append('r')
                    self.edges.append((step-1, step))

                a = step if self._debug else None
                residual, step = self._v3_RecBP(
                    lr, level-1, a,
                    loss_metric=None,
                    lr_out=None, lr_ref=None,
                    hr_out=None, hr_ref=None
                )

                new_hr = new_hr + self.net['Upscaling'](residual)
                if loss_metric is not None:
                    self._v3_loss_hr[level] += loss_metric(
                        self.net['Synthesis'](new_hr) + hr_out,
                        hr_ref
                    )
                    self._v3_loss_hr_count[level] += 1.

                if self._debug:
                    step += 1
                    if k < self._v3_mu-1:
                        self.pos[step] = (step, level)
                    else:
                        self.pos[step] = (step, level+0.125)
                    self.color.append('r')
                    self.edges.append((new_hr_step, step))
                    new_hr_step = step
                    self.edges.append((step-1, step))
            return new_hr, step
        else:
            if self._debug:
                step += 1
                self.pos[step] = (step, level)
                self.color.append('r')
                self.edges.append((self.lr_pos, step-1))
                self.edges.append((step-1, step))
            return hr - self._v3_r0, step

    def forward(self, x, pad=False):
        x_pad = x
        if pad:
            x_pad = self.reflection(x)
        out = x_pad

        step = 0
        if self._debug:
            self.pos = {step: (step, 0.5)}
            self.color = ['b']
            self.edges = []

        self._v3_r0 = self.net['Analysis'](x_pad)

        if self._debug:
            step += 1
            self.pos[step] = (step, 0.+0.25)
            self.lr_pos = step
            self.color.append('g')
            self.edges.append((step-1, step))
            last_res = step

        out = out + self.net['Synthesis'](self._v3_r0)

        if self._debug:
            step += 1
            self.pos[step] = (step, 0.5)
            self.color.append('b')
            self.edges.append((0, step))
            self.edges.append((step-1, step))

        res = self._v3_r0
        for lr_level in range(self._levels):
            out = self.net['Classic'](out, padding=True)

            if self._debug:
                step += 1
                self.pos[step] = (step, (lr_level+1)+0.5)
                self.color.append('b')
                self.edges.append((step-1, step))
                last_classic = step

            res = self.net['Upscaling'](res)

            if self._debug:
                step += 1
                self.pos[step] = (step, (lr_level+1)+0.25)
                self.color.append('g')
                self.edges.append((last_res, step))

            a = step if self._debug else None
            res, step = self._v3_RecBP(res, lr_level+1, a)

            if self._debug:
                step += 1
                self.pos[step] = (step, (lr_level+1)+0.25)
                self.color.append('g')
                self.edges.append((step-1, step))

            out = out + self.net['Synthesis'](res)

            if self._debug:
                step += 1
                self.pos[step] = (step, (lr_level+1)+0.5)
                self.color.append('b')
                self.edges.append((last_classic, step))
                self.edges.append((step-1, step))
                last_res = step

        if pad:
            return Cutborder(out, self.border_cut)
        return out

    def all_out(self, x, ref=None, pad=False, loss_metric=None):
        assert self.model_id.endswith('_ms')
        x_pad = x
        if pad:
            x_pad = self.reflection(x)

        ret = [{}, {}]
        out = x_pad
        self._v3_r0 = self.net['Analysis'](x_pad)
        out = out + self.net['Synthesis'](self._v3_r0)

        res = self._v3_r0
        if loss_metric is not None:
            self._v3_loss_hr = (self._levels+1) * [0.]
            self._v3_loss_hr_count = (self._levels+1) * [0.]
            self._v3_loss_lr = (self._levels+1) * [0.]
            self._v3_loss_lr_count = (self._levels+1) * [0.]
        for lr_level in range(self._levels):
            lr_out = out
            hr_out = self.net['Classic'](out, padding=True)
            lr_ref, hr_ref = None, None
            res = self.net['Upscaling'](res)
            if loss_metric is not None:
                lr_ref = ref[2**(lr_level)]
                hr_ref = ref[2**(lr_level+1)]
                self._v3_loss_hr[lr_level+1] += loss_metric(
                    self.net['Synthesis'](res) + hr_out,
                    hr_ref
                )
                self._v3_loss_hr_count[lr_level+1] += 1.
            res, _ = self._v3_RecBP(
                res, lr_level+1, None,
                loss_metric=loss_metric,
                lr_out=lr_out, lr_ref=lr_ref,
                hr_out=hr_out, hr_ref=hr_ref
            )
            out = hr_out + self.net['Synthesis'](res)

            ret[0][2**(lr_level+1)] = out
            if lr_level == 0:
                ret[0][2**0] = self.net['Synthesis'](self.net['Downscaling'](res)) + lr_out
        if loss_metric is not None:
            for level in range(self._levels+1):
                ret[1]['HR_'+str(2**(level))] = self._v3_loss_hr[level] / max(self._v3_loss_hr_count[level], 1.)
                ret[1]['LR_'+str(2**(level))] = self._v3_loss_lr[level] / max(self._v3_loss_lr_count[level], 1.)

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
        if self.model_id.endswith('_v3_ms'):
            self._v3_mu = mu
            self.set_factor(force=True)

    def set_factor(self, factor=None, force=False):
        if factor is None:
            factor = self.factor
        s = '%dx%d' % (factor[0], factor[1])
        if s in self._cached_border_in and not force:
            self._levels = int(np.log2(factor[0]))
            self.factor = factor
            self.border_in = self._cached_border_in[s]
            self.border_out = self._cached_border_out[s]
            self.border_cut = self._cached_border_cut[s]
            self.reflection = torch.nn.ReflectionPad2d(self.border_in)
        else:
            assert len(factor) == 2
            assert np.log2(factor[0]) == np.floor(np.log2(factor[0]))
            assert np.log2(factor[1]) == np.floor(np.log2(factor[1]))
            if self.vlevel > 0:
                print(self.str_tab, self.name,
                      '- Setting factor %dx%d' % (factor[0], factor[1]))
            assert self.factor[0] == self.factor[1]
            assert factor[0] == factor[1]
            self._levels = int(np.log2(factor[0]))
            self.factor = factor

            self.border_in = (0, 0, 0, 0)
            self.border_out = (0, 0, 0, 0)
            self.border_cut = (0, 0, 0, 0)

            probe_shape = (1, self.input_channels, 32, 32)
            probe = Variable(
                torch.FloatTensor(np.zeros(probe_shape)),
                requires_grad=False
            )
            if self.device >= 0:
                probe = probe.cuda(self.device)
            if self.dtype == 'float16':
                probe = probe.half()

            out_size = self.forward(probe, pad=False).size()
            pad_v = probe.size()[2]*self.factor[0]-out_size[2]
            if pad_v > 0:
                pad_v += (self.factor[0]*2) - (pad_v % (self.factor[0]*2))
            pad_h = probe.size()[3]*self.factor[1]-out_size[3]
            if pad_h > 0:
                pad_h += (self.factor[1]*2) - (pad_h % (self.factor[1]*2))
            self.border_out = (
                pad_h//2, pad_h-pad_h//2,
                pad_v//2, pad_v-pad_v//2
            )
            assert np.all(np.asarray(self.border_out) >= 0)

            assert pad_v % (2*self.factor[0]) == 0
            assert pad_h % (2*self.factor[1]) == 0
            pad_v = pad_v // self.factor[0]
            pad_h = pad_h // self.factor[1]
            self.border_in = (
                pad_h//2, pad_h-pad_h//2,
                pad_v//2, pad_v-pad_v//2
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
                      '- outwithpad_size', outwithpad_size)
                print(self.str_tab, self.name,
                      '- border_out', self.border_out)
                print(self.str_tab, self.name,
                      '- border_in', self.border_in)
                print(self.str_tab, self.name,
                      '- border_cut', self.border_cut)
        self._cached_border_in[s] = self.border_in
        self._cached_border_out[s] = self.border_out
        self._cached_border_cut[s] = self.border_cut
