#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 23:00:37 2018

@author: Pablo Navarrete Michelini
"""
from os.path import basename
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from sr_model import Upscaler as Upscaler


if __name__ == '__main__':
    gpu = 0
    track = 1
    input_file = 'sample.tif'

    if track == 1:
        set_factor = [8, 8]
        from sr_model_track1 import Upscaler as Upscaler
        mu, model_file = 2, 'models/track_1_mu_0_CH3_LZ6_FE32_LEAK0.2_BN0_Analysis#L3#X1#K3#D1_Upscaling#L10#M5#X1#K3#D1_Synthesis#L3#X1#K3#D1_v1_ms.pkl'
    elif track == 2:
        set_factor = [4, 4]
        mu, model_file = 2, 'models/track_2_mu_2_CH3_LZ6_FE32_LEAK0.2_MX0_BN0_Analysis#L3#X1#K3#D1_Upscaling#L10#M5#X1#K3#D1_Downscaling#L10#M5#K3#D1_Synthesis#L3#X1#K3#D1_v3_ms.pkl'
    elif track == 3:
        set_factor = [4, 4]
        mu, model_file = 8, 'models/track_3_mu_8_CH3_LZ6_FE32_LEAK0.2_MX1_BN0_Analysis#L3#X1#K3#D1_Upscaling#L10#M5#X1#K3#D1_Downscaling#L10#M5#K3#D1_Synthesis#L3#X1#K3#D1_v3_ms.pkl'
    elif track == 4:
        set_factor = [4, 4]
        mu, model_file = 2, 'models/track_4_mu_2_CH3_LZ6_FE32_LEAK0.2_MX0_BN0_Analysis#L3#X1#K3#D1_Upscaling#L10#M5#X1#K3#D1_Downscaling#L10#M5#K3#D1_Synthesis#L3#X1#K3#D1_v3_ms.pkl'
    else:
        assert False, "Not supported"

    PIL_to_Tensor = transforms.ToTensor()
    model_id = 'CH' + model_file.split('_CH')[1][:-4]
    if not torch.cuda.is_available():
        gpu = -1
    torch.backends.cudnn.benchmark = True

    print('\n- Load model')
    model = Upscaler(model_id, device=gpu)
    model.set_factor(set_factor)
    model.set_mu(mu)
    model.load_state_dict(
        torch.load(model_file, map_location=lambda storage, loc: storage)
    )
    for param in model.parameters():
        param.requires_grad = False
    model.train(False)
    if gpu >= 0:
        model.cuda(gpu)

    print('\n- Testing', flush=True)
    cut = tuple([6 + model.factor[0]] * 4)
    net_input_pil = Image.open(input_file).convert('RGB')

    net_input_pil = net_input_pil.crop((
        0, 0,
        (net_input_pil.size[0]//model.factor[0])*model.factor[0],
        (net_input_pil.size[1]//model.factor[0])*model.factor[0]
    ))

    net_input_tensor = Variable(
        PIL_to_Tensor(net_input_pil).unsqueeze(0),
        requires_grad=False
    )
    if gpu >= 0:
        net_input_tensor = net_input_tensor.cuda(gpu)

    net_output_rgb = model(
        net_input_tensor, pad=True
    ).data[0].clamp(0, 1.).permute(1, 2, 0).cpu().numpy() * 255.
    sr_filename = basename(input_file)[:-4] + \
        '_%dx%d_SR' % (model.factor[0], model.factor[1]) + '.png'
    Image.fromarray(
        np.uint8(np.round(net_output_rgb[cut[0]:-cut[1], cut[2]:-cut[3]]))
    ).save(sr_filename)
