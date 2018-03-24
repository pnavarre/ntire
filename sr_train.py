#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 15:02:39 2015

@author: Pablo Navarrete Michelini
"""
import os
import time
import math
import PIL
import numpy as np
import torch
import torch.optim as optim
import shutil
import torchvision.utils as vutils
from os.path import expanduser
from torch.autograd import Variable
from tqdm import tqdm
from tensorboardX import SummaryWriter

from Sampler import Sampler_UpDown, DatasetFromFolder
from layers import MSE_pix, Charbonnier_pix, SSIM_pix, PSNR_RGB, \
    L1WbLoss, weight_bias_init, LR_scheduler
from sr_model import Upscaler as Upscaler


def prime_factorize(n):
    factors = []
    number = math.fabs(n)
    while number > 1:
        factor = get_next_prime_factor(number)
        factors.append(factor)
        number /= factor
    if n < -1:
        factors[0] = -factors[0]
    return tuple(factors)


def get_next_prime_factor(n):
    if n % 2 == 0:
        return 2
    for x in range(3, int(math.ceil(math.sqrt(n)) + 1), 2):
        if n % x == 0:
            return x
    return int(n)


def to_np(x):
    return x.data.cpu().numpy()


if __name__ == '__main__':
    home = expanduser('~')
    with open(home+'/../tensorboard_dir.cfg', 'r') as myfile:
        tbdir = myfile.readline()[:-1]
    os.chdir('../../build')
    MSE = MSE_pix()
    Charbonnier = Charbonnier_pix()
    SSIM = SSIM_pix()
    PsnrRGB = PSNR_RGB(maxv=1., cut=None)
    model_id, init_model = None, None

    gpu = 0
    debug_plot = False
    vlevel = 1
    mu = 0
    model_file, init_model, mu, model_id = None, 'normal_7e-2', 4, 'CH3_LZ6_FE32_LEAK0.2_MX0_BN0_Analysis#L3#X1#K3#D1_Upscaling#L10#M5#X1#K3#D1_Downscaling#L10#M5#K3#D1_Synthesis#L3#X1#K3#D1_v3_ms'
    #model_file, init_model, mu, model_id = None, 'normal_7e-2', 4, 'CH3_LZ6_FE32_LEAK0.2_MX1_BN0_Analysis#L3#X1#K3#D1_Upscaling#L10#M5#X1#K3#D1_Downscaling#L10#M5#K3#D1_Synthesis#L3#X1#K3#D1_v3_ms'

    HR_shape = [128, 128]

    if model_id is None and model_file is not None:
        model_id = 'CH' + model_file.split('_CH')[1][:-4]

    minibatch_size = 116
    train_samples = 1000 * minibatch_size
    train_resample = '100%'
    reject_factor = 0.
    learn_rate = 1e-3
    op_algorithm = 'adam'
    train_scheduler = 'sqrt'
    train_tau = 1_000
    weight_decay = 1e-4
    valid_samples = -1
    rgb_weight = [0.299, 0.587, 0.114]
    g_label = {'weight': 0, 'bias': 1}
    g_lr_factor = {'weight': 1., 'bias': 2.}
    downscaler_mode = PIL.Image.BICUBIC
    train_metric = Charbonnier
    train_metric_name = str(train_metric).split('_')[0]
    train_weight = {
        'backprojection_%s_2x_out_2x' % train_metric_name: 1.,
        'backprojection_%s_4x_out_2x' % train_metric_name: 1.,
        'backprojection_%s_4x_out_4x' % train_metric_name: 1.,
        'backprojection_%s_8x_out_2x' % train_metric_name: 1.,
        'backprojection_%s_8x_out_4x' % train_metric_name: 1.,
        'backprojection_%s_8x_out_8x' % train_metric_name: 1.,
#        '%s_2x_out_2x' % train_metric_name: 1.,
#        '%s_2x_out_4x' % train_metric_name: 1.,
#        '%s_2x_out_8x' % train_metric_name: 1.,
        '%s_4x_out_2x' % train_metric_name: 1.,
        '%s_4x_out_4x' % train_metric_name: 1.,
#        '%s_4x_out_8x' % train_metric_name: 1.,
        '%s_8x_out_2x' % train_metric_name: 1.,
        '%s_8x_out_4x' % train_metric_name: 1.,
        '%s_8x_out_8x' % train_metric_name: 1.,
        # 'l1wb': 1e-6,
    }
    val_metric = PsnrRGB
    val_metric_name = str(val_metric).split('_')[0]
    val_best = {
        '%s_2x_out_2x' % val_metric_name: 0.,
#        '%s_2x_out_4x' % val_metric_name: 0.,
#        '%s_2x_out_8x' % val_metric_name: 0.,
        '%s_4x_out_2x' % val_metric_name: 0.,
        '%s_4x_out_4x' % val_metric_name: 0.,
#        '%s_4x_out_8x' % val_metric_name: 0.,
        '%s_8x_out_2x' % val_metric_name: 0.,
        '%s_8x_out_4x' % val_metric_name: 0.,
        '%s_8x_out_8x' % val_metric_name: 0.,
    }
    path_train = [
        '/opt/dataset/DIV2K_2017_shifted_split/Track 1 - Clean/DIV2K_train_HR',
        #'/opt/dataset/DIV2K_2017_shifted_split/Track 2 - Real World/DIV2K_train_HR'
        #'/opt/dataset/DIV2K_2017_shifted_split/Track 3 - Realistic Difficult/DIV2K_train_HR'
        #'/opt/dataset/DIV2K_2017_shifted_split/Track 4 - Realistic Wild/DIV2K_train_HR'
    ]
    ref_train = {
        'R8': '/opt/dataset/DIV2K_2017_shifted_split/Track 1 - Clean/DIV2K_train_LR_x8'
        #'R4': '/opt/dataset/DIV2K_2017_shifted_split/Track 2 - Real World/DIV2K_train_LR_mild'
        #'R4': '/opt/dataset/DIV2K_2017_shifted_split/Track 3 - Realistic Difficult/DIV2K_train_LR_difficult'
        #'R4': '/opt/dataset/DIV2K_2017_shifted_split/Track 4 - Realistic Wild/DIV2K_train_LR_wild'
    }
    path_val = [
        #'/opt/dataset/SRCNN/Set14',
        '/opt/dataset/DIV2K_2017_shifted_split/Track 1 - Clean/DIV2K_valid_HR_patch1k'
        #'/opt/dataset/DIV2K_2017_shifted_split/Track 2 - Real World/DIV2K_valid_HR_patch1k'
        #'/opt/dataset/DIV2K_2017_shifted_split/Track 3 - Realistic Difficult/DIV2K_valid_HR_patch1k'
        #'/opt/dataset/DIV2K_2017_shifted_split/Track 4 - Realistic Wild/DIV2K_valid_HR_patch4k'
    ]
    ref_valid = {
        'R8': '/opt/dataset/DIV2K_2017_shifted_split/Track 1 - Clean/DIV2K_valid_LR_clean_patch1k'
        #'R4': '/opt/dataset/DIV2K_2017_shifted_split/Track 2 - Real World/DIV2K_valid_LR_mild_patch1k'
        #'R4': '/opt/dataset/DIV2K_2017_shifted_split/Track 3 - Realistic Difficult/DIV2K_valid_LR_difficult_patch1k'
        #'R4': '/opt/dataset/DIV2K_2017_shifted_split/Track 4 - Realistic Wild/DIV2K_valid_LR_wild_patch4k'
    }

    assert len(ref_train) <= 1
    train_ref = [d for d in ref_train][0] if len(ref_train) == 1 else None
    val_ref = [d for d in ref_valid][0] if len(ref_valid) == 1 else None

    if model_id is None and model_file is not None:
        model_id = 'CH' + model_file.split('_CH')[1][:-4]

    impsamp = (reject_factor > 0.)
    train_print = None
    m = -np.inf
    for k, d in train_weight.items():
        if d >= m:
            m, train_print = d, k
    key_scheduler = None
    for k, _ in val_best.items():
        key_scheduler = k

    if not torch.cuda.is_available():
        gpu = -1
    torch.backends.cudnn.benchmark = True
    logdir = tbdir + '/upscaler_gpu%d_%s' % (gpu, model_id)
    shutil.rmtree(logdir, ignore_errors=True)
    logger = SummaryWriter(logdir)
    set_factor = [1, 1]
    for k, d in train_weight.items():
        if k.find('x_out_') > 0:
            f = int(k.split('x_out_')[0].split('_')[-1])
            if f > set_factor[0]:
                set_factor = [f, f]

    model = Upscaler(
        model_id,
        device=gpu,
        name='[gpu%d] upscaler' % gpu,
        str_tab='  >', vlevel=vlevel
    )
    model.set_mu(mu)
    model.set_factor(set_factor)
    factors = [1]
    if model_id.endswith('_v5_ms'):
        for k in prime_factorize(2**model._levels):
            factors.append(factors[-1] * k)
    else:
        for k in prime_factorize(model.factor[0]):
            factors.append(factors[-1] * k)
    max_factor = max(factors)
    probe_shape = (
        1, model.input_channels,
        HR_shape[0]//max_factor + model.border_in[2] + model.border_in[3],
        HR_shape[1]//max_factor + model.border_in[0] + model.border_in[1]
    )
    probe = Variable(
        torch.FloatTensor(np.zeros(probe_shape)),
        requires_grad=False
    )
    if gpu >= 0:
        probe = probe.cuda(gpu)
    probe_out, _ = model.all_out(probe)
    HR_shape[0] = probe_out[max_factor].shape[2]
    HR_shape[1] = probe_out[max_factor].shape[3]

    label = 'upscaler_gpu%d_%s' % (gpu, model_id)
    logger.add_text(label, 'model_id: %s - mu: %d - gpu: %d - nparam: %d' % (model_id, mu, gpu, model.stat_nparam), 0)
    logger.add_text(label, 'set_factor: %dx%d - model: %s' % (set_factor[0], set_factor[1], str(model)), 0)
    logger.add_text(label, 'path_train: %s - path_val: %s' % (path_train, path_val), 0)
    logger.add_text(label, 'op_algorithm: %s - init_model: %s' % (op_algorithm, init_model), 0)
    logger.add_text(label, 'train_scheduler: %s - train_tau: %s' % (train_scheduler, train_tau), 0)
    logger.add_text(label, 'train_samples: %d - minibatch_size: %d' % (train_samples, minibatch_size), 0)
    logger.add_text(label, 'impsamp: %s - reject_factor: %s' % (str(impsamp), reject_factor), 0)
    logger.add_text(label, 'learn_rate: %s - train_resample: %s - weight_decay=%s' % (learn_rate, train_resample, weight_decay), 0)
    logger.add_text(label, 'rgb_weight: [%.4f %.4f %.4f]' % (rgb_weight[0], rgb_weight[1], rgb_weight[2]), 0)
    txt = 'g_lr_factor: '
    for k, d in g_lr_factor.items():
        txt += k + '=%s - ' % d
    logger.add_text(label, txt, 0)
    txt = 'train_weight: '
    for k, d in train_weight.items():
        txt += k + '=%s - ' % d
    logger.add_text(label, txt, 0)
    txt = 'val_best: '
    for k, d in val_best.items():
        txt += k + '=%s - ' % d
    logger.add_text(label, txt, 0)

    if model_file is not None:
        model.load_state_dict(
            torch.load(model_file, map_location=lambda storage, loc: storage)
        )
    else:
        weight_bias_init(
            model, init_model.split('_')[0],
            gain=torch.nn.init.calculate_gain(
                'leaky_relu', param=float(init_model.split('_')[1])
            ) if init_model.split('_')[0] == 'kaiming' or
            init_model.split('_')[0] == 'xavier' else
            float(init_model.split('_')[1])
        )

    L1Wb = L1WbLoss(model)

    if model_file is not None:
        print('\n- Load model', flush=True)
        model.load_state_dict(
            torch.load(model_file, map_location=lambda storage, loc: storage)
        )

    if gpu >= 0:
        model.cuda(gpu)
        MSE.cuda(gpu)
        Charbonnier.cuda(gpu)
        SSIM.cuda(gpu)
        PsnrRGB.cuda(gpu)
        L1Wb.cuda(gpu)
        if 'style' in train_weight:
            Style.cuda(gpu)

    batch_shape = (
        minibatch_size,
        model.input_channels,
        HR_shape[0], HR_shape[1]
    )
    down2up = {down: up for down, up in zip(factors, factors[::-1])}
    if train_ref is not None:
        down2up[train_ref] = train_ref
    shapes = {down2up[k]: d.shape for k, d in probe_out.items()}
    if not model_id.endswith('_v5_ms'):
        shapes[max_factor] = probe.shape
    if train_ref is not None:
        shapes[train_ref] = probe.shape

    train_sampler = Sampler_UpDown(
        rng=np.random.RandomState(12345),
        output_paths=path_train,
        reference_paths=ref_train,
        scale_list=[1.],
        impsamp=impsamp,
        reject_factor=reject_factor,
        downscale=downscaler_mode,

        mbatch=batch_shape[0],
        shape=shapes,

        transpose_modes=[
            None,
            PIL.Image.FLIP_LEFT_RIGHT,
            PIL.Image.FLIP_TOP_BOTTOM,
            PIL.Image.ROTATE_90,
            PIL.Image.ROTATE_180,
            PIL.Image.ROTATE_270,
            PIL.Image.TRANSPOSE
        ],

        enable_plot=debug_plot,
        dtype=torch.FloatTensor,
        name='[cpu] patch sampler',
        str_tab='  >',
        vlevel=vlevel
    )
    valid_sampler = DatasetFromFolder(
        epoch_size=valid_samples,
        output_paths=path_val,
        reference_paths=ref_valid,
        down_factors=tuple(factors),
        downscaler=downscaler_mode
    )
    dset = {'train': train_sampler, 'validation': valid_sampler}
    train_sampler(train_samples).shuffle()
    if debug_plot:
        train_sampler.show()
    train_sampler.info()

    g_param = []
    for label in g_label:
        g_param.append({'params': [], 'lr': g_lr_factor[label] * learn_rate})
    for tag, value in model.named_parameters():
        name = tag[::-1].partition('.')[0][::-1]
        assert name in g_label, 'unknown parameter \'%s\'' % name
        g_param[g_label[name]]['params'].append(value)
    if op_algorithm == 'sgd':
        optimizer = optim.SGD(
            g_param,
            lr=learn_rate,
            momentum=0.1,
            dampening=0, weight_decay=0, nesterov=True
        )
    elif op_algorithm == 'rmsprop':
        optimizer = optim.RMSprop(
            g_param,
            lr=learn_rate,
            alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False
        )
    elif op_algorithm == 'adam':
        optimizer = optim.Adam(
            g_param,
            lr=learn_rate,
            betas=(0.9, 0.999), eps=1e-08, weight_decay=0
        )
    else:
        assert False
    if train_scheduler.startswith('adapt'):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, train_scheduler.split('_')[1],
            factor=0.9,
            patience=int(train_tau)
        )
    elif train_scheduler.startswith('step'):
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=train_tau,
            gamma=0.5
        )
    else:
        scheduler = LR_scheduler(
            optimizer, mode=train_scheduler,
            g_label=g_label, g_lr_factor=g_lr_factor,
            init_lr=learn_rate, tau=train_tau
        )

    backup_best = {}
    for key in val_best:
        backup_best[key] = 'models/upscaler_%s_gpu%d_%s_best_%s_%s.pkl' % (
            os.uname()[1], gpu, time.strftime("%Y-%m-%d-%H%M%S"), key, model_id
        )
    backup_current = 'models/upscaler_%s_gpu%d_%s_current_%s.pkl' % (
        os.uname()[1], gpu, time.strftime("%Y-%m-%d-%H%M%S"), model_id
    )

    one = torch.FloatTensor([1]) \
        if gpu < 0 else torch.FloatTensor([1]).cuda(gpu)
    RGB_weight = Variable(
        torch.FloatTensor(rgb_weight).unsqueeze(0).expand(
            [minibatch_size, model.input_channels]
        ).cuda(gpu)
    )

    epoch = 0
    train_loss = {}
    stat_train = {}

    val_score = {}
    best_epoch = {}
    for k, _ in val_best.items():
        best_epoch[k] = 0
    stat_val = {}
    while True:
        print(
            '  > [gpu{}] epoch {} - Model {}'.format(gpu, epoch, model_id),
            flush=True
        )
        for phase in ['train', 'validation']:
            if phase == 'train':
                if train_resample is not None and epoch > 0:
                    if debug_plot:
                        train_sampler.show()
                    train_sampler.info()
                    train_sampler(train_resample)
                    scheduler.update = False
                train_sampler.shuffle()
                model.train(True)
            else:
                model.train(False)

            for k, _ in train_weight.items():
                stat_train[k] = 0.
            for k, _ in val_best.items():
                stat_val[k] = 0.

            minibatch = tqdm(dset[phase])
            minibatch.set_description(
                desc='  > [gpu{}] epoch {}'.format(gpu, epoch)
            )
            log_img = {}
            if phase == 'train':
                for key, mb in minibatch:
                    if gpu < 0:
                        mb = {k: Variable(d) for k, d in mb.items()}
                    else:
                        mb = {k: Variable(d.cuda(gpu)) for k, d in mb.items()}

                    for input_dfactor in factors[1:]:
                        model.set_factor([input_dfactor, input_dfactor])
                        if train_ref == 'R'+str(input_dfactor):
                            mb_input = mb[train_ref]
                        else:
                            mb_input = mb[input_dfactor]

                        if model_id.endswith('_v5_ms'):
                            target_downfactors = factors
                            mb_targets = {
                                max_factor//k: mb[k] for k in target_downfactors
                            }
                        else:
                            target_downfactors = []
                            for d in factors:
                                if d > input_dfactor:
                                    break
                                target_downfactors.append(d)
                            mb_targets = {
                                input_dfactor//k: mb[k] for k in target_downfactors
                            }

                        mb_outs, mb_losses = model.all_out(
                            mb_input,
                            ref=mb_targets,
                            loss_metric=train_metric
                        )
                        for target_factor in mb_targets:
                            s = '%s_%dx_out_%dx' % (
                                train_metric_name, input_dfactor, target_factor
                            )
                            if s in train_weight:
                                train_loss[s] = (
                                    mb_losses['HR_'+str(target_factor)] * RGB_weight
                                ).mean()
                            if 'backprojection_'+s in train_weight:
                                train_loss['backprojection_'+s] = (
                                    mb_losses['LR_'+str(target_factor)] * RGB_weight
                                ).mean()
                            if epoch == 0:
                                log_img['train_%dx_out_%dx_target' % (input_dfactor, target_factor)] = mb_targets[target_factor]
                            log_img['train_%dx_out_%dx_output' % (input_dfactor, target_factor)] = mb_outs[target_factor]

                    if 'l1wb' in train_weight:
                        train_loss['l1wb'] = L1Wb()
                    model.set_factor(set_factor)

                    for k, var in train_loss.items():
                        stat_train[k] += var.data[0]

                    optimizer.zero_grad()
                    for k, var in train_loss.items():
                        train_loss[k].backward(
                            train_weight[k]*one,
                            retain_graph=True
                        )
                    optimizer.step()
                    if train_scheduler.startswith('step'):
                        scheduler.step()
                    else:
                        scheduler.step(metrics=stat_val[key_scheduler])

                    minibatch.set_postfix(
                        metric=train_print,
                        loss=train_loss[train_print].data[0]
                    )
                del mb_outs, mb_input, mb_targets

                minibatch.refresh()
                minibatch.close()

                for key in stat_train:
                    stat_train[key] = stat_train[key] / len(train_sampler)

                torch.save(model.state_dict(), backup_current)

                info = {
                    'learn_rate': optimizer.param_groups[0]['lr']
                }
                for key in stat_train:
                    info['Loss_%s' % key] = stat_train[key]
                for tag, value in info.items():
                    logger.add_scalar(tag, value, epoch)

                print(
                    '  > [gpu{}] epoch {} - Learning rate: {}'.
                    format(
                        gpu, epoch, scheduler.optimizer.param_groups[0]['lr']
                    ), flush=True
                )
            elif phase == 'validation':
                for mb in minibatch:
                    if gpu < 0:
                        mb = {k: Variable(d, requires_grad=False) for k, d in mb.items()}
                    else:
                        mb = {k: Variable(d.cuda(gpu), requires_grad=False) for k, d in mb.items()}

                    list_dfactors = list(set([int(d.split('x_out')[0].split('_')[1]) for d in val_best]))
                    for input_dfactor in list_dfactors:
                        if val_ref == 'R'+str(input_dfactor):
                            mb_input = mb[val_ref]
                        else:
                            mb_input = mb[input_dfactor]

                        if model_id.endswith('_v5_ms'):
                            target_downfactors = factors
                            mb_targets = {
                                max_factor//k: mb[k] for k in target_downfactors
                            }
                        else:
                            target_downfactors = []
                            for d in factors:
                                if d > input_dfactor:
                                    break
                                target_downfactors.append(d)
                            mb_targets = {
                                input_dfactor//k: mb[k] for k in target_downfactors
                            }

                        for out_factor, out_target in mb_targets.items():
                            if model_id.endswith('_v5_ms'):
                                model.set_factor([input_dfactor, input_dfactor])
                                mb_outs, _ = model.all_out(mb_input, pad=True)
                                mb_out = mb_outs[out_factor]
                            else:
                                model.set_factor([out_factor, out_factor])
                                mb_out = model(mb_input, pad=True)
                            val_score['%s_%dx_out_%dx' % (
                                val_metric_name, input_dfactor, out_factor
                            )] = (val_metric(mb_out, out_target) * RGB_weight).mean()

                            if epoch == 0:
                                log_img['validation%dx_out_%dx_target' % (input_dfactor, out_factor)] = out_target
                            log_img['validation%dx_out_%dx_output' % (input_dfactor, out_factor)] = mb_out
                    model.set_factor([max_factor, max_factor])

                    for k in stat_val:
                        stat_val[k] += val_score[k].data[0]

                    minibatch.set_postfix(
                        h=mb_input.shape[2], w=mb_input.shape[3],
                        H=mb_out.shape[2], W=mb_out.shape[3],
                    )
                del mb_out, mb_input, mb_targets

                minibatch.refresh()
                minibatch.close()

                for key in stat_val:
                    stat_val[key] = stat_val[key] / valid_samples

                for key in stat_val:
                    if stat_val[key] > val_best[key]:
                        val_best[key] = stat_val[key]
                        best_epoch[key] = epoch
                        print(
                            '  > [gpu{}] epoch {} - Validation: best {}'.
                            format(gpu, epoch, key)
                        )
                        torch.save(model.state_dict(), backup_best[key])
                    else:
                        print(
                            '  > [gpu{}] epoch {} - Validation: '
                            'best {} {} epochs behind.'.format(
                                gpu, epoch, key, epoch - best_epoch[key]
                            )
                        )

                info = {}
                for key in stat_val:
                    info['%s %s' % (phase, key)] = stat_val[key]
                for tag, value in info.items():
                    logger.add_scalar(tag, value, epoch)

                info = {}
                for s, img in log_img.items():
                    info[s] = vutils.make_grid(
                        log_img[s].contiguous().view(
                            -1, 3, log_img[s].shape[2], log_img[s].shape[3]
                        ).data.clamp_(0., 1.), normalize=False, pad_value=1.
                    )
                for tag, images in info.items():
                    logger.add_image(tag, images, epoch)
            else:
                assert False

        epoch += 1

    logger.close()
