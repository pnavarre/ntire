#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 16:22:43 2017

@author: pablo
"""
import PIL
import time
import h5py
import random
import torch
import numpy as np
import torch.utils.data as data
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from tqdm import trange
from pathlib import Path
from torchvision import transforms


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [
        '.png', '.tif', '.jpg', '.jpeg', '.bmp', '.pgm'
    ])


class DatasetFromFolder(data.Dataset):
    def __init__(self, output_paths,
                 reference_paths={},
                 epoch_size=-1, down_factors=[1],
                 downscaler=PIL.Image.BICUBIC):
        super().__init__()
        self.down_factors = tuple(down_factors)
        self.downscaler = downscaler

        self.filelist = [
            str(f) for path in output_paths for f in Path(path).iterdir()
            if is_image_file(str(f))
        ]

        self.epoch_size = len(self.filelist)
        if epoch_size > 0:
            assert epoch_size <= len(self.filelist)
            self.epoch_size = epoch_size
            self.filelist = self.filelist[:epoch_size]

        for f in self.down_factors:
            assert isinstance(f, int)

        max_factor = max(self.down_factors)
        self.reference_paths = reference_paths
        for k, path in self.reference_paths.items():
            assert k[0] == 'R' and max_factor % int(k[1:]) == 0, '%s %s' % (k, max_factor)
            p = Path(path)
            for f in self.filelist:
                fp = Path(f)
                assert len(
                    [x for x in p.glob(fp.name[:-4]+'*'+fp.suffix)]
                ) > 0, '%s not found in %s' % (fp.name[:-4]+'*'+fp.suffix, p)

    def __getitem__(self, index):
        img = Image.open(self.filelist[index]).convert('RGB')
        max_factor = max(self.down_factors)
        self.transform = []
        in_size = (np.asarray(img.size)//max_factor)*max_factor
        for f in self.down_factors:
            assert np.all(in_size % f == 0)
        out_size = in_size
        for f in self.down_factors:
            out_size = in_size//f
            self.transform.append(
                transforms.Compose([
                    transforms.CenterCrop(in_size),
                    transforms.Resize(
                        tuple(out_size), interpolation=self.downscaler
                    ),
                    transforms.ToTensor()
                ])
            )

        outputs = {}
        for tr, f in zip(self.transform, self.down_factors):
            outputs[f] = tr(img).expand(1, 3, -1, -1)
        for f, _ in outputs.items():
            outputs[f] = torch.round(255.*outputs[f])/255.

        in_path = Path(self.filelist[index])
        for k, path in self.reference_paths.items():
            f = int(k[1:])
            assert np.all(in_size % f == 0)
            out_size = in_size//f
            transform = transforms.Compose([
                transforms.CenterCrop(out_size),
                transforms.ToTensor()
            ])
            matches = [
                x for x in Path(path).glob(
                    in_path.name[:-4] + '*' + in_path.suffix
                )
            ]
            assert len(matches) == 1
            ref_filename = str(matches[0])

            rimg = Image.open(ref_filename).convert('RGB')
            outputs[k] = transform(rimg).expand(1, 3, -1, -1)

        return outputs

    def __len__(self):
        return self.epoch_size


def block_inside(H, W, i, j, h, w):
    return i >= 0 and j >= 0 and i + h <= H and j + w <= W


class Sampler_Base(object):
    def __init__(self,
                 output_paths,
                 scale_list,
                 impsamp=True,
                 mbatch=1,
                 rng=np.random.RandomState(12345),
                 enable_plot=True,
                 dtype=torch.FloatTensor,
                 name='sampler',
                 str_tab='',
                 vlevel=np.inf):
        self.rng = rng
        self.dtype = dtype
        self.name = name
        self.str_tab = str_tab
        self.vlevel = vlevel
        self.enable_plot = enable_plot
        self.mbatch = mbatch

        self.output_paths = output_paths
        self.scale_list = scale_list
        self._impsamp = impsamp

        self.num = 0
        self.samples = {}
        self.samples['score'] = -torch.ones(0).type(self.dtype)
        self.samples['label'] = torch.ones(0).long()
        self.label_idx = {}

        self.loops = 0
        self.filelist = [
            str(f) for path in output_paths for f in Path(path).iterdir()
            if is_image_file(str(f))
        ]
        self.pool_size = len(self.filelist)

        random.shuffle(self.filelist)

        if self.vlevel > 0:
            print(self.str_tab, self.name, '- Path: ', self.output_paths)
            print(self.str_tab, self.name, '- Number of files in pool:',
                  '{:_}'.format(self.pool_size), flush=True)

    def __call__(self, new_samples, add={}, impdrop=True):
        if self.num + new_samples < 0:
            new_samples = -self.num
        self.num += new_samples
        if new_samples < 0 and impdrop:
            x = self.samples['score']
            max_idx = [
                i[0] for i in sorted(enumerate(x), key=lambda x:x[1])
            ][:self.num]
        else:
            max_idx = np.arange(self.num)
        for key, data in add.items():
            if new_samples > 0:
                self.samples[key] = torch.cat(
                    (self.samples[key], data[:new_samples]),
                    dim=0
                )
            else:
                self.samples[key] = self.samples[key][max_idx]
        if new_samples > 0:
            self.samples['score'] = torch.cat(
                (self.samples['score'],
                 -torch.ones(new_samples).type(self.dtype)),
                dim=0
            )
        elif self.num > 0:
            self.samples['score'] = self.samples['score'][max_idx]
        else:
            for key in self.samples:
                if self.samples[key].ndimension() == 4:
                    self.samples[key] = torch.zeros(torch.Size(
                        [0] + [x for x in self.samples[key].shape[1:]]
                    )).type(self.dtype)
            self.samples['score'] = -torch.ones(0).type(self.dtype)

        if self.vlevel > 0:
            print(self.str_tab, self.name, '- Move to Pinned Memory')
        for d in self.samples:
            self.samples[d] = self.samples[d].pin_memory()

        if self.vlevel > 0:
            print(self.str_tab, self.name, '- Number of files in pool:',
                  '{:_}'.format(len(self.filelist)))
            print(self.str_tab, self.name, '- Added batches:',
                  '{:_}'.format(new_samples))
            print(self.str_tab, self.name, '- Total batches:',
                  '{:_}'.format(self.num), flush=True)

        return self

    def __len__(self):
        return self.num//self.mbatch

    def __setitem__(self, key, item):
        for d in self.samples:
            self.samples[d][key] = item[d]

    def __getitem__(self, key):
        if key < 0 or key >= len(self):
            raise IndexError('index {} is out of range'.format(key))
        r = {}
        for d in self.samples:
            r[d] = self.samples[d][key*self.mbatch:(key+1)*self.mbatch]
        return key, r

    def shuffle(self, hflip=False, vflip=False):
        if self.num > 1:
            if self.vlevel > 0:
                print(self.str_tab, self.name, '- Shuffling', flush=True)
            idx = torch.arange(0, self.samples['score'].size()[0]).long()
            random.shuffle(idx)
            idx = idx[:self.num]
            if self.vlevel > 0 and (hflip or vflip):
                print(self.str_tab, self.name, '- Flipping', flush=True)
            hidx = torch.LongTensor(
                np.random.choice(np.arange(self.num),
                                 self.num//2, replace=False)
            )
            vidx = torch.LongTensor(
                np.random.choice(np.arange(self.num),
                                 self.num//2, replace=False)
            )
            for key, _ in self.samples.items():
                self.samples[key] = self.samples[key][idx]
                if self.samples[key].ndimension() == 4:
                    if hflip:
                        width = self.samples[key].size()[3]
                        hrev = torch.arange(width-1, -1, -1).long()
                        self.samples[key][hidx] = \
                            self.samples[key][hidx][:, :, :, hrev]
                    if vflip:
                        height = self.samples[key].size()[3]
                        vrev = torch.arange(height-1, -1, -1).long()
                        self.samples[key][vidx] = \
                            self.samples[key][vidx][:, :, vrev, :]
            if self.vlevel > 0:
                print(self.str_tab, self.name,
                      '- Move to Pinned Memory', flush=True)
            for d in self.samples:
                self.samples[d] = self.samples[d].pin_memory()

        return self

    def get_file(self):
        f = self.filelist.pop()
        if len(self.filelist) == 0:
            self.filelist = [
                str(f) for path in self.output_paths
                for f in Path(path).iterdir() if is_image_file(str(f))
            ]
            random.shuffle(self.filelist)
            self.loops += 1
        return f

    def reset_filepool(self):
        self.filelist = [
            str(f) for path in self.output_paths
            for f in Path(path).iterdir() if is_image_file(str(f))
        ]
        random.shuffle(self.filelist)
        self.loops = 0

    def info(self):
        if self.vlevel > 0:
            print(self.str_tab, self.name,
                  '- Output paths: ', self.output_paths)
            print(self.str_tab, self.name, '- Number of files in pool:',
                  len(self.filelist))
            total_size = 0
            for key, data in self.samples.items():
                level_size = 4 * np.prod(data.size()) / (1024.*1000.)
                total_size += level_size
                print(self.str_tab, self.name, '- Samples', key, ': ',
                      data.size(), ' size: %.2f MB' % level_size)
            print(self.str_tab, self.name,
                  '- TOTAL SIZE: %.2f MB' % total_size, flush=True)

    def show(self, samples=8, plot=True, rand=False):
        if self.num > 0 and plot:
            if self.samples['score'].size()[0] < samples:
                samples = self.samples['score'].size()[0]

            image_num = 0
            scalar_list = []
            for key, data in self.samples.items():
                if len(data.size()) == 1:
                    scalar_list.append(key)
                else:
                    image_num += 1
            scalar_num = np.size(scalar_list)

            pindex = np.arange(samples)
            if rand:
                pindex = random.sample(
                    np.arange(self.samples['score'].size()[0]).tolist(), samples
                )
            plt.figure(figsize=(15., 20.*samples/8))
            k = 1
            col = max(2, image_num)
            for p in pindex:
                for key, data in self.samples.items():
                    data = data.numpy()
                    if data.ndim > 1:
                        plt.subplot(samples, col, k)
                        k += 1
                        if data.shape[1] == 1:
                            plt.imshow(data[p, 0, :, :],
                                       cm.Greys_r, interpolation='nearest',
                                       vmin=0., vmax=1.)
                        else:
                            plt.imshow(
                                np.array(
                                    np.clip(
                                        data[p, :3, :, :].transpose([1, 2, 0]),
                                        0., 1.
                                    )*255., dtype=np.uint8
                                ),
                                interpolation='nearest',
                                vmin=0., vmax=1.
                            )
                        title = ''
                        if image_num > 1:
                            title = str(key) + ' - '
                        for scalar in scalar_list:
                            title += str(scalar) + '=' + \
                                     str(self.samples[scalar][p])[:4] + ' - '
                        plt.title(title[:-3])

            plt.subplots_adjust(hspace=0.3)
            plt.show()

            k = 1
            plt.figure(figsize=(10., 2.))
            for scalar in scalar_list:
                plt.subplot(1, scalar_num, k)
                k += 1
                hist, bins = np.histogram(
                    self.samples[scalar].numpy(),
                    bins=50
                )
                width = 0.7 * (bins[1] - bins[0])
                center = (bins[:-1] + bins[1:]) / 2
                plt.bar(center, hist, align='center', width=width)
                plt.title("Histogram "+scalar)
            plt.show()

    def save(self, filename):
        if self.vlevel > 0:
            print(self.str_tab, self.name, '- Saving')
        fail = True
        while fail:
            try:
                with h5py.File(filename, 'w') as h5file:
                    for key, _ in self.samples.items():
                        h5file.create_dataset(
                            str(key),
                            data=np.asarray(self.samples[key]),
                            compression="gzip", compression_opts=9
                        )
                fail = False
            except OSError:
                time.sleep(5)
                if self.vlevel > 0:
                    print(self.str_tab, self.name, '- Waiting...')

    def load(self, filename):
        if self.vlevel > 0:
            print(self.str_tab, self.name, '- Loading')
        fail = True
        while fail:
            try:
                with h5py.File(filename, 'r') as h5file:
                    for key, _ in self.samples.items():
                        print(key)
                        assert h5file[str(key)].shape == self.samples[key].shape
                        self.samples[key] = torch.from_numpy(
                            np.asarray(h5file[str(key)])
                        ).type(self.dtype)
                fail = False
            except OSError:
                time.sleep(5)
                if self.vlevel > 0:
                    print(self.str_tab, self.name, '- Waiting...')

    def export(self, key, directory):
        assert key in self.samples
        if self.vlevel > 0:
            print(self.str_tab, self.name, '- Exporting')
        d = Path(directory)
        d.mkdir(parents=True)
        toimg = transforms.ToPILImage()
        k = 0
        for im in self.samples[key]:
            toimg(im).save(str(d / ('%05d.png' % k)))
            k += 1


class Sampler_UpDown(Sampler_Base):
    def __init__(self,
                 shape,
                 reference_paths={},
                 reject_factor=1.,
                 transpose_modes=[None],
                 downscale=PIL.Image.LANCZOS,
                 **kwds):
        super().__init__(**kwds)

        self.shape = shape
        self.transpose_modes = transpose_modes
        self.downscale = downscale
        self.reject_factor = reject_factor
        self.pil_to_tensor = transforms.ToTensor()

        self.down_factors = np.asarray(
            [d for d in self.shape if not isinstance(d, str)]
        )
        self.down_factors.sort()
        self.max_factor = int(self.down_factors.max())

        assert self.down_factors[0] == 1
        assert (self.down_factors == 1).sum() == 1
        for k in self.down_factors:
            assert self.max_factor % k == 0

        self.down_factors = tuple(self.down_factors)
        self.references = [
            d for d in self.shape
            if isinstance(d, str) and d != 'score' and d != 'label'
        ]

        for k in self.down_factors:
            self.samples[k] = torch.zeros(
                (0, self.shape[k][1], self.shape[k][2], self.shape[k][3])
            ).type(self.dtype)

        self.reference_paths = reference_paths
        for k, path in self.reference_paths.items():
            assert k in self.shape
            assert k[0] == 'R' and self.max_factor % int(k[1:]) == 0, '%s %s' % (k, self.max_factor)
            p = Path(path)
            for f in self.filelist:
                fp = Path(f)
                assert len(
                    [x for x in p.glob(fp.name[:-4]+'*'+fp.suffix)]
                ) > 0, '%s not found in %s' % (fp.name[:-4]+'*'+fp.suffix, p)
            self.samples[k] = torch.zeros(
                (0, self.shape[k][1], self.shape[k][2], self.shape[k][3])
            ).type(self.dtype)

        self.bar = trange(0, disable=True)

    def __call__(self, new_samples='resample',
                 samples_per_image=0, hflip=False, vflip=False, impdrop=True):
        HR = 1
        LR = self.down_factors[-1]

        sample = {'label': []}
        for f in self.shape:
            sample[f] = []

        if isinstance(new_samples, str):
            if new_samples[-1] == '%':
                assert float(new_samples.strip('%')) >= 0.
                new_samples = int(round(
                    (float(new_samples.strip('%'))/100.) * self.num
                ))
                if self.vlevel > 0:
                    print(self.str_tab, self.name,
                          '- Resampling %d images' % new_samples, flush=True)
                v = self.vlevel
                self.vlevel = 0
                super().__call__(
                    -new_samples,
                    add={},
                    impdrop=impdrop
                )
                self.vlevel = v
            else:
                assert new_samples == 'resample'
                new_samples = self.num
                if self.vlevel > 0:
                    print(self.str_tab, self.name, '- Resampling', flush=True)
                v = self.vlevel
                self.vlevel = 0
                super().__call__(
                    -self.num,
                    add={},
                    impdrop=impdrop
                )
                self.vlevel = v

        n_batches = 0
        self.bar.close()
        self.bar = trange(
            new_samples,
            disable=(self.vlevel == 0 or self.vlevel > 1 or new_samples == 0)
        )
        self.bar.set_description(self.str_tab + ' ' + self.name)
        stddev = None
        for progress in self.bar:
            while n_batches <= progress:
                fname = self.get_file()
                label = fname.split('/')[-2]
                if label not in self.label_idx:
                    self.label_idx[label] = len(self.label_idx)

                im_pil = {}
                im_pil[HR] = Image.open(fname).convert('RGB')
                scale = random.choice(self.scale_list)
                if scale != 1.:
                    im_pil[HR] = im_pil[HR].resize(
                        (int(round(im_pil[HR].size[0]/scale)),
                         int(round(im_pil[HR].size[1]/scale))),
                        resample=PIL.Image.LANCZOS
                    )

                if samples_per_image < 1:
                    try_samples_per_image = 10
                else:
                    try_samples_per_image = samples_per_image

                for f in self.down_factors:
                    im_pil[f] = im_pil[HR].resize(
                        (im_pil[HR].size[0]//f, im_pil[HR].size[1]//f),
                        resample=self.downscale
                    )
                for k in self.references:
                    p = Path(self.reference_paths[k])
                    fp = Path(fname)
                    rname = random.choice(
                        [x for x in p.glob(fp.name[:-4]+'*'+fp.suffix)]
                    )
                    im_pil[k] = Image.open(rname).convert('RGB')
                    if scale != 1.:
                        im_pil[k] = im_pil[k].resize(
                            (int(round(im_pil[k].size[0]/scale)),
                             int(round(im_pil[k].size[1]/scale))),
                            resample=PIL.Image.LANCZOS
                        )

                scale = random.choice(self.scale_list)
                tried_batch_samples = 0
                obtained_batch_samples = 0
                while tried_batch_samples < try_samples_per_image:
                    ready = False
                    it = 0
                    while not ready:
                        if it > try_samples_per_image:
                            break
                        top = {}
                        left = {}
                        min_top = int((self.shape[HR][2]/self.max_factor - self.shape[LR][2])/2)
                        min_left = int((self.shape[HR][3]/self.max_factor - self.shape[LR][3])/2)
                        top[LR] = np.random.randint(min_top, self.shape[LR][2]-min_top)
                        left[LR] = np.random.randint(min_left, self.shape[LR][3]-min_left)
                        ready = True
                        ready = ready and block_inside(
                            im_pil[LR].height, im_pil[LR].width,
                            top[LR], left[LR],
                            self.shape[LR][2], self.shape[LR][3]
                        )
                        for f in self.shape:
                            if f != LR:
                                if isinstance(f, int):
                                    factor = (self.max_factor//f)
                                else:
                                    factor = (self.max_factor//int(f[1:]))
                                top[f] = (
                                    (2*top[LR] + self.shape[LR][2]) * factor -
                                    self.shape[f][2]
                                )//2
                                left[f] = (
                                    (2*left[LR] + self.shape[LR][3]) * factor -
                                    self.shape[f][3]
                                )//2
                                ready = ready and block_inside(
                                    im_pil[f].height, im_pil[f].width,
                                    top[f], left[f],
                                    self.shape[f][2], self.shape[f][3]
                                )
                        if ready and self._impsamp:
                            stddev = np.mean(PIL.ImageStat.Stat(
                                im_pil[HR].crop((
                                    left[HR], top[HR],
                                    left[HR] + self.shape[HR][3],
                                    top[HR] + self.shape[HR][2]
                                ))
                            ).stddev)/255.
                            ready = (
                                stddev >
                                np.random.rand() * self.reject_factor
                            )
                        tried_batch_samples += 1
                        it += 1
                    if not ready:
                        break

                    sample['label'].append(self.label_idx[label])
                    option = random.choice(self.transpose_modes)
                    for f in self.shape:
                        if option is None:
                            patch_tensor = self.pil_to_tensor(
                                im_pil[f].crop((
                                    left[f], top[f],
                                    left[f] + self.shape[f][3],
                                    top[f] + self.shape[f][2]
                                ))
                            ).unsqueeze(0)
                        else:
                            patch_tensor = self.pil_to_tensor(
                                im_pil[f].crop((
                                    left[f], top[f],
                                    left[f] + self.shape[f][3],
                                    top[f] + self.shape[f][2]
                                )).transpose(option)
                            ).unsqueeze(0)
                        patch_tensor = (torch.round(patch_tensor*255.)/255.)

                        if self.shape[f][1] > 3:
                            sample[f].append(
                                torch.cat([
                                    patch_tensor,
                                    1e-3 * torch.randn(
                                        1, self.shape[f][1]-3,
                                        self.shape[f][2], self.shape[f][3]
                                    )
                                ], dim=1)
                            )
                        else:
                            sample[f].append(patch_tensor)
                    obtained_batch_samples += 1
                n_batches += obtained_batch_samples
                self.bar.set_postfix(
                    got=obtained_batch_samples,
                    tfit=tried_batch_samples,
                    std=stddev,
                    loop=self.loops,
                    tim=try_samples_per_image,
                )
                if self.vlevel > 1:
                    print(self.str_tab, self.name, '- file: %s' % f)
                    print(self.str_tab, self.name,
                          '-       batches obtained/tried: %d/%d' %
                          (obtained_batch_samples, tried_batch_samples))
        self.bar.close()
        self.reset_filepool()
        self.bar.refresh()

        return super().__call__(
            new_samples,
            add={
                **{'label': torch.LongTensor(sample['label'])},
                **{f: torch.cat(sample[f], dim=0) for f in self.shape}
            },
            impdrop=impdrop
        )

    def impsamp(self, val):
        assert isinstance(val, bool)
        self._impsamp = val
