import argparse
import pathlib
import copy
import os
import cv2

from PIL import Image

import matplotlib.pyplot as plt
import numpy as np

import torch
from torchvision import transforms

import models
import utils

def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, transforms.InterpolationMode.BICUBIC)(transforms.ToPILImage()(img))
    )


def read_img(img_path):
    return transforms.ToTensor()(
        transforms.ToPILImage()(transforms.ToTensor()(Image.open(img_path).convert('RGB')))
    )


def save_img(img, img_path):
    plt.imshow(img, cmap='viridis')
    plt.colorbar()
    plt.savefig(img_path)
    plt.close()


def system_scale(scale, scale_base=4):
    scale_list = []
    s = copy.copy(scale)

    if s <= scale_base:
        scale_list.append(s)
    else:
        scale_list.append(scale_base)
        s = s / scale_base
        while s > 1:
            if s >= scale_base:
                scale_list.append(scale_base)
            else:
                scale_list.append(s)
            s = s / scale_base

    return scale_list


def visualize(model, inp, coords, cell, save_dir, shape):
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    inp = inp.cuda()

    for idx in range(len(coords)):
        coords[idx] = coords[idx].cuda()

    cell = cell.cuda()

    model.eval()

    data_norm = {'inp': {'sub': [0], 'div': [1]}, 'gt': {'sub': [0], 'div': [1]}}
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    with torch.no_grad():
        preds = model.chop_forward(
            (inp - inp_sub) / inp_div, coords, cell, 30000
        )
    final_pred = preds[-1]
    final_pred = final_pred * gt_div + gt_sub
    final_pred.clamp_(0, 1)

    final_pred = final_pred.view(*shape).permute(0, 3, 1, 2)
    final_pred = final_pred[..., :shape[1], :shape[2]]

    return final_pred

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-input')
    parser.add_argument('-model')
    parser.add_argument('-resolution')
    parser.add_argument('-output', default='output.png')
    parser.add_argument('-gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    img_path = args.input


    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()
    h, w = list(map(int, args.resolution.split(',')))

    lr_img = read_img(img_path).cuda()
    hr_h, hr_w = h, w

    inp = lr_img.unsqueeze(0)

    inp_h, inp_w = inp.shape[-2:]

    coords = []



    coord = utils.make_coord((h, w), flatten=False).unsqueeze(0)

    coords.append(coord)

    cell = torch.ones(2).unsqueeze(0)
    cell[:, 0] *= 2. / hr_h
    cell[:, 1] *= 2. / hr_w

    save_dir = args.model.split('.pth')[0]
    shape = [1, h, w, 3]


    pred = visualize(model, inp, coords, cell, save_dir, shape)
    transforms.ToPILImage()(pred.squeeze().cpu()
                            ).save(args.output)
