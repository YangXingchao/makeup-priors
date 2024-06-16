#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import os.path as osp

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from .model import BiSeNet


def vis_parsing_maps(im, parsing_anno, stride):
    # Colors for all 20 parts
    part_colors = [[255, 255, 255],  # ?
                   [0, 0, 0],  # skin
                   [0, 0, 0],  # right eyebrows
                   [0, 0, 0],  # left eyebrows
                   [255, 255, 255],  # right eye
                   [255, 255, 255],  # left eye
                   [255, 255, 255],  # glass
                   [255, 255, 255],  # right ear
                   [255, 255, 255],  # left ear
                   [255, 255, 255],  # earrings
                   [0, 0, 0],  # nose
                   [255, 255, 255],  # teeth
                   [0, 0, 0],  # up lip
                   [0, 0, 0],  # down lip
                   [255, 255, 255],  # neck
                   [255, 255, 255],  # neckless
                   [255, 255, 255],  # cloth
                   [255, 255, 255],  # hair
                   [255, 255, 255],  # hat
                   [255, 255, 255],  # ?
                   [255, 255, 255],  # ?
                   [255, 255, 255],  # ?
                   [255, 255, 255],  # ?
                   [255, 255, 255]]  # ?

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    vis_im_mask = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.0, vis_parsing_anno_color, 1.0, 0)
    vis_im_mask = cv2.cvtColor(vis_im_mask, cv2.COLOR_BGR2GRAY)
    inv_mask = cv2.bitwise_not(vis_im_mask)

    face_masked = cv2.bitwise_and(im, im, mask=inv_mask)
    face_masked = cv2.cvtColor(face_masked, cv2.COLOR_RGB2BGR)

    return face_masked, inv_mask


def segment(img_path, img_size):
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()

    current_path = os.path.dirname(__file__)
    weight_path = f'{current_path}/module_weight/79999_iter.pth'
    net.load_state_dict(torch.load(weight_path))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        img = Image.open(img_path)
        image = img.resize((img_size, img_size), Image.BILINEAR)
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)

        vis, mask = vis_parsing_maps(image, parsing, stride=1)
        return vis, mask
