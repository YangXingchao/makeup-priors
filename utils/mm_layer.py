# -*- coding: utf-8 -*-

import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio

import config
from utils.FLAME import FLAME


def get_mm():
    model = {}
    scale = config.MM_SCALE
    flame = FLAME(config)
    diffuse = np.load(config.MM_DIFFUSE_PATH)
    specular = np.load(config.MM_SPECULAR_PATH)
    with open(config.MM_MASK_PATH, 'rb') as f:
        mask = pickle.load(f, encoding='latin1')

    tri = flame.faces_tensor.detach().clone().to(torch.int32)
    v_mask = np.array(mask['face'])

    model['meantex'] = diffuse['mean'].astype(np.float32) / 255.
    model['texBase'] = diffuse['tex_dir'][..., :config.n_tex].astype(np.float32) / 255.
    model['meanspec'] = specular['specMU'].astype(np.float32)
    model['specBase'] = specular['specPC'][..., :config.n_tex].astype(np.float32)

    shape_layer = ShapeModel(flame, scale)
    tex_layer = TexModel(model)
    spec_layer = SpecModel(model)

    uv_ids = diffuse['ft'].astype(np.int32)
    uvs = diffuse['vt'].astype(np.float32)

    uv_ids = torch.from_numpy(uv_ids)

    uvs[:, 1] = 1 - uvs[:, 1]
    uvs = uvs * 2 - 1

    uv_coords = np.hstack((uvs, np.zeros((uvs.shape[0], 1))))
    uv_coords = np.hstack((uv_coords, np.ones((uvs.shape[0], 1))))
    uv_coords = torch.from_numpy(uv_coords).to(torch.float32)[None]

    uvs = torch.from_numpy(uvs).to(torch.float32)[None]
    uvs = uvs / 2 - 0.5

    mm = {
        'shape_layer': shape_layer,
        'tex_layer': tex_layer,
        'spec_layer': spec_layer,
        'tri': tri,
        'v_mask': v_mask,
        'uvs': uvs,
        'uv_coords': uv_coords,
        'uv_ids': uv_ids
    }
    return mm


class ShapeModel(nn.Module):
    def __init__(self, model, scale=1.0, learnable=False):
        super(ShapeModel, self).__init__()
        self.flame = model
        self.scale = scale

        if not learnable:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, id, ex):
        vertices, lm2d, lm3d = self.flame(id, ex)
        vertices = vertices * self.scale
        lm3d = lm3d * self.scale

        return vertices, lm3d


class TexModel(nn.Module):
    def __init__(self, model, learnable=False):
        super(TexModel, self).__init__()

        meantex = torch.from_numpy(model['meantex']).float()[None, ...]
        tex = torch.from_numpy(model['texBase']).float()[None, ...]
        self.register_buffer('meantex', meantex)
        self.register_buffer('tex', tex)

        if not learnable:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        tex = self.meantex + (torch.einsum('bhwcx,bx->bhwcx', self.tex, x)).sum(-1)
        tex = tex.permute(0, 3, 1, 2)
        tex = F.interpolate(tex, [256, 256])
        return tex


class SpecModel(nn.Module):
    def __init__(self, model, learnable=False):
        super(SpecModel, self).__init__()

        meanspec = torch.from_numpy(model['meanspec']).float()[None, ...]
        spec = torch.from_numpy(model['specBase']).float()[None, ...]
        self.register_buffer('meanspec', meanspec)
        self.register_buffer('spec', spec)

        if not learnable:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        spec = self.meanspec + (torch.einsum('bhwcx,bx->bhwcx', self.spec, x)).sum(-1)
        spec = spec.permute(0, 3, 1, 2)
        spec = F.interpolate(spec, [256, 256])
        return spec


def get_mm_make(scale=1.0):
    model = {}
    makeup = sio.loadmat(config.MM_MAKE_PCA_PATH)
    model['meanmake'] = makeup['mean'].astype(np.float32) / 255.
    model['makeBase'] = makeup['base'].astype(np.float32) / 255.
    makeup_layer = MakeupModel(model, scale)

    return makeup_layer


class MakeupModel(nn.Module):
    def __init__(self, model, scale=1.0, learnable=False):
        super(MakeupModel, self).__init__()

        meanmake = torch.from_numpy(model['meanmake']).float()
        make = torch.from_numpy(model['makeBase']).float()[None, ...] * scale
        self.register_buffer('meanmake', meanmake)
        self.register_buffer('make', make)

        if not learnable:
            # this layer is not learnable
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        make = self.meanmake + (torch.einsum('bpx,bx->bpx', self.make, x)).sum(-1)
        make = make.reshape(-1, 256, 256, 4)
        make = make.permute(0, 3, 1, 2)
        make = F.interpolate(make, [256, 256])
        return make