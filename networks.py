import math

import torch
import torch.nn as nn
from torchvision.models import resnet50

import config
from utils.icosahedron import icosahedron_n


class CoarseReconsNet(nn.Module):
    def __init__(self, n_shape, n_exp, n_tex, n_spec):
        super().__init__()

        self.n_shape = n_shape
        self.n_exp = n_exp
        self.n_tex = n_tex
        self.n_spec = n_spec

        self.lp_init = torch.from_numpy(icosahedron_n)[None].to(torch.float32)

        backbone = resnet50()
        delattr(backbone, 'fc')

        def fit_forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            return x

        bound_method = fit_forward.__get__(backbone, backbone.__class__)
        setattr(backbone, 'forward', bound_method)

        last_dim = 2048
        self.final_layers = nn.ModuleList([
            conv1x1(last_dim, n_shape, bias=True),  # id
            conv1x1(last_dim, n_exp, bias=True),  # ex
            conv1x1(last_dim, n_tex, bias=True),  # tx
            conv1x1(last_dim, n_spec, bias=True),  # sp
            conv1x1(last_dim, 3, bias=True),  # r
            conv1x1(last_dim, 2, bias=True),  # tr
            conv1x1(last_dim, 1, bias=True),  # s
            conv1x1(last_dim, 27, bias=True),  # sh
            conv1x1(last_dim, 40, bias=True),  # p
            conv1x1(last_dim, 60, bias=True),  # ln
            conv1x1(last_dim, 3, bias=True),  # gain
            conv1x1(last_dim, 3, bias=True),  # bias
        ])
        for m in self.final_layers:
            nn.init.constant_(m.weight, 0.)
            nn.init.constant_(m.bias, 0.)

        self.backbone = backbone

    def forward(self, x):
        device = x.device
        x = self.backbone(x)
        output = []
        for layer in self.final_layers:
            output.append(layer(x))
        coeffs = torch.flatten(torch.cat(output, dim=1), 1)

        cnt = 0
        id = coeffs[:, 0:self.n_shape]
        cnt += self.n_shape

        ex = coeffs[:, cnt: cnt + self.n_exp]
        cnt += self.n_exp

        tx = coeffs[:, cnt: cnt + self.n_tex]
        cnt += self.n_tex

        sp = coeffs[:, cnt: cnt + self.n_spec]
        cnt += self.n_spec

        r = coeffs[:, cnt: cnt + 3]
        r += torch.tensor([1., 0.0, 0.0])[None].to(device)
        r = r * math.pi
        cnt += 3

        tr = coeffs[:, cnt:cnt + 2]
        tr *= config.FIT_SIZE // 2
        cnt += 2

        s = coeffs[:, cnt:cnt + 1]
        s += torch.ones(1, 1).to(device)
        cnt += 1

        sh = coeffs[:, cnt: cnt + 27].view(-1, 9, 3)
        sh += torch.tensor([0, 1., 0]).to(device)
        cnt += 27

        p = coeffs[:, cnt: cnt + 40].view(-1, 20, 2)
        p += torch.tensor([0, 1]).to(device)
        p *= torch.tensor([1., 200.]).to(device)
        cnt += 40

        ln = coeffs[:, cnt: cnt + 60].view(-1, 20, 3)
        ln *= 10
        ln += self.lp_init.to(device)
        cnt += 60

        gain = coeffs[:, cnt: cnt + 3]
        gain += 1
        cnt += 3

        bias = coeffs[:, cnt: cnt + 3]

        return {'id': id, 'tx': tx, 'sp': sp, 'ex': ex, 'r': r, 'tr': tr, 's': s, 'sh': sh, 'p': p, 'ln': ln,
                'gain': gain, 'bias': bias}


class MakeupEstimateNet(nn.Module):
    def __init__(self, n_make, is_train=False):
        super().__init__()

        self.n_make = n_make

        backbone = resnet50(pretrained=is_train)
        delattr(backbone, 'fc')

        def fit_forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            return x

        bound_method = fit_forward.__get__(backbone, backbone.__class__)
        setattr(backbone, 'forward', bound_method)

        last_dim = 2048
        self.final_layers = nn.ModuleList([
            conv1x1(last_dim, n_make, bias=True),  # mu
        ])
        for m in self.final_layers:
            nn.init.constant_(m.weight, 0.)
            nn.init.constant_(m.bias, 0.)

        self.backbone = backbone

    def forward(self, x):
        x = self.backbone(x)
        output = []
        for layer in self.final_layers:
            output.append(layer(x))
        coeffs = torch.flatten(torch.cat(output, dim=1), 1)

        mu = coeffs[:, 0:self.n_make]
        return {'mu': mu}


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, bias: bool = False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 1), stride=(stride, stride), bias=bias)
