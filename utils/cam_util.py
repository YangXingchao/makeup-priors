# -*- coding: utf-8 -*-

import math
import torch
import torch.nn.functional as F


def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    y = x.view(-1, *xsize[dim:])
    y = y.view(y.size(0), y.size(1), -1)[:, getattr(torch.arange(y.size(1) - 1,
                                                                 -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return y.view(xsize)


def create_K(w, h, FOV):
    f = 0.5 * h / math.tan(float(FOV) / 360.0 * math.pi)
    K = torch.eye(3)
    K[0, 0] = K[1, 1] = f
    K[0, 2] = 0.5 * float(w)
    K[1, 2] = 0.5 * float(h)

    return K


def compute_f_normal(Vs, Fs, norm=True, keepdim=True):
    B, nV, C = Vs.size()
    B, nF = Fs.size()[:2]
    Fs = Fs + (torch.arange(B, dtype=torch.int32).to(Vs.device) * nV)[:, None, None]
    Vs = Vs.reshape((B * nV, C))

    Vf = Vs[Fs.long()]

    Vf_reshape = Vf.view(Vf.size(0) * Vf.size(1), 3, 3)
    v10 = Vf_reshape[:, 0] - Vf_reshape[:, 1]
    v12 = Vf_reshape[:, 2] - Vf_reshape[:, 1]
    Ns = torch.cross(v10, v12).view(B, -1, 3)

    if norm:
        Ns = F.normalize(Ns, dim=2)

    if keepdim:
        Ns = Ns[:, :, None, :].expand_as(Vf)

    return Ns


def compute_v_normal(Vs, Fs, norm=True):
    Nf = compute_f_normal(Vs, Fs, keepdim=False)
    Ns = torch.zeros_like(Vs)  # (B, N, 3)
    Fs = Fs[:, :, :, None].expand(-1, -1, -1, 3)
    Nf = Nf[:, :, None].expand_as(Fs).contiguous()
    Ns = Ns.scatter_add_(1, Fs.long().view(Fs.size(0), -1, 3), Nf.view(Fs.size(0), -1, 3))

    if norm:
        Ns = F.normalize(Ns, dim=2)

    return Ns


def transform(x, R, t):
    if R is not None and t is not None:
        x = torch.bmm(x, R.transpose(1, 2)) + t[:, None, :]
    elif R is not None:
        x = torch.bmm(x, R.transpose(1, 2))
    elif t is not None:
        x = x + t[:, None, :]

    return x


def weak_perspective(x, R, t, s):
    if R is not None:
        x = torch.bmm(x, R.transpose(1, 2))
    if s is not None:
        x = s[:, :, None] * x
    if t is not None:
        t = torch.cat([t, torch.zeros_like(t[:, :1])], 1)
        x = x + t[:, None, :]
    return x


def projection(x, K):
    x = torch.bmm(x, K.tranpose(1, 2))

    px, py, pz = x[:, :, 0], x[:, :, 1], x[:, :, 2]
    px = px / pz
    py = py / pz

    return torch.stack([px, py], dim=2)


def get_K(p):
    B, C = p.size()
    K = torch.zeros(B, 3, 3).to(p.device)

    K[:, 0, 0] = p[:, 0]
    K[:, 1, 1] = p[:, 1]
    K[:, 0, 2] = p[:, 2]
    K[:, 1, 2] = p[:, 3]
    K[:, 2, 2] = 1.0

    return K


def get_K_param(K):
    B, _, _ = K.size()

    p = torch.zeros(B, 4).to(K.device)
    p[:, 0] = K[:, 0, 0]
    p[:, 1] = K[:, 1, 1]
    p[:, 2] = K[:, 0, 2]
    p[:, 2] = K[:, 1, 2]

    return p


def compute_euler(R):
    sy = (R[:, 0, 0] * R[:, 0, 0] + R[:, 1, 0] * R[:, 1, 0]).sqrt()

    singular = sy < 1e-6

    if not singular.data[0]:
        x = torch.atan2(R[:, 2, 1], R[:, 2, 2])
        y = torch.atan2(-R[:, 2, 0], sy)
        z = torch.atan2(R[:, 1, 0], R[:, 0, 0])
    else:
        x = torch.atan2(-R[:, 1, 2], R[:, 1, 1])
        y = torch.atan2(-R[:, 2, 0], sy)
        z = 0

    return torch.stack([x, y, z], -1)


def batch_euler(rot):
    B = rot.size(0)

    sin = torch.sin(rot)
    cos = torch.cos(rot)

    Rx = torch.zeros(B, 3, 3).type_as(rot)
    Rx[:, 0, 0] = 1.0
    Rx[:, 1, 1] = cos[:, 0]
    Rx[:, 1, 2] = -sin[:, 0]
    Rx[:, 2, 1] = sin[:, 0]
    Rx[:, 2, 2] = cos[:, 0]

    Ry = torch.zeros(B, 3, 3).type_as(rot)
    Ry[:, 0, 0] = cos[:, 1]
    Ry[:, 0, 2] = sin[:, 1]
    Ry[:, 1, 1] = 1.0
    Ry[:, 2, 0] = -sin[:, 1]
    Ry[:, 2, 2] = cos[:, 1]

    Rz = torch.zeros(B, 3, 3).type_as(rot)
    Rz[:, 0, 0] = cos[:, 2]
    Rz[:, 0, 1] = -sin[:, 2]
    Rz[:, 1, 0] = sin[:, 2]
    Rz[:, 1, 1] = cos[:, 2]
    Rz[:, 2, 2] = 1.0

    R = torch.bmm(torch.bmm(Rz, Ry), Rx)

    return R


def batch_rodrigues(rvec, epsilon=1e-8):
    B = rvec.shape[0]

    angle = torch.norm(rvec + 1e-8, dim=1, keepdim=True)
    rdir = rvec / angle

    cos = torch.cos(angle)[:, None]
    sin = torch.sin(angle)[:, None]

    rx, ry, rz = rdir[:, 0], rdir[:, 1], rdir[:, 2]
    K = torch.zeros((B, 3, 3)).type_as(rvec)

    zeros = torch.zeros((B, 1)).type_as(rvec)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1).view((B, 3, 3))

    ident = torch.eye(3).type_as(rvec)[None]
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)

    return rot_mat


def batch_quaternion(rvec):
    B = rvec.size(0)

    theta = torch.sqrt(1e-5 + torch.sum(rvec ** 2, dim=1))
    rvec = rvec / theta[:, None]
    return torch.stack((
        1. - 2. * rvec[:, 1] ** 2 - 2. * rvec[:, 2] ** 2,
        2. * (rvec[:, 0] * rvec[:, 1] - rvec[:, 2] * rvec[:, 3]),
        2. * (rvec[:, 0] * rvec[:, 2] + rvec[:, 1] * rvec[:, 3]),

        2. * (rvec[:, 0] * rvec[:, 1] + rvec[:, 2] * rvec[:, 3]),
        1. - 2. * rvec[:, 0] ** 2 - 2. * rvec[:, 2] ** 2,
        2. * (rvec[:, 1] * rvec[:, 2] - rvec[:, 0] * rvec[:, 3]),

        2. * (rvec[:, 0] * rvec[:, 2] - rvec[:, 1] * rvec[:, 3]),
        2. * (rvec[:, 0] * rvec[:, 3] + rvec[:, 1] * rvec[:, 2]),
        1. - 2. * rvec[:, 0] ** 2 - 2. * rvec[:, 1] ** 2
    ), dim=1).view(B, 3, 3)


def batch_rot6d(rvec):
    x_raw = rvec[:, 0:3]
    y_raw = rvec[:, 3:6]

    x = F.normalize(x_raw, dim=1)
    z = torch.cross(x, y_raw, dim=1)
    z = F.normalize(z, dim=1)
    y = torch.cross(z, x, dim=1)

    rotmat = torch.cat((x[:, :, None], y[:, :, None], z[:, :, None]), -1)  # (B, 3, 3)

    return rotmat
