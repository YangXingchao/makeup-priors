# -*- coding: utf-8 -*-


import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr

c1 = 0.429043
c2 = 0.511664
c3 = 0.743125
c4 = 0.886227
c5 = 0.247708

T = torch.zeros(16, 9)
T[15, 0] = c4
T[7, 1] = T[13, 1] = c2
T[11, 2] = T[14, 2] = c2
T[3, 3] = T[12, 3] = c2
T[1, 4] = T[4, 4] = c1
T[6, 5] = T[9, 5] = c1
T[10, 6] = c3
T[15, 6] = -c5
T[2, 7] = T[8, 7] = c1
T[0, 8] = c1
T[5, 8] = -c1


def sh_shading(nml, sh):
    global T

    L = torch.einsum('xn,bnc->bxc', T.to(sh.device), sh)
    nml_h = torch.cat([nml, torch.ones_like(nml[:, :, :1])], 2)
    L = L.view(L.shape[0], 4, 4, 3)
    nLn = torch.einsum('bns,bstc,bnt->bnc', nml_h, L, nml_h)

    return nLn


def diffuse_shading(nml, sh, pts=None, cull_thre=0.0):
    if pts is not None:
        view = F.normalize(pts, dim=2)
        ndotv = (nml * view).sum(2)
    else:
        ndotv = nml[:, :, 2]

    mask = (ndotv < cull_thre)
    diff = sh_shading(nml, sh)
    diff = diff.clamp(0, 1.)

    return mask[:, :, None].float(), diff


def specular_shading(nml, v_cam, p, ln):
    norm_v_cam = F.normalize(v_cam[:, None, :, :], dim=3)
    view_plane = norm_v_cam.clone()
    view_plane[:, :, :, 2] = 0
    view = F.normalize(view_plane - norm_v_cam, dim=3)
    l_pos = F.normalize(ln[:, :, None, :], dim=3)

    l_dir = l_pos
    h = l_dir + view
    h = F.normalize(h, dim=3)

    l_intensity = p[..., 0][..., None, None]
    l_intensity = 1 / (1 + torch.exp(-10 * (l_intensity - 0.5)))
    exponent = p[..., 1][..., None, None]
    exponent = F.relu(exponent)

    angle = torch.sum(nml[:, None, :, :] * h, dim=3)[..., None]
    refl = l_intensity * torch.max(torch.as_tensor(1e-8).to(angle.device), angle) ** exponent

    refl = torch.sum(refl, dim=1)
    refl = refl.clamp(0, 1).expand(-1, -1, 3)
    return refl


def texture_sampling(rast_out, texc, uv_ret, material):
    uv_tex = uv_ret[material]
    uv_tex = dr.texture(uv_tex.contiguous(), texc, filter_mode='linear')
    uv_tex *= torch.clamp(rast_out[..., -1:], 0, 1)
    return uv_tex


def compute_img_render(glctx, mm_ret, uv_ret, tri, uvs, uv_ids, fit_size):
    v_cam = mm_ret['v_cam']
    v_cam = v_cam / fit_size
    v_cam = torch.cat([v_cam, torch.ones([v_cam.shape[0], v_cam.shape[1], 1]).cuda()], axis=2)
    rast_out, _ = dr.rasterize(glctx, v_cam, tri, resolution=[fit_size, fit_size])
    texc, _ = dr.interpolate(uvs.contiguous(), rast_out, uv_ids.contiguous())

    albe = texture_sampling(rast_out, texc, uv_ret, 'albe')
    norm = texture_sampling(rast_out, texc, uv_ret, 'norm')
    diff = texture_sampling(rast_out, texc, uv_ret, 'diff')
    rndr = texture_sampling(rast_out, texc, uv_ret, 'rndr')
    spec = texture_sampling(rast_out, texc, uv_ret, 'spec')
    refl = texture_sampling(rast_out, texc, uv_ret, 'refl')
    rnsr = texture_sampling(rast_out, texc, uv_ret, 'rnsr')
    rncr = texture_sampling(rast_out, texc, uv_ret, 'rncr')
    bare = texture_sampling(rast_out, texc, uv_ret, 'bare')
    make = texture_sampling(rast_out, texc, uv_ret, 'make')
    alpha = texture_sampling(rast_out, texc, uv_ret, 'alpha')
    base = texture_sampling(rast_out, texc, uv_ret, 'base')

    ret = {'rncr': rncr, 'rndr': rndr, 'diff': diff, 'albe': albe, 'rnsr': rnsr, 'refl': refl, 'spec': spec, 'norm':norm, 'bare': bare, 'make': make, 'alpha': alpha, 'base': base}

    return ret


def compute_uv_render(mm_ret, rast_uv, tri, mask):
    mask_size = mask.shape[1]
    tex_size = mm_ret['albe'].shape[-1]
    B = mm_ret['albe'].shape[0]
    factor = mask_size // tex_size

    rast = rast_uv.expand(B, -1, -1, -1).contiguous()
    norm, _ = dr.interpolate(mm_ret['n_cam'], rast, tri)
    norm = norm * mask

    vert = F.normalize(mm_ret['v_cam'], dim=2)
    vert, _ = dr.interpolate(vert, rast, tri)
    vert = vert * mask

    albe = F.interpolate(mm_ret['albe'], scale_factor=factor, mode='bilinear')
    albe = albe.permute(0, 2, 3, 1) * mask

    bare = F.interpolate(mm_ret['bare'], scale_factor=factor, mode='bilinear')
    bare = bare.permute(0, 2, 3, 1) * mask

    make = F.interpolate(mm_ret['make'], scale_factor=factor, mode='bilinear')
    make = make.permute(0, 2, 3, 1) * mask

    base = F.interpolate(mm_ret['base'], scale_factor=factor, mode='bilinear')
    base = base.permute(0, 2, 3, 1) * mask

    alpha = F.interpolate(mm_ret['alpha'], scale_factor=factor, mode='bilinear')
    alpha = alpha.permute(0, 2, 3, 1) * mask

    diff, _ = dr.interpolate(mm_ret['diff'], rast, tri)
    diff = diff * mask

    rndr = albe * diff

    spec = F.interpolate(mm_ret['spec'], scale_factor=factor, mode='bilinear')
    spec = spec.permute(0, 2, 3, 1) * mask

    refl, _ = dr.interpolate(mm_ret['refl'].contiguous(), rast, tri)
    refl = refl * mask

    rnsr = spec * refl

    rncr = rnsr + rndr
    rncr = torch.clamp(rncr, 0, 1.)

    ret = {'rncr': rncr, 'rndr': rndr, 'diff': diff, 'albe': albe, 'rnsr': rnsr, 'refl': refl, 'spec': spec, 'norm': norm, 'vert': vert, 'bare': bare, 'make': make, 'alpha': alpha, 'base': base}

    return ret