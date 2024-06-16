# -*- coding: utf-8 -*-

from utils.cam_util import compute_v_normal, weak_perspective, batch_euler
from utils.render_util import diffuse_shading, specular_shading


def compute_pca_mm(shape_layer, tex_layer, spec_layer, make_layer, tri, id, ex, tx, mu, sp, r, t, s, sh, p, ln, gain, bias):
       v, lm = shape_layer(id, ex)
       bare = tex_layer(tx).permute(0, 2, 3, 1) * gain[:, None, None, :] + bias[:, None, None, :]
       bare = bare.permute(0, 3, 1, 2).clamp(0, 1.)
       spec = spec_layer(sp)

       make_full = make_layer(mu).clamp(0, 1.)
       base = make_full[:, :3, ...]
       alpha = make_full[:, 3, ...][:, None, ...]
       make = base * alpha
       albe = (bare * (1-alpha) + make).clamp(0, 1.)

       v_cam = weak_perspective(v, batch_euler(r), t, s)
       v_kpt = weak_perspective(lm, batch_euler(r), t, s)
       n_cam = compute_v_normal(v, tri[None, :, [2, 1, 0]].expand(v.size(0), -1, -1), True)

       mask, diff = diffuse_shading(n_cam, sh, pts=v_cam, cull_thre=1.)

       refl = specular_shading(n_cam, v_cam, p, ln)

       ret = {'mask': mask, 'diff': diff, 'v_cam': v_cam, 'n_cam': n_cam, 'v_kpt': v_kpt, 'albe': albe, 'v': v,
              'spec': spec, 'refl': refl, 'make': make, 'bare': bare, 'alpha': alpha, 'base': base}

       return ret


def compute_style_mm(shape_layer, tex_layer, spec_layer, make_layer, tri, id, ex, tx, mu, sp, r, t, s, sh, p, ln, gain, bias):
       v, lm = shape_layer(id, ex)
       bare = tex_layer(tx).permute(0, 2, 3, 1) * gain[:, None, None, :] + bias[:, None, None, :]
       bare = bare.permute(0, 3, 1, 2).clamp(0, 1.)
       spec = spec_layer(sp)
       
       make_full, make_crop = make_layer.inference(mu)
       make_full = make_full.clamp(0, 1.)

       base = make_full[:, :3, ...]
       alpha = make_full[:, 3, ...][:, None, ...]
       make = base * alpha

       albe = (bare * (1-alpha) + make).clamp(0, 1.)

       v_cam = weak_perspective(v, batch_euler(r), t, s)
       v_kpt = weak_perspective(lm, batch_euler(r), t, s)
       n_cam = compute_v_normal(v, tri[None, :, [2, 1, 0]].expand(v.size(0), -1, -1), True)

       mask, diff = diffuse_shading(n_cam, sh, pts=v_cam, cull_thre=1.)

       refl = specular_shading(n_cam, v_cam, p, ln)

       ret = {'mask': mask, 'diff': diff, 'v_cam': v_cam, 'n_cam': n_cam, 'v_kpt': v_kpt, 'albe': albe, 'v': v,
              'spec': spec, 'refl': refl, 'make': make, 'bare': bare, 'alpha': alpha, 'base': base, 'make_crop': make_crop}

       return ret