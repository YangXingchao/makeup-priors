import argparse
import os

import cv2
import numpy as np
import nvdiffrast.torch as dr
import torch

import config
from networks import CoarseReconsNet, MakeupEstimateNet
from utils.img_util import save_tensor
from utils.mm_layer import get_mm, get_mm_make
from utils.mm_util import compute_pca_mm
from utils.render_util import compute_uv_render, compute_img_render


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pca makeup reconstruction')
    parser.add_argument('--aligned_img_path', '-i', default='./results/align/aligned_img_0.png', help='Input aligned image')
    parser.add_argument('--out_dir', '-o', default='./results/pca_reconstruction/0', help='Output directory')
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    fit_size = config.FIT_SIZE

    # get mm components
    mm = get_mm()
    shape_layer = mm['shape_layer'].to(device)
    tex_layer = mm['tex_layer'].to(device)
    spec_layer = mm['spec_layer'].to(device)
    tri = mm['tri'].to(device)
    uvs = mm['uvs'].to(device)
    uv_coords = mm['uv_coords'].to(device)
    uv_ids = mm['uv_ids'].to(device)
    
    make_layer = get_mm_make()
    make_layer = make_layer.to(device)

    glctx = dr.RasterizeGLContext()
    rast_uv, _ = dr.rasterize(glctx, uv_coords.contiguous(), uv_ids.contiguous(), resolution=[fit_size, fit_size])

    fit_uv_mask = cv2.imread(config.SKIN_MASK_PATH)
    rend_uv_mask = cv2.resize(fit_uv_mask, (fit_size, fit_size))
    rend_uv_mask = torch.tensor(rend_uv_mask.astype(np.float32) / 255.0)[None].to(device)

    corse_recons_net = CoarseReconsNet(config.n_shape, config.n_exp, config.n_tex, config.n_tex).to(device)
    makeup_estimate_net = MakeupEstimateNet(config.n_make_pca).to(device)
    corse_recons_net.load_state_dict(torch.load('checkpoints/coarse_reconstruction.pkl'))
    makeup_estimate_net.load_state_dict(torch.load('checkpoints/pca_makeup_estimation.pkl'))
    corse_recons_net.eval()
    makeup_estimate_net.eval()

    img = cv2.imread(args.aligned_img_path)
    img = cv2.resize(img, (fit_size, fit_size))
    img = torch.tensor(img.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)

    res = corse_recons_net(img)
    res_m = makeup_estimate_net(img)

    mm_ret = compute_pca_mm(shape_layer, tex_layer, spec_layer, make_layer, tri, res['id'], res['ex'], res['tx'], res_m['mu'], res['sp'], res['r'],
                        res['tr'], res['s'], res['sh'], res['p'], res['ln'], res['gain'], res['bias'])
    uv_ret = compute_uv_render(mm_ret, rast_uv, tri, rend_uv_mask)
    rend_ret = compute_img_render(glctx, mm_ret, uv_ret, tri, uvs, uv_ids, fit_size)

    # img rendering
    save_tensor(rend_ret['rncr'][0], f'{out_dir}/rncr.png')
    save_tensor(rend_ret['albe'][0], f'{out_dir}/albe.png')
    save_tensor(rend_ret['bare'][0], f'{out_dir}/bare.png')
    save_tensor(rend_ret['alpha'][0], f'{out_dir}/alpha.png')
    save_tensor(rend_ret['base'][0], f'{out_dir}/base.png')

    # uv texture rendering
    save_tensor(uv_ret['rncr'][0], f'{out_dir}/uv_rncr.png')
    save_tensor(uv_ret['albe'][0], f'{out_dir}/uv_albe.png')
    save_tensor(uv_ret['bare'][0], f'{out_dir}/uv_bare.png')
    save_tensor(uv_ret['alpha'][0], f'{out_dir}/uv_alpha.png')
    save_tensor(uv_ret['base'][0], f'{out_dir}/uv_base.png')
    
    mu_param = res_m['mu'].detach().cpu().tolist()
    np.savez(f'{out_dir}/mu_param.npz', mu=mu_param)