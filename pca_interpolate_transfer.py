import argparse
import os

import cv2
import numpy as np
import nvdiffrast.torch as dr
import torch

import config
from networks import CoarseReconsNet
from utils.img_util import save_tensor
from utils.mm_layer import get_mm, get_mm_make
from utils.mm_util import compute_pca_mm
from utils.render_util import compute_uv_render, compute_img_render


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pca makeup reconstruction')
    parser.add_argument('--ref_mm_path_1', '-ref_1', default='./results/pca_reconstruction/0/mu_param.npz', help='Input makeup reference param 1')
    parser.add_argument('--ref_mm_path_2', '-ref_2', default='./results/pca_reconstruction/1/mu_param.npz', help='Input makeup reference param 2')
    parser.add_argument('--non_makeup_img_path', '-n', default='./results/align/aligned_img_non.png', help='Input nonmakeup image path')
    parser.add_argument('--out_dir', '-o', default='./results/pca_reconstruction/interpolate_transfer', help='Output directory')
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
    corse_recons_net.load_state_dict(torch.load('checkpoints/coarse_reconstruction.pkl'))
    corse_recons_net.eval()

    # load non makeup img
    img = cv2.imread(args.non_makeup_img_path)
    img = cv2.resize(img, (fit_size, fit_size))
    img = torch.tensor(img.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)

    res = corse_recons_net(img)
    
    # load makeup params
    mu_param_1 = np.load(args.ref_mm_path_1)
    mu_1 = torch.from_numpy(mu_param_1['mu']).float().to(device)
    mu_param_2 = np.load(args.ref_mm_path_2)
    mu_2 = torch.from_numpy(mu_param_2['mu']).float().to(device)
    
    # makeup interpolation and transfer
    lerp_n = 5
    for lerp_i in range(lerp_n):
        ratio = lerp_i / (lerp_n - 1)
        lerp_mu = torch.lerp(mu_1, mu_2, ratio)
        
        mm_ret = compute_pca_mm(shape_layer, tex_layer, spec_layer, make_layer, tri, res['id'], res['ex'], res['tx'], lerp_mu, res['sp'], res['r'],
                            res['tr'], res['s'], res['sh'], res['p'], res['ln'], res['gain'], res['bias'])
        uv_ret = compute_uv_render(mm_ret, rast_uv, tri, rend_uv_mask)
        rend_ret = compute_img_render(glctx, mm_ret, uv_ret, tri, uvs, uv_ids, fit_size)

        # img rendering
        save_tensor(rend_ret['rncr'][0], f'{out_dir}/rncr_{lerp_i}.png')
        save_tensor(uv_ret['rncr'][0], f'{out_dir}/uv_rncr_{lerp_i}.png')