import argparse
import os

import cv2
import numpy as np
import nvdiffrast.torch as dr
import torch

import config
from networks import CoarseReconsNet
from models.encoders import psp_encoders
from models.generator import makeup_generator
from utils.loss_util import VGGLoss, photo_loss, symmetry_loss, reg_loss
from utils.img_util import save_tensor
from utils.mm_layer import get_mm
from utils.mm_util import compute_style_mm
from utils.render_util import compute_uv_render, compute_img_render


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pca makeup reconstruction')
    parser.add_argument('--aligned_img_path', '-i', default='./results/align/aligned_img_0.png', help='Input aligned image')
    parser.add_argument('--aligned_mask_path', '-m', default='./results/align/aligned_mask_0.png', help='Input aligned mask')
    parser.add_argument('--out_dir', '-o', default='./results/style_reconstruction/0', help='Output directory')
    parser.add_argument('--n_itr', type=int, default=40, help='Input iteration number')
    parser.add_argument('--lr', type=float, default=1e-2, help='Input learning rate')
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # parameters
    n_itr = args.n_itr
    lr = args.lr
    
    w_dic = {'w_pho': 100., 'w_vgg': 1., 'w_reg': 1e-4, 'w_reg_apo': 1., 'w_sym': 8.}
    
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

    glctx = dr.RasterizeGLContext()
    rast_uv, _ = dr.rasterize(glctx, uv_coords.contiguous(), uv_ids.contiguous(), resolution=[fit_size, fit_size])

    fit_uv_mask = cv2.imread(config.SKIN_MASK_PATH)
    rend_uv_mask = cv2.resize(fit_uv_mask, (fit_size, fit_size))
    rend_uv_mask = torch.tensor(rend_uv_mask.astype(np.float32) / 255.0)[None].to(device)

    make_layer = makeup_generator(config.MM_MAKE_STYLE_PATH, config.SKIN_MASK_PATH)

    corse_recons_net = CoarseReconsNet(config.n_shape, config.n_exp, config.n_tex, config.n_tex).to(device)
    corse_recons_net.load_state_dict(torch.load('checkpoints/coarse_reconstruction.pkl'))
    makeup_estimate_net = psp_encoders.GradualStyleEncoder(50, 'ir_se').to(device)
    makeup_estimate_net.load_state_dict(torch.load('checkpoints/style_makeup_estimation.pkl'))
    
    corse_recons_net.eval()
    makeup_estimate_net.eval()
    
    vgg_loss = VGGLoss().to(device)

    img = cv2.imread(args.aligned_img_path)
    img = cv2.resize(img, (fit_size, fit_size))
    img = torch.tensor(img.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
    
    mask = cv2.imread(args.aligned_mask_path)
    mask = cv2.resize(mask, (fit_size, fit_size))
    mask = torch.tensor(mask.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)

    with torch.no_grad():
        res = corse_recons_net(img)
        res_m = makeup_estimate_net(img)
    
        mu = res_m.permute(1, 0, 2)
        mu_opt = mu.clone().detach().requires_grad_()
        
        
    optimizer = torch.optim.Adam([mu_opt], lr=lr)
    # start refine iteration
    for itr in range(n_itr):
        mm_ret = compute_style_mm(shape_layer, tex_layer, spec_layer, make_layer, tri, res['id'], res['ex'], res['tx'], mu_opt, res['sp'], res['r'],
                            res['tr'], res['s'], res['sh'], res['p'], res['ln'], res['gain'], res['bias'])
        uv_ret = compute_uv_render(mm_ret, rast_uv, tri, rend_uv_mask)
        rend_ret = compute_img_render(glctx, mm_ret, uv_ret, tri, uvs, uv_ids, fit_size)

        # photo loss
        rncr = rend_ret['rncr'].permute(0, 3, 1, 2)
        rncr_mask = rncr.ge(0.1)
        recons = rncr_mask * mask * rncr + ~rncr_mask * img + rncr_mask * (1 - mask) * img
        recons = recons.clamp(0, 1.)
        loss_pho = w_dic['w_pho'] * photo_loss(mask * recons, mask * img)
        
        # vgg loss
        loss_vgg = w_dic['w_vgg'] * vgg_loss(mask * recons, mask * img)
        
        # makeup parm regulartion loss
        loss_reg = w_dic['w_reg'] * reg_loss(mu_opt)

        # alpha matte regulation loss
        make_uv = mm_ret['make_crop']
        loss_reg_apo = w_dic['w_reg_apo'] * photo_loss(torch.zeros(make_uv[:, 3, ...].shape).to(device), make_uv[:, 3, ...])
        
        # symmetry loss
        loss_sym = w_dic['w_sym'] * symmetry_loss(make_uv)

        # total loss
        loss_total = loss_pho + loss_vgg + loss_reg + loss_reg_apo + loss_sym

        # optimizer update
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        print(f'itr [{itr:04}/{n_itr - 1:04}]', loss_total.item())

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
    
    mu_param = mu_opt.detach().cpu().tolist()
    np.savez(f'{out_dir}/mu_param.npz', mu=mu_param)