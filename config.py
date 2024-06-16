# -*- coding: utf-8 -*-

# 3DMM path
MM_PATH = './resources/generic_model.pkl'
LMK_PATH = './resources/landmark_embedding.npy'
MM_SPECULAR_PATH = './resources/albedoModel2020_FLAME_albedoPart.npz'
MM_DIFFUSE_PATH = './resources/FLAME_texture.npz'
MM_MASK_PATH = './resources/FLAME_masks.pkl'
SKIN_MASK_PATH = './resources/skin_mask.png'
MM_MAKE_PCA_PATH = './resources/priors/makeup_pca.mat'
MM_MAKE_STYLE_PATH = './resources/priors/makeup_style.pth'

# 3DMM definition
n_shape = 200
n_tex = 100
n_exp = 100
n_make_pca = 100
n_make_style = 512
MM_SCALE = 1600

CROP_RATIO = 1.6
KPT_SIZE = 256
FIT_SIZE = 256
SEG_SIZE = 512