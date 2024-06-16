# -*- coding: utf-8 -*-

import math
import numpy as np
import cv2

import torch
import torch.utils.data
import torch.nn.functional as F

from models.stylegan2 import Generator


class Generator_Configs():
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    generator: Generator

    d_latent: int = 512
    image_size: int = 256

    UV_SIZE = 1536
    uv_crop_w = 1152
    uv_crop_h = 1152
    UV_CROP_POS = [186, 186]
    tex_w = 512
    tex_h = 512

    uv_size = UV_SIZE * tex_w // uv_crop_w
    uv_crop_pos = [UV_CROP_POS[0] * tex_w / uv_crop_w, UV_CROP_POS[1] * tex_h / uv_crop_h]
    uv_crop_pos = np.around(uv_crop_pos).astype(np.int8)

    # Number of blocks in the generator (calculated based on image resolution)
    n_gen_blocks: int

    makeup_mask: torch.tensor

    # trained_generator_path = f'checkpoints/generator.pth'
    def init(self, generator_path, mask_path):
        log_resolution = int(math.log2(self.image_size))

        self.generator = Generator(log_resolution, self.d_latent).to(self.device)
        self.n_gen_blocks = self.generator.n_blocks
        self.generator.load_state_dict(torch.load(generator_path))

        self.generator.eval()

        makeup_mask = cv2.imread(mask_path)
        makeup_mask = cv2.resize(makeup_mask, (self.image_size, self.image_size))
        makeup_mask = cv2.cvtColor(makeup_mask, cv2.COLOR_BGR2GRAY)
        self.makeup_mask = torch.Tensor(makeup_mask.astype(np.float32)[..., None] / 255.0).permute(2, 0, 1).to(self.device)

    # Notice: we set noise to zero
    def get_noise(self, batch_size: int):
        noise = []
        resolution = 4

        for i in range(self.n_gen_blocks):
            if i == 0:
                n1 = None
            else:
                n1 = torch.zeros(batch_size, 1, resolution, resolution, device=self.device)
            n2 = torch.zeros(batch_size, 1, resolution, resolution, device=self.device)
            noise.append((n1, n2))
            resolution *= 2

        return noise

    def generate_images(self, batch_size: int, w: torch.Tensor):
        noise = self.get_noise(batch_size)
        images = self.generator(w, noise)
        return images, w

    def inference(self, w):
        batch_size = w.shape[1]
        generated_tex, _ = self.generate_images(batch_size, w)
        resized_tex = F.interpolate(generated_tex, size=(self.tex_h, self.tex_w),
                                             mode='bilinear', align_corners=False)
        resized_tex = resized_tex[:, [2, 1, 0, 3], ...]
        generated_full = torch.zeros([batch_size, 4, self.uv_size, self.uv_size]).to(self.device)
        generated_full[..., self.uv_crop_pos[1]:self.uv_crop_pos[1] + self.tex_h, self.uv_crop_pos[0]:self.uv_crop_pos[0] + self.tex_w,
      ] = resized_tex
        resized_full = F.interpolate(generated_full, size=(self.image_size, self.image_size),
                                             mode='bilinear', align_corners=False)

        return resized_full * self.makeup_mask, generated_tex


def makeup_generator(generator_path, mask_path):

    styleGAN2 = Generator_Configs()
    styleGAN2.init(generator_path, mask_path)

    return styleGAN2
