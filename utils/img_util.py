# -*- coding: utf-8 -*-

import cv2
import numpy as np


def save_tensor(img_tensor, save_path):
    save_tensor = img_tensor.clone().detach().cpu().numpy()
    save_img = (255.0 * save_tensor).clip(0, 255).astype(np.uint8)
    cv2.imwrite(save_path, save_img)
