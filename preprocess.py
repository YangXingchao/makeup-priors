import argparse
import os

import cv2
import numpy as np

import config
from libs.face_parsing.segment import segment
from libs.face_landmark import landmark

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess image')
    parser.add_argument('--img_path', '-i', default='./sample/img_non.png', help='Input image')
    parser.add_argument('--out_dir', '-o', default='./results/align', help='Output directory')
    parser.add_argument('--out_img_name', '-n', default='aligned_img_non.png', help='Output image name')
    parser.add_argument('--out_mask', '-m', default='aligned_mask_non.png', help='Output image mask')
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = f'{out_dir}/{args.out_img_name}'

    # crop
    img = cv2.imread(args.img_path)
    img_size = img.shape[0]
    kpt_size = config.KPT_SIZE
    resized_img = cv2.resize(img, (kpt_size, kpt_size))

    predicts = landmark.detect_landmark(resized_img)
    kpt = predicts * img_size / kpt_size
    kpt_width = abs(np.max(kpt[:, 0]) - np.min(kpt[:, 0]))
    kpt_hight = abs(np.max(kpt[:, 1]) - np.min(kpt[:, 1]))
    kpt_range = max(kpt_width, kpt_hight)

    crop_size = (kpt_range * config.CROP_RATIO).astype(np.int32)
    kpt_center = kpt[30].astype(np.int32)

    up = kpt_center[1] - crop_size // 2
    down = kpt_center[1] + crop_size // 2
    left = kpt_center[0] - crop_size // 2
    right = kpt_center[0] + crop_size // 2

    pad_u = abs(up - 0) if up - 0 < 0 else 0
    pad_d = abs(img.shape[1] - down) if img.shape[1] - down < 0 else 0
    pad_l = abs(left - 0) if left - 0 < 0 else 0
    pad_r = abs(img.shape[0] - right) if img.shape[0] - right < 0 else 0

    kpt_pad_center = kpt_center.copy()
    kpt_pad_center[1] = kpt_center[1] + pad_u
    kpt_pad_center[0] = kpt_center[0] + pad_l

    pad_img = cv2.copyMakeBorder(img, pad_u, pad_d, pad_l, pad_r, cv2.BORDER_CONSTANT, (0, 0, 0))

    crop_img = pad_img[kpt_pad_center[1] - crop_size // 2:kpt_pad_center[1] + crop_size // 2,
               kpt_pad_center[0] - crop_size // 2:kpt_pad_center[0] + crop_size // 2, :]

    crop_size_h = crop_img.shape[0]
    crop_size_w = crop_img.shape[1]
    half_min_crop_size = min(crop_size_h, crop_size_w) // 2
    half_max_crop_size = max(crop_size_h, crop_size_w) // 2
    margin = half_max_crop_size - half_min_crop_size
    if crop_size_w > crop_size_h:
        aligned_crop_img = crop_img[:, margin:crop_size_w - margin]
    else:
        aligned_crop_img = crop_img[margin:crop_size_h - margin, :]
    cv2.imwrite(out_path, aligned_crop_img)
    
    # segment
    seg_size = config.SEG_SIZE
    seg_img, mask_img = segment(out_path, seg_size)
    cv2.imwrite(f'{out_dir}/{args.out_mask}', mask_img)