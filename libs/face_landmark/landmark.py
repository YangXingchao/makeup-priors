from __future__ import division

import os
import time

import cv2
import numpy as np
import torch
from PIL import Image

from .models import basenet
from .src import detect_faces
from .common.utils import BBox

mean = np.asarray([0.485, 0.456, 0.406])
std = np.asarray([0.229, 0.224, 0.225])

if torch.cuda.is_available():
    map_location = lambda storage, loc: storage.cuda()
else:
    map_location = 'cpu'


def cv2pil(image):
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image


def load_model():
    model = basenet.MobileNet_GDConv(136)
    model = torch.nn.DataParallel(model)
    # Download from https://drive.google.com/file/d/1Le5UdpMkKOTRr1sTp4lwkw8263sbgdSe/view
    current_path = os.path.dirname(__file__)
    model_path = f'{current_path}/checkpoint/mobilenet_224_model_best_gdconv_external.pth.tar'
    checkpoint = torch.load(model_path, map_location=map_location)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def detect_landmark(img):
    out_size = 224
    model = load_model()
    model = model.eval()

    height, width, _ = img.shape
    # perform face detection using MTCNN

    image = cv2pil(img)
    faces, _ = detect_faces(image)
    ratio = 0
    if len(faces) == 0:
        print('NO face is detected!')
        return None

    face = faces[0]
    x1 = face[0]
    y1 = face[1]
    x2 = face[2]
    y2 = face[3]
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    size = int(min([w, h]) * 1.2)
    cx = x1 + w // 2
    cy = y1 + h // 2
    x1 = cx - size // 2
    x2 = x1 + size
    y1 = cy - size // 2
    y2 = y1 + size

    dx = max(0, -x1)
    dy = max(0, -y1)
    x1 = max(0, x1)
    y1 = max(0, y1)

    edx = max(0, x2 - width)
    edy = max(0, y2 - height)
    x2 = min(width, x2)
    y2 = min(height, y2)
    new_bbox = list(map(int, [x1, x2, y1, y2]))
    new_bbox = BBox(new_bbox)
    cropped = img[new_bbox.top:new_bbox.bottom, new_bbox.left:new_bbox.right]
    if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
        cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)
    cropped_face = cv2.resize(cropped, (out_size, out_size))

    if cropped_face.shape[0] <= 0 or cropped_face.shape[1] <= 0:
        return None
    test_face = cropped_face.copy()
    test_face = test_face / 255.0
    test_face = (test_face - mean) / std
    test_face = test_face.transpose((2, 0, 1))
    test_face = test_face.reshape((1,) + test_face.shape)
    input = torch.from_numpy(test_face).float()
    input = torch.autograd.Variable(input)
    start = time.time()
    landmark = model(input).cpu().data.numpy()
    end = time.time()
    print('Time: {:.6f}s.'.format(end - start))
    landmark = landmark.reshape(-1, 2)
    landmark = new_bbox.reprojectLandmark(landmark)
    return landmark
    # for x, y in landmark:
    #     cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), -1)
    #
    # cv2.imwrite(os.path.join('results', os.path.basename(imgname)), img)

