# Copyright 2020 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import cv2
import numpy as np
import torch


def bgr2rgb(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


def cam(x, size=256):
    x = x - np.min(x)
    cam_img = x / np.max(x)
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img, (size, size))
    cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
    return cam_img / 255.0


def denorm(x):
    return x * 0.5 + 0.5


def imagenet_norm(x):
    mean = [0.485, 0.456, 0.406]
    std = [0.299, 0.224, 0.225]
    mean = torch.FloatTensor(mean).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(x.device)
    std = torch.FloatTensor(std).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(x.device)
    return (x - mean) / std


def merge(images, size):
    image_height, image_width = images.shape[1], images.shape[2]
    image = np.zeros((image_height * size[0], image_width * size[1], 3))
    for index, image in enumerate(images):
        i = index % size[1]
        j = index // size[1]
        image[image_height * j:image_height * (j + 1), image_width * i:image_width * (i + 1), :] = image

    return image


def str2bool(x):
    return x.lower() in 'true'


def tensor2numpy(x):
    return x.detach().cpu().numpy().transpose(1, 2, 0)
