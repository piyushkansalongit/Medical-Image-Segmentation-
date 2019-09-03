import numpy as np
import cv2
import random

from skimage.io import imread

import torch
import torch.utils.data
from torchvision import datasets, models, transforms

IMG_HEIGHT = 256
IMG_WIDTH = 256

class Dataset(torch.utils.data.Dataset):

    def __init__(self, args, img_paths, mask_paths, aug=False):
        self.args = args
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.aug = aug

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        image = cv2.imread(img_path,1)[:,:,0]
        image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
        image = np.expand_dims(image, axis=-1)
        
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1))
        _mask = cv2.imread(mask_path)[:,:,0]
        _mask = cv2.resize(_mask, (IMG_HEIGHT, IMG_WIDTH))
        _mask = np.expand_dims(_mask, axis=-1)
        mask = np.maximum(mask, _mask)

        image = image.astype('float32') / 255
        mask = mask.astype('float32') / 255

        if self.aug:
            if random.uniform(0, 1) > 0.5:
                image = image[:, ::-1, :].copy()
                mask = mask[:, ::-1].copy()
            if random.uniform(0, 1) > 0.5:
                image = image[::-1, :, :].copy()
                mask = mask[::-1, :].copy()

        image = image.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))

        return image, mask