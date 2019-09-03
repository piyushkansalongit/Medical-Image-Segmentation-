# -*- coding: utf-8 -*-

import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime
import joblib
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from skimage.io import imread, imsave
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms

from dataset import Dataset

import archs
from metrics import dice_coef, batch_iou, mean_iou, iou_score
import losses
from utils import str2bool, count_params


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default="Lat_Vertebra",
                        help='model name')
    parser.add_argument('--batch_size', default=1,
                        help='batch size')
    args = parser.parse_args()

    return args


def main():
    val_args = parse_args()

    args = joblib.load('models/%s/args.pkl' %val_args.name)

    if not os.path.exists('output/%s' %args.name):
        os.makedirs('output/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    joblib.dump(args, 'models/%s/args.pkl' %args.name)

    # create model
    print("=> creating model %s" %args.arch)
    model = archs.__dict__[args.arch](args)

    model = model.cuda()


    DATA_PATH = '../../Test_Dataset/'

    img_paths = []
    mask_paths = []
    for class_folder in os.listdir(DATA_PATH):
        FOLDER_PATH = os.path.join(DATA_PATH, class_folder)
        for patient_folder in os.listdir(FOLDER_PATH):
            patient_folder = os.path.join(FOLDER_PATH, patient_folder)
            if os.path.isdir(patient_folder):
                img_paths.append(os.path.join(patient_folder, "LAT.jpg"))
                mask_paths.append(os.path.join(patient_folder, 'LAT/Lat_Vertebra.png'))

    c = list(zip(img_paths, mask_paths))

    random.shuffle(c)

    img_paths, mask_paths = zip(*c)
    img_paths = np.array(img_paths)
    mask_paths = np.array(mask_paths)
    input_paths = img_paths

    model.load_state_dict(torch.load('models/%s/model.pth' %args.name))
    model.eval()

    test_dataset = Dataset(args, img_paths, mask_paths)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        with torch.no_grad():
            for i, (input, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
                input = input.cuda()
                target = target.cuda()

                # compute output
                if args.deepsupervision:
                    output = model(input)[-1]
                else:
                    output = model(input)
            
                output = torch.sigmoid(output).data.cpu().numpy()
                img_paths = img_paths[args.batch_size*i:args.batch_size*(i+1)]
                
                imsave(os.path.join("./output/%s"%args.name, str(i)+".png"), (output[0,0,:,:]*255).astype('uint8'))    

        torch.cuda.empty_cache()

    # IoU
    ious = []
    for i in tqdm(range(len(mask_paths))):
        input_img = cv2.imread(input_paths[i],1)[:,:,0]
        input_img = cv2.resize(input_img, (256, 256))

        mask = np.zeros((256, 256))
        _mask = cv2.imread(mask_paths[i])[:,:,0]
        _mask = cv2.resize(_mask, (256, 256))
        mask = np.maximum(mask, _mask)

        pb = imread('output/%s/'%args.name+str(i)+".png")

        mask = mask.astype('float32') / 255
        pb = pb.astype('float32') / 255

        
        fig = plt.figure(figsize=(10,10))
        fig.subplots_adjust(hspace=0.5, wspace=0.5)
        ax = fig.add_subplot(2, 2, 1)
        ax.imshow( input_img, cmap="gray")
        ax.set_title('MRI Image')

        ax = fig.add_subplot(2, 2, 2)
        ax.imshow(mask, cmap="gray")
        ax.set_title('Expected Output')

        ax = fig.add_subplot(2, 2, 3)
        ax.imshow(pb, cmap="gray")
        ax.set_title('Model Output')

        ax = fig.add_subplot(2, 2, 4)
        ax.imshow(input_img, cmap="gray")
        ax.imshow(pb, cmap='jet', alpha=0.5)
        ax.set_title('Superimposition')
        plt.savefig(fname = os.path.join("./samples/Super-Imposed/%s"%args.name, str(i)+".png"))
        plt.show()
        
        iou = iou_score(pb, mask)
        ious.append(iou)
    print('IoU: %.4f' %np.mean(ious))


if __name__ == '__main__':
    main()