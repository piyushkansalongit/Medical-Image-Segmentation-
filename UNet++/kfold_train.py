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
import pandas as pd
from sklearn.model_selection import train_test_split
from skimage.io import imread, imsave
from sklearn.model_selection import KFold
import torch
import cv2
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


arch_names = list(archs.__dict__.keys())
loss_names = list(losses.__dict__.keys())
loss_names.append('BCEWithLogitsLoss')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default="Lat_Vertebra",
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='NestedUNet',
                        choices=arch_names,
                        help='model architecture: ' +
                            ' | '.join(arch_names) +
                            ' (default: NestedUNet)')
    parser.add_argument('--deepsupervision', default=False, type=str2bool)
    parser.add_argument('--dataset', default=None,
                        help='dataset name')
    parser.add_argument('--input-channels', default=1, type=int,
                        help='input channels')
    parser.add_argument('--image-ext', default='jpg',
                        help='image file extension')
    parser.add_argument('--mask-ext', default='png',
                        help='mask file extension')
    parser.add_argument('--aug', default=False, type=str2bool)
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=loss_names,
                        help='loss: ' +
                            ' | '.join(loss_names) +
                            ' (default: BCEDiceLoss)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=20, type=int,
                        metavar='N', help='early stopping (default: 20)')
    parser.add_argument('-b', '--batch-size', default=1, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    args = parser.parse_args()

    return args


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(args, train_loader, model, criterion, optimizer, epoch, scheduler=None):
    losses = AverageMeter()
    ious = AverageMeter()

    model.train()

    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        input = input.cuda()
        target = target.cuda()

        # compute output
        if args.deepsupervision:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou = iou_score(outputs[-1], target)
        else:
            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output, target)

        losses.update(loss.item(), input.size(0))
        ious.update(iou, input.size(0))

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
    ])

    return log


def validate(args, val_loader, model, criterion):
    losses = AverageMeter()
    ious = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            # compute output
            if args.deepsupervision:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou = iou_score(output, target)

            losses.update(loss.item(), input.size(0))
            ious.update(iou, input.size(0))

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
    ])

    return log


def main():
    args = parse_args()

    if args.name is None:
        if args.deepsupervision:
            args.name = '%s_%s_wDS' %(args.dataset, args.arch)
        else:
            args.name = '%s_%s_woDS' %(args.dataset, args.arch)
    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' %args.name)

    # define loss function (criterion)
    if args.loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[args.loss]().cuda()

    cudnn.benchmark = True



    DATA_PATH = '../../Datasets/'

    img_paths = []
    mask_paths = []
    for class_folder in os.listdir(DATA_PATH):
        FOLDER_PATH = os.path.join(DATA_PATH, class_folder)
        for patient_folder in os.listdir(FOLDER_PATH):
            patient_folder = os.path.join(FOLDER_PATH, patient_folder)
            if os.path.isdir(patient_folder):
                if(os.path.isfile(os.path.join(patient_folder, 'LAT/Lat_Vertebra.png'))):
                    mask_paths.append(os.path.join(patient_folder, 'LAT/Lat_Vertebra.png'))
                    img_paths.append(os.path.join(patient_folder, "LAT.jpg"))

    c = list(zip(img_paths, mask_paths))

    random.shuffle(c)

    img_paths, mask_paths = zip(*c)
    img_paths = np.array(img_paths)
    mask_paths = np.array(mask_paths)

    k = 10
    kf = KFold(n_splits=k)
    fold_num = 0
    mean_ious = []
    for train_index, test_index in kf.split(img_paths):
        train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
        train_test_split(img_paths[train_index], mask_paths[train_index], test_size=0.08, random_state=41)

        # create model
        print("=> creating model %s for fold %s" %(args.arch,fold_num))
        fold_num+=1
        model = archs.__dict__[args.arch](args)

        model = model.cuda()

        print(count_params(model))

        if args.optimizer == 'Adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        elif args.optimizer == 'SGD':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)

        train_dataset = Dataset(args, train_img_paths, train_mask_paths, args.aug)
        val_dataset = Dataset(args, val_img_paths, val_mask_paths)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False)

        log = pd.DataFrame(index=[], columns=[
            'epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou'
        ])

        best_iou = 0
        trigger = 0
        for epoch in range(args.epochs):
            print('Epoch [%d/%d]' %(epoch, args.epochs))

            # train for one epoch
            train_log = train(args, train_loader, model, criterion, optimizer, epoch)
            # evaluate on validation set
            val_log = validate(args, val_loader, model, criterion)

            print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
                %(train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

            tmp = pd.Series([
                epoch,
                args.lr,
                train_log['loss'],
                train_log['iou'],
                val_log['loss'],
                val_log['iou'],
            ], index=['epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou'])

            log = log.append(tmp, ignore_index=True)
            log.to_csv('models/%s/log.csv' %args.name, index=False)

            trigger += 1

            if val_log['iou'] > best_iou:
                torch.save(model.state_dict(), './models/%s/model.pth' %args.name)
                best_iou = val_log['iou']
                print("=> saved best model")
                trigger = 0

            # early stopping
            if not args.early_stop is None:
                if trigger >= args.early_stop:
                    print("=> early stopping")
                    break

            torch.cuda.empty_cache()

        args = joblib.load('models/%s/args.pkl' %args.name)

        if not os.path.exists('output/%s' %args.name):
            os.makedirs('output/%s' %args.name)

        joblib.dump(args, 'models/%s/args.pkl' %args.name)

        # create model
        print("=> Testing model %s" %args.arch)
        model = archs.__dict__[args.arch](args)

        model = model.cuda()

        test_img_paths, test_mask_paths = img_paths[test_index], mask_paths[test_index]
        input_paths = test_img_paths

        model.load_state_dict(torch.load('models/%s/model.pth' %args.name))
        model.eval()

        test_dataset = Dataset(args, test_img_paths, test_mask_paths)
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
                    test_img_paths = test_img_paths[args.batch_size*i:args.batch_size*(i+1)]
                    
                    imsave(os.path.join("./output/%s"%args.name, str(i)+".png"), (output[0,0,:,:]*255).astype('uint8'))    

            torch.cuda.empty_cache()

        # IoU
        ious = []
        for i in tqdm(range(len(test_mask_paths))):
            input_img = cv2.imread(input_paths[i],1)[:,:,0]
            input_img = cv2.resize(input_img, (256, 256))

            mask = np.zeros((256, 256))
            _mask = cv2.imread(test_mask_paths[i])[:,:,0]
            _mask = cv2.resize(_mask, (256, 256))
            mask = np.maximum(mask, _mask)

            pb = imread('output/%s/'%args.name+str(i)+".png")

            mask = mask.astype('float32') / 255
            pb = pb.astype('float32') / 255

            iou = iou_score(pb, mask)
            ious.append(iou)
        mean_ious.append(np.mean(ious))
        print("\n")
    print(mean_ious)
    print(np.mean(mean_ious))

if __name__ == '__main__':
    main()