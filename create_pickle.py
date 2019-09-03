import os
import sys
import numpy as np
import tensorflow as tf
import random
import math
import warnings
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

seed = 42
random.seed = seed
np.random.seed = seed

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 1

DATA_PATH = './Datasets'
Images = []
Masks = []
train_size = 30

for class_folder in os.listdir(DATA_PATH):
    FOLDER_PATH = os.path.join(DATA_PATH, class_folder)
    for patient_folder in os.listdir(FOLDER_PATH):
        patient_folder = os.path.join(FOLDER_PATH, patient_folder)
        if os.path.isdir(patient_folder):
            Images.append(os.path.join(patient_folder, "LAT.jpg"))
            Masks.append(os.path.join(patient_folder, 'LAT/Lat_Spinous_Process.png'))

images = []
labels = []

print("Getting and resizing train images and masks ... ")
sys.stdout.flush()

iter = 0
for n, path in tqdm(enumerate(Images), total=len(Images)):
    try:
        img = cv2.imread(path, 1)[:,:,0]
        img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
        img = np.expand_dims(img, axis=-1)
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1))
        _mask = cv2.imread(Masks[iter])[:,:,0]
        _mask = cv2.resize(_mask, (IMG_HEIGHT, IMG_WIDTH))
        _mask = np.expand_dims(_mask, axis=-1)
        mask = np.maximum(mask, _mask)
        img = img/255.0
        mask = mask/255.0
        
        images.append(img)
        labels.append(mask)

        iter+=1
    except:
        pass

images = np.array(images)
labels = np.array(labels)
shuffle_mask = np.arange(iter)
np.random.shuffle(shuffle_mask)
X_train = images[shuffle_mask[:train_size]]
X_test = images[shuffle_mask[train_size:]]
Y_train = labels[shuffle_mask[:train_size]]
Y_test = labels[shuffle_mask[train_size:]]

print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

dbfile = open('./pickle_files/Lat_Spinous_Process.pkl', 'ab')
pickle.dump({'X_train': X_train, 'X_test': X_test, 'Y_train':Y_train, 'Y_test':Y_test}, dbfile)
dbfile.close()

