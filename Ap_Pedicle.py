import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as keras
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import pickle

def unet(input_size=(256,256,1)):
    inputs = Input(input_size)

    nb_filter = [64, 128, 256, 512, 1024]
    dropout = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    conv1 = Conv2D(nb_filter[0], 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(nb_filter[0], 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)
    pool1 = Dropout(dropout[0])(pool1)

    conv2 = Conv2D(nb_filter[1], 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(nb_filter[1], 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)
    pool2 = Dropout(dropout[1])(pool2)

    conv3 = Conv2D(nb_filter[2], 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(nb_filter[2], 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2,2))(conv3)
    pool3 = Dropout(dropout[2])(pool3)

    conv4 = Conv2D(nb_filter[3], 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(nb_filter[3], 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2,2))(conv4)
    pool4 = Dropout(dropout[3])(pool4)

    conv5 = Conv2D(nb_filter[4], 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(nb_filter[4], 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Dropout(dropout[4])(conv5)

    up6 = Conv2D(nb_filter[3], 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5))
    up6 = BatchNormalization()(up6)
    merge6 = concatenate([conv4, up6], axis = 3)
    conv6 = Conv2D(nb_filter[3], 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(nb_filter[3], 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Dropout(dropout[5])(conv6)

    up7 = Conv2D(nb_filter[2], 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size=(2,2))(conv6))
    up7 = BatchNormalization()(up7)
    merge7 = concatenate([conv3, up7], axis = 3)
    conv7 = Conv2D(nb_filter[2], 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(nb_filter[2], 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Dropout(dropout[6])(conv7)

    up8 = Conv2D(nb_filter[1], 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size=(2,2))(conv7))
    up8 = BatchNormalization()(up8)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(nb_filter[1], 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(nb_filter[1], 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv2D(nb_filter[0], 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size=(2,2))(conv8))
    up9 = BatchNormalization()(up9)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(nb_filter[0], 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(nb_filter[0], 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Dropout(dropout[8])(conv9)

    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    sgd  = SGD(lr = 0.0001, momentum=0.99, decay=0.005)

    def dice_coeff(y_true, y_pred, smooth=1):
        intersection = keras.sum(keras.abs(y_true * y_pred), axis=-1)
        return (2* intersection + smooth) / (keras.sum(keras.square(y_true), -1) + keras.sum(keras.square(y_pred), -1) + smooth)

    def dice_loss(y_true, y_pred):
        return 1-dice_coeff(y_true, y_pred)

    def iou(y_true, y_pred, smooth=1):
        intersection = keras.sum(keras.abs(y_true * y_pred), axis=-1)
        union = keras.sum(y_true,-1) + keras.sum(y_pred,-1) - intersection
        iou = (intersection + smooth) / ( union + smooth)
        return iou    

    model.compile(optimizer = "adam" , loss = "binary_crossentropy", metrics = [iou])

    model.summary()

    return model

pkl_file = open('../pickle_files/Ap_Pedicle.pkl', 'rb')
data = pickle.load(pkl_file)
X_train = data['X_train'] 
X_test = data['X_test'] 
Y_train = data['Y_train'] 
Y_test = data['Y_test'] 
image_size = 256

model = unet()

earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('Ap_Pedicle.hdf5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

model.fit(X_train, Y_train, validation_data = (X_test, Y_test), batch_size=1, callbacks=[earlyStopping, mcp_save, reduce_lr_loss], epochs=25)

train_result = model.predict(X_train[:3])
# train_result = (train_result > 0.5).astype(np.uint8)

fig = plt.figure()
fig.subplots_adjust(hspace=0.2, wspace=0.2)

for i in range(3):
    ax = fig.add_subplot(3, 2, i*2 + 1)
    ax.imshow(np.reshape(Y_train[i]*255, (image_size, image_size)), cmap="gray")

    ax = fig.add_subplot(3, 2, i*2 + 2)
    ax.imshow(np.reshape(train_result[i]*255, (image_size, image_size)), cmap="gray")

plt.show()
test_result = model.predict(X_test)
test_result = (test_result > 0.2).astype(np.uint8)

for i in range(X_test.shape[0]):
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(np.reshape(Y_test[i]*255, (image_size, image_size)), cmap="gray")

    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(np.reshape(test_result[i]*255, (image_size, image_size)), cmap="gray")
    plt.savefig(fname = os.path.join("../samples/Ap_Pedicle", str(i)+".png"))
    plt.show()

##Best parameters dropout 0.5 each and lr 0.003