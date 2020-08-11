# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 18:59:08 2020

@author: rtgun
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization, SpatialDropout2D, Activation, Dropout
from tensorflow.keras.layers import  Conv3D, MaxPooling3D, Conv3DTranspose, SpatialDropout3D, Flatten, Dense, AveragePooling3D
from tensorflow.keras.layers import Lambda
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras import losses
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import History
from tensorflow.keras.layers import Reshape, Permute, Activation, Input, add, multiply

# from tensorflow.keras.mixed_precision import experimental as mixed_precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)

from config import default_configs
cf = default_configs()
batch_size = cf.batch_size
patch_size = cf.patch_size
nfilters = cf.nfilters
model_depth = cf.model_depth
dropout_rate = cf.dropout_rate
img_rows = cf.img_rows
img_cols = cf.img_cols

# 2DU Net
def UNet2D():
    inputs = Input((patch_size[0], patch_size[1], 1))
    x = inputs
    nf = nfilters
    depth = model_depth
    d_rate = dropout_rate
    skips = []
    for i in range(depth):
        x = Conv2D(nf, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(nf, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        skips.append(x)
        x = MaxPooling2D((2, 2))(x)
        x = SpatialDropout2D(d_rate)(x)
        nf = nf * 2

    x = Conv2D(nf, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(nf, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    for i in reversed(range(depth)):
        nf = nf // 2
        x = Conv2DTranspose(nf, (2, 2), strides=(2, 2), padding='same')(x)
        x = SpatialDropout2D(d_rate)(x)
        x = concatenate([skips[i], x])
        x = Conv2D(nf, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(nf, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)

    conv6 = Conv2D(1, (1, 1), padding='same')(x)
    conv7 = Activation('sigmoid', dtype='float32')(conv6)
    model = Model(inputs=inputs, outputs=conv7)

    return model

def UNet3D():
    inputs = Input((patch_size[0], patch_size[1], patch_size[2], 1))
    x = inputs
    nf = nfilters
    depth = model_depth
    d_rate = dropout_rate
    skips = []
    for i in range(depth):
        x = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        skips.append(x)
        x = MaxPooling3D((2, 2, 2))(x)
        x = SpatialDropout3D(d_rate)(x)
        nf = nf * 2

    x = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    for i in reversed(range(depth)):
        nf = nf // 2
        x = Conv3DTranspose(nf, (2, 2, 2), strides=(2, 2, 2), padding='same')(x)
        x = SpatialDropout3D(d_rate)(x)
        x = concatenate([skips[i], x])
        x = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)

    conv6 = Conv3D(1, (1, 1, 1), padding='same')(x)
    conv7 = Activation('sigmoid')(conv6)
    model = Model(inputs=inputs, outputs=conv7)

    return model

# 2DU Net
def Detect_2D():
    nf = nfilters
    depth = model_depth
    d_rate = dropout_rate
    skips = []

    inputs = Input((patch_size[0], patch_size[1], 1))
    x = inputs

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = SpatialDropout2D(d_rate)(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = SpatialDropout2D(d_rate)(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = SpatialDropout2D(d_rate)(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = SpatialDropout2D(d_rate)(x)
    
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = SpatialDropout2D(d_rate)(x)

    # for i in range(depth):
    #     x = Conv2D(nf, (3, 3), activation='relu', padding='same')(x)
    #     x = BatchNormalization()(x)
    #     x = Conv2D(nf, (3, 3), activation='relu', padding='same')(x)
    #     x = BatchNormalization()(x)
    #     x = MaxPooling2D((2, 2))(x)
    #     x = SpatialDropout2D(d_rate)(x)
    #     nf = nf * 2

    # x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    # x = Dense(4096, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(64, activation = 'relu')(x)
    x = Dense(1, activation='relu')(x)
    out = Activation('sigmoid', dtype = 'float32')(x)
    model = Model(inputs = inputs, outputs = out)
    return model

def Detect_3D():
    inputs = Input((patch_size[0], patch_size[1], patch_size[2], 1))
    x = inputs
    nf = nfilters
    depth = model_depth
    d_rate = dropout_rate

    for i in range(depth):
        x = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv3D(nf, (3, 3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling3D((2, 2, 2))(x)
        x = SpatialDropout3D(d_rate)(x)
        nf = nf * 2

    x = Flatten()(x)

    x = Dense(8192, activation='relu')(x)
    x = Dropout(d_rate)(x)
    x = BatchNormalization()(x)

    x = Dense(8192, activation='relu')(x)
    x = Dropout(d_rate)(x)
    x = BatchNormalization()(x)
    
    x = Dense(512, activation='relu')(x)
    x = Dropout(d_rate)(x)
    x = BatchNormalization()(x)

    x = Dense(64, activation='relu')(x)
    x = Dropout(d_rate)(x)
    x = BatchNormalization()(x)
    
    x = Dense(1)(x)
    out = Activation('sigmoid', dtype = 'float32')(x)
    model = Model(inputs = inputs, outputs = out)
    return model

def conv_bn_relu_drop(x, nf, ker_size, stride_size, d_rate):
  x = Conv3D(filters = nf, kernel_size = ker_size, padding='same', strides=stride_size)(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  # x = SpatialDropout3D(d_rate)(x)
  return x

def transconv_bn_relu_drop(x, nf, ker_size, str_size, d_rate):
  x = Conv3DTranspose(filters = nf, kernel_size = ker_size, padding='same', strides=str_size)(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  # x = SpatialDropout3D(d_rate)(x)
  return x

def dense_bn_relu_drop(x, nodes, d_rate):
  x = Dense(units = nodes)(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  # x = Dropout(d_rate)(x)
  return x

def VNet():
    inputs = Input((patch_size[0], patch_size[1], patch_size[2], 1))
    x = inputs
    nf = nfilters
    depth = model_depth
    d_rate = dropout_rate
    skips = []

    for i in range(depth):
        res = x
        for _ in range(min(i+1, depth-1)):
          x = conv_bn_relu_drop(x, nf, 5, 1, d_rate)
        x += conv_bn_relu_drop(res, nf, 1, 1, d_rate)
        skips.append(x)
        x = conv_bn_relu_drop(x, nf, 2, 2, d_rate)
        nf = nf * 2

    res = x
    for i in range(3):
      x = conv_bn_relu_drop(x, nf, 5, 1, d_rate)
    x += conv_bn_relu_drop(res, nf, 1, 1, d_rate)
    nf = nf * 2

    for i in reversed(range(depth)):
        nf = nf // 2
        x = transconv_bn_relu_drop(x, nf, 2, 2, d_rate)
        res = x
        x = concatenate([skips[i], x])
        for _ in range(min(i+1, depth-1)):
          x = conv_bn_relu_drop(x, nf, 5, 1, d_rate)
        x += conv_bn_relu_drop(res, nf, 1, 1, d_rate)

    conv6 = conv_bn_relu_drop(x, 1, 1, 1, d_rate)
    conv7 = Activation('sigmoid')(conv6)
    model = Model(inputs=inputs, outputs=conv7)

    return model

def Det_VNet():
    inputs = Input((patch_size[0], patch_size[1], patch_size[2], 1))
    x = inputs
    nf = nfilters
    depth = model_depth
    d_rate = dropout_rate
    skips = []

    for i in range(depth):
        res = x
        for _ in range(min(i+1, depth-1)):
          x = conv_bn_relu_drop(x, nf, 5, 1, d_rate)
        x += conv_bn_relu_drop(res, nf, 1, 1, d_rate)
        skips.append(x)
        x = conv_bn_relu_drop(x, nf, 2, 2, d_rate)
        nf = nf * 2

    res = x
    for i in range(3):
      x = conv_bn_relu_drop(x, nf, 5, 1, d_rate)
    x += conv_bn_relu_drop(res, nf, 1, 1, d_rate)
    nf = nf * 2

    # x = AveragePooling3D(4)(x)
    x = Flatten()(x)

    # Fully Connected Layers
    x = dense_bn_relu_drop(x, 8192, d_rate)
    x = dense_bn_relu_drop(x, 2048, d_rate)
    x = dense_bn_relu_drop(x, 512, d_rate)
    x = dense_bn_relu_drop(x, 32, d_rate)

    x = Dense(1)(x)
    out = Activation('sigmoid', dtype = 'float32')(x)
    model = Model(inputs = inputs, outputs = out)
    return model