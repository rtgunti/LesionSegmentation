# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 09:17:24 2020

@author: rtgun
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')  # TF dimension ordering in this code
from tensorflow.keras import losses
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import History, CSVLogger, TensorBoard, TerminateOnNaN

from loss_functions import loss_class

from data import create_train_data, load_train_data, load_pp_train1
from data import save_train2_data, load_train2_data, get_topk_data, get_sorted_index

from utility_functions import show_all, get_overlay, stat, show_imgs
from utility_functions import get_reservoir_sample, get_random_patch
from utility_functions import trim_data, check_queue, plot_history

from pipeline import exclude_lesions

from batchgenerators.augmentations.resample_augmentations import augment_linear_downsampling_scipy

from eda import plot_data_shape_distribution, plot_img_int_dist, plot_lesion_int_distribution
from eda import get_les_raw_data, lesion_summary, plot_img_int_dist_from_pp, plot_stats

from preprocessing import preprocess_resample, preprocess_znorm, get_global_data_stats, check_hist_eq, crop_liver_from_raw

from generators import get_list_of_patients, get_train_val_generators, get_train_transform
from generators import generator3D_patch, generator2D_patch

from models import UNet3D, UNet2D, Detect_2D, Detect_3D, VNet, Det_VNet

import os

import logging
import random
from glob import glob
import pickle

from config import default_configs
cf = default_configs()

data_path = cf.data_path

img_rows = cf.img_rows
img_cols = cf.img_cols

seg_path = cf.seg_path
project_path = cf.project_path
raw_data_path = cf.raw_data_path
raw_seg_path = cf.raw_seg_path

resampled_data_path = cf.resampled_data_path
resampled_seg_path = cf.resampled_seg_path

liver_cropped_data_path = cf.liver_cropped_data_path
liver_cropped_seg_path = cf.liver_cropped_seg_path

zNorm_data_path = cf.zNorm_data_path
zNorm_seg_path = cf.zNorm_seg_path

img_rows = cf.img_rows
img_cols = cf.img_cols
train_ind, val_ind = cf.get_train_val_ind()
batch_size = cf.batch_size
patch_size = cf.patch_size
project_path = cf.project_path
smooth = 1.
nslices = cf.nslices
rft = cf.rft
rfv = cf.rfv
epsilon = 1e-5
th_les_pix = cf.th_les_pix

train_steps_per_epoch = 40
val_steps_per_epoch = 10

os.chdir(project_path)
# os.remove(log_path + 'test.log')
logging.basicConfig(filename=cf.log_path + 'test.log',level=logging.DEBUG)

tr_gen, val_gen = get_train_val_generators()

os.chdir(data_path)
img_names = glob("*.npy")
os.chdir(seg_path)
mask_names = glob("*.npy")
os.chdir(project_path)
img_names.sort()
mask_names.sort()
print(len(img_names), len(mask_names), img_names[0], mask_names[0])

check_queue(tr_gen, val_gen)

train_generator = generator2D_patch(tr_gen)
val_generator = generator2D_patch(val_gen)

model = UNet3D()
loss_obj = loss_class
model.compile(optimizer=Adam(), loss = tf.keras.losses.binary_crossentropy, metrics=[loss_obj.f1_score, 'accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
model_id = 'SavedModels/06-06_xxx_Thesis_3D_8_64x64_16_4_0.0_500_Adam'
cb_checkpoint = ModelCheckpoint(model_id + '_cp.h5', monitor='val_dice_coef', save_best_only=False)
cb_terminateonNaN = TerminateOnNaN()
cb_csv = CSVLogger(model_id + '_CSV.csv', separator=',', append=False)

history = model.fit(
        train_generator,
        steps_per_epoch=train_steps_per_epoch,
        epochs = cf.epochs,
        validation_data=val_generator,
        validation_steps=val_steps_per_epoch,
        callbacks=[cb_terminateonNaN, cb_csv])

model.save(model_id + '_m.h5')
with open(model_id + '_History', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)