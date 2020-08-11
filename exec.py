# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 09:17:24 2020

@author: rtgun
"""
from config import Configs
cf = Configs()
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

from loss_functions import dice_coef, dice_coef_loss, f1_score, f1_loss, precision, recall
from loss_functions import t_score, bce_loss, wbce_loss, tp, tn, tversky, tversky_loss, focal_tversky

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

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

from skimage.segmentation import mark_boundaries
from skimage.transform import resize
from skimage.io import imsave
from sklearn.utils import class_weight
from skimage import io
from skimage.exposure import rescale_intensity
from skimage import img_as_ubyte
from skimage import exposure
from skimage import transform
from scipy.ndimage import zoom
import os

import logging
import random
from random import randint
from random import random
import nibabel
from nilearn.image import resample_img
from glob import glob
import pickle

import datetime
import time

from config import get_img_size, get_nslices, get_rft_rfv, get_th_les_pix, get_batch_size, get_patch_size, log_path
from config import data_path, seg_path, dataset_name, dim_type, generator, nfilters, loss, epochs, opt, model_depth, dropout_rate
from config import raw_data_path, raw_seg_path, get_train_val_ind, batch_size

img_rows, img_cols  = get_img_size()
train_ind, val_ind = get_train_val_ind()
batch_size = get_batch_size()
patch_size = get_patch_size()
smooth = 1.
nslices = get_nslices()
rft, rfv = get_rft_rfv()
epsilon = 1e-5
th_les_pix = get_th_les_pix()

train_steps_per_epoch = 40
val_steps_per_epoch = 10

from config import project_path
os.chdir(project_path)
# os.remove(log_path + 'test.log')
logging.basicConfig(filename=log_path + 'test.log',level=logging.DEBUG)

tr_gen, val_gen = get_train_val_generators()

os.chdir(data_path)
print(data_path)
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

model.compile(optimizer=Adam(), loss = tf.keras.losses.binary_crossentropy, metrics=[f1_score, 'accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
model_id = 'SavedModels/06-06_xxx_Thesis_3D_8_64x64_16_4_0.0_500_Adam'
cb_checkpoint = ModelCheckpoint(model_id + '_cp.h5', monitor='val_dice_coef', save_best_only=False)
cb_terminateonNaN = TerminateOnNaN()
# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# cb_tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
cb_csv = CSVLogger(model_id + '_CSV.csv', separator=',', append=False)

history = model.fit(
        train_generator,
        steps_per_epoch=train_steps_per_epoch,
        epochs = 100,
        validation_data=val_generator,
        validation_steps=val_steps_per_epoch,
        callbacks=[cb_terminateonNaN, cb_csv])

model.save(model_id + '_m.h5')
with open(model_id + '_History', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)