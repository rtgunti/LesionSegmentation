# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 19:26:29 2020

@author: rtgun
"""
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

import random
from random import randint
from random import random
import nibabel
from nilearn.image import resample_img

import seaborn as sns
import matplotlib.pyplot as plt 

import os
import numpy as np
from glob import glob

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


from eda import get_img_mask_names

from utility_functions import stat, trim_data, show_imgs
img_names, mask_names = get_img_mask_names()


def preprocess_resample():
  '''
  1. Load nii img and mask using nibabel
  2. Resample img(continuous) and mask(nearest) to 1x1x1mm using nilearn.resample_img
  3. Rotate and flip for better view
  4. min-max or z normalization for img
  5. Extract liver from img
  6. For mask, make liver as background and clip mask values to (0,1)
  7. Remove the blank voxels
  8. Save as img and mask as .npy files
  '''
  os.chdir(raw_data_path)
  img_names = glob("*pre*")
  os.chdir(raw_seg_path)
  mask_names = glob("*pre*")
  os.chdir(project_path)
  img_names.sort()
  mask_names.sort()
  print(len(img_names), len(mask_names), img_names[0], mask_names[0])
  for ind in range(len(img_names)):
    # ind = 6
    print(ind, img_names[ind], mask_names[ind])
    img = nibabel.load(raw_data_path + img_names[ind])
    mask = nibabel.load(raw_seg_path + mask_names[ind])

    img = resample_img(img, np.eye(3), interpolation='continuous')
    mask = resample_img(mask, np.eye(3), interpolation='nearest')

    img = img.get_fdata()
    mask = mask.get_fdata()
    
    img = np.rot90(img)
    mask = np.rot90(mask)
    
    img = np.fliplr(img)   #only for thesis
    mask = np.fliplr(mask)

    # min_, max_ = float(np.min(img)), float(np.max(img))
    # img = (img - min_) / (max_ - min_)

    img *= np.clip(mask, 0, 1)
    
    mask = np.clip(mask, 0, 2)
    mask[mask == 1] = 0
    mask[mask == 2] = 1

    img, mask = trim_data(img, mask)
    mask = mask.astype('uint8')
    img = img.astype('float32')

    # stat(img)
    # # Min Max Normalization
    # min_ = np.min(img)
    # max_ = np.max(img)
    # img = (img - min_)/(max_ - min_)

    # zScaling
    # mean_ = np.mean(img)
    # std_ = np.std(img)
    # img -= mean_
    # img /= std_ 
    # stat(img)

    np.save(resampled_data_path + img_names[ind].split('.')[0] + '.npy', img)
    np.save(resampled_seg_path + mask_names[ind].split('.')[0] + '.npy', mask)
    # return img, mask
    # break

def preprocess_znorm():
  '''
  Load resampled .npz data
  Caclulate global _mean and _std using get_global_stats()
  zNormalize each record with _mean and _std
  Save in 
  '''
  os.chdir(resampled_data_path)
  img_names = glob("*.npy*")
  os.chdir(resampled_seg_path)
  mask_names = glob("*.npy*")
  os.chdir(project_path)
  img_names.sort()
  mask_names.sort()
  
  #use get_global_stats() to compute global stats

  # #For Thesis pre Data
  # _min = -103.54093933105469
  # _max = 3386.0966796875
  # _avg = 250.36423532498463
  # _std = 434.0828201129888
  _min, _max, _mean, _std = get_global_data_stats()

  # #For Thesis Intervension Data
  # _min :  -106.36845397949219  
  # _max :  5758.24267578125  
  # _median :  0.0  
  # _mean :  280.6546859685511  
  # _std :  506.65324054321627
  # _min , _max, _mean, _std = get_global_data_stats()
  
  for ind in range(len(img_names)):
    print(ind)
    img = np.load(resampled_data_path + img_names[ind])
    mask = np.load(resampled_seg_path + mask_names[ind])
    stat(img)
    # zScore Normalization
    img -= mean_
    img /= std_
    # Min-Max Normalization
    # img = (img - min_)/(max_ - min_)
    stat(img)
    np.save(zNorm_data_path + img_names[ind] , img)
    np.save(zNorm_seg_path + mask_names[ind] , mask)
    # break

def get_global_data_stats():
  '''
  Print stats of whole data. This is used to compute global stats
  '''
  os.chdir(resampled_data_path)
  img_names = glob("*.npy")
  os.chdir(resampled_seg_path)
  mask_names = glob("*.npy")
  os.chdir(project_path)
  img_names.sort()
  mask_names.sort()
  print(len(img_names), len(mask_names))
  img_ = []
  img_ = np.array(img_)
  for ind in range(len(img_names)):
    print(ind)
    img_ = np.append(img_, np.load(resampled_data_path + img_names[ind]).flatten())
  stat(img_)
  # return np.min(img_), np.max(img_), np.mean(img_), np.std(img_)


def crop_liver_from_raw():
  '''
  1. Load nii img and mask using nibabel
  2.
  3. Rotate and flip for better view
  4. min-max or z normalization for img
  5. Extract liver from img
  6. For mask, make liver as background and clip mask values to (0,1)
  7. Remove the blank voxels
  8. Save as img and mask as .npy files
  '''
  os.chdir(raw_data_path)
  img_names = glob("*pre*")
  os.chdir(raw_seg_path)
  mask_names = glob("*pre*")
  os.chdir(project_path)
  img_names.sort()
  mask_names.sort()
  print(len(img_names), len(mask_names), img_names[0], mask_names[0])
  for ind in range(len(img_names)):
    # ind = 6
    print(ind, img_names[ind], mask_names[ind])
    img = nibabel.load(raw_data_path + img_names[ind])
    mask = nibabel.load(raw_seg_path + mask_names[ind])

    img = img.get_fdata()
    mask = mask.get_fdata()
    
    img = np.rot90(img)
    mask = np.rot90(mask)
    
    img = np.fliplr(img)   #only for thesis
    mask = np.fliplr(mask)

    img *= np.clip(mask, 0, 1)
    
    mask = np.clip(mask, 0, 2)
    mask[mask == 1] = 0
    mask[mask == 2] = 1

    img, mask = trim_data(img, mask)
    mask = mask.astype('uint8')
    img = img.astype('float32')

    np.save(liver_cropped_data_path + img_names[ind].split('.')[0] + '.npy', img)
    np.save(liver_cropped_seg_path + mask_names[ind].split('.')[0] + '.npy', mask)
    # return img, mask
    # break

def check_hist_eq():
  '''
  Check if Histogram Equalization is helpful
  '''

  os.chdir(raw_data_path)
  img_names = glob("*nii*")
  os.chdir(raw_seg_path)
  mask_names = glob("*nii*")
  os.chdir(project_path)

  img_names.sort()
  mask_names.sort()

  ind = 2
  print(img_names[ind], mask_names[ind])
  img = nibabel.load(raw_data_path + img_names[ind])
  mask = nibabel.load(raw_seg_path + mask_names[ind])

  img = img.get_fdata()
  mask = mask.get_fdata()

  img = np.rot90(img)
  mask = np.rot90(mask)

  img = np.clip(img, -100, 400)
  # img *= np.clip(mask, 0, 1)
  # img, mask = trim_data(img, mask)
  # img = np.squeeze(img)
  # mask = np.squeeze(mask)

  # Equalization
  img_eq = exposure.equalize_hist(img)

  # Contrast stretching
  p2, p98 = np.percentile(img, (2, 98))
  img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))

  stat(img)
  stat(img_eq)
  stat(img_rescale)
  
  sns.distplot(img, label = 'img')
  plt.figure()
  sns.distplot(img_eq, label = 'img_eq')
  plt.figure()
  sns.distplot(img_rescale, label = 'img_rescale')
  # plt.legend()
  plt.figure()
  print(img.shape, mask.shape)
  for ind in range(0, img.shape[-1], 2):
    show_imgs([img[:,:,ind], mask[:,:,ind], img_eq[:,:,ind], img_rescale[:,:,ind]])