# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 18:54:02 2020

@author: rtgun
"""

img_rows = int(512/2)
img_cols = int(512/2)
nslices = 86
rft = 44 #Records for training
rfv = 10 #Records for validation
th_les_pix = 100 #Threshold to exclude small lesions from training
epsilon = 1e-5
# patch_size = (img_rows, img_cols, nslices) #For 2D

root = '/content/drive/My Drive/Thesis/'
dataset_name = 'Thesis/'
# dataset_name = 'LiTS/'
# dataset_name = 'Brats/'
project_path = root + dataset_name

generic_path = root + 'Generic/' 

raw_folder = project_path + 'Dataset/'
# raw_folder = project_path + 'ResampledIntData/'
raw_data_path = raw_folder + 'data/'
raw_seg_path = raw_folder + 'seg/'

resampled_folder = project_path + 'Resampled/'
resampled_data_path = resampled_folder + 'data/'
resampled_seg_path = resampled_folder + 'seg/'

liver_cropped_folder = project_path + 'LiverCropped/'
liver_cropped_data_path = liver_cropped_folder + 'data/'
liver_cropped_seg_path = liver_cropped_folder + 'seg/'

zNorm_folder = project_path + 'zNormalized/'
zNorm_data_path = zNorm_folder + 'data/'
zNorm_seg_path = zNorm_folder + 'seg/'

# preprocessed_folder = project_path + 'PreprocessedmData/'
# preprocessed_folder = project_path + 'zNormalized/'
# preprocessed_folder = project_path + 'LiverCropped/'
preprocessed_folder = project_path + 'Resampled/'
# preprocessed_folder = project_path + 'ResampledIntzData/'
data_path = preprocessed_folder + 'data/'
seg_path = preprocessed_folder + 'seg/'

log_path = project_path + 'logs/'

patch_size = ( 64, 64, 64 )
batch_size = 8
mul_thread_flag = True

dim_type = '3D'
generator = 'BatGen'
nfilters = 16
loss = 'Dice'
epochs = 500
opt = 'Adam'
model_depth = 4
dropout_rate = 0.

def get_max_shape():
    return [int(round(i*1.2)) for i in patch_size]
    # return patch_size

def get_batch_size():
  return batch_size

def get_img_size():
  return img_rows, img_cols

def get_nslices():
  return nslices

def get_rft_rfv():
  return rft, rfv

def get_th_les_pix():
  return th_les_pix

def get_data_path():
  return data_path

def get_seg_path():
  return seg_path

def get_project_path():
  return project_path

def get_patch_size():
  return patch_size

def get_train_val_ind():
  if ('LiTS' in dataset_name):
    train_ind = [i for i in range(101)]
    val_ind = [i for i in range(101, 126)]
    if (not mul_thread_flag):
      train_ind = [6]
      val_ind = [6]
  elif('Brats' in dataset_name):
    train_ind = [0]
    val_ind = [0]
  else:

    # train_ind = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 29, 30, 31, 32, 33, 34, 35, 36, 37, 48]
    # val_ind = [ 41, 42, 43, 44, 45, 46, 38, 49, 52 ]
    # excluded_ind = [4, 16, 27, 28, 39, 40, 47, 50, 51]

    train_ind = [i for i in range(44)]
    val_ind = [i for i in range(44,54)]
    
    # train_ind = [ind for ind in train_ind if ind not in [4, 8, 9, 10, 27, 28]]
    # val_ind = [ind for ind in val_ind if ind not in [47, 50, 51]]
    
    # train_ind = [6]
    # val_ind = [6]

    #Intervension Data
    # train_ind = [i for i in range(60)] 
    # val_ind = [i for i in range(60,75)]
    if (not mul_thread_flag):
      train_ind = [6]
      val_ind = [6]
  return train_ind, val_ind
