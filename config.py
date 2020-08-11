# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 18:54:02 2020

@author: rtgun
"""
import os
class Configs():
    
    def __init__(self):
        print("Init")
        img_rows = int(512/2)
        img_cols = int(512/2)
        nslices = 86
        rft = 44 #Records for training
        rfv = 10 #Records for validation
        th_les_pix = 100 #Threshold to exclude small lesions from training
        epsilon = 1e-5
        # patch_size = (img_rows, img_cols, nslices) #For 2D
        
        root = "C:/Users/rtgun/Google Drive/Thesis/"
        # root = '/content/drive/My Drive/Thesis/'
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


# def get_train_val_ind():
#   if ('LiTS' in dataset_name):
#     train_ind = [i for i in range(101)]
#     val_ind = [i for i in range(101, 126)]
#     if (not mul_thread_flag):
#       train_ind = [6]
#       val_ind = [6]
#   else:
#     train_ind = [i for i in range(44)]
#     val_ind = [i for i in range(44,54)]
#     if (not mul_thread_flag):
#       train_ind = [6]
#       val_ind = [6]
#   return train_ind, val_ind
