# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 18:54:02 2020

@author: rtgun
"""
import os


class default_configs():

    def __init__(self):
        print("Init")
        self.img_rows = int(512 / 2)
        self.img_cols = int(512 / 2)
        self.nslices = 86
        self.rft = 44  # Records for training
        self.rfv = 10  # Records for validation
        self.th_les_pix = 100  # Threshold to exclude small lesions from training
        self.epsilon = 1e-5
        # patch_size = (img_rows, img_cols, nslices) #For 2D

        self.root = "C:/Users/rtgun/Google Drive/Thesis/"
        # root = '/content/drive/My Drive/Thesis/'
        self.dataset_name = 'Thesis/'
        # dataset_name = 'LiTS/'
        # dataset_name = 'Brats/'
        self.project_path = os.path.join(self.root, self.dataset_name)

        self.generic_path = os.path.join(self.root, 'Generic/')

        self.raw_folder = os.path.join(self.project_path, 'Dataset/')
        # raw_folder = project_path + 'ResampledIntData/'
        self.raw_data_path = os.path.join(self.raw_folder, 'data/')
        self.raw_seg_path = os.path.join(self.raw_folder, 'seg/')

        self.resampled_folder = self.project_path + 'Resampled/'
        self.resampled_data_path = self.resampled_folder + 'data/'
        self.resampled_seg_path = self.resampled_folder + 'seg/'

        self.liver_cropped_folder = self.project_path + 'LiverCropped/'
        self.liver_cropped_data_path = self.liver_cropped_folder + 'data/'
        self.liver_cropped_seg_path = self.liver_cropped_folder + 'seg/'

        self.zNorm_folder = self.project_path + 'zNormalized/'
        self.zNorm_data_path = self.zNorm_folder + 'data/'
        self.zNorm_seg_path = self.zNorm_folder + 'seg/'

        # preprocessed_folder = project_path + 'PreprocessedmData/'
        # preprocessed_folder = project_path + 'zNormalized/'
        # preprocessed_folder = project_path + 'LiverCropped/'
        self.preprocessed_folder = self.project_path + 'Resampled/'
        # preprocessed_folder = project_path + 'ResampledIntzData/'
        self.data_path = self.preprocessed_folder + 'data/'
        self.seg_path = self.preprocessed_folder + 'seg/'

        self.log_path = self.project_path + 'logs/'

        self.patch_size = (256, 256, 16)
        self.max_shape = (256, 256, 16)
        self.batch_size = 1
        self.mul_thread_flag = False

        self.dim_type = '3D'
        self.generator = 'BatGen'
        self.nfilters = 2
        self.loss = 'Dice'
        self.epochs = 500
        self.opt = 'Adam'
        self.model_depth = 3
        self.dropout_rate = 0

    def get_train_val_ind(self):
        if ('LiTS' in self.dataset_name):
            train_ind = [i for i in range(101)]
            val_ind = [i for i in range(101, 126)]
            if (not self.mul_thread_flag):
                train_ind = [6]
                val_ind = [6]
        else:
            train_ind = [i for i in range(44)]
            val_ind = [i for i in range(44, 54)]
            if (not self.mul_thread_flag):
                train_ind = [6]
                val_ind = [6]
        self.train_ind = train_ind
        self.val_ind = val_ind
        return train_ind, val_ind
