# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 12:30:15 2020
@author: rtgun
"""
from batchviewer import view_batch
import numpy as np
import nibabel as nib
from generators import get_train_val_generators, generator3D_patch
tr_gen, val_gen = get_train_val_generators()

train_generator = generator3D_patch(tr_gen)
imgs, masks = next(train_generator)

imgs = np.squeeze(imgs)
masks = np.squeeze(masks)

print(imgs.shape, masks.shape)

# imgs = np.rot90(imgs, k=2, axes=(-2, -1))
# masks = np.rot90(masks, k=2, axes=(-2, -1))

print(imgs.shape, masks.shape)

data_f = np.concatenate((imgs[None], masks[None]), axis=0)
print(data_f.shape)

data = nib.load("d010_pre0_dat.nii.gz")
data = data.get_fdata()
print(data.shape)

data = data[None]

# view_batch(data, width = 500, height = 500)

data1 = np.transpose(data, axes = (0,3,1,2))
print(data1.shape)

data2 = np.transpose(data, axes = (0,3,2,1))
print(data2.shape)

data3 = np.rot90(data1, k = 2, axes = (-2, -1))
print(data3.shape)

data4 = np.rot90(data2, k = 2, axes = (-1, -2))
print(data4.shape)

data_f = np.concatenate((data1, data2, data3, data4))

print(data_f.shape)
view_batch(data_f, width = 500, height = 500)