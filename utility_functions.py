# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 18:49:09 2020

@author: rtgun
"""
import os
import numpy as np
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
from skimage.measure import label, regionprops, regionprops_table
from scipy.ndimage import zoom

from glob import glob

import random
from random import randint
from random import random
import nibabel

from config import get_img_size, get_nslices, get_rft_rfv, get_project_path, patch_size
from config import raw_data_path, raw_seg_path

img_rows, img_cols = get_img_size()
nslices = get_nslices()
rft, rfv = get_rft_rfv()
project_path = get_project_path()
  
def get_overlay(imgs, masks, preds):
    for ind in range(preds.shape[-1]):
        img = rescale_intensity(imgs[:,:,ind],out_range=(0,1))
        mask =(masks[:,:,ind]).astype('uint8')
        pred=(preds[:,:,ind]).astype('uint8')
        
        if(len(np.unique(pred)) == 1 and len(np.unique(mask)) == 1):
          continue
        print('Counter : ' + str(ind) + '/' + str(len(preds)-1))
        mask_o = mark_boundaries(img, mask, color=(1,0,0))
        pred_o = mark_boundaries(img, pred, color=(1,0,0))
        w=20
        h=20
        fig=plt.figure(figsize=(w, h))
        columns = 4
        rows = 1

        fig.add_subplot(rows, columns, 1)
        plt.imshow(mask)

        fig.add_subplot(rows, columns, 2)
        plt.imshow(pred)

        fig.add_subplot(rows, columns, 3)
        plt.imshow(mask_o)

        fig.add_subplot(rows, columns, 4)
        plt.imshow(pred_o)

        plt.show()

def stat(array):
    print ('min : ',np.min(array), ' max : ',np.max(array),' median : ',np.median(array),' avg : ',np.mean(array), ' std : ', np.std(array))

def show_imgs(img_list):
  w=15
  h=15
  fig=plt.figure(figsize=(w, h))
  columns = len(img_list)
  rows = 1
  for i in range(columns*rows):
    img = img_list[i].reshape(img_list[i].shape[0],img_list[i].shape[1])
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(img)
  plt.show()

def show_all(imgs, masks, preds):
  for ind in range(len(imgs)):
    print(str(ind) + '/' + str(len(imgs)))
    stat(imgs[ind])
    stat(masks[ind])
    stat(preds[ind])
    show_imgs([imgs[ind], masks[ind], preds[ind]])

def trim_data(imgs, masks):
    '''
    Args:
    imgs : one 3D Image (b,c,x,y,z)
    masks : one 3D Mask (b,c,x,y,z)

    # Trims the data, crops and pads with 5 units on all directions

    Return:
    cropped img, mask with same number of dimensions (b,c,x,y,z)

    '''
    print(imgs.shape, masks.shape)
    imgs = np.squeeze(imgs)
    masks = np.squeeze(masks)
    
    x = np.any(imgs, axis=(1, 2))
    y = np.any(imgs, axis=(0, 2))
    z = np.any(imgs, axis=(0, 1))

    xmin, xmax = np.where(x)[0][[0, -1]]
    ymin, ymax = np.where(y)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    imgs = imgs[xmin:xmax, ymin:ymax, zmin:zmax]
    masks = masks[xmin:xmax, ymin:ymax, zmin:zmax]

    # imgs = np.pad(imgs, (5,), mode='constant')
    # masks = np.pad(masks, (5,), mode='constant')
    
    imgs = np.reshape(imgs, (1,1,imgs.shape[0], imgs.shape[1], imgs.shape[2]))
    masks = np.reshape(masks, (1,1,masks.shape[0], masks.shape[1], masks.shape[2]))
    
    print(imgs.shape, masks.shape)
    return imgs,masks 

def check_queue(tr_gen, val_gen):
  tr_queue = [tr_gen._queues[i].qsize() for i in range(tr_gen.num_processes)]
  val_queue = [val_gen._queues[i].qsize() for i in range(val_gen.num_processes)]
  print(tr_queue, val_queue)

def numpy_to_nii(imgs, masks):
  imgs = np.squeeze(imgs)
  masks = np.squeeze(masks)
  img_nii = nibabel.Nifti1Image(imgs, affine = np.eye(4))
  mask_nii = nibabel.Nifti1Image(masks, affine= np.eye(4))
  nibabel.save(img_nii, "img.nii.gz")
  nibabel.save(mask_nii, "mask.nii.gz")

def plot_history(hist):
  for ind in hist.keys():
    if('val' in ind):
      break
    plt.plot(hist[ind])
    plt.plot(hist['val_'+ind])
    plt.title('Model '+ ind)
    plt.ylabel(ind)
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.grid()
    plt.show()
    # break

def get_reservoir_sample(y):
  pa = [k.area for k in y]
  tot_rand_ind = np.random.randint(sum(pa))
  curr_ind = pa[0]
  for ind in range(len(y)):
    if (tot_rand_ind < curr_ind):
      return ind, tot_rand_ind
    else:
      curr_ind += pa[ind + 1]

def get_random_patch(img, mask, patch_size):
  img = np.squeeze(img)
  mask = np.squeeze(mask)
  r_props = regionprops(label(mask))
  # print(len(r_props))
  # print([i.bbox_area for i in r_props])
  id, _ = get_reservoir_sample(r_props)
  # print(id)
  bbox = np.array(r_props[id].bbox)
  # print(bbox)
  for i in range(3):
    if(bbox[i+3] - patch_size[i] <= bbox[i]):
      # pad_value = int((patch_size[i] - (bbox[i+3] - bbox[i]))/2) + 1
      pad_value = int((patch_size[i] - (bbox[i+3] - bbox[i]))/2) + 1
      # print(bbox[i+3] - bbox[i])
      # print(pad_value)
      if (bbox[i] - pad_value) >= 0 :
        bbox[i] -= pad_value  #TODO :  For better random crops, pad unevenly on both sides
      else:
        bbox[i+3] += pad_value
      if (bbox[i+3] + pad_value <= mask.shape[i]):
        bbox[i+3] += pad_value
      else:
        bbox[i] -= pad_value
  indice = [np.random.randint(bbox[i], bbox[i+3] - patch_size[i]) for i in range(3)]
  # print(indice)
  random_img = img[indice[0]:indice[0]+patch_size[0], indice[1]:indice[1]+patch_size[1], indice[2]:indice[2]+patch_size[2]]
  random_mask = mask[indice[0]:indice[0]+patch_size[0], indice[1]:indice[1]+patch_size[1], indice[2]:indice[2]+patch_size[2]]
  return random_img[None][None], random_mask[None][None]