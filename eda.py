import os
from config import get_train_val_ind, raw_data_path, raw_seg_path, project_path
from config import data_path, seg_path, generic_path
import nibabel
import numpy as np
from glob import glob

# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

def get_img_mask_names():
  # os.chdir(data_path)
  img_names = os.listdir(data_path)
  # os.chdir(seg_path)
  mask_names = os.listdir(seg_path)
  # os.chdir(generic_path)
  img_names.sort()
  mask_names.sort()
  return img_names, mask_names

def plot_data_shape_distribution():
  '''
  Each record is of shape (b, c, x, y, z)
  Plots the distribution of x, y, z of all records 
  '''
  os.chdir(data_path)
  img_names = glob("*.npy")
  os.chdir(seg_path)
  mask_names = glob("*.npy")
  os.chdir(project_path)
  img_names.sort()
  mask_names.sort()
  shapes = []
  for ind in range(len(img_names)):
    imgs = np.load(data_path + img_names[ind])
    masks = np.load(seg_path + mask_names[ind])
    shapes.append(imgs.shape[2:]) #Shape is (b,c,x,y,z)
  shapes = np.array(shapes)
  f, axes = plt.subplots(1, 3, figsize=(7*3, 7))
  sns.despine(left=True)
  for ind in range(3):
    sns.distplot(shapes[:,ind], ax=axes[ind])
    axes[ind].set_xlabel('img.shape[' + str(ind)+ ']')
    axes[ind].set_ylabel('Frequency')
  plt.legend()
  plt.suptitle('Data Shape distribution per axis')
  plt.figure()
  # [(i, np.min(sizes[:,i]), np.max(sizes[:,i]), round(np.mean(sizes[:,i]))) for i in range(3)]

def plot_img_int_dist():
  '''
  Plots intensity values vs normalized bin count for liver of all records in single plot
  This is to check the overlap of intensity distributions
  '''
  data_path = raw_data_path
  seg_path = raw_seg_path
  os.chdir(raw_data_path)
  img_names = glob("*pre*")
  os.chdir(raw_seg_path)
  mask_names = glob("*pre*")
  os.chdir(project_path)

  img_names.sort()
  mask_names.sort()
  for ind in range(len(img_names)):
    img = nibabel.load(os.path.join(data_path, img_names[ind]))
    mask = nibabel.load(os.path.join(seg_path, mask_names[ind]))
    img = np.array(img.get_fdata())
    mask = np.array(mask.get_fdata())
    mask = np.clip(mask, 0,1)
    img *= mask
    img, mask = trim_data(img, mask)
    sns.distplot(np.unique(np.squeeze(img)), label=ind)
  plt.xlabel("Intensity values")
  plt.ylabel("Normalized bin count")
  plt.figure()

def plot_lesion_int_distribution(ind_ = 6):
  '''
  Plots intensity values vs normalized bin count for liver and lesion of each raw record
  This is to check the overlapping intensities of liver and lesion for each record
  '''
  os.chdir(raw_data_path)
  img_names = glob("*pre*")
  os.chdir(raw_seg_path)
  mask_names = glob("*pre*")
  os.chdir(project_path)
  img_names.sort()
  mask_names.sort()
  for ind in range(len(img_names)):
    # ind = ind_
    img = nibabel.load(os.path.join(raw_data_path, img_names[ind]))
    mask = nibabel.load(os.path.join(raw_seg_path, mask_names[ind]))
    img = np.array(img.get_fdata())
    mask = np.array(mask.get_fdata())
    sns.distplot((img[mask == 1]), label='Liver')
    sns.distplot((img[mask > 1]), label='Lesions')
    plt.title(str(ind) + " : " + img_names[ind])
    plt.xlabel("Intensity values")
    plt.ylabel("Normalized bin count")
    plt.legend()
    plt.title(ind)
    plt.figure()

def get_les_raw_data(ind_ = None):
  '''
  Plots #voxels per lesion for each record.
  Ind 30 is corrupted. To be fixed
  '''
  os.chdir(raw_data_path)
  img_names = glob("*pre*")
  os.chdir(raw_seg_path)
  mask_names = glob("*pre*")
  os.chdir(project_path)
  img_names.sort()
  mask_names.sort()
  for ind in range(len(mask_names)):
    if(ind == 30):
      continue
    ind = ind_
    mask_name  = mask_names[ind]
    mask = nibabel.load(os.path.join(raw_seg_path, mask_name))
    mask = mask.get_fdata()
    print(ind, np.unique(mask, return_counts=True))
    num_lesions = len(np.unique(mask))-2
    les_pix = np.ndarray((num_lesions))
    for k in range(num_lesions):
      les_pix[k] = np.unique(mask[mask == (k+10)], return_counts=True)[1]
    sns.barplot(x = np.arange(10, 10+num_lesions), y = np.asarray(les_pix)).set_title(str(ind) + " : " + mask_name)
    plt.figure()
    break

def lesion_summary():
    '''
    Prints the following for each record:
    Sum of lesion voxels
    Number of voxels for each lesion 
    '''
    print('-'*30)
    print('Lesion Summary')
    print('-'*30)
    masks = os.listdir(raw_seg_path)
    masks = [x for x in masks if "int" not in x]
    masks.sort()
    num_vox_dist = np.array([])

    for ind in range(len(masks)):
        training_mask = nibabel.load(os.path.join(raw_seg_path, masks[ind]))
        mask = training_mask.get_fdata()
        les_id, num_vox = np.unique(mask, return_counts=True)
        num_vox_dist = np.append(num_vox_dist, num_vox[2:])
        print(ind, masks[ind], sum(num_vox[2:]), les_id[2:], num_vox[2:])
        # break
    ax = sns.distplot(num_vox_dist)
    ax.set(xlabel='Lesion Size(in terms of voxels)', ylabel='Frequency')
    plt.show()
    # return num_vox_dist

def plot_img_int_dist_from_pp(ind_ = 6):
  '''
  Plots intensity values vs normalized bin count for liver for preprocessed records
  '''
  os.chdir(data_path)
  img_names = glob("*pre*")
  os.chdir(seg_path)
  mask_names = glob("*pre*")
  os.chdir(project_path)
  img_names.sort()
  mask_names.sort()
  for ind in range(len(img_names)):
    img = np.load(os.path.join(data_path, img_names[ind]))
    sns.distplot(np.unique(np.squeeze(img)), label=ind)
  plt.xlabel("Intensity values")
  plt.ylabel("Normalized bin count")
  # plt.legend()
  plt.figure()

def plot_stats(stat_):
  '''
  Plot stats() of each record
  Params : stat_ - np array of stat() of each record
  ToDo : Write a method to create stat_array
  '''
  f, axes = plt.subplots(1, 4, figsize=(7*4, 7))
  sns.despine(left=True)
  for ind in range(4):
    sns.distplot(stat_[:,ind], ax=axes[ind])
    axes[ind].set_xlabel('stat_[' + str(ind)+ ']')
    axes[ind].set_ylabel('Frequency')
  plt.legend()
  plt.suptitle('Stat Distribution')
  plt.figure()

def xplot_train_val_les_dist():
  '''
  Plots intensity distripution of lesion pixels of train, val and overall.
  This is to check if train and val are from almost same distribution 
  '''
  data_path = raw_data_path
  seg_path = raw_seg_path
  os.chdir(data_path)
  img_names = glob("*pre*")
  os.chdir(seg_path)
  mask_names = glob("*pre*")
  os.chdir(project_path)
  img_names.sort()
  mask_names.sort()
  total_ = []
  total_ = np.array(total_)
  train_ = []
  train_ = np.array(train_)
  val_ = []
  val_ = np.array(val_)
  for ind in range(len(img_names)):
    img = nibabel.load(os.path.join(data_path, img_names[ind]))
    mask = nibabel.load(os.path.join(seg_path, mask_names[ind]))
    img = np.array(img.get_fdata())
    mask = np.array(mask.get_fdata())
    print(ind, img[mask > 1].shape)
    total_ = np.append(total_, (img[mask > 1]).flatten())
    if(ind in train_ind):
      train_ = np.append(train_, (img[mask > 1]).flatten())
    else:
      val_ = np.append(val_, img[mask > 1].flatten())
    # sns.distplot(img[mask > 1 ], label=ind)
    # sns.distplot(img, label = ind+1)
    # break 
  print(total_.shape, train_.shape, val_.shape)
  # sns.distplot(img, label='img')
  sns.distplot(total_, label = 'total')
  sns.distplot(train_, label='train_')
  sns.distplot(val_, label='val_')
  plt.legend()
  plt.xlabel("Intensity values")
  plt.ylabel("Normalized bin count")
  plt.figure()
