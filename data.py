import os
import numpy as np
import nibabel
import re
import nilearn
from PIL import Image
IMG_DTYPE = np.float
SEG_DTYPE = np.uint8
from skimage.transform import resize
import pre_pro as pp
from config import default_configs
cf = default_configs()

image_rows = cf.img_rows
image_cols = cf.img_cols
project_path = cf.project_path
data_path = project_path + 'Dataset/data/'
seg_path = project_path + 'Dataset/seg/'

def purge(dir, pattern):
    for f in os.listdir(dir):
        if re.search(pattern, f):
            os.remove(os.path.join(dir, f))

def remove_post_int():
    purge(os.path.join(data_path,'train'), "int")
    purge(os.path.join(data_path,'test'), "int")

def shuffle_in_unison(a, b):
     n_elem = a.shape[0]
     indeces = np.random.choice(n_elem, size=n_elem, replace=False)
     return a[indeces], b[indeces]
    
def create_train_data():
    print('-'*30)
    print('Creating training data...')
    print('-'*30)
    # train_data_path = os.path.join(data_path, 'train')
    images = os.listdir(data_path)
    masks = os.listdir(seg_path)

    images = [x for x in images if "int" not in x] #Exlude Intervension Data
    masks = [x for x in masks if "int" not in x]
    images.sort()
    masks.sort()
    gslice_ind = []
    gslice_count = 0
    spr = []  #Slices per record
    pre_blanks_ind = []
    non_blanks_ind = []
    post_blanks_ind = []
    imgs_train = []
    record_dat = []
    record_seg = []
    fsr = np.ndarray((len(images)), dtype = 'uint16')
    fsr[0] = 0
    # training masks (corresponding to the liver)
    masks_train = []    
    # file names corresponding to training images
    # training_images = images[::2]
    # training_images = [x for x in images if "dat" in x]
    # split = int(len(images)//(0.875)) #To split train val before training. 
    split = len(images) #Putting everything into training
    print("Split value : "+str(split))
    training_images = images[:split]
    # file names corresponding to training masks
    # training_masks = images[1::2]
    # training_masks = [x for x in images if "seg" in x]
    training_masks = masks[:split]
    print("Train size")
    print(len(training_images))
    print(len(training_masks))
    blank_ctr = []
    for ind, (liver, orig) in enumerate(zip(training_masks, training_images)):   
        blank_flag = 0
        pre_blanks = 0
        post_blanks = 0
        non_blank_count = 0
        print(liver, orig)
        training_mask = nibabel.load(os.path.join(seg_path, liver))
        training_image = nibabel.load(os.path.join(data_path, orig))
        record_dat.append(orig)
        record_seg.append(liver)
        blk_cnt = 0
        spr.append(training_mask.shape[2])
        if ind == 0:
          fsr[ind] = 0
        else:
          fsr[ind] = fsr[ind-1] + spr[ind-1]
        for k in range(training_mask.shape[2]):
            mask_2d = np.array(training_mask.get_data()[::, ::, k])
            mask_2d = resize(mask_2d, (image_rows, image_cols), order = 0, anti_aliasing=False, preserve_range=True)
            image_2d = np.array(training_image.get_data()[::, ::, k])
            image_2d = resize(image_2d, (image_rows, image_cols), order = 3, preserve_range=True)
            if len(np.unique(mask_2d)) == 1:
                if(blank_flag == 0):
                    pre_blanks += 1
                else:
                    post_blanks += 1
                blk_cnt += 1  
            else:
                blank_flag = 1
                non_blank_count += 1       
            gslice_count += 1            
            #Rotate and flip for better view (Liver on top left corner)
            image_2d = np.rot90(image_2d)
            #image_2d = np.fliplr(image_2d)
            mask_2d = np.rot90(mask_2d)
            #mask_2d = np.fliplr(mask_2d)         
            masks_train.append(mask_2d)
            imgs_train.append(image_2d)
            gslice_ind.append(gslice_count)
        print('Record : '+ str(ind + 1) + '/' + str(len(training_images)))
        print('No. of slices : '+ str(spr[-1]))
        print('Blank slices : '+ str(blk_cnt))
        print('Pre Blanks :' + str(pre_blanks))
        print('Non Blanks : ' + str(non_blank_count))
        print('Post Blanks : ' + str(post_blanks))
        blank_ctr.append(blk_cnt)
        pre_blanks_ind.append(pre_blanks)
        post_blanks_ind.append(post_blanks)
        non_blanks_ind.append(non_blank_count)
        print('-'*10)
        #break
    imgs = np.ndarray(
            (len(imgs_train), image_rows, image_cols), dtype=np.float32)
    imgs_mask = np.ndarray(
            (len(masks_train), image_rows, image_cols), dtype=np.uint8)
    for index, img in enumerate(imgs_train):
        imgs[index, :, :] = img
        
    for index, img in enumerate(masks_train):
        imgs_mask[index, :, :] = img
    np.save('inter_files/imgs_train.npy', imgs)
    np.save('inter_files/masks_train.npy', imgs_mask)
    np.save('eda_files/gslice_ind.npy', gslice_ind)
    np.save('eda_files/spr.npy', spr)
    np.save('eda_files/pre_blanks_ind.npy', pre_blanks_ind)
    np.save('eda_files/post_blanks_ind.npy', post_blanks_ind)
    np.save('eda_files/non_blanks_ind.npy', non_blanks_ind)
    np.save('eda_files/record_dat.npy', record_dat)
    np.save('eda_files/record_seg.npy', record_seg)
    np.save('eda_files/fsr.npy', fsr)
    print('Saving to .npy files done.')
    return imgs, imgs_mask, gslice_ind

def load_train_data():
    imgs_train = np.load('inter_files/imgs_train.npy')
    masks_train = np.load('inter_files/masks_train.npy')
    gslice_ind = np.load('eda_files/gslice_ind.npy')
    return imgs_train, masks_train, gslice_ind

def load_val_data():
    imgs_val = np.load('imgs_val.npy')
    masks_val = np.load('masks_val.npy')
    return imgs_val, masks_val

def load_pp_train1():
    imgs_train = np.load('inter_files/pp_imgs_train1.npy')
    masks_train = np.load('inter_files/pp_masks_train1.npy')
    masks_train = masks_train.astype('float32')
    gslice_ind = np.load('eda_files/pp_gslice_ind1.npy')
    return imgs_train, masks_train, gslice_ind

def load_pp_val1():
    imgs_val = np.load('pp_imgs_val1.npy')
    masks_val = np.load('pp_imgs_val1.npy')
    return imgs_val, masks_val

def save_train2_data(imgs_train2, masks_train2, gslice_ind2):
  np.save('inter_files/imgs_train2.npy', imgs_train2)
  np.save('inter_files/masks_train2.npy', masks_train2)
  np.save('inter_files/gslice_ind2.npy', gslice_ind2)

def load_train2_data():
  imgs_train2 = np.load('inter_files/imgs_train2.npy')
  masks_train2 = np.load('inter_files/masks_train2.npy')
  gslice_ind2 = np.load('inter_files/gslice_ind2.npy')
  return imgs_train2, masks_train2, gslice_ind2

def get_class_weights(masks_train2):
  uniq_v , uniq_c  = np.unique(masks_train2, return_counts=True)
  weight_for_0 = (1 / uniq_c[0])*(np.sum(uniq_c))/2.0 
  weight_for_1 = (1 / uniq_c[1])*(np.sum(uniq_c))/2.0
  class_weight = {0: weight_for_0, 1: weight_for_1}
  # print('Weight for class 0: {:.2f}'.format(weight_for_0))
  # print('Weight for class 1: {:.2f}'.format(weight_for_1))
  return class_weight

def pad_reflect(img, img_rows, img_cols, x_offset, y_offset):
  result = np.zeros((img_rows, img_cols))
  # result = np.pad((img,((x_offset-int(img.shape[0]//2),y_offset-int(img.shape[1])//2)),(x_offset-int(img.shape[0]//2),y_offset-int(img.shape[1])//2)),mode='reflect')
  result   = np.pad(img,((92,92),(92,92)),mode='reflect')
  return result

def get_sorted_index(masks_train2):
  les_pix = []
  for k in range(len(masks_train2)):
    les_pix.append(np.count_nonzero(masks_train2[k] == 1))
  # print(k, np.count_nonzero(masks_train2[k] == 1))
  les_pix = np.asarray(les_pix)
  sort_ind = np.argsort(les_pix)
  return sort_ind

def get_topk_data(k, imgs_train2, masks_train2):
  img_rows = 256
  img_cols = 256
  imgs_train2_t = np.ndarray((k, img_rows, img_cols, 1), dtype = np.float32)
  masks_train2_t = np.ndarray((k, img_rows, img_cols, 1), dtype = np.uint8)
  # imgs_train2_t = []
  # masks_train2_t = []
  sort_ind = get_sorted_index(masks_train2)
  for i in range(k):
    imgs_train2_t[i] = imgs_train2[sort_ind[i]]
    masks_train2_t[i] = masks_train2[sort_ind[i]]
    # imgs_train2_t.append(imgs_train2[-i])
    # masks_train2_t.append(masks_train2[-i])
  return imgs_train2_t, masks_train2_t

def to_scale(img):    
    
    target_shape = [512 , 512 , 40]
#    print(img.affine)
    target_affine = get_target_affine(img)
#    print(target_affine)
    
    voxel_dims = [1 , 1 , 2.5]
    resampled_img = nilearn.image.resample_img(img, target_affine = target_affine,target_shape=target_shape, interpolation = 'continuous')
    resampled_img.header.set_zooms((np.absolute(voxel_dims)))
    return resampled_img  

def pad_copy(img, img_rows, img_cols):
    # nh = img_cols//img.shape[0]
    # nv = img_rows//img.shape[1]
    # ncopies = img_rows*img_cols//img.shape[0]*img.shape[1]
    result = np.zeros((img_rows, img_cols))
    y_offset = 0
    while(y_offset + img.shape[1] < img_rows):#for x in range(nh-1):#(int(math.sqrt(ncopies))):
        x_offset = 0
        while(x_offset + img.shape[0] < img_cols):#for y in range(nv-1):# (int(math.sqrt(ncopies))):
            print(x_offset, y_offset)
            result[x_offset:img.shape[0]+x_offset,y_offset:img.shape[1]+y_offset] = img
            x_offset+=img.shape[0]
        y_offset+=img.shape[1]
    return result

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector

def get_target_affine(img):
    target_affine = img.affine.copy()
    
    # Calculate the translation part of the affine
    spatial_dimensions = (img.header['dim'] * img.header['pixdim'])[1:4]
#    print(img.header['dim'])
#    print(img.header['pixdim'])
#    print(img.affine)
#    print(spatial_dimensions)
  
    # Calculate the translation affine as a proportion of the real world
    # spatial dimensions
    image_center_as_prop = img.affine[0:3,3] / spatial_dimensions
#    print(image_center_as_prop)
    # Calculate the equivalent center coordinates in the target image
    voxel_dims=[1, 1, 2.5]
    target_shape = [500,500,40]
    dimensions_of_target_image = (np.array(voxel_dims) * np.array(target_shape))
#    print(dimensions_of_target_image)
    target_center_coords =  dimensions_of_target_image * image_center_as_prop 
#    print(target_center_coords)
#    print("Affine before rescale")
#    print(target_affine)
    target_affine = rescale_affine(target_affine,voxel_dims,target_center_coords)
    return target_affine

def rescale_affine(input_affine, voxel_dims=[1, 1, 2.5], target_center_coords= None):
    
#    print("input affine")
#    print(input_affine)
    """
    This function uses a generic approach to rescaling an affine to arbitrary
    voxel dimensions. It allows for affines with off-diagonal elements by
    decomposing the affine matrix into u,s,v (or rather the numpy equivalents)
    and applying the scaling to the scaling matrix (s).

    Parameters
    ----------
    input_affine : np.array of shape 4,4
        Result of nibabel.nifti1.Nifti1Image.affine
    voxel_dims : list
        Length in mm for x,y, and z dimensions of each voxel.
    target_center_coords: list of float
        3 numbers to specify the translation part of the affine if not using the same as the input_affine.

    Returns
    -------
    target_affine : 4x4matrix
        The resampled image.
    """
    # Initialize target_affine
    target_affine = input_affine.copy()
    # Decompose the image affine to allow scaling
    u,s,v = np.linalg.svd(target_affine[:3,:3],full_matrices=False)
    
#    print("u,s,v")
#    print(u)
#    print(s)
#    print(v)
    # Rescale the image to the appropriate voxel dimensions
#    s = voxel_dims
#    print(s)
    # Reconstruct the affine
    target_affine[:3,:3] = u @ np.diag(s) @ v
#    print("Target Affine")
#    print(target_affine)
    # Set the translation component of the affine computed from the input
    # image affine if coordinates are specified by the user.
    if target_center_coords is not None:
        target_affine[:3,3] = target_center_coords
#        print("Target Affine")
#        print(target_affine)
    return target_affine
