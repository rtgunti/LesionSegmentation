from time import time
import numpy as np
from batchgenerators.augmentations.crop_and_pad_augmentations import crop
from batchgenerators.dataloading import MultiThreadedAugmenter, SingleThreadedAugmenter
from batchgenerators.examples.brats2017.config import brats_preprocessed_folder, num_threads_for_brats_example
from batchgenerators.transforms import Compose
from batchgenerators.utilities.data_splitting import get_split_deterministic
from batchgenerators.utilities.file_and_folder_operations import *
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.augmentations.utils import pad_nd_image
from batchgenerators.augmentations.normalizations import zero_mean_unit_variance_normalization
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2, MirrorTransform, Rot90Transform, SpatialTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
import logging
from glob import glob
from datetime import datetime
from sklearn.utils import shuffle

from utility_functions import get_reservoir_sample, get_random_patch


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

mul_thread_flag = cf.mul_thread_flag

data_path = cf.data_path
seg_path = cf.seg_path
preprocessed_folder = cf.preprocessed_folder
project_path = cf.project_path
mul_thread_flag = cf.mul_thread_flag
batch_size = cf.batch_size

get_nslices = cf.nslices
get_batch_size = cf.batch_size

# max_shape = cf.max_shape

img_rows = cf.img_rows
img_cols = cf.img_cols
nslices = cf.nslices
max_shape = cf.max_shape
patch_size = cf.patch_size
train_ind, val_ind = cf.get_train_val_ind()

def get_train_transform(patch_size):

    tr_transforms = []
    degrees = 5
    tr_transforms.append(
        SpatialTransform_2(
            patch_size, [i // 2 for i in patch_size],
            do_elastic_deform=True, deformation_scale=(0, 0.15),
            do_rotation=True,
            angle_x=( - degrees / 360. * 2 * np.pi, degrees / 360. * 2 * np.pi),
            angle_y=( - degrees / 360. * 2 * np.pi, degrees / 360. * 2 * np.pi),
            angle_z=( - degrees / 360. * 2 * np.pi, degrees / 360. * 2 * np.pi),
            do_scale=True, scale=(0.8, 1.2),
            border_mode_data='constant', border_cval_data=0,
            border_mode_seg='constant', border_cval_seg=0,
            order_seg=1, order_data=3,
            random_crop=True,
            p_el_per_sample=0.1, 
            p_rot_per_sample=0.1, 
            p_scale_per_sample=0.1
        )
    )

    # now we mirror along all axes
    tr_transforms.append(MirrorTransform(axes=(0, 1, 2)))

    # brightness transform for 15% of samples
    # tr_transforms.append(BrightnessMultiplicativeTransform((0.7, 1.5), per_channel=False, p_per_sample=0.15))

    # Rotate 90 Degrees
    if(len(np.unique(patch_size)) == 1):
      tr_transforms.append(Rot90Transform())

    # gamma transform. This is a nonlinear transformation of intensity values
    # (https://en.wikipedia.org/wiki/Gamma_correction)
    # tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=True, p_per_sample=0.15))
    # we can also invert the image, apply the transform and then invert back
    # tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=False, p_per_sample=0.15))

    # Gaussian Noise
    # tr_transforms.append(GaussianNoiseTransform(noise_variance=(0, 0.005), p_per_sample=0.15))

    # now we compose these transforms together
    tr_transforms = Compose(tr_transforms)

    # tr_transforms = []
    return tr_transforms

def get_train_val_generators():
    os.chdir(data_path)
    patients = glob("*.npy")
    patients.sort()
    # patients = [i[:-4] for i in patients]
    os.chdir(project_path)
    train = []
    val = []
    for i in train_ind:
      train.append(patients[i])
    for i in val_ind:
      val.append(patients[i])    
    #train, val = get_split_deterministic(patients, fold=0, num_splits=5, random_state=12345)
    print(len(train), (train))
    print(len(val), (val))

    # dataloader = BraTS2017DataLoader3D(train, batch_size, patch_size, 1)
    # batch = next(dataloader)
    # # now we have some DataLoader. Let's go an get some augmentations
    # # first let's collect all shapes, you will see why later
    # shapes = [BraTS2017DataLoader3D.load_patient(i)[0].shape[1:] for i in patients]
    # max_shape = np.max(shapes, 0)
    # print('Max Shape : '+str(max_shape))
    # max_shape = np.max((max_shape, patch_size), 0)
    max_shape = cf.max_shape
    print("Max Shape : " + str(max_shape))
    # max_shape = patch_size


    # we create a new instance of DataLoader. This one will return batches of shape max_shape. Cropping/padding is
    # now done by SpatialTransform. If we do it this way we avoid border artifacts (the entire brain of all cases will
    # be in the batch and SpatialTransform will use zeros which is exactly what we have outside the brain)
    # this is viable here but not viable if you work with different data. If you work for example with CT scans that
    # can be up to 500x500x500 voxels large then you should do this differently. There, instead of using max_shape you
    # should estimate what shape you need to extract so that subsequent SpatialTransform does not introduce border
    # artifacts
    dataloader_train = DataLoader3D(train, batch_size, max_shape, num_threads_for_brats_example, purpose = "tr")
    # during training I like to run a validation from time to time to see where I am standing. This is not a correct
    # validation because just like training this is patch-based but it's good enough. We don't do augmentation for the
    # validation, so patch_size is used as shape target here
    dataloader_validation = DataLoader3D(val, batch_size, patch_size, max(1, num_threads_for_brats_example // 2), purpose = "val")
    tr_transforms = get_train_transform(patch_size)
    # finally we can create multithreaded transforms that we can actually use for training
    # we don't pin memory here because this is pytorch specific.
    if mul_thread_flag:
      tr_gen = MultiThreadedAugmenter(dataloader_train, tr_transforms, 
                                    num_processes=num_threads_for_brats_example, num_cached_per_queue=5,
                                    seeds=None, pin_memory=False, purpose = "tr")
      val_gen = MultiThreadedAugmenter(dataloader_validation, None,
                                     num_processes=max(1, num_threads_for_brats_example//2), num_cached_per_queue=5,
                                     seeds=None, pin_memory=False, purpose = "val")
      tr_gen.restart()
      val_gen.restart() 
    else:                               
      tr_gen = SingleThreadedAugmenter(dataloader_train, tr_transforms)
      val_gen = SingleThreadedAugmenter(dataloader_validation, None)
    
    return tr_gen, val_gen

def get_list_of_patients(preprocessed_data_folder):
    patients = subfiles(data_path, suffix=".npy", join=True)
    # patients = glob("*.nii.npy")
    patients.sort()
    # npy_files = np.load('records_dat.npy')
    # remove npy file extension
    patients = [i[:-4] for i in patients]
    print(patients)
    return patients

class DataLoader3D(DataLoader):
    def __init__(self, data, batch_size, patch_size, num_threads_in_multithreaded, seed_for_shuffle=1234,
                 return_incomplete=False, shuffle=True, infinite=True, purpose = ""):
        """
        data must be a list of patients as returned by get_list_of_patients (and split by get_split_deterministic)
        patch_size is the spatial size the retured batch will have
        """
        super().__init__(data, batch_size, num_threads_in_multithreaded, seed_for_shuffle, return_incomplete, shuffle,
                         infinite)
        self.patch_size = patch_size
        self.num_modalities = 1
        self.indices = list(range(len(data)))
        self.purpose = purpose
        # print('Patch Size : ' + str(patch_size))

    @staticmethod
    def load_patient(patient):
        dat = np.load(data_path + patient, mmap_mode="r")
        # Data saved in the format (b, c, x, y, z)
        if("dat" in patient): 
          seg = np.load(seg_path + patient.replace("dat","seg", 1), mmap_mode="r")# For Thesis and Brats Data
        elif("volume" in patient):
          seg = np.load(seg_path + patient.replace("volume","segmentation", 1), mmap_mode="r") # For LiTS Data
        return dat, seg

    def generate_train_batch(self):
        # DataLoader has its own methods for selecting what patients to use next
        if (self.purpose == "val"):
            logging.debug(str(datetime.now().time()) + " " + self.purpose +" "+ str(self.thread_id) + ' : lets get a batch')
        idx = self.get_indices()
        patients_for_batch = [self._data[i] for i in idx]
        # initialize empty array for data and seg #(b, c, x, y, z)
        data = np.zeros((self.batch_size, self.num_modalities, *self.patch_size), dtype=np.float32)
        # data = np.full((self.batch_size, self.num_modalities, *self.patch_size), -103.54093933105469, dtype=np.float32)
        seg = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.float32)
        patient_names = []
        # iterate over patients_for_batch and include them in the batch
        for i, j in enumerate(patients_for_batch):
            patient_data, patient_seg = self.load_patient(j)
            
            # this will only pad patient_data if its shape is smaller than self.patch_size
            # print('Before Padding : ' + str(patient_data.shape) + str(patient_seg.shape))
            patient_data = pad_nd_image(patient_data, self.patch_size)
            patient_seg = pad_nd_image(patient_seg, self.patch_size)
            # print('After Padding : ' + str(patient_data.shape) + str(patient_seg.shape))

            # now random crop to self.patch_size
            # crop expects the data to be (b, c, x, y, z) but patient_data is (c, x, y, z) so we need to add one
            # dummy dimension in order for it to work (@Todo, could be improved)
            # patient_data = patient_data[None]
            # patient_seg = patient_seg[None]
            
            while(True):
              # print(patient_seg.shape, patch_size)
              patient_data_c, patient_seg_c = crop(patient_data, patient_seg, crop_size = self.patch_size, crop_type="random")
              # break #Break to have no constraints on the crop. otherwise, have balanced patches with and without lesions
              # dat_uval, dat_uval_cnt = np.unique(patient_data_c, return_counts = True)
              seg_uval, seg_uval_cnt = np.unique(patient_seg_c, return_counts = True)
              if(len(seg_uval) == 1):
                seg_ratio = seg_uval[0]
              else:
                seg_ratio = seg_uval_cnt[1]/sum(seg_uval_cnt)
              # dat_ratio = 1. - dat_cnt[0]/sum(dat_cnt)
              # seg_ratio = 1. - seg_cnt[0]/sum(seg_cnt)
              # if(i < batch_size//2):
              #   if(len(np.unique(patient_seg)) == 1 and dat_ratio > 0.4): #For records that do not have lesions, just get a good patch
              #     break
              #   if(len(np.unique(patient_seg_c)) != 1):
              #     break
              # else:
              #   if(dat_ratio > 0.4):
              #     break
              if(len(np.unique(patient_seg)) == 1):
                  logging.debug(str(datetime.now().time()) + " " + self.purpose +" "+ str(self.thread_id) + ' : Initial mask is empty')
                  break
              
              if(i < batch_size//2):     #Wait till random crop contains lesion
                  patient_data_c, patient_seg_c = get_random_patch(patient_data, patient_seg, self.patch_size)
                  if(not np.count_nonzero(patient_seg_c)):
                    continue
                  logging.debug(str(datetime.now().time()) + " " + self.purpose +" "+ str(self.thread_id) + ' : Got pre - patch with lesion!!')
                  break 
              # else:
              #   logging.debug(str(datetime.now().time()) + " " + self.purpose +" "+ str(self.thread_id) + ' : Got pre - patch without lesion!!')


              # if(i < batch_size//2 and seg_ratio > 0.05):     #Wait till random crop contains lesion
              #     logging.debug(str(datetime.now().time()) + " " + self.purpose +" "+ str(self.thread_id) + ' : Got pre- patch with lesion!!')
              #     break 
              # else:
              #   logging.debug(str(datetime.now().time()) + " " + self.purpose +" "+ str(self.thread_id) + ' : Got pre - patch without lesion!!')
              #for class balancing, half batch with lesion and the other half without lesion
              if(i >= batch_size//2):   # Wait till random crop doesn't contain lesion
                  if (seg_ratio):
                    logging.debug(str(datetime.now().time()) + " " + self.purpose +" "+ str(self.thread_id) + ' : Got post - patch WITH lesion!!')
                    continue
                  else:
                    logging.debug(str(datetime.now().time()) + " " + self.purpose +" "+ str(self.thread_id) + ' : Got post - patch without lesion!!')
                    break           
              if (self.purpose == "val"):
                logging.debug(str(datetime.now().time()) + " " + self.purpose +" "+ str(self.thread_id) + ' : Look for another patch!!')
            # print('After Cropping : ' + str(patient_data_c.shape) + str(patch_size))
            data[i] = patient_data_c[0]
            seg[i] = patient_seg_c[0]
            #metadata.append(patient_metadata)
            patient_names.append(j)
            # print(i, data[i].shape, seg[i].shape)
        if (self.purpose == "val"):
            logging.debug(str(datetime.now().time()) + " " + self.purpose +" "+ str(self.thread_id) + ' : batch ready to return')
        return {'data': data, 'seg':seg, 'names':patient_names}

def generator2D_patch(gen):
    while True:
        item = next(gen)
        # st_timer = time()
        # print(item['data'].shape)

        # Change format from (b, c, x, y, z) to (b, z, x, y, c)
        item['data'] = np.swapaxes(item['data'], -1, 1) 
        item['seg'] = np.swapaxes(item['seg'], -1, 1)        
        # Change format from (b, z, x, y, c) to (b*z, x, y, c)
        item['data'] = np.reshape(item['data'], (batch_size*patch_size[2], patch_size[0], patch_size[1], 1))
        item['seg'] = np.reshape(item['seg'], (batch_size*patch_size[2], patch_size[0], patch_size[1], 1))
        if ('Resampled' not in preprocessed_folder):
          item['data'] = zero_mean_unit_variance_normalization(item['data'], per_channel = False)
                 
        # For 2.5D UNet. Input has 5 slices with center slice as single label
        # # Change format from (b, x, y, z, c) to (b, x, y, z*c )
        # item['data'] = np.reshape(item['data'], (batch_size, patch_size[0], patch_size[1], patch_size[2] * 1))
        # item['seg'] = np.reshape(item['seg'], (batch_size, patch_size[0], patch_size[1], patch_size[2] * 1))
        # item['seg'] = item['seg'][:,:,:,2]
        # item['seg'] = item['seg'][...,None]
        # logging.debug('Time for swap and reshape : ' + str(time() - st_timer))

        item['data'], item['seg'] = shuffle(item['data'], item['seg'], random_state=0)
        yield(item['data'], item['seg'])

        # #Prep for detection
        # seg = item['seg']
        # seg_clf = [len(np.unique(seg[i]))-1  for i in range(seg.shape[0])]
        # seg_clf = np.array(seg_clf)
        # seg_clf = seg_clf[...,np.newaxis]
        # item['data'], seg_clf = shuffle(item['data'], seg_clf, random_state=0)
        # yield(item['data'], seg_clf)

def generator3D_patch(gen):
    while True:
        item = next(gen)
        # st_timer = time()
        # print(item['data'].shape)
        # item['data'] = np.swapaxes(item['data'], -1, 1) #Change format from (b, c, x, y, z) to (b, z, x, y, c)
        # item['seg'] = np.swapaxes(item['seg'], -1, 1)
        item['data'] = np.moveaxis(item['data'], 1, -1) #Change format from (b, c, x, y, z) to (b, x, y, z, c)
        item['seg'] = np.moveaxis(item['seg'], 1, -1)
        if ('Resampled' in preprocessed_folder):
          item['data'] = zero_mean_unit_variance_normalization(item['data'], per_channel = False)
        # logging.debug('Time for swap and reshape : ' + str(time() - st_timer))

        # # For segmentation
        # yield(item['data'], item['seg'])

        # Prep for detection
        seg = item['seg']
        # seg_clf = [len(np.unique(i))-1 for i in seg]
        seg_clf = [np.max(i) for i in seg]
        seg_clf = np.array(seg_clf)
        yield(item['data'], seg_clf)   

        # seg_clf = np.zeros((batch_size))
        # for ind in range(batch_size): 
        #   seg_uval, seg_uval_cnt = np.unique(item['seg'][ind], return_counts = True)
        #   if(len(seg_uval) == 1):
        #     seg_clf[ind] = seg_uval[0]
        #   else:
        #     seg_clf[ind] = np.round(seg_uval_cnt[1]/sum(seg_uval_cnt))
        #     # seg_clf[ind] = seg_uval_cnt[1]/sum(seg_uval_cnt)
        # seg_clf = seg_clf[...,np.newaxis]
        # yield(item['data'], seg_clf)


        # seg_clf = np.zeros((batch_size))
        # print(seg_clf)
        # for ind in range(batch_size): 
        #   seg_uval, seg_uval_cnt = np.unique(item['seg'][ind], return_counts = True)
        #   if(len(seg_uval) == 1):
        #     seg_clf[ind] = seg_uval[0]
        #   else:
        #     seg_clf[ind] = np.round(seg_uval_cnt[1]/sum(seg_uval_cnt))
        #     # print(np.round(seg_uval_cnt[0]/sum(seg_uval_cnt)))
        #   print(ind, seg_uval, seg_uval_cnt, seg_clf[ind])
        # seg_clf = seg_clf[...,np.newaxis]
        # yield(item['data'], seg_clf)

        # yield(item['data'], seg_clf)


