# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 18:49:09 2020

@author: rtgun
"""


import numpy as np
from config import default_configs
cf = default_configs()

th_les_pix = cf.th_les_pix
nslices = cf.nslices

def exclude_lesions(masks):
  for ind in range(int(masks.shape[0]/nslices)):
    mask = masks[ind*nslices:(ind+1)*nslices]
    num_lesions = len(np.unique(mask))-2
    ##Correcting the noise point
    if(ind == 30):
          mask[mask == 20] = 14
    # print(x, liver, num_lesions)
    les_pix = np.ndarray((num_lesions))
    for k in range(num_lesions):
    # print(k, np.unique(masks[masks == (k+10)], return_counts=True)[1])
      les_pix[k] = np.unique(mask[mask == (k+10)], return_counts=True)[1]
      if(les_pix[k]<th_les_pix): #To exclude smaller lesions
        mask[mask == (k+10)] = 1
    masks[ind*nslices:(ind+1)*nslices] = mask
    #print(ind, np.unique(mask, return_counts=True)[0][2:], np.unique(mask, return_counts=True)[1][2:])
  return masks