# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import print_function

from future import standard_library

standard_library.install_aliases()
from builtins import object
import numpy as np
from config import default_configs
cf = default_configs()
patch_size = cf.patch_size
batch_size = cf.batch_size

class SingleThreadedAugmenter(object):
    """
    Use this for debugging custom transforms. It does not use a background thread and you can therefore easily debug
    into your augmentations. This should not be used for training. If you want a generator that uses (a) background
    process(es), use MultiThreadedAugmenter.
    Args:
        data_loader (generator or DataLoaderBase instance): Your data loader. Must have a .next() function and return
        a dict that complies with our data structure

        transform (Transform instance): Any of our transformations. If you want to use multiple transformations then
        use our Compose transform! Can be None (in that case no transform will be applied)
    """
    def __init__(self, data_loader, transform):
        self.data_loader = data_loader
        self.transform = transform

    def __iter__(self):
        return self


    def __next__(self):
        item = next(self.data_loader)
        if self.transform is not None:
            item = self.transform(**item)
            # # print('Pre Swap : ' + str(item['data'].shape))
            # item['data'] = np.swapaxes(item['data'], -1, 1)
            # # print('Post Swap : ' + str(item['data'].shape))
            # item['seg'] = np.swapaxes(item['seg'], -1, 1)
            # item['data'] = np.reshape(item['data'], (batch_size*patch_size[2], patch_size[0], patch_size[1], 1))
            # item['seg'] = np.reshape(item['seg'], (batch_size*patch_size[2], patch_size[0], patch_size[1], 1))  
            # print(item['data'].shape, item['seg'].shape)          
        # return item['data'], item['seg']
        return item
