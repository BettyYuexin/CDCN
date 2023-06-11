# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os
import torch
import pandas as pd
#from skimage import io, transform
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pdb
import math
import os 


frames_total = 8    # each video 8 uniform samples

face_scale = 1.3  #default for test and val 
#face_scale = 1.1  #default for test and val

# region

class Normaliztion_valtest(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """
    def __call__(self, sample):
        image_x, val_map_x, spoofing_label = sample['image_x'],sample['val_map_x'] ,sample['spoofing_label']
        new_image_x = (image_x - 127.5)/128     # [-1,1]
        return {'image_x': new_image_x, 'val_map_x': val_map_x , 'spoofing_label': spoofing_label}


class ToTensor_valtest(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample):
        image_x, val_map_x, spoofing_label = sample['image_x'],sample['val_map_x'] ,sample['spoofing_label']
        
        # swap color axis because    BGR2RGB
        # numpy image: (batch_size) x T x H x W x C
        # torch image: (batch_size) x T x C X H X W
        image_x = image_x[:,:,:,::-1].transpose((0, 3, 1, 2))
        image_x = np.array(image_x)
        
        val_map_x = np.array(val_map_x)
                        
        spoofing_label_np = np.array([0],dtype=np.long)
        spoofing_label_np[0] = spoofing_label
        
        return {'image_x': torch.from_numpy(image_x.astype(np.float)).float(), 'val_map_x': torch.from_numpy(val_map_x.astype(np.float)).float(),'spoofing_label': torch.from_numpy(spoofing_label_np.astype(np.long)).long()} 


# /home/ztyu/FAS_dataset/OULU/Train_images/          6_3_20_5_121_scene.jpg        6_3_20_5_121_scene.dat
# /home/ztyu/FAS_dataset/OULU/IJCB_re/OULUtrain_images/        6_3_20_5_121_depth1D.jpg
class Spoofing_valtest(Dataset):

    def __init__(self, info_list, root_dir, val_map_dir, transform=None):

        self.landmarks_frame = pd.read_csv(info_list, delimiter=',', header=None)
        self.root_dir = root_dir
        self.val_map_dir = val_map_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        #print(self.landmarks_frame.iloc[idx, 0])
        videoname = str(self.landmarks_frame.iloc[idx, 0])
        image_path = os.path.join(self.root_dir, videoname)
        val_map_path = os.path.join(self.val_map_dir, videoname)
        
        spoofing_label = self.landmarks_frame.iloc[idx, 2]
        image_x, val_map_x = self.get_single_image_x(image_path, val_map_path, videoname, spoofing_label, self.landmarks_frame.iloc[idx, 1])

        if spoofing_label == 1:
            spoofing_label = 1            # real
        else:
            spoofing_label = 0
            
        sample = {'image_x': image_x, 'val_map_x':val_map_x , 'spoofing_label': spoofing_label}

        if self.transform:
            sample = self.transform(sample)
        return sample
        
    def get_single_image_x(self, image_path, val_map_path, videoname, spoofing_label, files_total):
        interval = files_total // 10
        image_x = np.zeros((frames_total, 256, 256, 3))
        val_map_x = np.ones((frames_total, 32, 32))
        
        # random choose 1 frame
        for ii in range(frames_total):
            image_id = ii * interval + 1 
            
            for temp in range(50):
                s = "_frame%d" % image_id
                s1 = "_frame%d_depth" % image_id
                image_name = videoname + s + '.jpg'
                map_name = videoname + s1 + '.jpg'
                image_path2 = os.path.join(self.root_dir, image_name)
                image_x_temp = cv2.imread(image_path2)
                val_map_path2 = os.path.join(self.val_map_dir, map_name)
                val_map_x_temp2 = cv2.imread(val_map_path2, 0)
                
                if os.path.exists(image_path2) and image_x_temp is not None:
                    if spoofing_label == 0:    
                        break
                    if os.path.exists(val_map_path2) and val_map_x_temp2 is not None:
                        break
                    else:
                        image_id += 1
                else:
                    image_id += 1
                    
                    
            # RGB
            image_x_temp = cv2.imread(image_path2)
            
            # gray-map
            val_map_x_temp = cv2.imread(val_map_path2, 0)
            # if image_x_temp is None:
            #     print(f"Warning: image None!{ii} {temp}")
            #     print(image_path2)
            #     print(files_total)
            #     print(interval)
            image_x[ii,:,:,:] = cv2.resize(image_x_temp, (256, 256))
            # transform to binary mask --> threshold = 0
            if val_map_x_temp is not None:
                temp = cv2.resize(val_map_x_temp, (32, 32))
                # 表示<1时返回本身，>=1的时候返回1
                np.where(temp < 1, temp, 1)
                val_map_x[ii,:,:] = temp
            else:
                val_map_x[ii,:,:] = np.zeros((32, 32))
                
        return image_x, val_map_x
# endregion
