# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os
import torch
import pandas as pd
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pdb
import math
import os 
import imgaug.augmenters as iaa

# face_scale = 1.3  #default for test, for training , can be set from [1.2 to 1.5]

# data augment from 'imgaug' --> Add (value=(-40,40), per_channel=True), GammaContrast (gamma=(0.5,1.5))
# 进行图像增强的库中的成员，定义一个变换序列后直接将图像batch传入即可
seq = iaa.Sequential([
    iaa.Add(value=(-40,40), per_channel=True), # Add color  每个像素随机加上-40到40之间的数字
    iaa.GammaContrast(gamma=(0.5,1.5)) # GammaContrast with a gamma of 0.5 to 1.5
    #  iaa.GammaContrast通过缩放像素值来调整图像亮度，
])

# region =image[x1:x2,y1:y2]
class Normaliztion_valtest(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """
    def __call__(self, sample):
        image_x, val_map_x, spoofing_label = sample['image_x'],sample['val_map_x'] ,sample['spoofing_label']
        new_image_x = (image_x - 127.5)/128     # [-1,1]
        return {'image_x': new_image_x, 'val_map_x': val_map_x , 'spoofing_label': spoofing_label}


class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample):
        image_x, map_x, spoofing_label = sample['image_x'], sample['val_map_x'],sample['spoofing_label']
        
        # swap color axis because
        # numpy image: (batch_size) x H x W x C
        # torch image: (batch_size) x C X H X W
        image_x = image_x[:,:,::-1].transpose((2, 0, 1))
        image_x = np.array(image_x)
        
        map_x = np.array(map_x)
        
                        
        spoofing_label_np = np.array([0],dtype=np.long)
        spoofing_label_np[0] = spoofing_label
        
        
        return {'image_x': torch.from_numpy(image_x.astype(np.float)).float(), 'val_map_x': torch.from_numpy(map_x.astype(np.float)).float(), 'spoofing_label': torch.from_numpy(spoofing_label_np.astype(np.long)).long()}

class Spoofing_train(Dataset):
    # 分别对应train_list, train_image_dir, map_dir
    def __init__(self, info_list, root_dir, map_dir, dat_dir, transform=None):

        self.landmarks_frame = pd.read_csv(info_list, delimiter=',', header=None)
        self.root_dir = root_dir    # /mnt/hdd.user/datasets/Oulu-NPU/Train_files/
        self.map_dir = map_dir      # /mnt/hdd.user/datasets/Oulu-NPU/Train_depth/
        self.transform = transform
        self.dat_dir = dat_dir

    def __len__(self):
        return len(self.landmarks_frame)


def get_single_image_x(image_path, map_path, videoname):
    image_id = 0
    s = "_frame%d" % image_id
    image_name = videoname + s + '.jpg'
    s = "_frame%d_depth" % image_id
    map_name = videoname + s + '.jpg'

    # random scale from [1.2 to 1.5]
    
    image_x = np.zeros((256, 256, 3))
    map_x = np.zeros((32, 32))

    # RGB
    image_path = os.path.join(image_path, image_name)
    image_x_temp = cv2.imread(image_path)

    # gray-map
    map_path = os.path.join(map_path, map_name)
    map_x_temp = cv2.imread(map_path, 0)
    
    if image_x_temp is not None:
        image_x = cv2.resize(image_x_temp, (256, 256))
    if map_x_temp is not None:
        map_x = cv2.resize(map_x_temp, (32, 32))

    return image_x, map_x


def getTestImg(image_path, map_path, videoname, spoofing_label):
    # #print(self.landmarks_frame.iloc[idx, 0])
    # videoname = str(self.landmarks_frame.iloc[idx, 1])  # 返回类似1_1_01_1的视频名称
    # image_path = os.path.join(self.root_dir, videoname) # /mnt/hdd.user/datasets/Oulu-NPU/Train_files/1_1_01_1
    # map_path = os.path.join(self.map_dir, videoname)    # /mnt/hdd.user/datasets/Oulu-NPU/Train_depth/1_1_01_1
      
    image_x, map_x = get_single_image_x(image_path, map_path, videoname)

    # spoofing_label = self.landmarks_frame.iloc[idx, 0]  # 得到+1表示真实图像，-1表示fake，会自动转换为整数
    if spoofing_label == 1:
        spoofing_label = 1            # real
    else:
        spoofing_label = 0
        map_x = np.zeros((32, 32))    # fake


    sample = {'image_x': image_x, 'val_map_x': map_x, 'spoofing_label': spoofing_label}
    transform = transforms.Compose([Normaliztion_valtest(), ToTensor()])
    if transform:
        sample = transform(sample)
    return sample["image_x"], sample["val_map_x"]
# endregion
