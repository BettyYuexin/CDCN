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
# array
class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, probability = 0.5, sl = 0.01, sh = 0.05, r1 = 0.5, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, sample):
        img, map_x, spoofing_label = sample['image_x'], sample['map_x'],sample['spoofing_label']
        
        if random.uniform(0, 1) < self.probability:
            attempts = np.random.randint(1, 3)
            for attempt in range(attempts):
                area = img.shape[0] * img.shape[1]
           
                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.r1, 1/self.r1)
    
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
    
                if w < img.shape[1] and h < img.shape[0]:
                    x1 = random.randint(0, img.shape[0] - h)
                    y1 = random.randint(0, img.shape[1] - w)

                    img[x1:x1+h, y1:y1+w, 0] = self.mean[0]
                    img[x1:x1+h, y1:y1+w, 1] = self.mean[1]
                    img[x1:x1+h, y1:y1+w, 2] = self.mean[2]
                    
        return {'image_x': img, 'map_x': map_x, 'spoofing_label': spoofing_label}


# Tensor
class Cutout(object):
    def __init__(self, length=50):
        self.length = length

    def __call__(self, sample):
        img, map_x, spoofing_label = sample['image_x'], sample['map_x'],sample['spoofing_label']
        h, w = img.shape[1], img.shape[2]    # Tensor [1][2],  nparray [0][1]
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        length_new = np.random.randint(1, self.length)
        
        y1 = np.clip(y - length_new // 2, 0, h)
        y2 = np.clip(y + length_new // 2, 0, h)
        x1 = np.clip(x - length_new // 2, 0, w)
        x2 = np.clip(x + length_new // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return {'image_x': img, 'map_x': map_x, 'spoofing_label': spoofing_label}


class Normaliztion(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """
    def __call__(self, sample):
        image_x, map_x, spoofing_label = sample['image_x'], sample['map_x'],sample['spoofing_label']
        new_image_x = (image_x - 127.5)/128     # [-1,1]
        new_map_x = map_x/255.0                 # [0,1]
        return {'image_x': new_image_x, 'map_x': new_map_x, 'spoofing_label': spoofing_label}



class RandomHorizontalFlip(object):
    """Horizontally flip the given Image randomly with a probability of 0.5."""
    def __call__(self, sample):
        image_x, map_x, spoofing_label = sample['image_x'], sample['map_x'],sample['spoofing_label']
        
        new_image_x = np.zeros((256, 256, 3))
        new_map_x = np.zeros((32, 32))

        p = random.random()
        if p < 0.5:
            #print('Flip')

            new_image_x = cv2.flip(image_x, 1)
            new_map_x = cv2.flip(map_x, 1)

                
            return {'image_x': new_image_x, 'map_x': new_map_x, 'spoofing_label': spoofing_label}
        else:
            #print('no Flip')
            return {'image_x': image_x, 'map_x': map_x, 'spoofing_label': spoofing_label}



class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample):
        image_x, map_x, spoofing_label = sample['image_x'], sample['map_x'],sample['spoofing_label']
        
        # swap color axis because
        # numpy image: (batch_size) x H x W x C
        # torch image: (batch_size) x C X H X W
        image_x = image_x[:,:,::-1].transpose((2, 0, 1))
        image_x = np.array(image_x)
        
        map_x = np.array(map_x)
        
                        
        spoofing_label_np = np.array([0],dtype=np.long)
        spoofing_label_np[0] = spoofing_label
        
        
        return {'image_x': torch.from_numpy(image_x.astype(np.float)).float(), 'map_x': torch.from_numpy(map_x.astype(np.float)).float(), 'spoofing_label': torch.from_numpy(spoofing_label_np.astype(np.long)).long()}


# ./CASIA-FASD_videoname_len.txt
# /mnt/hdd.user/datasets/CASIA-FASD/train_release_frame/
# /mnt/hdd.user/datasets/CASIA-FASD/train_release_depth/      1_2_11_3_frame140_depth.jpg
class Spoofing_train(Dataset):
    # 分别对应train_list, train_image_dir, map_dir
    def __init__(self, info_list, root_dir, map_dir, transform=None):

        self.landmarks_frame = pd.read_csv(info_list, delimiter = ',', header=None)
        self.root_dir = root_dir    # /mnt/hdd.user/datasets/Oulu-NPU/Train_files/
        self.map_dir = map_dir      # /mnt/hdd.user/datasets/Oulu-NPU/Train_depth/
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)
    
    def __getitem__(self, idx):
        videoname = str(self.landmarks_frame.iloc[idx, 0])  # 返回类似1_1_01_1的视频名称
        image_path = os.path.join(self.root_dir, videoname) 
        map_path = os.path.join(self.map_dir, videoname)    
        
        spoofing_label = self.landmarks_frame.iloc[idx, 2]  # 得到+1表示真实图像，0表示fake，会自动转换为整数

        image_x, map_x = self.get_single_image_x(image_path, map_path, videoname, spoofing_label, self.landmarks_frame.iloc[idx, 1])
        if spoofing_label == 1:
            spoofing_label = 1            # real
        else:
            spoofing_label = 0
            map_x = np.zeros((32, 32))    # fake


        sample = {'image_x': image_x, 'map_x': map_x, 'spoofing_label': spoofing_label}

        if self.transform:
            sample = self.transform(sample)
        return sample
    
    
    def get_single_image_x(self, image_path, map_path, videoname, spoofing_label, frames_total):
        # 不适用于现在的文件结构，原文实现中将一个视频的图像存放在了同一个文件夹下
        # random choose 1 frame
        for temp in range(500):
            image_id = np.random.randint(0, frames_total-1)

            s = "_frame%d" % image_id
            image_name = videoname + s + '.jpg'
            s = "_frame%d_depth" % image_id
            map_name = videoname + s + '.jpg'
            map_path2 = os.path.join(self.map_dir, map_name)
            image_path2 = os.path.join(self.root_dir, image_name)

            if os.path.exists(image_path2):
                image_x_temp2 = cv2.imread(image_path2, 0)
                if image_x_temp2 is not None:
                    if spoofing_label == 0 or os.path.exists(map_path2):
                        map_x_temp2 = cv2.imread(map_path2, 0)
                        if spoofing_label == 0 or map_x_temp2 is not None:
                            break
        
        
        # random scale from [1.2 to 1.5]
        face_scale = np.random.randint(12, 15)
        face_scale = face_scale/10.0
        
        image_x = np.zeros((256, 256, 3))
        map_x = np.zeros((32, 32))

        # RGB
        image_path = os.path.join(self.root_dir, image_name)
        image_x_temp = cv2.imread(image_path)

        # gray-map
        map_path = os.path.join(self.map_dir, map_name)
        map_x_temp = cv2.imread(map_path, 0)
        

        image_x = cv2.resize(image_x_temp, (256, 256))
        image_x_aug = seq.augment_image(image_x) 
        
        if map_x_temp is not None:
            map_x = cv2.resize(map_x_temp, (32, 32))
    
        
        # 返回增强后的图像和裁剪出的脸部深度图/进行了resize
        return image_x_aug, map_x




# endregion
