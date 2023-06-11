# -*- coding: utf-8 -*-
# +
from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
import argparse,os
import pandas as pd
import cv2
import numpy as np
import random
import math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from models.CDCNs import Conv2d_cd, CDCN, CDCNpp

from Load_OULUNPU_train import Spoofing_train, Normaliztion, ToTensor, RandomHorizontalFlip, Cutout, RandomErasing
from Load_CASIA_FASD_valtest import Spoofing_valtest, Normaliztion_valtest, ToTensor_valtest
from Load_Single_Img import getTestImg

import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import copy
import pdb
from utils import AvgrageMeter, accuracy, performances
import torchvision.utils as vutils


# +
# Dataset root
# Dataset root
train_image_dir = '/mnt/hdd.user/datasets/CASIA-FASD/train_release_frame/'
test_image_dir = '/mnt/hdd.user/datasets/CASIA-FASD/test_release_frame/'
val_image_dir = test_image_dir

map_dir = '/mnt/hdd.user/datasets/CASIA-FASD/train_release_depth/'
test_map_dir = '/mnt/hdd.user/datasets/CASIA-FASD/test_release_depth/'
val_map_dir = test_map_dir
# 文件名1_2_11_3_frame140_depth.jpg

train_list = "../CASIA-FASD_videoname_len.txt"
# train_list = "/mnt/hdd.user/datasets/CASIA-FASD/Protocols/Train.txt"
test_list = "../CASIA-FASD_videoname_len_test.txt"
val_list = "../CASIA-FASD_videoname_len_val.txt"
protocol_num = 3

log_file = open(f'cross_on_casia_P_{protocol_num}.txt', 'w')

print(f"cross_on_casia-P{protocol_num}:\n ")
log_file.write(f'cross_on_casia-P{protocol_num}:\n ')
log_file.flush()


# -


cuda_list = ["","cuda:7", "cuda:5", "cuda:4", "cuda:6", "cuda:4", "cuda:1"]
trained_list = [5,6]

# +
device = torch.device("cuda:2")
model = CDCNpp(basic_conv=Conv2d_cd, theta=0.7).to(device)


val_data = Spoofing_valtest(val_list, val_image_dir, val_map_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
dataloader_val = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=1)
test_data = Spoofing_valtest(test_list, test_image_dir, test_map_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
dataloader_test = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)

for test_camera_num in trained_list:
    val_list = f'/mnt/hdd.user/datasets/Oulu-NPU/Protocols/Protocol_{protocol_num}/Dev_{test_camera_num}.txt'
    test_list = f'/mnt/hdd.user/datasets/Oulu-NPU/Protocols/Protocol_{protocol_num}/Test_{test_camera_num}.txt'
#     log_file = open(f'validation_and_test_P_{protocol_num}_{test_camera_num}.txt', 'w')

    file_name = f"CDCNpp_OULU_NPU_P{protocol_num}-{test_camera_num}"

    model_state_dict = torch.load(f"../{file_name}/{file_name}_400.pkl", map_location={cuda_list[test_camera_num]:"cuda:2"})
    model.load_state_dict(model_state_dict)
    model.eval()
    
    print(f"cross_on_casia-P{protocol_num}-{test_camera_num}:\n ")
    log_file.write(f'cross_on_casia-P{protocol_num}-{test_camera_num}:\n ')
    log_file.flush()
    
    
    with torch.no_grad():
        ###########################################
        '''                val             '''
        ###########################################
        # val for threshold

        map_score_list = []

        for i, sample_batched in enumerate(dataloader_val):
            # get the inputs
            inputs, spoof_label = sample_batched['image_x'].to(device), sample_batched['spoofing_label'].to(device)
            val_maps = sample_batched['val_map_x'].to(device)   # binary map from PRNet


            #pdb.set_trace()
            map_score = 0.0
            for frame_t in range(inputs.shape[1]):
                map_x, embedding, x_Block1, x_Block2, x_Block3, x_input =  model(inputs[:,frame_t,:,:,:])
                score_norm = torch.sum(map_x)
                map_score += score_norm
            map_score = map_score/inputs.shape[1]

            map_score_list.append('{} {}\n'.format(map_score, spoof_label[0][0]))

            #pdb.set_trace()
        map_score_val_filename = './' + file_name + 'casia_map_score_val.txt'
        with open(map_score_val_filename, 'w') as file:
            file.writelines(map_score_list)                

        ###########################################
        '''                test             '''
        ##########################################
        # test for ACC

        map_score_list = []

        for i, sample_batched in enumerate(dataloader_test):
            # get the inputs
            inputs, spoof_label = sample_batched['image_x'].to(device), sample_batched['spoofing_label'].to(device)
            test_maps = sample_batched['val_map_x'].to(device)   # binary map from PRNet 


            #pdb.set_trace()
            map_score = 0.0
            for frame_t in range(inputs.shape[1]):
                map_x, embedding, x_Block1, x_Block2, x_Block3, x_input =  model(inputs[:,frame_t,:,:,:])
                score_norm = torch.sum(map_x)
                map_score += score_norm
            map_score = map_score/inputs.shape[1]

            map_score_list.append('{} {}\n'.format(map_score, spoof_label[0][0]))

        map_score_test_filename = './' + file_name + 'casia_map_score_test.txt'
        with open(map_score_test_filename, 'w') as file:
            file.writelines(map_score_list)    

        #############################################################     
        #       performance measurement both val and test
        #############################################################     
        val_threshold, test_threshold, val_ACC, val_ACER, test_ACC, test_APCER, test_BPCER, test_ACER, test_ACER_test_threshold = performances(map_score_val_filename, map_score_test_filename)

        print('%s, Val:  val_threshold= %.4f, val_ACC= %.4f, val_ACER= %.4f' % (file_name, val_threshold, val_ACC, val_ACER))
        log_file.write('\n Val:  val_threshold= %.4f, val_ACC= %.4f, val_ACER= %.4f \n' % ( val_threshold, val_ACC, val_ACER))
        print('%s, Test:  ACC= %.4f, APCER= %.4f, BPCER= %.4f, ACER= %.4f' % (file_name, test_ACC, test_APCER, test_BPCER, test_ACER))
        log_file.write('Test:  ACC= %.4f, APCER= %.4f, BPCER= %.4f, ACER= %.4f \n' % ( test_ACC, test_APCER, test_BPCER, test_ACER))
        log_file.flush()
        
log_file.close()
    
    

# -



