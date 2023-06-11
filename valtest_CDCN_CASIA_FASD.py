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

from Load_CASIA_FASD_train import Spoofing_train, Normaliztion, ToTensor, RandomHorizontalFlip, Cutout, RandomErasing
from Load_CASIA_FASD_valtest import Spoofing_valtest, Normaliztion_valtest, ToTensor_valtest
# from Load_Single_Img import getTestImg

import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import copy
import pdb
from utils import AvgrageMeter, accuracy, performances
import torchvision.utils as vutils


# +
# CASIA-FASD: 
# Protocols      test_release_dat         test_release_frame       train_release_dat         train_release_frame
# Protocols-bak  test_release_depth       test_release_freme_full  train_release_depth       train_release_frame_full
# test_release   test_release_depth_full  train_release            train_release_depth_full

# train_release: 1....20:1.avi...
# train_release_frame: 1_0_11_2_frame182.jpg
# train_release_depth: 1_0_4_1_frame41_depth.jpg
# train_release_dat: 1_0_11_2_frame179.dat
# 已经处理好了ROI
# label path = "/mnt/hdd.user/datasets/CASIA-FASD/Protocols/Train.txt"
# df = pd.read_csv(path, delimiter = ' ', header=None)
# print(df[2])


# +
# Dataset root
train_image_dir = '/mnt/hdd.user/datasets/CASIA-FASD/train_release_frame/'
test_image_dir = '/mnt/hdd.user/datasets/CASIA-FASD/test_release_frame/'
val_image_dir = test_image_dir

map_dir = '/mnt/hdd.user/datasets/CASIA-FASD/train_release_depth/'
test_map_dir = '/mnt/hdd.user/datasets/CASIA-FASD/test_release_depth/'
val_map_dir = test_map_dir
# 文件名1_2_11_3_frame140_depth.jpg

train_list = "./CASIA-FASD_videoname_len.txt"
# train_list = "/mnt/hdd.user/datasets/CASIA-FASD/Protocols/Train.txt"
test_list = "./CASIA-FASD_videoname_len_test.txt"
val_list = "./CASIA-FASD_videoname_len_val.txt"

device = torch.device("cuda:9")

# + 1,filename_i
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1, help='the gpu id used for predict')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')  
parser.add_argument('--batchsize', type=int, default=7, help='initial batchsize')  
parser.add_argument('--step_size', type=int, default=500, help='how many epochs lr decays once')  # 500 
parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
parser.add_argument('--echo_batches', type=int, default=50, help='how many batches display once')  # 50
parser.add_argument('--epochs', type=int, default=800, help='total training epochs')
parser.add_argument('--log', type=str, default="CDCNpp_CASIA_FASD", help='log and save model name')
parser.add_argument('--finetune', action='store_true', default=False, help='whether finetune other models')

args = parser.parse_args(args=[])
print(args)

epsilon = 1e-2
# +
# main function

# GPU  & log file  -->   if use DataParallel, please comment this command
#os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % (args.gpu)
# 创建日志
isExists = os.path.exists(args.log)
if not isExists:
    os.makedirs(args.log)
log_file = open(args.log+'/'+ args.log+'_log_valtest.txt', 'w')

echo_batches = args.echo_batches

print("CASIA_FASD-val:\n ")

log_file.write('CASIA_FASD-val:\n ')
log_file.flush()


ACER_save = 1.0
predict_maps_real = []
predict_maps_fake = []
 

for i in range(10):
    epoch = i * 100
    model = CDCNpp(basic_conv=Conv2d_cd, theta=0.7).to(device)
    model_state_dict = torch.load(f"./CDCNpp_CASIA_FASD/CDCNpp_CASIA_FASD_{epoch}.pkl", map_location={'cuda:8':'cuda:9'})
    model.load_state_dict(model_state_dict)
    model.eval()

    with torch.no_grad():
        ###########################################
        '''                val             '''
        ###########################################
        # val for threshold
        val_data = Spoofing_valtest(val_list, val_image_dir, val_map_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
        dataloader_val = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=1)

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
        map_score_val_filename = args.log+'/'+ args.log+'_map_score_val.txt'
        with open(map_score_val_filename, 'w') as file:
            file.writelines(map_score_list)                

        ###########################################
        '''                test             '''
        ##########################################
        # test for ACC
        test_data = Spoofing_valtest(test_list, test_image_dir, test_map_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
        dataloader_test = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)

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

        map_score_test_filename = args.log+'/'+ args.log+'_map_score_test.txt'
        with open(map_score_test_filename, 'w') as file:
            file.writelines(map_score_list)    

        #############################################################     
        #       performance measurement both val and test
        #############################################################     
        val_threshold, test_threshold, val_ACC, val_ACER, test_ACC, test_APCER, test_BPCER, test_ACER, test_ACER_test_threshold = performances(map_score_val_filename, map_score_test_filename)

        print('epoch:%d, Val:  val_threshold= %.4f, val_ACC= %.4f, val_ACER= %.4f' % (epoch, val_threshold, val_ACC, val_ACER))
        log_file.write('\n epoch:%d, Val:  val_threshold= %.4f, val_ACC= %.4f, val_ACER= %.4f \n' % (epoch + 1, val_threshold, val_ACC, val_ACER))
        print('epoch:%d, Test:  ACC= %.4f, APCER= %.4f, BPCER= %.4f, ACER= %.4f' % (epoch, test_ACC, test_APCER, test_BPCER, test_ACER))
        log_file.write('epoch:%d, Test:  ACC= %.4f, APCER= %.4f, BPCER= %.4f, ACER= %.4f \n' % (epoch + 1, test_ACC, test_APCER, test_BPCER, test_ACER))
        log_file.flush()

#         predict_real_map = model(test_real_img.unsqueeze(0))
#         predict_fake_map = model(test_fake_img.unsqueeze(0))
#         predict_maps_real.append(predict_real_map[0][0].detach())
#         predict_maps_fake.append(predict_fake_map[0][0].detach())
        

log_file.close()
# -

