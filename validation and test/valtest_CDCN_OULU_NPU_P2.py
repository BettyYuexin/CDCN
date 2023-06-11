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
from Load_OULUNPU_valtest import Spoofing_valtest, Normaliztion_valtest, ToTensor_valtest
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
protocol_num = 2
# Dataset root
train_image_dir = '/mnt/hdd.user/datasets/Oulu-NPU/Train_files/'
val_image_dir = '/mnt/hdd.user/datasets/Oulu-NPU/Dev_files/'
test_image_dir = '/mnt/hdd.user/datasets/Oulu-NPU/Test_files/'
# 文件夹内包含原来的视频.avi文件和其中每一帧的图像1_1_01_1_frame92.jpg
# 以及每一帧的ROi region of interest在.txt文件中 文件内容类似0,455,779,638,756
train_dat_dir = '/mnt/hdd.user/datasets/Oulu-NPU/Train_dat/'
val_dat_dir = '/mnt/hdd.user/datasets/Oulu-NPU/Dev_dat/'
test_dat_dir = '/mnt/hdd.user/datasets/Oulu-NPU/Test_dat/'

map_dir = '/mnt/hdd.user/datasets/Oulu-NPU/Train_depth/'
val_map_dir = '/mnt/hdd.user/datasets/Oulu-NPU/Dev_depth/'
test_map_dir = '/mnt/hdd.user/datasets/Oulu-NPU/Test_depth/'
# 文件名1_2_11_3_frame140_depth.jpg

train_list = f'/mnt/hdd.user/datasets/Oulu-NPU/Protocols/Protocol_{protocol_num}/Train.txt'
val_list = f'/mnt/hdd.user/datasets/Oulu-NPU/Protocols/Protocol_{protocol_num}/Dev.txt'
test_list = f'/mnt/hdd.user/datasets/Oulu-NPU/Protocols/Protocol_{protocol_num}/Test.txt'
# 文件都会在’train.txt’、'Dev.txt’和’Test.txt’中存放训练、验证、测试建议方法的视频文件列表。这些文件的组织如下：
# -1,1_1_01_2
# …
# 其中+1代表真实人脸，-1代表攻击。

# + 1,filename_i
# 最后运算的结果相当于对于原本32*32的图像进行了中心之外的8个像素的差分，作为8个channel的数据输出
def contrast_depth_conv(input):
    ''' compute contrast depth in both of (out, label) '''
    '''
        input  32x32
        output 8x32x32
    '''

    # [8, 3, 3]
    kernel_filter_list =[
                        [[1,0,0],[0,-1,0],[0,0,0]], [[0,1,0],[0,-1,0],[0,0,0]], [[0,0,1],[0,-1,0],[0,0,0]],
                        [[0,0,0],[1,-1,0],[0,0,0]], [[0,0,0],[0,-1,1],[0,0,0]],
                        [[0,0,0],[0,-1,0],[1,0,0]], [[0,0,0],[0,-1,0],[0,1,0]], [[0,0,0],[0,-1,0],[0,0,1]]
                        ]
    
    kernel_filter = np.array(kernel_filter_list, np.float32)

    kernel_filter = torch.from_numpy(kernel_filter.astype(np.float)).float().to(device)
    # weights (in_channel, out_channel, kernel, kernel)
    kernel_filter = kernel_filter.unsqueeze(dim=1) # [1, 8, 3, 3]
    # [32, 32]--[1, 32, 32]--[8,32,32](进行完全拷贝的维度扩展)
    input = input.unsqueeze(dim=1).expand(input.shape[0], 8, input.shape[1],input.shape[2])
    # 输出[8,32,32]，group=8相当于每个channel分别和对应的channel的kernel进行卷积运算，得到输出对应的channel
    contrast_depth = F.conv2d(input, weight=kernel_filter, groups=8)  # depthwise conv

    return contrast_depth

# TODO 理解为什么depth loss是这么定义的
class Contrast_depth_loss(nn.Module):    # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(Contrast_depth_loss,self).__init__()
        return
    def forward(self, out, label): 
        '''
        compute contrast depth in both of (out, label),
        then get the loss of them
        tf.atrous_convd match tf-versions: 1.4
        '''
        # 根据out和label分别构造depth_conv，之后计算mseloss
        contrast_out = contrast_depth_conv(out)
        contrast_label = contrast_depth_conv(label)
        
        
        criterion_MSE = nn.MSELoss().to(device)
    
        loss = criterion_MSE(contrast_out, contrast_label)
        #loss = torch.pow(contrast_out - contrast_label, 2)
        #loss = torch.mean(loss)
    
        return loss

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1, help='the gpu id used for predict')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')  
parser.add_argument('--batchsize', type=int, default=7, help='initial batchsize')  
parser.add_argument('--step_size', type=int, default=500, help='how many epochs lr decays once')  # 500 
parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
parser.add_argument('--echo_batches', type=int, default=50, help='how many batches display once')  # 50
parser.add_argument('--epochs', type=int, default=800, help='total training epochs')
parser.add_argument('--log', type=str, default=f"CDCNpp_OULU_NPU_P{protocol_num}", help='log and save model name')
parser.add_argument('--finetune', action='store_true', default=False, help='whether finetune other models')

args = parser.parse_args(args=[])
print(args)

epsilon = 1e-2
# +
# main function

# GPU  & log file  -->   if use DataParallel, please comment this command
#os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % (args.gpu)
# 创建日志
isExists = os.path.exists('../' + args.log)
if not isExists:
    os.makedirs(args.log)
log_file = open('../' + args.log+'/'+ args.log+f'_log_valtest_P{protocol_num}.txt', 'w')

echo_batches = args.echo_batches

print(f"Oulu-NPU-val-P{protocol_num}:\n ")

log_file.write(f'Oulu-NPU-val-P{protocol_num}:\n ')
log_file.flush()


ACER_save = 1.0
predict_maps_real = []
predict_maps_fake = []
 

for i in range(9):
    epoch = i * 100

    if not os.path.exists(f"../{args.log}/{args.log}_{epoch}.pkl"):
        if epoch == 800:
            model_state_dict = torch.load(f"./{args.log}/{args.log}_final.pkl")
            if not os.path.exists(f"../{args.log}/{args.log}_final.pkl"):
                continue
    else:
        model_state_dict = torch.load(f"../{args.log}/{args.log}_{epoch}.pkl")
    device = model_state_dict['conv1.0.conv.weight'].device
    model = CDCNpp(basic_conv=Conv2d_cd, theta=0.7).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()


    with torch.no_grad():
        ###########################################
        '''                val             '''
        ###########################################
        # val for threshold
        val_data = Spoofing_valtest(val_list, val_image_dir, val_map_dir, val_dat_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
        dataloader_val = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=1)

        map_score_list = []
        criterion_MSE = nn.MSELoss().to(device)

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
        map_score_val_filename = '../' + args.log+'/'+ args.log+'_map_score_val_sum.txt'
        with open(map_score_val_filename, 'w') as file:
            file.writelines(map_score_list)                

        ###########################################
        '''                test             '''
        ##########################################
        # test for ACC
        test_data = Spoofing_valtest(test_list, test_image_dir, test_map_dir, test_dat_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
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

        map_score_test_filename = '../' + args.log +'/'+ args.log+'_map_score_test_sum.txt'
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



