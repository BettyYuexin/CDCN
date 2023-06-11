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

device = torch.device("cuda:8")


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
parser.add_argument('--epochs', type=int, default=3200, help='total training epochs')
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
log_file = open(args.log+'/'+ args.log+'_log_P2.txt', 'w')

echo_batches = args.echo_batches

print("CASIA_FASD, P2:\n ")

log_file.write('CASIA_FASD, P2:\n ')
log_file.flush()

# load the network, load the pre-trained model in UCF101?
finetune = args.finetune
if finetune==True:
    print('finetune!\n')
    log_file.write('finetune!\n')
    log_file.flush()

    model = CDCN()
    model = model.to(device)
    # 将模型放在多个GPU上并行
    model = nn.DataParallel(model, device_ids=device, output_device=device[0])
    # Todo 载入训练好/预训练的参数
    model.load_state_dict(torch.load('xxx.pkl'))

    lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)


else:
    print('train from scratch!\n')
    log_file.write('train from scratch!\n')
    log_file.flush()

    model = CDCNpp(basic_conv=Conv2d_cd, theta=0.7)

    model = model.to(device)

    lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    model_state_dict = torch.load("./CDCNpp_CASIA_FASD/CDCNpp_CASIA_FASD_final.pkl", map_location={'cuda:7':'cpu'})
    model.load_state_dict(model_state_dict)


criterion_absolute_loss = nn.MSELoss().to(device)
criterion_contrastive_loss = Contrast_depth_loss().to(device)

ACER_save = 1.0
predict_maps_real = []
predict_maps_fake = []

for epoch in range(1600):
    scheduler.step()
    if epoch % args.step_size == args.step_size - 1:
        lr *= args.gamma
        
for epoch in range(1600, args.epochs):  # loop over the dataset multiple times
    scheduler.step()
    if epoch % args.step_size == args.step_size - 1:
        lr *= args.gamma


    loss_absolute = AvgrageMeter()
    loss_contra =  AvgrageMeter()
    #top5 = utils.AvgrageMeter()


    ###########################################
    '''                train             '''
    ###########################################
    model.train()

    # load random 16-frame clip data every epoch
    train_data = Spoofing_train(train_list, train_image_dir, map_dir, transform=transforms.Compose([RandomErasing(), RandomHorizontalFlip(),  ToTensor(), Cutout(), Normaliztion()]))
    dataloader_train = DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=1)

    for i, sample_batched in enumerate(dataloader_train):
        # get the inputs
        # inputs: 增强后的图像[256,256]
        # map_label: crop后的深度图 [32,32]
        # spoof_label: 1表示是real image, 0表示是fake face
        inputs, map_label, spoof_label = sample_batched['image_x'].to(device), sample_batched['map_x'].to(device), sample_batched['spoofing_label'].to(device)

        optimizer.zero_grad()

        # forward + backward + optimize
        map_x, embedding, x_Block1, x_Block2, x_Block3, x_input =  model(inputs)

        absolute_loss = criterion_absolute_loss(map_x, map_label)
        contrastive_loss = criterion_contrastive_loss(map_x, map_label)

        loss =  absolute_loss + contrastive_loss
        loss.backward()
        optimizer.step()

        n = inputs.size(0)
        loss_absolute.update(absolute_loss.data, n)
        loss_contra.update(contrastive_loss.data, n)

        if i % echo_batches == echo_batches-1:    # print every 50 mini-batches
            print('epoch:%d, mini-batch:%3d, lr=%f, Absolute_Depth_loss= %.4f, Contrastive_Depth_loss= %.4f' % ((epoch), i, lr,  loss_absolute.avg, loss_contra.avg))


    # whole epoch average
    print('epoch:%d, Train:  Absolute_Depth_loss= %.4f, Contrastive_Depth_loss= %.4f\n' % ((epoch), loss_absolute.avg, loss_contra.avg))
    log_file.write('epoch:%d, Train: Absolute_Depth_loss= %.4f, Contrastive_Depth_loss= %.4f \n' % ((epoch), loss_absolute.avg, loss_contra.avg))
    log_file.flush()
            
    if epoch % 100 == 0:
        torch.save(model.state_dict(), args.log+'/'+args.log+'_%d.pkl' % (epoch))
        


print('Finished Training')
log_file.close()
# -
torch.save(model.state_dict(), args.log+'/'+args.log+'_final.pkl')
