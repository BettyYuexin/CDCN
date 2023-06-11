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
test_camera_num = 5
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

train_list = f'/mnt/hdd.user/datasets/Oulu-NPU/Protocols/Protocol_4/Train_{test_camera_num}.txt'
val_list = f'/mnt/hdd.user/datasets/Oulu-NPU/Protocols/Protocol_4/Dev_{test_camera_num}.txt'
test_list = f'/mnt/hdd.user/datasets/Oulu-NPU/Protocols/Protocol_4/Test_{test_camera_num}.txt'
# 文件都会在’train.txt’、'Dev.txt’和’Test.txt’中存放训练、验证、测试建议方法的视频文件列表。这些文件的组织如下：
# -1,1_1_01_2
# …
# 其中+1代表真实人脸，-1代表攻击。
device = torch.device("cuda:4")


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
parser.add_argument('--log', type=str, default=f"CDCNpp_OULU_NPU_P3-{test_camera_num}", help='log and save model name')
parser.add_argument('--finetune', action='store_true', default=False, help='whether finetune other models')

args = parser.parse_args(args=[])
print(args)

epsilon = 1e-2
# -
test_real_img, test_real_depth = getTestImg(val_image_dir, val_map_dir, val_dat_dir, "1_1_21_1", 1)
test_real_img, test_real_depth = test_real_img.to(device), test_real_depth.to(device)
test_fake_img, test_fake_depth = getTestImg(val_image_dir, val_map_dir, val_dat_dir, "1_1_21_2", -1)
test_fake_img, test_fake_depth = test_fake_img.to(device), test_fake_depth.to(device)
# plt.imshow(test_real_img.transpose(0, 2).transpose(0, 1).numpy())
# plt.imshow(test_real_depth, cmap="gray")
# # +1, 1_1_21_1
# -1, 1_1_21_2

# +
# main function

# GPU  & log file  -->   if use DataParallel, please comment this command
#os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % (args.gpu)
# 创建日志
isExists = os.path.exists(args.log)
if not isExists:
    os.makedirs(args.log)
log_file = open(args.log+'/'+ args.log+f'_log_P4_{test_camera_num}.txt', 'w')

echo_batches = args.echo_batches

print(f"Oulu-NPU, P4-{test_camera_num}:\n ")

log_file.write(f'Oulu-NPU, P4-{test_camera_num}:\n ')
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
#     model_state_dict = torch.load("./CDCNpp_OULU_NPU/CDCNpp_OULU_NPU_final.pkl", map_location={'cuda:7':'cpu'})
#     model.load_state_dict(model_state_dict)


criterion_absolute_loss = nn.MSELoss().to(device)
criterion_contrastive_loss = Contrast_depth_loss().to(device)

ACER_save = 1.0
predict_maps_real = []
predict_maps_fake = []

# for epoch in range(800):
#     scheduler.step()
#     if epoch % args.step_size == args.step_size - 1:
#         lr *= args.gamma

for epoch in range(args.epochs):  # loop over the dataset multiple times
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
    train_data = Spoofing_train(train_list, train_image_dir, map_dir, train_dat_dir, transform=transforms.Compose([RandomErasing(), RandomHorizontalFlip(),  ToTensor(), Cutout(), Normaliztion()]))
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
            print('epoch:%d, mini-batch:%3d, lr=%f, Absolute_Depth_loss= %.4f, Contrastive_Depth_loss= %.4f' % (epoch, i, lr,  loss_absolute.avg, loss_contra.avg))


    # whole epoch average
    print('epoch:%d, Train:  Absolute_Depth_loss= %.4f, Contrastive_Depth_loss= %.4f\n' % (epoch, loss_absolute.avg, loss_contra.avg))
    log_file.write('epoch:%d, Train: Absolute_Depth_loss= %.4f, Contrastive_Depth_loss= %.4f \n' % (epoch, loss_absolute.avg, loss_contra.avg))
    log_file.flush()
    if epoch % 50 == 0:
        torch.save(model.state_dict(), args.log+'/'+args.log+'_%d.pkl' % (epoch))
        


print('Finished Training')
log_file.close()
# -


torch.save(model.state_dict(), args.log+'/'+args.log+'_final.pkl')
