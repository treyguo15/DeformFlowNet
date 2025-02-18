#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 19:35:14 2023

@author: bme
"""

#from Unet3D import UNet3D

import xlwt
from diceloss import dice_coeff
from torchvision import transforms
from torch.utils.data import DataLoader
import os
#from resunet mport resunet
from PIL import Image
import numpy as np
#from Unet import Unet
import torch
import cv2
from sample3dflow import SeqDataset_flow
import csv
from skimage.transform import resize
from DeformFlowNet import  ResTranUnet
#from ResTranUnet_flow import ResTranUnet
#from Unet3D_flow import UNet3D
from opts import parser      
import re

transform_list = [
    transforms.ToTensor()]
data_transforms = transforms.Compose( transform_list )

def default_loader(path):
        Images= Image.open(path)
        Images1=Images.resize((224,224))
        return Images.size,Images1.convert('RGB')


def optical_flow(img):
    flowall=[]   
    if img.ndim==3:
        for i in range(0,int(img.shape[0]/2)):
            img1=img[i,:,:]
            img2=img[int(img.shape[0]/2),:,:]
            flow = cv2.calcOpticalFlowFarneback(img2, img1, None, 0.5, 3, 10, 3, 5, 1.2,0)
            flowall.append(flow)
            
        for i in range(int(img.shape[0]/2)+1,img.shape[0]):
            img2=img[int(img.shape[0]/2),:,:] 
            img3=img[i,:,:]
            flow = cv2.calcOpticalFlowFarneback(img2, img3, None, 0.5, 3, 10, 3, 5, 1.2,0)
            flowall.append(flow)
        flowall=np.stack(flowall,axis=0)
        return flowall        

if __name__ == '__main__':
    transform_list = [transforms.ToTensor()]
    data_transforms = transforms.Compose(transform_list)
    # GPU 
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    i=1
    NAME=[]
    #module = UNet3D(residual='conv').cuda()
    #module = Unet(in_channels=1,n_classes=1,is_deconv=True)
    #module = resunet(in_channels=1,n_classes=1,is_deconv=True)
    module = ResTranUnet(norm_cfg='BN', activation_cfg='ReLU', num_classes=1,).cuda()
    args = parser.parse_args()
    args.snapshot_pref = os.path.join(args.snapshot_dir, args.arch)
    #module = torch.nn.DataParallel(module).cuda()
    checkpoint=torch.load('./checkpoints/ourproposed_of/DeformFlowNet_model_best.pth.tar')
    module.load_state_dict(checkpoint['state_dict'])
      
    
    dir_path=r'..//framing_data/HHD'#HHD_高血压心脏病,HCM_肥厚型心肌病
    imgfile_list = os.listdir(dir_path)
    imgfile_list.sort(key= lambda x:str(x[:]))
    #print(img_list)
    seqsize =8
    gap=1
    for imgfile in imgfile_list[:]:
        filepath = os.path.join(dir_path,imgfile)
        img_list = os.listdir(filepath)
        img_list.sort(key=lambda x: int(re.findall('\d+',x)[0]))
        print(imgfile)
        #滑窗取序列
        for i in range(0, len(img_list)-(seqsize*gap)+1, 1):
            current_imgs = []
            #print(i)
            for j in range(i,i+(seqsize*gap),gap):
                  
                  path = os.path.join(filepath, img_list[j])
                  s,img = default_loader(path)
                  img = data_transforms(img)[0,:,:]
                  current_imgs.append(img)
            batch_cur_imgs = np.stack(current_imgs, axis=0)
            a=batch_cur_imgs
            of=optical_flow(batch_cur_imgs)
            of1=torch.tensor(of).clone().detach().unsqueeze(0).cuda()
            batchimg=torch.tensor(a).clone().detach().unsqueeze(0).unsqueeze(2).cuda()
            output,inneroutput= module(batchimg,of1)
            #print(output.size(),inneroutput.size())
            
            inneroutput=inneroutput.squeeze().cpu()
            inneroutput =  inneroutput.detach().numpy()
            
            test_img1 = cv2.resize(inneroutput, (s[0], s[1]))
            mask_obj = test_img1 >= 0.5
            mask_noobj = test_img1 < 0.5
            test_img1[mask_obj] = 255
            test_img1[mask_noobj] = 0

        
            save_path=os.path.join('./infer/',path.split('/',3)[-1])
           # print(save_path)
            # if not  os.path.exists(save_path.rsplit('/',1)[0]):
            #     os.makedirs(save_path.rsplit('/',1)[0])
           # print(save_path.rsplit('/',1)[0])
            #cv2.imwrite(save_path,test_img1)
        #print('save')


