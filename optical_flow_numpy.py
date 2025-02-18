#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 20:36:05 2022

@author: bme
"""

import cv2
import numpy as np
import os
import re
from PIL import Image
from sample3dflow import SeqDataset
#from sample3d import SeqDataset
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms
transform_list = [
    transforms.ToTensor()]

data_transforms = transforms.Compose(transform_list)

def cv_warp(input, flow):
    h, w = flow.shape[:2]
    warp_grid_x, warp_grid_y = np.meshgrid(np.linspace(0, w-1, w), np.linspace(0, h-1, h))
    flow_inv = flow + np.stack((warp_grid_x, warp_grid_y), axis=-1)
    flow_inv = flow_inv.astype(np.float32)
    warped = cv2.remap(input, flow_inv, None, cv2.INTER_LINEAR)
    return warped

def optical_flow_swap(inputs,flos): 
        D,B, C, H, W = inputs.size()
        #inputs=inputs*255
        #print(D)
        segout=[]     
        for i in range(int(D/2)):
            x=inputs[i,...]
            flo=flos[i,...]
            #mesh grid 
            xx = torch.arange(0, W).view(1,-1).repeat(H,1)
            yy = torch.arange(0, H).view(-1,1).repeat(1,W)
            xx = xx.view(1,1,H,W).repeat(B,1,1,1)
            yy = yy.view(1,1,H,W).repeat(B,1,1,1)
            grid = torch.cat((xx,yy),1).float()
     
            grid = grid.cuda()
            vgrid = Variable(grid) + flo # B,2,H,W
            #图二的每个像素坐标加上它的光流即为该像素点对应在图一的坐标
            # scale grid to [-1,1] 
            ##2019 code
            vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone()/max(W-1,1)-1.0 
            #取出光流v这个维度，原来范围是0~W-1，再除以W-1，范围是0~1，再乘以2，范围是0~2，再-1，范围是-1~1
            vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone()/max(H-1,1)-1.0 #取出光流u这个维度，同上
    
            vgrid = vgrid.permute(0,2,3,1)#from B,2,H,W -> B,H,W,2，为什么要这么变呢？是因为要配合grid_sample这个函数的使用
            output = nn.functional.grid_sample(x, vgrid,align_corners=True)
            mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
            mask = nn.functional.grid_sample(mask, vgrid,align_corners=True)   
            ##2019 author
            mask[mask<0.9999] = 0
            mask[mask>0] = 1    
            segout.append(output*mask)
        segout.append(inputs[int(D/2),...])    
        for i in range(int(D/2),D-1):
            x=inputs[i+1,...]
            flo=flos[i,...]
            # mesh grid 
            xx = torch.arange(0, W).view(1,-1).repeat(H,1)
            yy = torch.arange(0, H).view(-1,1).repeat(1,W)
            xx = xx.view(1,1,H,W).repeat(B,1,1,1)
            yy = yy.view(1,1,H,W).repeat(B,1,1,1)
            grid = torch.cat((xx,yy),1).float()

            grid = grid.cuda()
            vgrid = Variable(grid) + flo # B,2,H,W
            #图二的每个像素坐标加上它的光流即为该像素点对应在图一的坐标
    
            # scale grid to [-1,1] 
            ##2019 code
            vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone()/max(W-1,1)-1.0 
            #取出光流v这个维度，原来范围是0~W-1，再除以W-1，范围是0~1，再乘以2，范围是0~2，再-1，范围是-1~1
            vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone()/max(H-1,1)-1.0 #取出光流u这个维度，同上
    
            vgrid = vgrid.permute(0,2,3,1)#from B,2,H,W -> B,H,W,2，为什么要这么变呢？是因为要配合grid_sample这个函数的使用
            output = nn.functional.grid_sample(x, vgrid,align_corners=True)
            mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
            mask = nn.functional.grid_sample(mask, vgrid,align_corners=True)
    
             ##2019 author
            mask[mask<0.9999] = 0
            mask[mask>0] = 1
            
            ##2019 code
            # mask = torch.floor(torch.clamp(mask, 0 ,1))
    
            segout.append(output*mask)
        #print(segout)    
        sgeoutlall=torch.stack(segout,dim=0) 
        #sgeoutlall=sgeoutlall/255
        #print(sgeoutlall.size()) 
        #sgeoutlall=np.stack(segout,axis=0) 

        return sgeoutlall

def optical_flow(img):
    flowall=[]   
    if img.ndim==3:
        prelabel_all=[]
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
    
    s = SeqDataset('./path/all_img_path_n8_g1.txt',transform=data_transforms)
    #for i in range(1118) :  
    i=0
    c,d = s.__getitem__(i)
    path=s.imgseqs[i]
    e=path.split('*')[1]
    b=e.replace('image','optical_flow').rsplit('/',1)[0]
    f=e.rsplit('/',1)[1].split('.')[0]
    if not  os.path.exists(b):
        os.makedirs(b)
    c*=255
    a=torch.tensor(optical_flow(c))
    
    # d=torch.tensor(d)
    # inn=d.numpy()
    # a=a.unsqueeze(0)
    # flow=a.permute(1,0,4,2,3)
    # optical_flow_input_label=d.repeat(8,1,1).unsqueeze(1).unsqueeze(1)
    # #optical_flow_input_label=d.permute(0,1,2,3,4)
    # optical_flow_output=optical_flow_swap(optical_flow_input_label,flow)
    # output=optical_flow_output[0,0,0,:,:].numpy()
           
    # pre=optical_flow_swap(c,a)
    
    save_path=os.path.join(b,f+'.npz')
        
        #torch.save(a,save_path)

        