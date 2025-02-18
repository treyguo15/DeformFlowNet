# ------------------------------------------------------------------------
# CoTr
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from CoTr_package.CoTr.network_architecture import CNNBackbone_v2
from CoTr_package.CoTr.network_architecture.neural_network import SegmentationNetwork
from CoTr_package.CoTr.network_architecture.DeTrans.DeformableTrans import DeformableTransformer
from CoTr_package.CoTr.network_architecture.DeTrans.position_encoding import build_position_encoding
from optical_flow_numpy import optical_flow_swap
# import CNNBackbone
# from# neural_network import SegmentationNetwork
# from DeTrans.DeformableTrans import DeformableTransformer
# from DeTrans.position_encoding import build_position_encoding
class Conv3d_wd(nn.Conv3d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1), groups=1, bias=False):
        super(Conv3d_wd, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True)
        weight = weight - weight_mean
        # std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1, 1) + 1e-5
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(-1, 1, 1, 1, 1)
        weight = weight / std.expand_as(weight)
        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3x3(in_planes, out_planes, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), groups=1, bias=False, weight_std=False):
    "3x3x3 convolution with padding"
    if weight_std:
        return Conv3d_wd(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)


def Norm_layer(norm_cfg, inplanes):

    if norm_cfg == 'BN':
        out = nn.BatchNorm3d(inplanes)
    elif norm_cfg == 'SyncBN':
        out = nn.SyncBatchNorm(inplanes)
    elif norm_cfg == 'GN':
        out = nn.GroupNorm(16, inplanes)
    elif norm_cfg == 'IN':
        out = nn.InstanceNorm3d(inplanes,affine=True)

    return out


def Activation_layer(activation_cfg, inplace=True):

    if activation_cfg == 'ReLU':
        out = nn.ReLU(inplace=inplace)
    elif activation_cfg == 'LeakyReLU':
        out = nn.LeakyReLU(negative_slope=1e-2, inplace=inplace)

    return out


class Conv3dBlock(nn.Module):
    def __init__(self,in_channels,out_channels,norm_cfg,activation_cfg,kernel_size,stride=(1, 1, 1),padding=(0, 0, 0),dilation=(1, 1, 1),bias=False,weight_std=False):
        super(Conv3dBlock,self).__init__()
        self.conv = conv3x3x3(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, weight_std=weight_std)
        self.norm = Norm_layer(norm_cfg, out_channels)
        self.nonlin = Activation_layer(activation_cfg, inplace=True)
    def forward(self,x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.nonlin(x)
        return x

class ResBlock(nn.Module):

    def __init__(self, inplanes, planes, norm_cfg, activation_cfg, weight_std=False):
        super(ResBlock, self).__init__()
        self.resconv1 = Conv3dBlock(inplanes, planes, norm_cfg, activation_cfg, kernel_size=3, stride=1, padding=1, bias=False, weight_std=weight_std)
        self.resconv2 = Conv3dBlock(planes, planes, norm_cfg, activation_cfg, kernel_size=3, stride=1, padding=1, bias=False, weight_std=weight_std)

    def forward(self, x):
        residual = x

        out = self.resconv1(x)
        out = self.resconv2(out)
        out = out + residual

        return out

class U_ResTran3D(nn.Module):
    def __init__(self, norm_cfg='BN', activation_cfg='ReLU', img_size=None, num_classes=None, weight_std=False):
        super(U_ResTran3D, self).__init__()

        self.MODEL_NUM_CLASSES = num_classes

        self.upsamplex2 = nn.Upsample(scale_factor=(1,2,2), mode='trilinear')
        self.transposeconv_stage3 = nn.ConvTranspose3d(768, 384, kernel_size=(2,2,2), stride=(2,2,2), bias=False)
        self.transposeconv_stage2 = nn.ConvTranspose3d(384, 384, kernel_size=(2,2,2), stride=(2,2,2), bias=False)
        self.transposeconv_stage1 = nn.ConvTranspose3d(192, 96, kernel_size=(2,2,2), stride=(2,2,2), bias=False)
        self.transposeconv_stage0 = nn.ConvTranspose3d(96, 48, kernel_size=(2,2,2), stride=(2,2,2), bias=False)

        self.stage2_de = ResBlock(384, 384, norm_cfg, activation_cfg, weight_std=weight_std)
        self.stage1_de = ResBlock(96, 96, norm_cfg, activation_cfg, weight_std=weight_std)
        self.stage0_de = ResBlock(48, 48, norm_cfg, activation_cfg, weight_std=weight_std)

        self.ds2_cls_conv = nn.Conv3d(384, self.MODEL_NUM_CLASSES, kernel_size=1)
        self.ds1_cls_conv = nn.Conv3d(192, self.MODEL_NUM_CLASSES, kernel_size=1)
        self.ds0_cls_conv = nn.Conv3d(48, self.MODEL_NUM_CLASSES, kernel_size=1)

        self.cls_conv = nn.Conv3d(48, self.MODEL_NUM_CLASSES, kernel_size=1)
        
        # Final 1*1 Conv Segmentation map
        self.one_conv = nn.Conv3d(8, 1, kernel_size=1, stride=1, padding=0, bias=True)
        
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, (nn.Conv3d, Conv3d_wd, nn.ConvTranspose3d)):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm3d, nn.SyncBatchNorm, nn.InstanceNorm3d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.backbone = CNNBackbone_v2.Backbone(depth=9, norm_cfg=norm_cfg, activation_cfg=activation_cfg, weight_std=weight_std)
        total = sum([param.nelement() for param in self.backbone.parameters()])
        print('  + Number of Backbone Params: %.2f(e6)' % (total / 1e6))

        self.position_embed = build_position_encoding(mode='v2', hidden_dim=192)
        self.position_embed1 = build_position_encoding(mode='v2', hidden_dim=192)
        self.encoder_Detrans = DeformableTransformer(d_model=192, dim_feedforward=1536, dropout=0.1, activation='gelu', num_feature_levels=1, nhead=6, num_encoder_layers=6, enc_n_points=4)
        self.encoder_Detrans1 = DeformableTransformer(d_model=192, dim_feedforward=1536, dropout=0.1, activation='gelu', num_feature_levels=1, nhead=6, num_encoder_layers=6, enc_n_points=4)
        total = sum([param.nelement() for param in self.encoder_Detrans.parameters()])
        print('  + Number of Transformer Params: %.2f(e6)' % (total / 1e6))

    def posi_mask(self, x):

        x_fea = []
        x_posemb = []
        masks = []
        for lvl, fea in enumerate(x):
            # if lvl > 1:
            #     x_fea.append(fea)
            #     x_posemb.append(self.position_embed(fea))
            #     masks.append(torch.zeros((fea.shape[0], fea.shape[2], fea.shape[3], fea.shape[4]), dtype=torch.bool).cuda())
            if lvl == 1:
                x_fea.append(fea)
                x_posemb.append(self.position_embed(fea))
                masks.append(torch.zeros((fea.shape[0], fea.shape[2], fea.shape[3], fea.shape[4]), dtype=torch.bool).cuda())
            if lvl == 2:
                x_fea.append(fea)
                x_posemb.append(self.position_embed1(fea))
                masks.append(torch.zeros((fea.shape[0], fea.shape[2], fea.shape[3], fea.shape[4]), dtype=torch.bool).cuda())
            # if lvl == 4:
            #     x_fea.append(fea)
            #     x_posemb.append(self.position_embed1(fea))
            #     masks.append(torch.zeros((fea.shape[0], fea.shape[2], fea.shape[3], fea.shape[4]), dtype=torch.bool).cuda())    
            

        return x_fea, masks, x_posemb


    def forward(self, inputs):        # # %%%%%%%%%%%%% CoTr
        x_convs = self.backbone(inputs)
        #print(x_convs[0].size(),x_convs[1].size(),x_convs[2].size(),x_convs[3].size())
        x_fea, masks, x_posemb = self.posi_mask(x_convs)
        #x_trans = self.encoder_Detrans([x_fea[0]], [masks[0]], [x_posemb[0]])
        #print('2',x_fea[-1].size(), masks[-1].size(), x_posemb[-1].size())
        x_trans = self.encoder_Detrans1([x_fea[-1]], [masks[-1]], [x_posemb[-1]])
        #print(x_trans.size())
        # print(masks[1].shape)

        x= self.transposeconv_stage1(x_trans.transpose(-1, -2).view(x_convs[-1].shape))#+ x_convs[-1]
     #   print(x.size())
        #skip2 = self.encoder_Detrans([x_fea[-2]], [masks[-2]], [x_posemb[-2]]).transpose(-1, -2).view(x_convs[-2].shape)
        skip2=x_convs[-2]
        # # Single_scale
        # # x = self.transposeconv_stage2(x_trans.transpose(-1, -2).view(x_convs[-1].shape))
        # # skip2 = x_convs[-2]
        # Multi-scale   
        # x = self.transposeconv_stage2(x_trans[:, x_fea[0].shape[-3]*x_fea[0].shape[-2]*x_fea[0].shape[-1]::].transpose(-1, -2).view(x_convs[-1].shape)) # x_trans length: 12*24*24+6*12*12=7776
       
        # skip2 = x_trans[:, 0:x_fea[0].shape[-3]*x_fea[0].shape[-2]*x_fea[0].shape[-1]].transpose(-1, -2).view(x_convs[-2].shape)
        #print( 'skip2',skip2.size())
        x = x + skip2
        x = self.stage1_de(x)
        # #print( 'x',x.size())
        #ds2 = self.ds2_cls_conv(x)
        # #print( 'ds2',ds2.size())
        x = self.transposeconv_stage0(x)
        #print( 'x',x.size())
        skip1 = x_convs[-3]
        # print( 'skip1',skip1.size())
        # print( 'skip1',x.size())
        x = x + skip1
        #print( 'x',x.size())
        x = self.stage0_de(x)
        #ds1 = self.ds1_cls_conv(x)

        # x = self.transposeconv_stage0(x)
        # skip0 = x_convs[-4]
        # x = x + skip0
        # x = self.stage0_de(x)
        # #ds0 = self.ds0_cls_conv(x)

        # x = self.transposeconv_stage0(x)
        # skip = x_convs[-5]
        # # print( 'x',x.size())
        # # print( 'skip0',skip0.size())
        # x = x + skip
        # x = self.stage0_de(x)
        # ds = self.ds0_cls_conv(x)
        
        result = self.upsamplex2(x)
        result = self.cls_conv(result)
        result1=result.transpose(1,2)
        seg = self.sigmoid(result1)

        return seg
class ResTranUnet(SegmentationNetwork):
    """
    ResTran-3D Unet
    """
    def __init__(self, norm_cfg='BN', activation_cfg='ReLU', img_size=None, num_classes=None, weight_std=False, deep_supervision=False):
        super().__init__()
        self.do_ds = False
        self.U_ResTran3D = U_ResTran3D(norm_cfg, activation_cfg, img_size, num_classes, weight_std) # U_ResTran3D

        if weight_std==False:
            self.conv_op = nn.Conv3d
        else:
            self.conv_op = Conv3d_wd
        if norm_cfg=='BN':
            self.norm_op = nn.BatchNorm3d
        if norm_cfg=='SyncBN':
            self.norm_op = nn.SyncBatchNorm
        if norm_cfg=='GN':
            self.norm_op = nn.GroupNorm
        if norm_cfg=='IN':
            self.norm_op = nn.InstanceNorm3d
        self.dropout_op = nn.Dropout3d
        self.num_classes = num_classes
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        
        self.one_conv = nn.Conv3d(8, 1, kernel_size=1, stride=1, padding=0, bias=True)       
        self.sigmoid = nn.Sigmoid()
    def forward(self, x,y):
        x = self.U_ResTran3D(x)
        #print(x.size())
        # if self._deep_supervision and self.do_ds:
        #     return seg_output
        # else:
        #     return seg_output[0]

        output=optical_flow_swap(x.permute(1,0,2,3,4),y.permute(1,0,4,2,3))
        output=output.permute(1,0,2,3,4)
        #print(optical_flow_output.size())
 
        seg = self.sigmoid(self.one_conv(output))[:,0,0,:,:]
        return  x,seg
        # imgs=x.squeeze()
        # seg_out=seg_output[0].sequence()
        # output=optical_flow(imgs,seg_output)

if __name__=='__main__':

    net = ResTranUnet(norm_cfg='BN', activation_cfg='ReLU', num_classes=1,deep_supervision=False).cuda()

    import torch
    x = torch.ones(4, 1,8, 112, 112).cuda() #batchsize, channel,depth, h,w

    print (net.forward(x).size())

    