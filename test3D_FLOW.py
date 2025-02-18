from evaluation import get_DC
from evaluation import get_JS
from evaluation import get_sensitivity
from evaluation import get_specificity
from evaluation import get_precision
from evaluation import get_accuracy
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
#rom Unet3D_flow import UNet3D
from opts import parser        
def default_loader(path):
        Images= Image.open(path)
        Images1=Images.resize((224,224))
        return Images.size,Images1.convert('RGB')

if __name__ == '__main__':
    transform_list = [transforms.ToTensor()]
    data_transforms = transforms.Compose(transform_list)
    # GPU 
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    i=1
    DICE,SEN,PRE,ACC,SPE,JS=[],[],[],[],[],[]
    NAME=[]
    #module = UNet3D(residual=None).cuda()
    #module = Unet(in_channels=1,n_classes=1,is_deconv=True)
    #module = resunet(in_channels=1,n_classes=1,is_deconv=True)
    module = ResTranUnet(norm_cfg='BN', activation_cfg='ReLU', num_classes=1,).cuda()
    args = parser.parse_args()
    args.snapshot_pref = os.path.join(args.snapshot_dir, args.arch)
    #module = torch.nn.DataParallel(module).cuda()
    checkpoint=torch.load('./checkpoints/ourproposed_of/DeformFlowNet_model_best.pth.tar')
    module.load_state_dict(checkpoint['state_dict'])
      
    test_path='./path/test_3dflow_img_path_n8_g1.txt'
    fh = open(test_path, 'r')
    #imgseqs = []
    for line in fh:
        line = line.strip('\n')
        line = line.rstrip()
        
        imgs_path = line.split('*')[:-1]
        optical_flow_path=imgs_path[4].replace('image', 'optical_flow').replace('png','npz')
        current_imgs = []
        current_imgs_path = imgs_path[:int(len(imgs_path)/2)]
        innerlabel_path=imgs_path[12]
        s,innerlabel=default_loader(innerlabel_path)
        innerlabel=np.copy(innerlabel)[:,:,0]
        test_label = cv2.resize(innerlabel, (s[0], s[1]))
        
        mask_obj = test_label >= 128
        mask_noobj = test_label < 128
        test_label[mask_obj] = 255
        test_label[mask_noobj] = 0
        
        for frame in current_imgs_path:
            n,img =default_loader(frame)
            img1 = data_transforms(img)[0,:,:]
            current_imgs.append(img1)

        batch_cur_imgs =torch.stack(current_imgs,dim=0).cuda()
        #print(batch_cur_imgs.size(),test_label.shape)#.unsqueeze(0)
        optical_flow=torch.load(optical_flow_path).unsqueeze(0).cuda()
        
        #print(optical_flow.size())
        batch_cur_imgs = batch_cur_imgs.unsqueeze(0).unsqueeze(0).cuda()
        #print(batch_cur_imgs.size())
        output,inneroutput= module(batch_cur_imgs,optical_flow)
        #print(output.size(),inneroutput.size())
        
        inneroutput=inneroutput.squeeze().cpu()
        inneroutput =  inneroutput.detach().numpy()
        
        test_img1 = cv2.resize(inneroutput, (s[0], s[1]))
        mask_obj = test_img1 >= 0.5
        mask_noobj = test_img1 < 0.5
        test_img1[mask_obj] = 255
        test_img1[mask_noobj] = 0
        
        dice = get_DC(test_img1, test_label)
        DICE.append(dice)
        
        sensitivity = get_sensitivity(test_img1, test_label)      
        spe = get_specificity(test_img1, test_label) 
        js = get_JS(test_img1, test_label) 
        pre = get_precision(test_img1, test_label)   
        Acc= get_accuracy(test_img1, test_label)
        #pre=get_precision(test_img1, test_labe)
        SEN.append(sensitivity) 
        JS.append(js)
        PRE.append(pre) 
        SPE.append(spe)
        ACC.append(Acc)
        NAME.append(innerlabel_path.split('/')[-1])
        print(js)
      
        #cv2.imwrite(os.path.join('./prediction/cotr_of',innerlabel_path.split('/')[-1]),test_img1)
        #print('save')
    print(np.average(DICE),np.average(SEN),np.average(PRE),np.average(ACC),np.average(JS),np.average(SPE))

