from torch.utils.data import Dataset
import os
import numpy as np
from skimage.transform import resize #skimage.transform.resize(image，output_shape)调整图像的大小以匹配一定的大小。
import torch
import cv2
from PIL import Image
from transforms import *
from torchvision import  transforms
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# data_dir = r'G:\f\imgs'
# label_dir = r'G:\f\labels'

# transform_list = [
#         transforms.ToTensor(),
#         ]
# input_mean = [103.939, 116.779, 123.68]
# input_std = [1]
# normalize = GroupNormalize(input_mean, input_std)

transform_list = [
    transforms.ToTensor()]
data_transforms = transforms.Compose( transform_list )


def default_loader(path):
    Images= cv2.imread(path,0)
#    Images=cv2.cvtColor(Images, cv2.COLOR_BGR2GRAY)
    Images1=cv2.resize(Images,dsize=(224,224)) 
    return Images1
   
# def default_loader(path):
#     Images= Image.open(path)
#     Images1=Images.resize((224,224))
#     return Images1.convert('RGB')
   
def default_loader_test(path):
    Images= Image.open(path)
    Images1=Images.resize((224,224))
    return Images.size,Images1.convert('RGB')


def to_img(x):
    x = 0.5 * (x + 1.)  # 将-1~1转成0-1
    x = x.clamp(0, 1)
    x = x.view(x.shape[0], 3, 128, 128)
    return x


class SeqDataset(Dataset):
    def __init__(self, txt, transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgseqs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            imgseqs.append(line)
        self.num_samples = len(imgseqs)
        self.imgseqs = imgseqs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        current_index = np.random.choice(range(0, self.num_samples))
        imgs_path = self.imgseqs[current_index].split('*')
        current_imgs = []
        current_imgs_path = imgs_path[:len(imgs_path)-1]
        current_label_path = imgs_path[len(imgs_path)-1]
        current_label = self.loader(current_label_path)
        current_label = self.transform(current_label)[0,:,:]
        for frame in current_imgs_path:
            img = self.loader(frame)
            if self.transform is not None:
                img = self.transform(img)[0,:,:]
            current_imgs.append(img)
        #current_label = current_label
        #print(len(self.imgseqs))
        batch_cur_imgs = np.stack(current_imgs, axis=0)
        #atch_cur_labels = np.stack(current_labels, axis=0)
        #print(batch_cur_imgs.shape,current_label.shape)
        return batch_cur_imgs, current_label

    def __len__(self):
        return len(self.imgseqs)

class SeqDataset_flow(Dataset):
    def __init__(self, txt, transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgseqs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            imgseqs.append(line)
        self.num_samples = len(imgseqs)
        self.imgseqs = imgseqs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        current_index = np.random.choice(range(0, self.num_samples))
        imgs_path = self.imgseqs[current_index].split('*')[:-1]
        optical_flow_path=imgs_path[4].replace('image', 'optical_flow').replace('png','npz')
        current_imgs = []
        current_label = []
        current_imgs_path = imgs_path[:int(len(imgs_path)/2)]
        current_label_path = imgs_path[int(len(imgs_path)/2):len(imgs_path)]
        #print(current_label_path)
        innerlabel_path=current_label_path[4]
        innerlabel=self.loader(innerlabel_path)
        innerlabel=self.transform(innerlabel)[0,:,:]
        
        for l in current_label_path:
            label = self.loader(l)
            if self.transform is not None:
                label = self.transform(label)[0,:,:]
            current_label.append(label)

        for frame in current_imgs_path:
            img = self.loader(frame)
            if self.transform is not None:
                img = self.transform(img)[0,:,:]
            current_imgs.append(img)
        #print(len(self.imgseqs))
        batch_cur_imgs =torch.stack(current_imgs,dim=0).unsqueeze(0)#unet 0 
        batch_cur_labels = torch.stack(current_label,dim=0) 
        #print(batch_cur_imgs.size())
        #batch_cur_imgs = np.stack(current_imgs, axis=0)
        optical_flow=torch.load(optical_flow_path)
        return batch_cur_imgs, batch_cur_labels,optical_flow,innerlabel

    def __len__(self):
        return len(self.imgseqs)



class SeqDataset_test(Dataset):
    def __init__(self, txt, transform=None, loader=default_loader_test):
        fh = open(txt, 'r')
        imgseqs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            imgseqs.append(line)
        self.num_samples = len(imgseqs)
        self.imgseqs = imgseqs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        current_index = np.random.choice(range(0, self.num_samples))
        imgs_path = self.imgseqs[current_index].split('*')[:-1]
        current_imgs = []
        
        shapes=[]
        current_imgs_path = imgs_path[:len(imgs_path)-1]
        current_label_path = imgs_path[len(imgs_path)-1]

        n,current_label = self.loader(current_label_path)

        for frame in current_imgs_path:
            s,img = self.loader(frame)
            if self.transform is not None:
                img = self.transform(img)[0,:,:]
            current_imgs.append(img)
        current_label = self.transform(current_label)[0,:,:]
        #print(len(self.imgseqs))
        batch_cur_imgs = np.stack(current_imgs, axis=0)
        #print(batch_cur_imgs.shape)
        shapes.append(s)
        return shapes,batch_cur_imgs, current_label,
        #print(len(imgs_path))
        

    def __len__(self):
        return len(self.imgseqs)

if __name__ == '__main__':
    s = SeqDataset_flow('./path/train_img_path_n3_g1.txt',transform=data_transforms)
    #s = MydataSet()
    i=1
    c,d,e= s.__getitem__(i)
    #f=e[0,...].numpy()
    
    
    # for i in range(s.__len__()):
    #     a,b = s.__getitem__(i)
    #     #exit()
    #     # print(a.shape,b.shape)
    #     # exit()
