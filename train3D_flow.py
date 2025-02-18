import torch
from torch import nn
#from Unet3D import UNet3D
#from resunet import resunet
import os
import numpy as np
import shutil
from sample3dflow import SeqDataset_flow
#from evaluation import get_DC
from diceloss import dice_coeff,dice_loss
from torch.utils.data import DataLoader
from torchvision import transforms
#from transforms import *
from torch import optim
#import torchvision
#from torch.autograd import Variable
#from tensorboardX import SummaryWriter
import time
#from utils import *
#from torch.utils import AverageMeter
from opts import parser
#from CSWin_Unet import CSWin_Unet
#from vision_transformer import SwinUnet
#from config import get_config
#from Unet import Unet
from DeformFlowNet import  ResTranUnet
from optical_flow_numpy import optical_flow_swap
#from Unet3D_flow import UNet3D

# input_mean = [103.939, 116.779, 123.68]
# input_std = [1]
# normalize = GroupNormalize(input_mean, input_std)

transform_list = [
    transforms.ToTensor()]

data_transforms = transforms.Compose(transform_list)


best_dice=0
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    global args, best_dice
    args = parser.parse_args()
    args.snapshot_pref = os.path.join(args.snapshot_dir, args.arch)
    #config = get_config(args)
    #module = UNet3D(residual='pool').cuda()
    #module = resunet(in_channels=1,n_classes=1,is_deconv=True).cuda()
    module = ResTranUnet(norm_cfg='BN', activation_cfg='ReLU', num_classes=1,).cuda()
    #module = Unet(in_channels=1,n_classes=1,is_deconv=True).cuda()
    train_dataset=SeqDataset_flow(args.train_list,transform=data_transforms)
    val_dataset=SeqDataset_flow(args.val_list,transform=data_transforms)
    
    train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=2,drop_last=True)
    val_loader = DataLoader(val_dataset,batch_size=args.batch_size,shuffle=True,num_workers=2,drop_last=True)
   
    criticizer=nn.MSELoss().to(device)
    #criticizer=nn.BCELoss().to(device)
    #criticizer=nn.BCEWithLogitsLoss().to(device)    
    #optimizer = optim.Adam(module.parameters(),lr=args.lr)
    optimizer = optim.SGD(module.parameters(), lr=args.lr, momentum=0.99, weight_decay=0.0001)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer, args.lr_steps)
    model = torch.nn.DataParallel(module).to(device)
    # train_writer = SummaryWriter(os.path.join(args.log_dir, 'train'))
    # val_writer = SummaryWriter(os.path.join(args.log_dir, 'val'))
    
    if args.evaluate:
        validate(val_loader, model, criticizer, 0)
        return
    
    
    # checkpoint=torch.load('./checkpoints/cotr_2_checkpoint.pth.tar')
    #module.load_state_dict(torch.load('/media/bme/2.0T/Reserach/xz/3d/checkpoints/3dunet_model_best.pth.tar'))
    #module.load_state_dict(checkpoint['state_dict'])
    # args.start_epoch=checkpoint['epoch']
    # best_dice=checkpoint['best_dice']
    # optimizer.load_state_dict(checkpoint['optim'])
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        score = train(train_loader, model, criticizer,
                      optimizer, epoch)
        #train_score.append(score)
        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or (epoch + 1) >= args.epochs * 0.9:
            dice = validate(val_loader, model, criticizer,optimizer,
                             epoch + 1)
            #val_score.append(dice)
            # remember best prec@1 and save checkpoint
            is_best = dice > best_dice
            best_dice = max(dice, best_dice)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,   #
                'state_dict': model.module.state_dict(),
                'best_dice': best_dice,
                'optim': optimizer.state_dict(),
            }, is_best=is_best,epoch=epoch + 1)

        elif (epoch + 1) % args.save_freq == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.module.state_dict(),
                'best_dice': best_dice,
                'optim': optimizer.state_dict(),
            }, is_best=False,epoch=epoch + 1)    
        scheduler.step()
def train(train_loader, module, criticizer, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    dices = AverageMeter()

    # switch to train mode
    module.train()

    #end = time.time()
    for i, (inputs, label,optical_flows,innerlabel) in enumerate(train_loader):
        # measure data loading time
        #data_time.update(time.time() - end)

        inputs, label,optical_flows,innerlabel = inputs.to(device), label.to(device),optical_flows.to(device),innerlabel.to(device)
        #inputs, label = inputs.to(device), label.to(device)
        #print(inputs.size(),label.size(),optical_flows.size())
        # compute output
        output,inneroutput= module(inputs,optical_flows)#.squeeze(1).squeeze(1)
        #print(outputs.size())
        output=output.squeeze(2)
        #print(inneroutput.shape,innerlabel.shape)  
        celoss = criticizer(output, label)
        #    
        diceloss = dice_loss(output, label)
        #innerloss=dice_loss(inneroutput,innerlabel)
        innerloss=dice_loss(inneroutput,innerlabel)
        #diceloss=dice_loss(output, label)
        loss=celoss+innerloss+diceloss#+diceloss*0.1+innerloss*0.1
        
        # loss =get_losses_weights(torch.tensor([celoss, diceloss, innerloss]))	# loss_w: [0.8649, 0.7568, 1.4054, 0.9730] 
        # loss.requires_grad_(True)
        
        dice = dice_coeff(output, label)
        losses.update(loss.item(), inputs.size(0))
        dices.update(dice, inputs.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())
        # measure elapsed time
        #batch_time.update(time.time() - end)
        #end = time.time()
    
        if i > 0 and (i % args.print_freq == 0 or i + 1 == len(train_loader)):
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'dice {dices.val:.3f} ({dices.avg:.3f})\t'
                    .format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, dices=dices, lr=optimizer.param_groups[-1]['lr'])))
    return dices.avg

def validate(val_loader, module, criticizer, optimizer, epoch):
    batch_time = AverageMeter()
    #data_time = AverageMeter()
    losses = AverageMeter()
    dice = AverageMeter()
    dices = AverageMeter()

    # switch to train mode
    module.eval()

    #end = time.time()
    with torch.no_grad():
        for i, (inputs, label,optical_flows,innerlabel) in enumerate(val_loader):  
            inputs, label,optical_flows,innerlabel = inputs.to(device), label.to(device),optical_flows.to(device),innerlabel.to(device)
            #inputs, label = inputs.to(device), label.to(device)
            #output = module(inputs).squeeze(1).squeeze(1)
            output,inneroutput= module(inputs,optical_flows)#.squeeze(1).squeeze(1)
               
            output=output.squeeze()
            celoss = criticizer(output, label)/label.shape[0]
            innerloss = dice_loss(inneroutput, innerlabel)
            diceloss=dice_loss(output, label)
            loss=celoss+diceloss+innerloss
            dice = dice_coeff(output, label)
            losses.update(loss.item(), inputs.size(0))
            dices.update(dice, inputs.size(0))
    
            #batch_time.update(time.time() - end)
            #end = time.time()
            
            if i > 0 and (i % args.print_freq == 0 or i + 1 == len(val_loader)):
                #step = epoch * len(val_loader) + i
                print(('Test: [{0}/{1}]\t'
                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                       'dice@1 {dices.val:.3f} ({dices.avg:.3f})\t'
                       .format(
                           i, len(val_loader), batch_time=batch_time, loss=losses,
                           dices=dices)))


    print(('Testing Results: dice@1 {dices.avg:.3f}  Loss {loss.avg:.5f}'
           .format(dices=dices,loss=losses)))
    return dices.avg

def save_checkpoint(state, is_best, epoch,filename='checkpoint.pth.tar'):
    a='_'.join((str(epoch),filename))
    filename = '_'.join((args.snapshot_pref, a))
    torch.save(state, filename)
    if is_best:
        best_name = '_'.join(
            (args.snapshot_pref, 'model_best.pth.tar'))
        shutil.copyfile(filename, best_name)      
class AverageMeter(object):
    def __init__(self):
        self.reset()
 
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
 
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count 
        
# def get_losses_weights(losses:[list, np.ndarray, torch.Tensor]):
#     weights = torch.div(losses, torch.sum(losses)) * 3
#     loss=torch.sum(weights*losses)
#     return loss





if __name__ == '__main__':
    main()