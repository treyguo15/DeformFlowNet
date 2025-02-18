import argparse
# import models
# model_names = sorted(name for name in models.__dict__
#                      if name.islower() and not name.startswith("__")
#                      and callable(models.__dict__[name]))
parser = argparse.ArgumentParser(
    description="PyTorch implementation of Temporal Segment Networks")
parser.add_argument('--train_list',default='path/train_3dflow_img_path_n8_g1.txt',type=str)
parser.add_argument('--val_list',default='path/val_3dflow_img_path_n8_g1.txt', type=str)
parser.add_argument('--root_path', default="", type=str)
parser.add_argument('--log_dir', default='log', type=str)


# # ========================= Dataset Configs ==========================
# parser.add_argument('--num_segments', default= 5, type=int) # 取多少个序列（每个视频）
# parser.add_argument('--seq_length', default= 8, type=int) # 每个序列取多少帧, 此时batchsize最大为1
# parser.add_argument('--sample_rate', default= 2, type=int,  # 采样间隔
#                     help='video sample rate')
# parser.add_argument('--read_mode', default='img',
#                     choices=['img', 'video'])
# parser.add_argument('--num_classes', default=2, type=int)

# # ========================= Model Configs ==========================
parser.add_argument('--arch', type=str, default="example")
# parser.add_argument('--dropout', '--do', default=0.5, type=float,
#                     metavar='DO', help='dropout ratio (default: 0.5)')
# parser.add_argument('--nonlocal_mod', default=[1000], type=int, nargs="+")
# parser.add_argument('--nltype', default='nl3d', type=str)
# parser.add_argument('--k', default=4, type=int)
# parser.add_argument('--tk', default=0, type=int)
# parser.add_argument('--ts', default=4, type=int)
# parser.add_argument('--nl_drop', default=0.2, type=float)
# parser.add_argument('--freeze_bn', action='store_true')

# # ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=5, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=2, type=int,
                     metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                     metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default= 20, type=float, 
                     metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--warmup_epochs', default=0, type=int)
parser.add_argument('--coslr', action='store_true')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                     metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--clip_gradient', '--gd', default=20, type=float,
                    metavar='W', help='gradient norm clipping (default: disabled)')
parser.add_argument('--nesterov', action='store_true',
                    help='enables Nesterov momentum')
parser.add_argument('--use_affine', action='store_true', help='freeze BN')
parser.add_argument('--drop_rate', default=0.)

# # ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=40, type=int,
                     metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', '-ef', default=1, type=int,
                    metavar='N', help='evaluation frequency (default: 5)')
parser.add_argument('--save-freq', '-sf', default=10, type=int,
                  metavar='N', help='evaluation frequency (default: 5)')
parser.add_argument('--save-vg', help='save the summary of value and gradient',
                     default=False, action='store_true')


# # ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# parser.add_argument('--resume', default='', type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')
parser.add_argument('--soft_resume', action='store_true')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--snapshot_dir', type=str, default="checkpoints")
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--seed', default=-1, type=int, help='random seed')

# # ========================= swin-unet Configs ==========================
parser.add_argument('--cfg', type=str, #required=True, 
                    default='swin_tiny_patch4_window7_224_lite.yaml',
                    metavar="FILE", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')