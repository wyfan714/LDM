import torch, math, time, argparse, json, os, sys
import random, dataset, utils, losses, net
import numpy as np

from dataset.Inshop import Inshop_Dataset
from net.resnet import *
from net.googlenet import *
from net.bn_inception import *
from dataset import sampler
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.dataloader import default_collate

from tqdm import *
import wandb

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # set random seed for all gpus

# parser = argparse.ArgumentParser(description=
#     'Official implementation of `Proxy Anchor Loss for Deep Metric Learning`'
#     + 'Our code is modified from `https://github.com/dichotomies/proxy-nca`'
# )
# parser.add_argument('--dataset',
#     default='cub',
#     help = 'Training dataset, e.g. cub, cars, SOP, Inshop'
# )
# parser.add_argument('--embedding-size', default = 512, type = int,
#     dest = 'sz_embedding',
#     help = 'Size of embedding that is appended to backbone model.'
# )
# parser.add_argument('--batch-size', default = 150, type = int,
#     dest = 'sz_batch',
#     help = 'Number of samples per batch.'
# )
# parser.add_argument('--gpu-id', default = 0, type = int,
#     help = 'ID of GPU that is used for training.'
# )
# parser.add_argument('--workers', default = 4, type = int,
#     dest = 'nb_workers',
#     help = 'Number of workers for dataloader.'
# )
# parser.add_argument('--model', default = 'bn_inception',
#     help = 'Model for training'
# )
# parser.add_argument('--l2-norm', default = 1, type = int,
#     help = 'L2 normlization'
# )
# parser.add_argument('--resume', default = '',
#     help = 'Path of resuming model'
# )
# parser.add_argument('--remark', default = '',
#     help = 'Any reamrk'
# )
parser = argparse.ArgumentParser(description=
    'Official implementation of `Learnable Margin in Deep Metric Learning`'
    + 'Our code is modified from `https://github.com/tjddus9597/Proxy-Anchor-CVPR2020`'
)
# export directory, training and val datasets, test datasets
parser.add_argument('--LOG_DIR',
    default='../Proxy_Anchor/Newlogs',
    help = 'Path to log folder'
)
parser.add_argument('--dataset',
    default='cub',
    help = 'Training dataset, e.g. cub, cars, SOP, Inshop'
)
parser.add_argument('--embedding-size', default = 512, type = int,
    dest = 'sz_embedding',
    help = 'Size of embedding that is appended to backbone model.'
)
parser.add_argument('--batch-size', default = 90, type = int,
    dest = 'sz_batch',
    help = 'Number of samples per batch.'
)
parser.add_argument('--epochs', default = 40, type = int,
    dest = 'nb_epochs',
    help = 'Number of training epochs.'
)
parser.add_argument('--gpu-id', default = 1, type = int,
    help = 'ID of GPU that is used for training.'
)
parser.add_argument('--workers', default = 2, type = int,
    dest = 'nb_workers',
    help = 'Number of workers for dataloader.'
)
parser.add_argument('--model', default = 'bn_inception',
    help = 'Model for training'
)
parser.add_argument('--loss', default = 'Proxy_Anchor',
    help = 'Criterion for training'
)

parser.add_argument('--optimizer', default = 'adamw',
    help = 'Optimizer setting'
)
parser.add_argument('--lr', default = 1e-4, type =float,
    help = 'Learning rate setting'
)
parser.add_argument('--mrg_lr', default = 1e-4, type =float,
    help = 'dynamic mrg Learning rate setting'
)
parser.add_argument('--weight-decay', default = 1e-4, type =float,
    help = 'Weight decay setting'

)
parser.add_argument('--weight_lambda', default=0.3, type=float, dest='weight_lambda',
                    help='weight_lambda'
)
parser.add_argument('--lr-decay-step', default = 10, type =int,
    help = 'Learning decay step setting'
)
parser.add_argument('--lr-decay-gamma', default = 0.5, type =float,
    help = 'Learning decay gamma setting'
)
parser.add_argument('--mrg-lr-decay-step', default = 10, type =int,
    help = 'dynamic mrg Learning decay step setting'
)
parser.add_argument('--mrg-lr-decay-gamma', default = 0.5, type =float,
    help = 'dynamic mrg Learning decay gamma setting'
)
parser.add_argument('--alphap', default = 32, type = float,
    help = 'Positive Scaling Parameter setting'
)
parser.add_argument('--alphan', default = 32, type = float,
    help = 'Negative Scaling Parameter setting'
)
parser.add_argument('--mrg', default = 0.1, type = float,
    help = 'Margin parameter setting'
)
parser.add_argument('--IPC', type = int,
    help = 'Balanced sampling, images per class'
)
parser.add_argument('--warm', default = 1, type = int,
    help = 'Warmup training epochs'
)
parser.add_argument('--bn-freeze', default = 1, type = int,
    help = 'Batch normalization parameter freeze'
)

parser.add_argument('--l2-norm', default = 1, type = int,
    help = 'L2 normlization'
)
parser.add_argument('--resume', default = '/home/wyf/Proxy_Anchor/Newlogs/logs_cub/512_embedding/90_batchsize/bn_inception_Proxy_Anchor_alpha48.0_mrg0.1_adamw_lr0.0001_mrglr0.0005_epochs40_delta0.05_T1.0_lambda1.0/cub_bn_inception_best.pth',
    help = 'Path of resuming model'
)
# /home/wyf/Proxy_Anchor/Newlogs/logs_cub/512_embedding/90_batchsize/bn_inception_Proxy_Anchor_alpha48.0_mrg0.1_adamw_lr0.0001_mrglr0.0005_epochs40_delta0.05_T1.0_lambda1.0/cub_bn_inception_best.pth
# /home/wyf/Proxy_Anchor/Newlogs/logs_cars/512_embedding/90_batchsize/bn_inception_Proxy_Anchor_alpha48.0_mrg0.1_adamw_lr0.0001_mrglr0.0001_epochs40_delta-0.1_T1.0_lambda0.8/cars_bn_inception_best.pth
# /home/wyf/Proxy_Anchor/Newlogs/logs_Inshop/512_embedding/90_batchsize/bn_inception_Proxy_Anchor_alphap32_alphan32_mrg0.1_adamw_lr0.0006_mrglr5e-05_epochs60_delta0.1_T1.0_lambda1.0/Inshop_bn_inception_best.pth
# /home/wyf/Proxy_Anchor/Newlogs/logs_SOP/512_embedding/90_batchsize/bn_inception_Proxy_Anchor_alphap32_alphan32_mrg0.1_adamw_lr0.0006_mrglr0.0001_epochs60_delta0.1_T1.0_lambda1.0/SOP_bn_inception_best.pth
parser.add_argument('--remark', default = '',
    help = 'Any reamrk'
)

parser.add_argument('--delta',  default=-0.1, type = float,
    help='delta in proxy&proxy')
parser.add_argument('--T',  default=1.0, type = float,
    help='temperature in proxy&proxy')
parser.add_argument('--lam',  default=1.0, type = float,
    help='lambda in proxy&proxy')
args = parser.parse_args()

if args.gpu_id != -1:
    torch.cuda.set_device(args.gpu_id)

# Data Root Directory
data_root = os.path.join('/media/', 'wyf')  # wyf
    
# Dataset Loader and Sampler
if args.dataset != 'Inshop':
    ev_dataset = dataset.load(
            name = args.dataset,
            root = data_root,
            mode = 'eval',
            transform = dataset.utils.make_transform(
                is_train = False, 
                is_inception = (args.model == 'bn_inception')
            ))

    dl_ev = torch.utils.data.DataLoader(
        ev_dataset,
        batch_size = args.sz_batch,
        shuffle = False,
        num_workers = args.nb_workers,
        pin_memory = True
    )
    
else:
    query_dataset = Inshop_Dataset(
            root = data_root,
            mode = 'query',
            transform = dataset.utils.make_transform(
                is_train = False, 
                is_inception = (args.model == 'bn_inception')
    ))
    
    dl_query = torch.utils.data.DataLoader(
        query_dataset,
        batch_size = args.sz_batch,
        shuffle = False,
        num_workers = args.nb_workers,
        pin_memory = True
    )

    gallery_dataset = Inshop_Dataset(
            root = data_root,
            mode = 'gallery',
            transform = dataset.utils.make_transform(
                is_train = False, 
                is_inception = (args.model == 'bn_inception')
    ))
    
    dl_gallery = torch.utils.data.DataLoader(
        gallery_dataset,
        batch_size = args.sz_batch,
        shuffle = False,
        num_workers = args.nb_workers,
        pin_memory = True
    )

# Backbone Model
if args.model.find('googlenet')+1:
    model = googlenet(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze = 1)
elif args.model.find('bn_inception')+1:
    model = bn_inception(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze = 1)
elif args.model.find('resnet18')+1:
    model = Resnet18(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze = 1)
elif args.model.find('resnet50')+1:
    model = Resnet50(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze = 1)
elif args.model.find('resnet101')+1:
    model = Resnet101(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze = 1)
model = model.cuda()

if args.gpu_id == -1:
    model = nn.DataParallel(model)

if os.path.isfile(args.resume):
    print('=> loading checkpoint {}'.format(args.resume))
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    print('=> No checkpoint found at {}'.format(args.resume))
    sys.exit(0)
                    
with torch.no_grad():
    print("**Evaluating...**")
    if args.dataset == 'Inshop':
        Recalls = utils.evaluate_cos_Inshop(model, dl_query, dl_gallery)

    elif args.dataset != 'SOP':
        Recalls = utils.evaluate_cos(model, dl_ev)

    else:
        Recalls = utils.evaluate_cos_SOP(model, dl_ev)

    
