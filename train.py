import torch, math, time, argparse, os
import random, dataset, utils, losses, net
import numpy as np
from dataset.Inshop import Inshop_Dataset
from torchvision import datasets, transforms
from net.resnet import *
from net.googlenet import *
from net.bn_inception import *
from dataset import sampler
from torch.utils.data.sampler import BatchSampler
from tqdm import *
import sys
import wandb

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # set random seed for all gpus

parser = argparse.ArgumentParser(description=
    'Official implementation of `Learnable Dynamic Margin in Deep Metric Learning`'  
    + 'Our code is modified from `https://github.com/tjddus9597/Proxy-Anchor-CVPR2020`'
)
# export directory, training and val datasets, test datasets
parser.add_argument('--LOG_DIR', 
    default='./AMDML/logs',
    help = 'Path to log folder'
)
parser.add_argument('--dataset', 
    default='cars',
    help = 'Training dataset, e.g. cub, cars, SOP, Inshop, ucmd, aid, pattern'
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
    help = 'Model for training, resnet50'
)
parser.add_argument('--loss', default = 'AMLoss',
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
parser.add_argument('--remark', default = '',
    help = 'Any reamrk'
)
parser.add_argument('--delta',  default = 0.1, type = float,
    help ='delta in proxy&proxy')
parser.add_argument('--lam',  default = 1.0, type = float,
    help ='lambda in proxy&proxy')



args = parser.parse_args()
def main():
    if args.gpu_id != -1:
        torch.cuda.set_device(args.gpu_id)

    # Directory for Log
    LOG_DIR = args.LOG_DIR + '/logs_{}/{}_embedding/{}_batchsize/{}_{}_alphap{}_alphan{}_mrg{}_{}_lr{}_mrglr{}_epochs{}_delta{}_lambda{}'.format(args.dataset,
                                                                                                 args.sz_embedding,
                                                                                                 args.sz_batch,
                                                                                                 args.model, args.loss,            
                                                                                                 args.alphap,
                                                                                                 args.alphan,
                                                                                                 args.mrg,
                                                                                                 args.optimizer,
                                                                                                 args.lr, 
                                                                                                 args.mrg_lr,
                                                                                                 args.nb_epochs,
                                                                                                 args.delta,
                                                                                                 args.lam)

    

    # Wandb Initialization
    wandb.init(project=args.dataset + '_LDM', notes=LOG_DIR)
    wandb.config.update(args)
    data_root = os.path.join('/media/', 'wyf')  # wyf
    # Dataset Loader and Sampler
    if args.dataset == 'Inshop':
        trn_dataset = Inshop_Dataset(
                root = data_root,
                mode = 'train',
                transform = dataset.utils.make_transform(
                    is_train = True, 
                    is_inception = (args.model == 'bn_inception')
                ))
    else:
        trn_dataset = dataset.load(
                name = args.dataset,
                root = data_root,
                mode = 'train',
                transform = dataset.utils.make_transform(
                    is_train = True, 
                    is_inception = (args.model == 'bn_inception')
                ))

    if args.IPC:
        balanced_sampler = sampler.BalancedSampler(trn_dataset, batch_size=args.sz_batch, images_per_class=args.IPC)
        batch_sampler = BatchSampler(balanced_sampler, batch_size=args.sz_batch, drop_last=True)
        dl_tr = torch.utils.data.DataLoader(
            trn_dataset,
            num_workers=args.nb_workers,
            pin_memory=True,
            batch_sampler=batch_sampler
        )
        print('Balanced Sampling')

    else:
        dl_tr = torch.utils.data.DataLoader(
            trn_dataset,
            batch_size=args.sz_batch,
            shuffle=True,
            num_workers=args.nb_workers,
            drop_last=True,
            pin_memory=True
        )
        print('Random Sampling')

    if args.dataset == 'Inshop':
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
    else:
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
    nb_classes = trn_dataset.nb_classes()
    

    # Backbone Model
    if args.model.find('googlenet') + 1:
        model = googlenet(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm,
                          bn_freeze=args.bn_freeze)
    elif args.model.find('bn_inception') + 1:
        model = bn_inception(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm,
                             bn_freeze=args.bn_freeze)
    elif args.model.find('resnet18') + 1:
        model = Resnet18(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm,
                         bn_freeze=args.bn_freeze)
    elif args.model.find('resnet50') + 1:
        model = Resnet50(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm,
                         bn_freeze=args.bn_freeze)
    elif args.model.find('resnet101') + 1:
        model = Resnet101(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm,
                          bn_freeze=args.bn_freeze)
    model = model.cuda()

    if args.gpu_id == -1:
        model = nn.DataParallel(model)

    # DML Losses
    if args.loss == 'AMLoss':
        criterion = losses.AMLoss(nb_classes=nb_classes, sz_embed=args.sz_embedding, mrg=args.mrg, alphap=args.alphap, alphan=args.alphan, delta=args.delta, lam=args.lam).cuda()
    elif args.loss == 'Proxy_Anchor':
        criterion = losses.Proxy_Anchor(nb_classes = nb_classes, sz_embed = args.sz_embedding,mrg = args.mrg, alpha = args.alphap).cuda()
    elif args.loss == 'Circle':
        criterion = losses.CircleLoss(m = 0.4, gamma = 80 ).cuda()
    elif args.loss == 'ProxyGML':
        criterion = losses.ProxyGML(C=nb_classes, N=args.N, dim=args.sz_embedding,weight_lambda=args.weight_lambda, r=args.r).cuda()
    elif args.loss == 'Proxy_NCA':
        criterion = losses.Proxy_NCA(nb_classes=nb_classes, sz_embed=args.sz_embedding).cuda()
    elif args.loss == 'MS':
        criterion = losses.MultiSimilarityLoss().cuda()
    elif args.loss == 'Contrastive':
        criterion = losses.ContrastiveLoss().cuda()
    elif args.loss == 'Triplet':
        criterion = losses.TripletLoss().cuda()
    elif args.loss == 'NPair':
        criterion = losses.NPairLoss().cuda()
    elif args.loss == 'SoftTripleLoss':
        criterion = losses.SoftTripleLoss(nb_classes=nb_classes,sz_embed=args.sz_embedding).cuda()
    elif args.loss == 'ProxyAnchorLoss':
        criterion = losses.ProxyAnchorLoss(nb_classes=nb_classes, sz_embed=args.sz_embedding, mrg=args.mrg,alpha=args.alpha).cuda()
    elif args.loss == 'MarginLoss':
        criterion = losses.MarginLoss(num_classes=nb_classes, learn_beta=True).cuda()


    # Train Parameters
    param_groups = [
        {'params': list(
            set(model.parameters()).difference(set(model.model.embedding.parameters()))) if args.gpu_id != -1 else
        list(set(model.module.parameters()).difference(set(model.module.model.embedding.parameters())))},
        {
            'params': model.model.embedding.parameters() if args.gpu_id != -1 else model.module.model.embedding.parameters(),
            'lr': float(args.lr) * 1},
    ]
    if args.loss == 'AMLoss':
        param_groups.append({'params': criterion.proxies, 'lr': float(args.lr) * 100})
        mrg_param=[{'params': criterion.mrg_list, 'lr': float(args.mrg_lr) }]
    if args.loss == 'Proxy_Anchor':
        param_groups.append({'params': criterion.proxies, 'lr': float(args.lr) * 100})
    if args.loss == 'Proxywyf':
        param_groups.append({'params': criterion.proxies, 'lr': float(args.lr) * 100})
        param_groups.append({'params': criterion.mrg_list, 'lr': float(args.lr) * 5})
    if args.loss == 'ProxyGML':
        param_groups.append({'params': criterion.Proxies, 'lr': float(args.lr) * 100})
    if args.loss == 'SoftTripleLoss':
        param_groups.append({'params': criterion.loss_func.fc, 'lr': float(args.lr) * 100})
    if args.loss == 'ProxyAnchorLoss':
        param_groups.append({'params': criterion.loss_func.proxies, 'lr': float(args.lr) * 100})
    if args.loss == 'MarginLoss':
        param_groups.append({'params': criterion.loss_func.beta, 'lr': float(args.lr) * 100})

    # Optimizer Setting
    if args.optimizer == 'sgd':
        opt = torch.optim.SGD(param_groups, lr=float(args.lr), weight_decay=args.weight_decay, momentum=0.9,
                              nesterov=True)
    elif args.optimizer == 'adam':
        opt = torch.optim.Adam(param_groups, lr=float(args.lr), weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        opt = torch.optim.RMSprop(param_groups, lr=float(args.lr), alpha=0.9, weight_decay=args.weight_decay,
                                  momentum=0.9)
    elif args.optimizer == 'adamw':
        opt = torch.optim.AdamW(param_groups, lr=float(args.lr), weight_decay = args.weight_decay)
        if args.loss == 'AMLoss':
            mrg_opt = torch.optim.AdamW(mrg_param, lr=float(args.mrg_lr), weight_decay = args.weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)
    if args.loss == 'AMLoss':
        mrg_scheduler = torch.optim.lr_scheduler.StepLR(mrg_opt, step_size=args.mrg_lr_decay_step, gamma=args.mrg_lr_decay_gamma)
    print("Training parameters: {}".format(vars(args)))
    print("Training for {} epochs.".format(args.nb_epochs))
    losses_list = []
    best_recall = [0]
    best_rp = 0
    best_epoch = 0
    best_mapr = 0

    for epoch in range(0, args.nb_epochs):
        model.train()
        bn_freeze = args.bn_freeze
        if bn_freeze:
            modules = model.model.modules() if args.gpu_id != -1 else model.module.model.modules()
            for m in modules:
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        losses_per_epoch = []

        # Warmup: Train only new params, helps stabilize learning.
        if args.warm > 0:
            if args.gpu_id != -1:
                unfreeze_model_param = list(model.model.embedding.parameters()) + list(criterion.parameters())
            else:
                unfreeze_model_param = list(model.module.model.embedding.parameters()) + list(criterion.parameters())

            if epoch == 0:
                for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
                    param.requires_grad = False
            if epoch == args.warm:
                for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
                    param.requires_grad = True
        
       # if epoch == 0:
       #     criterion.mrg_list.requires_grad = False
       # if epoch == 5:
       #     criterion.mrg_list.requires_grad - True
        pbar = tqdm(enumerate(dl_tr))
        for batch_idx, (x, y, path) in pbar:
            m = model(x.squeeze().cuda())
            if args.loss == 'ProxyGML':
                loss, loss_samples = criterion(m, y.squeeze().cuda())
            else:
                loss = criterion(m, y.squeeze().cuda())
    
            opt.zero_grad()
            if args.loss == 'AMLoss':
                mrg_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 10)
            if args.loss == 'AMLoss':
                torch.nn.utils.clip_grad_value_(criterion.parameters(), 10)
            if args.loss == 'Proxy_Anchor':
                torch.nn.utils.clip_grad_value_(criterion.parameters(), 10)
            if args.loss == 'ProxyGML':
                torch.nn.utils.clip_grad_value_(criterion.parameters(), 10)
            if args.loss == 'MarginLoss':
                torch.nn.utils.clip_grad_value_(criterion.loss_func.beta, 10)
            losses_per_epoch.append(loss.data.cpu().numpy())
            opt.step()
            if args.loss == 'AMLoss':
                mrg_opt.step()
            pbar.set_description(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                        epoch, batch_idx + 1, len(dl_tr),
                               100. * batch_idx / len(dl_tr),
                        loss.item()))

        losses_list.append(np.mean(losses_per_epoch))
        wandb.log({'loss': losses_list[-1]}, step=epoch)
        scheduler.step()
        if args.loss == 'AMLoss':
            mrg_scheduler.step()
        if (epoch >= 0):
            with torch.no_grad():
                print("**Evaluating...**")
                if args.dataset == 'Inshop':
                    Recalls,rp, mapr = utils.evaluate_cos_Inshop(model, dl_query, dl_gallery)
                elif args.dataset != 'SOP':
                    Recalls, rp, mapr = utils.evaluate_cos(model, dl_ev)
                else:
                    Recalls, rp, mapr = utils.evaluate_cos_SOP(model, dl_ev)

            # Logging Evaluation Score
            if args.dataset == 'Inshop':
                 for i, K in enumerate([1, 10, 20, 30, 40, 50]):
                     #print('args.dataset == Inshop')
                     wandb.log({"R@{}".format(K): Recalls[i]}, step=epoch)
            elif args.dataset != 'SOP':
                 for i in range(6):
                     #print('args.dataset == SOP')
                     wandb.log({"R@{}".format(2**i): Recalls[i]}, step=epoch)
            else:
                 for i in range(4):
                     #print('args.dataset == else')
                     wandb.log({"R@{}".format(10**i): Recalls[i]}, step=epoch)

            # Best model save
            if best_rp < rp:
                best_rp = rp
                if not os.path.exists('{}'.format(LOG_DIR)):
                    os.makedirs('{}'.format(LOG_DIR))
                with open('{}/{}_{}_best_rp.txt'.format(LOG_DIR, args.dataset, args.model), 'w') as f:
                    f.write('best epoch: {}\n'.format(epoch))
                    f.write('best rp: {}\n'.format(best_rp))
            if best_mapr < mapr:
                best_mapr = mapr
                if not os.path.exists('{}'.format(LOG_DIR)):
                    os.makedirs('{}'.format(LOG_DIR))
                with open('{}/{}_{}_best_mapr.txt'.format(LOG_DIR, args.dataset, args.model), 'w') as f:
                    f.write('best epoch: {}\n'.format(epoch))
                    f.write('best mapr: {}\n'.format(best_mapr))
            if best_recall[0] < Recalls[0]:
                best_recall = Recalls
                best_epoch = epoch
                if not os.path.exists('{}'.format(LOG_DIR)):
                    os.makedirs('{}'.format(LOG_DIR))
                torch.save({'model_state_dict': model.state_dict()},
                           '{}/{}_{}_best.pth'.format(LOG_DIR, args.dataset, args.model))
                with open('{}/{}_{}_best_results.txt'.format(LOG_DIR, args.dataset, args.model), 'w') as f:
                    f.write('Best Epoch: {}\n'.format(best_epoch))
                    if args.dataset == 'Inshop':
                        for i, K in enumerate([1, 10, 20, 30, 40, 50]):
                            f.write("Best Recall@{}: {:.4f}\n".format(K, best_recall[i] * 100))
                    elif args.dataset == 'market':
                        f.write("Best Recall@{}: {:.4f}\n".format(1, best_recall[0] * 100))
                    elif args.dataset != 'SOP':
                        for i in range(6):
                            f.write("Best Recall@{}: {:.4f}\n".format(2 ** i, best_recall[i] * 100))
                    else:
                        for i in range(4):
                            f.write("Best Recall@{}: {:.4f}\n".format(10 ** i, best_recall[i] * 100))


if __name__ == '__main__':
    main()
