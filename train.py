# run train.py --dataset cifar10 --model resnet18 --data_augmentation --cutout --length 16
# run train.py --dataset cifar100 --model resnet18 --data_augmentation --cutout --length 8
# run train.py --dataset svhn --model wideresnet --learning_rate 0.01 --epochs 160 --cutout --length 20

import sys
sys.path.append('./utils/')
import pdb
import argparse
import numpy as np
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR,CosineAnnealingLR
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.dataset import Subset

from models.resnet import ResNet18
from models import resnet_nobn
from models.wideresnet import WideResNet
from models.mlp import MLP
from models.create_model import create_model
import wandb
import warmup_scheduler

import asdfghjkl as asdl
from asdfghjkl import FISHER_EXACT, FISHER_MC, FISHER_EMP
from asdfghjkl import SHAPE_FULL, SHAPE_LAYER_WISE, SHAPE_KRON, SHAPE_UNIT_WISE, SHAPE_DIAG
from asdfghjkl import LOSS_CROSS_ENTROPY
from asdfghjkl.precondition import Shampoo,ShampooHyperParams
from asdfghjkl.interval import StepIntervalScheduler,LinearIntervalScheduler

from timm.data.mixup import Mixup
from utils.loss import SoftTargetCrossEntropy
from utils.cutout import Cutout
from utils.autoaugment import CIFAR10Policy

import pandas as pd
import os
import copy

model_options = ['mlp','resnet18','resnet18_nobn', 'wideresnet']
dataset_options = ['MNIST','CIFAR10', 'CIFAR100', 'svhn']

OPTIM_SGD = 'sgd'
OPTIM_ADAM = 'adamw'
OPTIM_KFAC_MC = 'kfac_mc'
OPTIM_KFAC_EMP = 'kfac_emp'
OPTIM_SKFAC_MC = 'skfac_mc'
OPTIM_SKFAC_EMP= 'skfac_emp'
OPTIM_SMW_NGD = 'smw_ng'
OPTIM_FULL_PSGD = 'full_psgd'
OPTIM_KRON_PSGD = 'psgd'
OPTIM_NEWTON = 'newton'
OPTIM_ABS_NEWTON = 'abs_newton'
OPTiM_KBFGS = 'kbfgs'
OPTIM_SHAMPOO='shampoo'

def decide_interval(fix,free,maxratio,max_interval,min_interval):
    prop = max_interval
    while prop>1 and prop >= min_interval:
        step_time = fix+free/prop
        step_time_bfr = fix+free/(prop-1)
        if step_time/step_time_bfr>maxratio:
            prop-=1
        else:
            break
    return prop
    
def main():
    total_train_time = 0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train(epoch)
        total_train_time += time.time() - start
        val(epoch)
        test(epoch)

    print(f'total_train_time: {total_train_time:.2f}s')
    print(f'avg_epoch_time: {total_train_time / args.epochs:.2f}s')
    print(f'avg_step_time: {total_train_time / args.epochs / num_steps_per_epoch * 1000:.2f}ms')
    if args.wandb:
        wandb.run.summary['total_train_time'] = total_train_time
        wandb.run.summary['avg_epoch_time'] = total_train_time / args.epochs
        wandb.run.summary['avg_step_time'] = total_train_time / args.epochs / num_steps_per_epoch

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    if args.wandb:
        log = {'epoch': epoch,
               'iteration': epoch * num_steps_per_epoch,
               'test_loss': test_loss,
               'test_accuracy': test_accuracy}
        wandb.log(log)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_accuracy))

def val(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val_loader.dataset)
    test_accuracy = 100. * correct / len(val_loader.dataset)
    if args.wandb:
        log = {'epoch': epoch,
               'iteration': epoch * num_steps_per_epoch,
               'val_loss': test_loss,
               'val_accuracy': test_accuracy}
        wandb.log(log)
    print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset), test_accuracy))

def train(epoch):
    for batch_idx, (x, t) in enumerate(train_loader):
        torch.cuda.manual_seed(int(torch.rand(1) * 100))

        model.train()
        x, t = x.to(device), t.to(device)
        optimizer.zero_grad(set_to_none=True)

        # y = model(x)
        # loss = F.cross_entropy(y, t)
        # loss.backward()

        dummy_y = grad_maker.setup_model_call(model, x)
        loss_func=torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        mixup = None
        if args.mixup > 0 or args.cutmix > 0:
            mixup = Mixup(mixup_alpha=args.mixup,
                          cutmix_alpha=args.cutmix,
                          label_smoothing=args.label_smoothing,
                          num_classes=10)
            loss_func = SoftTargetCrossEntropy()
        if mixup is not None:
            x, t = mixup(x, t)
        
        grad_maker.setup_loss_call(loss_func, dummy_y, t)

        y, loss = grad_maker.forward_and_backward()
        if args.gradient_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.kl_clip)
        optimizer.step()
        scheduler.step()

        if batch_idx % args.log_interval == 0:
            if args.wandb:
                log = {'epoch': epoch,
                       'iteration': (epoch - 1) * num_steps_per_epoch + batch_idx + 1,
                       'train_loss': float(loss),
                       'learning_rate': optimizer.param_groups[0]['lr']}
                wandb.log(log)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(train_loader.dataset),
                100. * batch_idx / num_steps_per_epoch, float(loss)))

        if args.fisher_eig and batch_idx % args.eig_interval == 0:
            culc_fisher_eig(model,epoch, batch_idx, x, t, loss_func)

def culc_fisher_eig(model,epoch, batch_idx, x, t, loss_func):
    dummy_yf = fisher_maker.setup_model_call(model, x)
    fisher_maker.setup_loss_call(loss_func, dummy_yf, t)
    fisher_eigs,_ = fisher_maker.fisher_eig(top_n=1)

    if args.wandb:
        log = {'epoch': epoch,
                'iteration': (epoch - 1) * num_steps_per_epoch + batch_idx + 1,
                'max_fisher_eig': float(fisher_eigs[0]),
                }
        print(log)
        wandb.log(log)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', default='CIFAR10',
                        choices=dataset_options)
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--warmup', default=0, type=int, metavar='N', help='number of warmup epochs')
    parser.add_argument('--data_augmentation', action='store_false', default=True,
                        help='augment data by flipping and cropping')
    parser.add_argument('--auto_augment', action='store_true', default=True)
    parser.add_argument('--cutout', action='store_true', default=True,
                        help='apply cutout')
    parser.add_argument('--mixup', type=float, default=0)
    parser.add_argument('--cutmix', type=float, default=0)

    parser.add_argument('--n_holes', type=int, default=1,
                        help='number of holes to cut out from image')
    parser.add_argument('--length', type=int, default=16,
                        help='length of the holes')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--train_size', type=int, default=45056)
    parser.add_argument('--img_size', type=int, default=32)

    parser.add_argument('--lr', type=float, default=0.03,
                        help='learning rate')
    parser.add_argument('--lr_ratio', type=float, default=0,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--optim', default=OPTIM_KRON_PSGD, choices=[OPTIM_KFAC_EMP,OPTIM_KFAC_MC,OPTIM_SKFAC_EMP,OPTIM_SKFAC_MC, OPTIM_SMW_NGD, OPTIM_KRON_PSGD,OPTIM_SHAMPOO,OPTIM_SGD,OPTIM_ADAM])
    parser.add_argument('--damping', type=float, default=1e-3)
    parser.add_argument('--ema_decay', type=float, default=-1,
                        help='ema_decay')

    parser.add_argument('--nesterov', action='store_true', default=False)

    parser.add_argument('--gradient_clipping', action='store_true', default=True)
    parser.add_argument('--kl_clip', type=float, default=10,
                        help='kl_clip')

    parser.add_argument('--label_smoothing', type=float, default=0.1)

    parser.add_argument('--interval', type=int, default=1)
    parser.add_argument('--interval_scheduler', type=str, default='step')
    parser.add_argument('--interval_inverse', action='store_true',default=False)
    parser.add_argument('--warmup_ratio', type=float, default=0)

    parser.add_argument('--width', type=int, default=2048)
    parser.add_argument('--depth', type=int, default=3)

    parser.add_argument('--log_interval', type=int, default=1,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--wandb', action='store_false', default=True)

    parser.add_argument('--sd', default=0.1, type=float, help='rate of stochastic depth')
    parser.add_argument('--is_LSA',default=True, action='store_false', help='Locality Self-Attention')
    parser.add_argument('--is_SPT',default=True, action='store_false', help='Shifted Patch Tokenization')

    parser.add_argument('--fisher_eig', action='store_true', default=False)
    parser.add_argument('--eig_interval', type=int, default=10)
    parser.add_argument('--fisher_eig_type', type=str, default=FISHER_MC)
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    cudnn.benchmark = True  # Should make training should go faster for large models
    print(args)

    config = vars(args).copy()
    if args.wandb:
        wandb.init(config=config,
                   entity=os.environ.get('WANDB_ENTITY', None),
                   project=os.environ.get('WANDB_PROJECT', None),
                   )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    device = torch.device('cuda')
    
    if args.dataset == 'CIFAR10':
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        train_transform = transforms.Compose([])
        if args.data_augmentation:
            train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
            train_transform.transforms.append(transforms.RandomHorizontalFlip())
            if args.auto_augment:
                train_transform.transforms.append(CIFAR10Policy())
        train_transform.transforms.append(transforms.ToTensor())
        train_transform.transforms.append(normalize)
        if args.cutout:
            train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])

        num_classes = 10
        train_dataset = datasets.CIFAR10(root='data/',
                                         train=True,
                                         download=True,
                                         transform=train_transform,)
        val_dataset = datasets.CIFAR10(root='data/',
                                         train=True,
                                         download=True,
                                         transform=test_transform,)
        test_dataset = datasets.CIFAR10(root='data/',
                                        train=False,
                                        transform=test_transform,
                                        download=True)

    elif args.dataset == 'MNIST':
        train_transform = transforms.Compose([])
        if args.data_augmentation:
            train_transform.transforms.append(transforms.RandomAffine([-15,15], scale=(0.8, 1.2)))
        train_transform.transforms.append(transforms.ToTensor())
        train_transform.transforms.append(transforms.Normalize((0.1307,), (0.3081,)))

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        num_classes = 10
        train_dataset = datasets.MNIST(root='data/',
                                         train=True,
                                         download=True,
                                         transform=train_transform,)
        val_dataset = datasets.MNIST(root='data/',
                                         train=True,
                                         download=True,
                                         transform=test_transform,)
        test_dataset = datasets.MNIST(root='data/',
                                        train=False,
                                        transform=test_transform,
                                        download=True)

    ## split dataset
    indices = list(range(len(train_dataset)))
    np.random.shuffle(indices)
    train_idx, val_idx = indices[:args.train_size], indices[args.train_size:]
    train_dataset = Subset(train_dataset, train_idx)
    val_dataset   = Subset(val_dataset, val_idx)


    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=args.num_workers)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=args.num_workers)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=args.num_workers)
    num_steps_per_epoch = len(train_loader)
    
    if args.model == 'mlp':
        model = MLP(n_hid=args.width,depth=args.depth)
    elif args.model == 'resnet18':
        model = ResNet18(num_classes=num_classes)
    elif args.model == 'resnet18_nobn':
        model = resnet_nobn.ResNet18(num_classes=num_classes)
    elif args.model == 'wideresnet':
        model = WideResNet(depth=28, num_classes=num_classes, widen_factor=10,
                             dropRate=0.3)
    elif args.model in ['vit','cait','pit','t2t','swin']:
        model = create_model(img_size=32,n_classes=10,args=args)

    model = model.cuda()

    if args.interval_scheduler == 'step':
        intervalscheduler=StepIntervalScheduler(interval=args.interval,T_max=args.epochs*num_steps_per_epoch,warmup_ratio=args.warmup_ratio,inverse=args.interval_inverse)
    elif args.interval_scheduler == 'linear':
        intervalscheduler=LinearIntervalScheduler(interval=args.interval,T_max=args.epochs*num_steps_per_epoch,warmup_ratio=args.warmup_ratio,inverse=args.interval_inverse)

    if args.optim == OPTIM_ADAM:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == OPTIM_SHAMPOO:
        config = ShampooHyperParams(weight_decay=args.weight_decay,interval_scheduler = intervalscheduler,nesterov=args.nesterov)
        optimizer = Shampoo(model.parameters(),lr=args.lr,momentum=args.momentum,hyperparams=config)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,momentum=args.momentum, weight_decay=args.weight_decay,nesterov=args.nesterov)

    if args.optim == OPTIM_KFAC_MC:
        config = asdl.NaturalGradientConfig(data_size=args.batch_size,
                                            fisher_type=FISHER_MC,
                                            damping=args.damping,
                                            interval_scheduler=intervalscheduler,
                                            ignore_modules=[nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d,nn.LayerNorm],
                                            ema_decay=args.ema_decay)
        grad_maker = asdl.KfacGradientMaker(model, config,swift=False)
    elif args.optim == OPTIM_KFAC_EMP:
        config = asdl.NaturalGradientConfig(data_size=args.batch_size,
                                            fisher_type=FISHER_EMP,
                                            damping=args.damping,
                                            interval_scheduler=intervalscheduler,
                                            ignore_modules=[nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d,nn.LayerNorm],
                                            ema_decay=args.ema_decay)
        grad_maker = asdl.KfacGradientMaker(model, config,swift=False)
    elif args.optim == OPTIM_SKFAC_MC:
        config = asdl.NaturalGradientConfig(data_size=args.batch_size,
                                            fisher_type=FISHER_MC,
                                            damping=args.damping,
                                            interval_scheduler=intervalscheduler,
                                            ignore_modules=[nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d,nn.LayerNorm])
        grad_maker = asdl.KfacGradientMaker(model, config,swift=True)
    elif args.optim == OPTIM_SKFAC_EMP:
        config = asdl.NaturalGradientConfig(data_size=args.batch_size,
                                            fisher_type=FISHER_EMP,
                                            damping=args.damping,
                                            interval_scheduler=intervalscheduler,
                                            ignore_modules=[nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d,nn.LayerNorm])
        grad_maker = asdl.KfacGradientMaker(model, config,swift=True)
    elif args.optim == OPTIM_SMW_NGD:
        config = asdl.SmwEmpNaturalGradientConfig(data_size=args.batch_size,
                                                  damping=args.damping)
        grad_maker = asdl.SmwEmpNaturalGradientMaker(model, config)
    elif args.optim == OPTIM_FULL_PSGD:
        config = asdl.PsgdGradientConfig(interval_scheduler=intervalscheduler,)
        grad_maker = asdl.PsgdGradientMaker(model,config)
    elif args.optim == OPTIM_KRON_PSGD:
        config = asdl.PsgdGradientConfig(interval_scheduler=intervalscheduler,)
        grad_maker = asdl.KronPsgdGradientMaker(model,config)
    elif args.optim == OPTIM_NEWTON:
        config = asdl.NewtonGradientConfig(damping=args.damping)
        grad_maker = asdl.NewtonGradientMaker(model, config)
    elif args.optim == OPTIM_ABS_NEWTON:
        config = asdl.NewtonGradientConfig(damping=args.damping, absolute=True)
        grad_maker = asdl.NewtonGradientMaker(model, config)
    elif args.optim == OPTiM_KBFGS:
        config = asdl.KronBfgsGradientConfig(data_size=args.batch_size,
                                             damping=args.damping)
        grad_maker = asdl.KronBfgsGradientMaker(model, config)
    else:
        grad_maker = asdl.GradientMaker(model)

    fisher_config = asdl.FisherConfig(fisher_type = args.fisher_eig_type,
                                        loss_type = LOSS_CROSS_ENTROPY,
                                        fisher_shapes=SHAPE_LAYER_WISE)
    fisher_maker = asdl.get_fisher_maker(model,config=fisher_config)

    hessian_config=asdl.HessianConfig(hessian_shapes=SHAPE_LAYER_WISE)
    hessian_maker = asdl.HessianMaker(model,config=hessian_config)

    if args.warmup > 0:
        base_scheduler=CosineAnnealingLR(optimizer, T_max=args.epochs*num_steps_per_epoch,eta_min=args.lr*args.lr_ratio)
        scheduler = warmup_scheduler.GradualWarmupScheduler(optimizer, multiplier=1., total_epoch=args.warmup*num_steps_per_epoch, after_scheduler=base_scheduler)
    if args.warmup==0:
        scheduler=CosineAnnealingLR(optimizer, T_max=args.epochs*num_steps_per_epoch,eta_min=args.lr*args.lr_ratio)

    torch.cuda.synchronize()
    try:
        main()
        max_memory = torch.cuda.max_memory_allocated()
    except RuntimeError as err:
        if 'CUDA out of memory' in str(err):
            print(err)
            max_memory = -1  # OOM
        else:
            raise RuntimeError(err)

    print(f'cuda_max_memory: {max_memory/float(1<<30):.2f}GB')
    if args.wandb:
        wandb.run.summary['cuda_max_memory'] = max_memory
