import sys,os
sys.path.append(os.path.abspath('../../'))
sys.path.append('./utils/')

import argparse
import numpy as np
import time,math
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.dataset import Subset

from models.resnet import ResNet18,ResNet34
from models.wideresnet import WideResNet
from models.mlp import MLP
from models.cnn import CNN

import asdl
from utils.cutout import Cutout
from utils.autoaugment import CIFAR10Policy

import os,json,timm,wandb

dataset_options = ['MNIST','CIFAR10']
OPTIM_SGD = 'sgd'
OPTIM_ADAMW = 'adamw'
OPTIM_SHAMPOO='shampoo'
OPTIM_KFAC_MC = 'kfac_mc'
OPTIM_NOISY_KFAC_MC = 'noisy_kfac_mc'
OPTIM_SMW_NGD = 'smw_ngd'
OPTIM_FULL_PSGD = 'full_psgd'
OPTIM_KRON_PSGD = 'psgd'
OPTIM_NEWTON = 'newton'
OPTIM_ABS_NEWTON = 'abs_newton'
OPTIM_KBFGS = 'kbfgs'
OPTIM_CURVE_BALL = 'curve_ball'
OPTIM_SENG = 'seng'

max_validation_acc=0
max_test_acc=0
min_validation_loss=np.inf
min_test_loss=np.inf

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
    global max_test_acc,min_test_loss

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
    if test_accuracy>max_test_acc:
        max_test_acc=test_accuracy
    if test_loss<min_test_loss:
        min_test_loss=test_loss

def val(epoch):
    global max_validation_acc,min_validation_loss

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

    if test_accuracy>max_validation_acc:
        max_validation_acc=test_accuracy
    if test_loss<min_validation_loss:
        min_validation_loss=test_loss

def train(epoch):
    for batch_idx, (x, t) in enumerate(train_loader):
        torch.cuda.manual_seed(int(torch.rand(1) * 100))

        model.train()
        x, t = x.to(device), t.to(device)
        optimizer.zero_grad(set_to_none=True)
        dummy_y = grad_maker.setup_model_call(model, x)
        grad_maker.setup_loss_call(F.cross_entropy, dummy_y, t, label_smoothing=args.label_smoothing)
        y, loss = grad_maker.forward_and_backward()
        if args.gradient_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clipping_norm)
        optimizer.step()
        
        pred = y.data.max(1)[1]
        acc = 100. * pred.eq(t.data).cpu().sum() / t.size(0)
        
        if batch_idx % args.log_interval == 0:
            if args.wandb:
                log = {'epoch': epoch,
                       'iteration': (epoch - 1) * num_steps_per_epoch + batch_idx + 1,
                       'train_loss': float(loss),
                       'train_accuracy':float(acc),
                       'learning_rate': optimizer.param_groups[0]['lr']}
                wandb.log(log)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(train_loader.dataset),
                100. * batch_idx / num_steps_per_epoch, float(loss)))

        if math.isnan(loss):
            print('Error: Train loss is nan', file=sys.stderr)
            sys.exit(0)
    
    scheduler.step()


class ParseAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        print('%r %r %r' % (namespace, values, option_string))
        values = list(map(int, values.split()))
        setattr(namespace, self.dest, values)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', default='MNIST',
                        choices=dataset_options)
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--width', type=int, default=2048)
    parser.add_argument('--depth', type=int, default=3)

    parser.add_argument('--batch_size', type=int, default=2048,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')

    parser.add_argument('--data_augmentation', action='store_false', default=True,
                        help='augment data by flipping and cropping')
    parser.add_argument('--auto_augment', action='store_true', default=True)
    parser.add_argument('--cutout', action='store_false', default=True,
                        help='apply cutout')
    parser.add_argument('--n_holes', type=int, default=1,
                        help='number of holes to cut out from image')
    parser.add_argument('--length', type=int, default=16,
                        help='length of the holes')
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--optim', default=OPTIM_KFAC_MC, choices=[OPTIM_KFAC_MC, OPTIM_SMW_NGD, OPTIM_KRON_PSGD,OPTIM_SHAMPOO,OPTIM_SGD,OPTIM_ADAMW,OPTIM_KBFGS,OPTIM_SENG,OPTIM_NOISY_KFAC_MC])
    parser.add_argument('--damping', type=float, default=1e-3)
    parser.add_argument('--ema_decay', type=float, default=-1,
                        help='ema_decay')
    parser.add_argument('--nesterov', action='store_true', default=False)

    parser.add_argument('--gradient_clipping', action='store_true', default=False)
    parser.add_argument('--clipping_norm', type=float, default=10,
                        help='global norm of gradient_clipping')
    
    parser.add_argument('--curvature_update_interval', type=int, default=1)
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--train_size', type=int, default=45056)

    parser.add_argument('--wandb', action='store_false', default=True)
    parser.add_argument('--config', default=None,
                        help='config file path')

    args = parser.parse_args()
    dict_args = vars(args)
    # Load config file
    if args.config is not None:
        with open(args.config) as f:
            config = json.load(f)
        dict_args.update(config)
    print(args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    cudnn.benchmark = True  # Should make training should go faster for large models
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
    
    if args.clipping_norm !=-1:
        args.gradient_clipping=True

    device = torch.device('cuda')
    
    if args.dataset == 'CIFAR10':
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

        # train transform
        train_transform = transforms.Compose([])
        if args.data_augmentation:
            if 'vit' in args.model or 'mixer' in args.model:
                train_transform.transforms.append(transforms.RandomResizedCrop(224))
            else:
                train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
            train_transform.transforms.append(transforms.RandomHorizontalFlip())
            if args.auto_augment:
                train_transform.transforms.append(CIFAR10Policy())
        train_transform.transforms.append(transforms.ToTensor())
        train_transform.transforms.append(normalize)
        if args.cutout:
            train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))

        # test transform
        test_transform = transforms.Compose([])
        if 'vit' in args.model or 'mixer' in args.model:
            test_transform.transforms.append(transforms.Resize(256))
            test_transform.transforms.append(transforms.CenterCrop(224))
        test_transform.transforms.append(transforms.ToTensor())
        test_transform.transforms.append(normalize)

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
                                        download=True,
                                        transform=test_transform,)
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
    elif args.model == 'cnn':
        model = CNN()
    elif args.model == 'resnet18':
        model = ResNet18(num_classes=num_classes)
    elif args.model == 'resnet34':
        model = ResNet34(num_classes=num_classes)
    elif args.model == 'wideresnet28':
        model = WideResNet(depth=28, num_classes=num_classes, widen_factor=10,
                             dropRate=0.3)
    else:
        model = timm.create_model(args.model,pretrained=True,num_classes=num_classes)

    model = model.cuda()

    if args.optim == OPTIM_ADAMW:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,momentum=args.momentum, weight_decay=args.weight_decay)

    config = asdl.PreconditioningConfig(data_size=args.batch_size,
                                    damping=args.damping,
                                    ema_decay = args.ema_decay,
                                    preconditioner_upd_interval=args.curvature_update_interval,
                                    curvature_upd_interval=args.curvature_update_interval,
                                    ignore_modules=[nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d,nn.LayerNorm]
                                    )
                                    
    if args.optim == OPTIM_KFAC_MC:
        grad_maker = asdl.KfacGradientMaker(model, config)
    elif args.optim == OPTIM_SHAMPOO:
        grad_maker = asdl.ShampooGradientMaker(model,config)
    elif args.optim == OPTIM_SMW_NGD:
        grad_maker = asdl.SmwEmpNaturalGradientMaker(model, config)
    elif args.optim == OPTIM_FULL_PSGD:
        grad_maker = asdl.PsgdGradientMaker(model)
    elif args.optim == OPTIM_KRON_PSGD:
        grad_maker = asdl.KronPsgdGradientMaker(model,config)
    elif args.optim == OPTIM_NEWTON:
        grad_maker = asdl.NewtonGradientMaker(model, config)
    elif args.optim == OPTIM_ABS_NEWTON:
        grad_maker = asdl.NewtonGradientMaker(model, config)
    elif args.optim == OPTIM_KBFGS:
        grad_maker = asdl.KronBfgsGradientMaker(model, config)
    elif args.optim == OPTIM_CURVE_BALL:
        grad_maker = asdl.CurveBallGradientMaker(model, config)
    elif args.optim == OPTIM_SENG:
        grad_maker = asdl.SengGradientMaker(model,config=config)
    else:
        grad_maker = asdl.GradientMaker(model)

    scheduler=CosineAnnealingLR(optimizer, T_max=args.epochs,eta_min=0)
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
        wandb.run.summary['max_val_accuracy'] = max_validation_acc
        wandb.run.summary['max_test_accuracy'] = max_test_acc
        wandb.run.summary['max_val_loss'] = min_validation_loss
        wandb.run.summary['max_test_loss'] = min_test_loss