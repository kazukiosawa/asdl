
import torch
import timm
from torch import nn
from asdfghjkl import KfacGradientMaker
from asdfghjkl.precondition.natural_gradient import NaturalGradientConfig
from asdfghjkl import FISHER_EMP
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
from timm.data.transforms_factory import create_transform
from torch.utils.data import RandomSampler,SequentialSampler, DataLoader

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='kfac')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument("--model", default="deit_tiny_patch16_224", type=str, choices=["deit_tiny_patch16_224", "vit_base_patch16_224"])
    parser.add_argument("--optimizer", default="sgd", type=str, choices=["sgd", "rmsprop", "adamw", "kfac_mc", "kfac_emp"])
    parser.add_argument('--label-smoothing', default=0.1, type=float)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--cov-update-freq', type=int, default=10)
    parser.add_argument('--inv-update-freq', type=int, default=100)
    parser.add_argument('--damping', type=float, default=0.001)
    parser.add_argument("--mixup-alpha", default=0.8, type=float)
    parser.add_argument("--cutmix-alpha", default=1.0, type=float)
    parser.add_argument('--log_interval', type=int, default=50,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--clip-grad-norm', type = float,  default=0.1)
    parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
    return parser.parse_args()

def train(epoch, datasets, model, criterion_out_define, optimizer, grad_maker, args):
    model.train()
    lr = optimizer.param_groups[0]['lr']
    metric = Metric(args.device)
    for i, (inputs, targets) in enumerate(datasets.train_loader):
        inputs,targets = inputs.to(args.device),targets.to(args.device)
        optimizer.zero_grad()
        dummy_y = grad_maker.setup_model_call(model, inputs)
        
        ### FIXME loss unreasonably large after epoch0 when use criterion defined from outside of function.
        grad_maker.setup_loss_call(criterion_out_define, dummy_y, targets)
        
        ### FIXME define critertion inside fucntion can aviod the issue.
        # criterion_in_define = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        # grad_maker.setup_loss_call(criterion_in_define, dummy_y, targets)
        
        outputs, loss = grad_maker.forward_and_backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        optimizer.step() 
        metric.update(inputs.shape[0], loss, outputs, targets)
        if i % args.log_interval == 0:
            print(f'Epoch{epoch} Iter{i} Train Loss{loss:.4f}')
    print(f'Epoch {epoch} Train {metric} LR: {lr}')
    return metric,lr

def eval(epoch, eval_dataloader, model, criterion, args):
    model.eval()
    metric = Metric(args.device)
    with torch.inference_mode():
        for i, (inputs, targets) in enumerate(eval_dataloader):
            inputs,targets = inputs.to(args.device),targets.to(args.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            metric.update(inputs.shape[0], loss, outputs, targets)

    print(f'Epoch {epoch} Val {metric}')
    return  metric


class Dataset(object):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.pin_memory = True
        self.train_sampler = RandomSampler(self.train_dataset)
        self.train_loader = DataLoader(self.train_dataset,
                                    batch_size=self.batch_size,
                                    sampler=self.train_sampler,
                                    num_workers=args.num_workers,
                                    pin_memory=True)
        self.eval_sampler = SequentialSampler(self.val_dataset)
        self.val_loader = DataLoader(self.val_dataset,
                                    batch_size=self.batch_size,
                                    sampler=self.eval_sampler,
                                    num_workers=args.num_workers,
                                    pin_memory=True)
        self.test_loader = DataLoader(self.test_dataset,
                                    batch_size=self.batch_size,
                                    sampler=self.eval_sampler,
                                    num_workers=args.num_workers,
                                    pin_memory=True)
class CIFAR10(Dataset):
    def __init__(self, args):      
        TrainCIFAR10Transforms = create_transform(
            224,
            is_training=True,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bilinear',
            re_prob=0.0,
        )        
        ValCIFAR10Transforms = create_transform(
            224,
            is_training=False,
            interpolation='bilinear',
            crop_pct=1
        ) 
        self.train_dataset = torchvision.datasets.CIFAR10(root="data", transform=TrainCIFAR10Transforms, download=True)
        self.val_dataset = torchvision.datasets.CIFAR10(root="data", train=False, transform=ValCIFAR10Transforms)        
        self.test_dataset = torchvision.datasets.CIFAR10(root="data", train=False, transform=ValCIFAR10Transforms)        
        super().__init__(args)

class Metric(object):
    def __init__(self, device):
        self._n = torch.tensor([0.0]).to(device)
        self._loss = torch.tensor([0.0]).to(device)
        self._corrects = torch.tensor([0.0]).to(device)

    def update(self, n, loss, outputs, targets):
        with torch.inference_mode():
            self._n += n
            self._loss += loss * n 
            _, preds = torch.max(outputs, dim=1)
            if targets.dim() != 1: # use mix up
                _, targets = torch.max(targets, dim=1)
            self._corrects += torch.sum(preds == targets)
    @property
    def loss(self):
        return (self._loss / self._n).item()
    @property
    def accuracy(self):
        return (self._corrects / self._n).item()
    def __str__(self):
        return f'Loss: {self.loss:.4f}, Acc: {self.accuracy:.4f}'
                
if __name__ == '__main__':
    args = parse_args()
    print(args)
    torch.manual_seed(args.seed)
    # ========== DATA ==========
    datasets = CIFAR10(args)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    num_classes=len(datasets.train_dataset.classes)
    # ========== MODEL ==========
    model = timm.create_model(args.model, pretrained=args.pretrained, num_classes = num_classes) ### Add num_classes!
    model.to(args.device)
    # ========== OPTIMIZER ==========
    optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    config = NaturalGradientConfig(data_size=args.batch_size,
                                    fisher_type=FISHER_EMP,
                                    damping=args.damping,
                                    curvature_upd_interval=args.cov_update_freq,
                                    preconditioner_upd_interval=args.inv_update_freq,
                                    ignore_modules=[nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm],
    )
    grad_maker = KfacGradientMaker(model, config, swift=False)
    # ========== LEARNING RATE SCHEDULER ==========
    lr_scheduler = CosineAnnealingLR(optimizer,T_max=args.epochs)
    # ========== TRAINING ==========
    # training 
    for epoch in range(args.epochs):
        metric,lr = train(epoch, datasets, model, criterion, optimizer, grad_maker, args)
        metric = eval(epoch, datasets.val_loader, model, criterion, args)
        lr_scheduler.step()
