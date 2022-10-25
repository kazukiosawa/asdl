import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

from asdfghjkl import KfacGradientMaker, data_loader_gradient
from asdfghjkl import PseudoBatchLoaderGenerator
from asdfghjkl.fisher import LOSS_CROSS_ENTROPY


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def time_f(f):
    print(f.__name__)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()
    f()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f'elapsed: {elapsed:.3f}s')


def train_by_sgd(print_log=True):
    model = Net()
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

    # 1 epoch training
    for i, pseudo_batch_loader in enumerate(psl_generator):
        optimizer.zero_grad(set_to_none=True)
        # forward + backward
        loss_fn = nn.CrossEntropyLoss(reduction='sum')
        loss = data_loader_gradient(model, loss_fn, pseudo_batch_loader)
        # update param by param.grad
        optimizer.step()

        if print_log:
            print(f'step {i} (pseudo-batch-size {len(pseudo_batch_loader.dataset)}): loss {loss}')


def _train_by_kfac(fisher_type):
    model = Net()
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    ng = KfacGradientMaker(model, fisher_type, loss_type=LOSS_CROSS_ENTROPY)

    # 1 epoch training
    for i, pseudo_batch_loader in enumerate(psl_generator):
        optimizer.zero_grad(set_to_none=True)
        # forward + backward
        # (By setting calc_emp_loss_grad=True, param.grad
        #  for the empirical loss will be calculated together with curvature.)
        loss = ng.refresh_curvature(data_loader=pseudo_batch_loader,
                                    calc_emp_loss_grad=True)
        # invert curvature
        ng.update_preconditioner()
        # precondition param.grad
        ng.precondition()
        # update param by param.grad
        optimizer.step()

        print(f'step {i} (pseudo-batch-size {len(pseudo_batch_loader.dataset)}): loss {loss}')


def train_by_kfac_fisher_emp():
    _train_by_kfac('fisher_emp')


def train_by_kfac_fisher_mc():
    _train_by_kfac('fisher_mc')


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    torch.random.manual_seed(0)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data',
                                            train=True,
                                            download=True,
                                            transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=4)

    psl_generator = PseudoBatchLoaderGenerator(trainloader,
                                               pseudo_batch_size=2**13,
                                               drop_last=True)
    train_by_sgd(print_log=False)  # warmup run
    time_f(train_by_sgd)
    time_f(train_by_kfac_fisher_emp)
    time_f(train_by_kfac_fisher_mc)

