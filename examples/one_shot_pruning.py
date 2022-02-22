import argparse
from itertools import islice

import torch
from torch import nn
import torch.nn.functional as F
import torchvision

from asdfghjkl.vector import ParamVector
from asdfghjkl import KFAC
from asdfghjkl import SHAPE_KRON


def parse_args():
    parser = argparse.ArgumentParser(description="prnning")
    parser.add_argument("--dir", type=str, default="/tmp/")
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument('--skip-pretrain', action='store_true', default=False)

    parser.add_argument("--sparsity", type=float, default=0.8)
    parser.add_argument("--n_recompute", type=int, default=16)
    parser.add_argument("--n_recompute_samples", type=int, default=64)

    return parser.parse_args()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def to_vector(parameters):
    return nn.utils.parameters_to_vector(parameters)


def train(model, loader, optimizer, criterion, args):
    model.train()

    n = torch.tensor([0]).to(args.device)
    loss = torch.tensor([0.0]).to(args.device)
    corrects = torch.tensor([0]).to(args.device)

    m = len(loader)
    for i, (inputs, labels) in enumerate(loader):
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        optimizer.zero_grad()

        outputs = model(inputs)
        batch_loss = criterion(outputs, labels)
        batch_loss.backward()
        optimizer.step()
        _, preds = torch.max(outputs, 1)

        n = n + inputs.size(0)
        loss += batch_loss.item() * inputs.size(0)
        corrects += torch.sum(preds == labels.data)

    _loss = (loss / n).item()
    _acc = (corrects / n).item()
    print(f"Train Loss: {_loss:.4f} Acc: {_acc:.4f}")


def test(model, loader, criterion, args, prefix=""):
    model.eval()

    n = torch.tensor([0]).to(args.device)
    loss = torch.tensor([0.0]).to(args.device)
    corrects = torch.tensor([0]).to(args.device)

    for i, (inputs, labels) in enumerate(loader):
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        outputs = model(inputs)
        batch_loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)

        n = n + inputs.size(0)
        loss += batch_loss.item() * inputs.size(0)
        corrects += torch.sum(preds == labels.data)

    loss = (loss / n).item()
    acc = (corrects.double() / n).item()
    print(f"{prefix} Test Loss: {loss:.4f} Acc: {acc:.4f}")


def polynomial_schedule(start, end, i, n):
    scale = end - start
    progress = min(float(i) / n, 1.0)
    remaining_progress = (1.0 - progress)**2
    return end - scale * remaining_progress


def main():
    args = parse_args()

    # model
    model = Net()
    model = model.to(args.device)

    # data
    train_dataset = torchvision.datasets.MNIST(
        args.dir,
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(32),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307, ), (0.3081, ))
        ]))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=512,
                                               shuffle=True)
    fisher_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=32,
                                                shuffle=True)
    test_dataset = torchvision.datasets.MNIST(
        args.dir,
        train=False,
        download=False,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(32),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307, ), (0.3081, ))
        ]))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512)

    criterion = nn.CrossEntropyLoss()

    # pretrain
    if not args.skip_pretrain:
        opt = torch.optim.SGD(model.parameters(),
                              lr=0.01,
                              momentum=0.9,
                              weight_decay=1e-4)
        for e in range(30):
            print(f"Epoch {e}/30")
            train(model, train_loader, opt, criterion, args)
            test(model, test_loader, criterion, args)
            if e == 19:
                opt.param_groups[0]["lr"] /= 10.
        torch.save(model, args.dir + "best")
    else:
        model = torch.load(args.dir + "best")
    test(model, test_loader, criterion, args, "Pretrain")

    # pruning
    ng = KFAC(model, "fisher_emp")

    param_vector = [p for p in ng.parameters_for(SHAPE_KRON)]
    param_vector = ParamVector(param_vector, [x.data for x in param_vector])

    # setup mask
    mask_vector = []
    for module in ng.modules_for(SHAPE_KRON):
        module.register_buffer("weight_mask", torch.ones_like(module.weight))
        module.weight.register_hook(lambda g: g * module.weight_mask)
        module.register_buffer("bias_mask", torch.ones_like(module.bias))
        module.bias.register_hook(lambda g: g * module.bias_mask)
        mask_vector += [module.weight_mask, module.bias_mask]
    mask_vector = ParamVector(mask_vector, [x.data for x in mask_vector])

    sparsity = 0.0
    for i in range(1, args.n_recompute + 1):
        # calculate inverse fisher
        for j, (inputs, targets) in islice(enumerate(fisher_loader),
                                           args.n_recompute_samples):
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            ng.update_curvature(inputs,
                                targets,
                                accumulate=j > 0,
                                scale=1. / args.n_recompute_samples)
        ng.update_inv(1e-4)

        # extract parameter, mask, and the diagonal of the inverse fisher
        param = param_vector.get_flatten_vector()
        mask = mask_vector.get_flatten_vector()
        ifisher_diag = []
        for module in ng.modules_for(SHAPE_KRON):
            kron = ng._get_module_fisher(module).kron
            ifisher_diag.append(
                torch.kron(kron.A_inv.diag(), kron.B_inv.diag()))
        ifisher_diag = to_vector(ifisher_diag)

        # calculate optimal brain surgeon score
        score = param.pow(2) / ifisher_diag
        score = score.masked_fill(mask == 0.0, float("inf"))

        # get indices of minimum score
        new_sparsity = polynomial_schedule(0.0, args.sparsity, i,
                                           args.n_recompute)
        n_pruned = int((new_sparsity - sparsity) * torch.numel(score))
        sparsity = new_sparsity
        _, indices = torch.sort(score)
        indices = indices[:n_pruned]

        # calculate the prunig direction
        pruning_direction = torch.zeros_like(param)
        pruning_direction[indices] = -param[indices] / ifisher_diag[indices]
        pruning_direction = ParamVector(param_vector.params(),
                                        pruning_direction)
        ng.precondition(pruning_direction)
        pruning_direction = pruning_direction.get_flatten_vector()

        # add pruning direction and set mask
        param += pruning_direction
        mask[indices] = 0.0
        param *= mask

        # store parameter and mask back
        param_vector = ParamVector(param_vector.params(), param)
        mask_vector = ParamVector(mask_vector.params(), mask)
        for p, v in zip(param_vector.params(), param_vector.values()):
            p.data = v
        for p, v in zip(mask_vector.params(), mask_vector.values()):
            p.data = v

        test(model, test_loader, criterion, args, f"[sparsity={sparsity:.2f}]")


if __name__ == "__main__":
    main()
