import argparse
from itertools import islice
import io
import urllib

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import MNIST

from asdfghjkl.vector import ParamVector
from asdfghjkl import KfacGradientMaker
from asdfghjkl import SHAPE_KRON

MODEL_URL = "https://github.com/Cecilwang/models/raw/main/net-mnist-0.9320"


def parse_args():
    parser = argparse.ArgumentParser(description="prnning")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--sparsity", type=float, default=0.8)
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


def test(model, loader, args, prefix=""):
    model.eval()
    n = loss = corrects = 0.
    for i, (inputs, labels) in enumerate(loader):
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        outputs = model(inputs)
        batch_loss = nn.CrossEntropyLoss()(outputs, labels)
        _, preds = torch.max(outputs, 1)

        n = n + inputs.shape[0]
        loss += batch_loss.item() * inputs.shape[0]
        corrects += torch.sum(preds == labels.data).item()
    print(f"{prefix} Test Loss: {loss/n:.4f} Acc: {corrects/n:.4f}")


def poly(start, end, i, n):
    scale = end - start
    progress = min(float(i) / n, 1.0)
    remaining_progress = (1.0 - progress)**2
    return end - scale * remaining_progress


def main():
    args = parse_args()

    # data
    transform = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(32),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307, ), (0.3081, ))
    ])
    dataset = MNIST("/tmp", train=True, download=True, transform=transform)
    fisher_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    dataset = MNIST("/tmp", train=False, download=False, transform=transform)
    test_loader = DataLoader(dataset, batch_size=512)

    # model
    print("creating model")
    model = Net()
    model = torch.load(io.BytesIO(urllib.request.urlopen(MODEL_URL).read()))
    model = model.to(args.device)
    test(model, test_loader, args, "Pretrain")

    # pruning
    ng = KfacGradientMaker(model, "fisher_emp")

    param_vector = [p for p in ng.parameters_for(SHAPE_KRON)]
    param_vector = ParamVector(param_vector, [x.data for x in param_vector])
    mask = torch.ones(param_vector.numel())

    pruning_itertaion = 16
    sparsity = 0.0
    for i in range(pruning_itertaion):
        # calculate inverse fisher
        samples = 64
        scale = 1. / samples
        for j, (inputs, targets) in islice(enumerate(fisher_loader), samples):
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            ng.update_curvature(inputs, targets, accumulate=j > 0, scale=scale)
        ng.update_preconditioner(1e-4)

        # construct parameter and the diagonal of the inverse fisher
        param = param_vector.get_flatten_vector()
        diag = []
        for module in ng.modules_for(SHAPE_KRON):
            kron = ng._get_module_fisher(module).kron
            diag.append(torch.kron(kron.A_inv.diag(), kron.B_inv.diag()))
        diag = to_vector(diag)

        # calculate optimal brain surgeon score
        score = param.pow(2) / diag
        score = score.masked_fill(mask == 0.0, float("inf"))

        # get indices of minimum score
        new_sparsity = poly(0.0, args.sparsity, i + 1, pruning_itertaion)
        n_pruned = int((new_sparsity - sparsity) * torch.numel(score))
        sparsity = new_sparsity
        _, indices = torch.sort(score)
        indices = indices[:n_pruned]

        # calculate the prunig direction
        direction = torch.zeros_like(param)
        direction[indices] = -param[indices] / diag[indices]
        direction = ParamVector(param_vector.params(), direction)
        ng.precondition(direction)
        direction = direction.get_flatten_vector()

        # add pruning direction and set mask
        param += direction
        mask[indices] = 0.0
        param *= mask

        # write parameter back
        param_vector = ParamVector(param_vector.params(), param)
        for p, v in zip(param_vector.params(), param_vector.values()):
            p.data = v

        test(model, test_loader, args, f"[sparsity={sparsity:.2f}]")


if __name__ == "__main__":
    main()
