import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Subset
import torchvision
from torchvision import transforms
import asdl


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 5, bias=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 6, 5)
        self.fc1 = nn.Linear(6 * 5 * 5, 6, bias=False)
        self.fc2 = nn.Linear(6, 8)
        self.fc3 = nn.Linear(8, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    density = {}
    for fisher_type in [asdl.FISHER_EMP, asdl.FISHER_MC, asdl.FISHER_EXACT]:
        density[fisher_type] = {}
        for fisher_shape in [asdl.SHAPE_FULL, asdl.SHAPE_LAYER_WISE]:
            density[fisher_type][fisher_shape] = {}
            density_slq, grids_slq = asdl.fisher_esd_for_cross_entropy(
                model, fisher_type, fisher_shape, data_loader=trainloader, seed=1)
            density[fisher_type][fisher_shape]['slq'] = (density_slq, grids_slq)
            
            f = asdl.fisher_for_cross_entropy(model, fisher_type, fisher_shape, data_loader=trainloader, seed=1)
            eigvals = f.get_eigenvalues(fisher_type, fisher_shape).tolist()
            lambda_max = np.max(eigvals)
            lambda_min = np.min(eigvals)
            grids_exact = np.linspace(lambda_min, lambda_max, num=10000)
            sigma_squared = 1e-5 * max(1, (lambda_max - lambda_min))
            density_exact = np.zeros(10000)
            for j in range(10000):
                x = grids_exact[j]
                tmp_result = np.exp(-(x - eigvals)**2 /
                                      (2.0 * sigma_squared)) / np.sqrt(2 * np.pi * sigma_squared)
                density_exact[j] = np.sum(tmp_result)
            normalization = np.sum(density_exact) * (grids_exact[1] - grids_exact[0])
            density_exact = density_exact / normalization
            density[fisher_type][fisher_shape]['exact'] = (density_exact, grids_exact)

    fig, axs = plt.subplots(2, 3)
    for i, fisher_shape in enumerate([asdl.SHAPE_FULL, asdl.SHAPE_LAYER_WISE]):
        for j, fisher_type in enumerate([asdl.FISHER_EMP, asdl.FISHER_MC, asdl.FISHER_EXACT]):
            density_slq, grids_slq = density[fisher_type][fisher_shape]['slq']
            density_exact, grids_exact = density[fisher_type][fisher_shape]['exact']
            axs[i, j].semilogy(grids_slq, density_slq+1.0e-7)
            axs[i, j].semilogy(grids_exact, density_exact+1.0e-7)
            axs[i, j].set_title(fisher_type+', '+fisher_shape, fontsize=10)
            if j==0:
                axs[i, j].set_ylabel('Density (Log Scale)')
            if i==1:
                axs[i, j].set_xlabel('Eigenvlaue')
    lgd = fig.legend(['slq', 'exact'], bbox_to_anchor=(1.13,0.8))
    fig.tight_layout()
    fig.savefig('example_esd.png', bbox_extra_artists=(lgd,), bbox_inches='tight')


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    torch.random.manual_seed(1)
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data',
                                            train=True,
                                            download=True,
                                            transform=transform)
    
    n_examples = 128
    trainset = Subset(trainset, range(n_examples))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=4)
    model = Net()
    for i, p in enumerate(model.parameters()):
        if i % 2 == 0:
            p.requires_grad_(False)
    model.to(device)
    main()
