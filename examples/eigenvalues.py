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
    top_k = 10
    for fisher_type in [asdl.FISHER_EMP, asdl.FISHER_MC, asdl.FISHER_EXACT]:
        for fisher_shape in [asdl.SHAPE_FULL, asdl.SHAPE_LAYER_WISE]:
            print('=============================')
            print(f'fisher_type: {fisher_type}, fisher_shape: {fisher_shape}')
            eigvals1, _ = asdl.fisher_eig_for_cross_entropy(model, fisher_type, fisher_shape,
                                                            data_loader=trainloader, top_n=top_k, seed=1)
            print(f'Top-{top_k} eigenvalues by power method:')
            print(eigvals1)
            f = asdl.fisher_for_cross_entropy(model, fisher_type, fisher_shape, data_loader=trainloader, seed=1)
            eigvals2 = f.get_eigenvalues(fisher_type, fisher_shape)
            print(f'Top-{top_k} eigenvalues by torch.linalg.eigvalh:')
            print(eigvals2[:top_k].tolist())


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
