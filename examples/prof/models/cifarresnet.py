import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


__all__ = ['CifarResNet', 'cifar_resnet14', 'cifar_resnet32', 'cifar_resnet56', 'cifar_resnet110', 'cifar_resnet218', 'cifar_resnet434', 'cifar_resnet866']


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CifarResNet(nn.Module):
    def __init__(self, num_blocks, num_classes=10, width_scale=1):
        super(CifarResNet, self).__init__()
        self.in_planes = 16 * width_scale
        if isinstance(num_blocks, int):
            num_blocks = [num_blocks] * 3

        self.conv1 = nn.Conv2d(3, 16 * width_scale, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16 * width_scale)
        self.layer1 = self._make_layer(16 * width_scale, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(32 * width_scale, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(64 * width_scale, num_blocks[2], stride=2)
        self.linear = nn.Linear(64 * width_scale, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def cifar_resnet14(**kwargs):
    return CifarResNet(2, **kwargs)


def cifar_resnet32(**kwargs):
    return CifarResNet(5, **kwargs)


def cifar_resnet56(**kwargs):
    return CifarResNet(9, **kwargs)


def cifar_resnet110(**kwargs):
    return CifarResNet(18, **kwargs)


def cifar_resnet218(**kwargs):
    return CifarResNet(36, **kwargs)


def cifar_resnet434(**kwargs):
    return CifarResNet(72, **kwargs)


def cifar_resnet866(**kwargs):
    return CifarResNet(144, **kwargs)
