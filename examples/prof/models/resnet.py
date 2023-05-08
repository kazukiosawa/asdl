import torch.nn
import torch.nn as nn
from torchvision.models.resnet import Bottleneck, conv1x1


def resnet50_last():
    downsample = nn.Sequential(
        conv1x1(1024, 2048, 2),
        nn.BatchNorm2d(2048),
    )

    block1 = Bottleneck(inplanes=1024,
                        planes=512,
                        stride=2,
                        downsample=downsample)
    block2 = Bottleneck(inplanes=2048,
                        planes=512)
    block3 = Bottleneck(inplanes=2048,
                        planes=512)
    avgpool = nn.AdaptiveAvgPool2d((1, 1))
    fc = nn.Linear(2048, 1000)
    model = nn.Sequential()
    model.add_module('block1', block1)
    model.add_module('block2', block2)
    model.add_module('block3', block3)
    model.add_module('avgpool', avgpool)
    model.add_module('flatten', torch.nn.Flatten())
    model.add_module('fc', fc)

    return model
