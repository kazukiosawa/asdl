
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

__all__ = ['mlp']

class MLP(nn.Module):
    def __init__(self,n_hid = 1000,depth=3,img_size=28,num_classes=10,bias=True):
        super().__init__()
        self.img_size=img_size
        layers=[]
        layers.append(nn.Linear(img_size*img_size, n_hid,bias=bias))
        layers.append(nn.ReLU(inplace=True))
        for i in range(depth-2):
            layers.append(nn.Linear(n_hid, n_hid,bias=bias))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(n_hid, num_classes,bias=bias))
        self.l1 = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        x = x.view([-1, self.img_size*self.img_size])
        x = self.l1(x)
        return x

def mlp(**kwargs):
    model = MLP(**kwargs)
    return model