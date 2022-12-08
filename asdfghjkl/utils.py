from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda import nvtx

_REQUIRES_GRAD_ATTR = '_original_requires_grad'

__all__ = [
    'original_requires_grad',
    'record_original_requires_grad',
    'restore_original_requires_grad',
    'disable_param_grad',
    'im2col_2d',
    'add_value_to_diagonal',
    'nvtx_range'
]


def original_requires_grad(module, param_name):
    param = getattr(module, param_name, None)
    return param is not None and getattr(param, _REQUIRES_GRAD_ATTR)


def record_original_requires_grad(param):
    setattr(param, _REQUIRES_GRAD_ATTR, param.requires_grad)


def restore_original_requires_grad(param):
    param.requires_grad = getattr(
        param, _REQUIRES_GRAD_ATTR, param.requires_grad
    )


@contextmanager
def disable_param_grad(model):

    for param in model.parameters():
        record_original_requires_grad(param)
        param.requires_grad = False

    yield
    for param in model.parameters():
        restore_original_requires_grad(param)


def im2col_2d(x: torch.Tensor, conv2d: nn.Module):
    assert x.ndimension() == 4  # n x c x h_in x w_in
    assert isinstance(conv2d, (nn.Conv2d, nn.ConvTranspose2d))
    assert conv2d.dilation == (1, 1)

    ph, pw = conv2d.padding
    kh, kw = conv2d.kernel_size
    sy, sx = conv2d.stride
    if ph + pw > 0:
        x = F.pad(x, (pw, pw, ph, ph))
    x = x.unfold(2, kh, sy)  # n x c x h_out x w_in x kh
    x = x.unfold(3, kw, sx)  # n x c x h_out x w_out x kh x kw
    x = x.permute(0, 1, 4, 5, 2,
                  3).contiguous()  # n x c x kh x kw x h_out x w_out
    x = x.view(x.size(0),
               x.size(1) * x.size(2) * x.size(3),
               x.size(4) * x.size(5))  # n x c(kh)(kw) x (h_out)(w_out)
    return x


def im2col_2d_aug(x, conv2d):
    n, k_aug = x.shape[:2]
    x = im2col_2d(x.flatten(start_dim=0, end_dim=1), conv2d)
    return x.view(n, k_aug, *x.shape[1:])

    
def arr2col_1d(x: torch.Tensor, conv1d: nn.Module):
    assert x.ndimension() == 3  # n x c x w_in
    assert isinstance(conv1d, (nn.Conv1d, nn.ConvTranspose1d))
    assert conv1d.dilation == (1,)

    pw = conv1d.padding[0]
    kw = conv1d.kernel_size[0]
    sw = conv1d.stride[0]
    if pw > 0:
        x = F.pad(x, (pw,))
    x = x.unfold(2, kw, sw)  # n x c x w_out x kw
    x = x.permute(0, 1, 3, 2).contiguous()
    x = x.view(x.size(0), x.size(1) * x.size(2), x.size(3))  # n x c(kw) x w_out
    return x

    
def arr2col_1d_aug(x, conv1d):
    n, k_aug = x.shape[:2]
    x = arr2col_1d(x.flatten(start_dim=0, end_dim=1), conv1d)
    return x.view(n, k_aug, *x.shape[1:])


def add_value_to_diagonal(x: torch.Tensor, value):
    ndim = x.ndim
    assert ndim >= 2
    eye = torch.eye(x.shape[-1], device=x.device)
    if ndim > 2:
        shape = tuple(x.shape[:-2]) + (1, 1)
        eye = eye.repeat(*shape)
    return x.add_(eye, alpha=value)


@contextmanager
def nvtx_range(msg):
    try:
        nvtx.range_push(msg)
        yield
    finally:
        nvtx.range_pop()


def flatten_after_batch(tensor: torch.Tensor):
    if tensor.ndim == 1:
        return tensor.unsqueeze(-1)
    else:
        return tensor.flatten(start_dim=1)
