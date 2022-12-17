from typing import Iterable, Union, List
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.distributed as dist

__all__ = ['ParamVector', 'reduce_vectors', 'normalization', 'orthnormal']


class ParamVector:
    def __init__(self, params: Iterable[torch.Tensor], values: Union[torch.Tensor, Iterable[torch.Tensor]]):
        if not isinstance(params, list):
            params = list(params)
        assert len(params) > 0, 'params cannot be empty.'
        self.vectors: OrderedDict[torch.Tensor, torch.Tensor] = OrderedDict()

        if isinstance(values, torch.Tensor):
            assert values.ndim == 1
            pointer = 0
            for p in params:
                numel = p.numel()
                v = values[pointer: pointer + numel]
                self.vectors[p] = v.view_as(p)
                pointer += numel
            assert pointer == values.numel()
        elif isinstance(values, Iterable):
            for p, v in zip(params, values):
                assert p.shape == v.shape
                self.vectors[p] = v
        else:
            raise TypeError(f'Invalid vectors type: {type(values)}')

    def params(self):
        return self.vectors.keys()

    def values(self):
        return self.vectors.values()

    def __add__(self, other):
        vectors = [v1 + v2 for v1, v2 in zip(self.values(), other.values())]
        return ParamVector(self.params(), vectors)

    def __iadd__(self, other):
        for v1, v2 in zip(self.values(), other.values()):
            v1 += v2
        return self

    def add(self, other, alpha=1):
        vectors = [v1.add(v2, alpha=alpha) for v1, v2 in zip(self.values(), other.values())]
        return ParamVector(self.params(), vectors)

    def add_(self, other, alpha=1):
        for v1, v2 in zip(self.values(), other.values()):
            v1.add_(v2, alpha=alpha)
        return self

    def extend(self, other):
        assert not set(self.params()) & set(other.params()), \
            'self.params and other.params cannot have a common element.'
        self.vectors.update(other.vectors)

    def mul(self, value):
        return ParamVector(self.params(), [v.mul(value) for v in self.values()])

    def mul_(self, value):
        for key in self.vectors:
            self.vectors[key].mul_(value)
        return self

    def dot(self, other):
        return torch.sum(self.get_flatten_vector().mul(other.get_flatten_vector()))

    def norm(self):
        return torch.norm(self.get_flatten_vector())

    def get_vectors_by_module(self, module: nn.Module):
        params = [p for p in module.parameters()]
        return self.get_vectors_by_params(params)

    def get_vectors_by_params(self, params: List[torch.Tensor]):
        vectors = {p: self.vectors[p] for p in params if p in self.vectors}
        if len(vectors) == 0:
            return None
        return ParamVector(vectors.keys(), vectors.values())

    def get_vector_by_param(self, param: torch.Tensor, default=None) -> torch.Tensor:
        return self.vectors.get(param, default)

    def get_flatten_vector(self):
        flat_vecs = [v.flatten() for v in self.values()]
        return torch.cat(flat_vecs)

    def numel(self):
        return sum(v.numel() for v in self.values())

    def copy(self):
        return ParamVector(self.params(), [v.clone().detach() for v in self.values()])


def reduce_vectors(vectors: ParamVector, is_master=True, all_reduce=False) -> ParamVector:
    # pack
    packed_tensor = vectors.get_flatten_vector()
    if all_reduce:
        # all-reduce
        dist.all_reduce(packed_tensor)
    else:
        dist.reduce(packed_tensor, dst=0)
    if all_reduce or is_master:
        # unpack
        rst = ParamVector(vectors.params(), packed_tensor)
    else:
        rst = None

    dist.barrier()

    return rst


def normalization(v: ParamVector) -> ParamVector:
    s = v.dot(v)
    s = s**0.5
    s = s.cpu().item()
    v.mul_(1 / (s + 1e-6))
    return v


def orthnormal(w: ParamVector, v_list: List[ParamVector]) -> ParamVector:
    for v in v_list:
        w = w.add(v, alpha=-w.dot(v))
    return normalization(w)
