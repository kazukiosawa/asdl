from typing import List, Dict

import torch
import torch.nn as nn
import torch.distributed as dist

__all__ = ['ParamVector', 'reduce_vectors', 'normalization', 'orthnormal']


class ParamVector:
    def __init__(self, params: List[torch.Tensor], vectors):
        self.params: List[torch.Tensor] = params
        self.vectors: Dict[torch.Tensor, torch.Tensor] = {}

        if isinstance(vectors, torch.Tensor):
            assert vectors.ndim == 1
            pointer = 0
            for p in params:
                numel = p.numel()
                v = vectors[pointer: pointer + numel]
                self.vectors[p] = v.view_as(p)
                pointer += numel
            assert pointer == vectors.numel()
        elif isinstance(vectors, list):
            for p, v in zip(params, vectors):
                assert p.shape == v.shape
                self.vectors[p] = v
        elif isinstance(vectors, dict):
            for p in params:
                assert p.shape == vectors[p].shape
            self.vectors = vectors

        raise TypeError(f'Invalid vectors type: {type(vectors)}')

    def __add__(self, other):
        assert self.params == other.params
        vectors = {}
        for p in self.params:
            vectors[p] = self.vectors[p] + other.vectors[p]
        return ParamVector(self.params, vectors)

    def add(self, other, alpha=1):
        assert self.params == other.params
        vectors = {}
        for p in self.params:
            vectors[p] = self.vectors[p].add(other.vectors[p], alpha=alpha)
        return ParamVector(self.params, vectors)

    def __iadd__(self, other):
        assert self.params == other.params
        for p in self.params:
            self.vectors[p] += other.vectors[p]

    def extend(self, other):
        self.params.extend(other.params)
        self.vectors.update(other.vectors)

    def mul(self, value):
        return ParamVector(self.params, [v.mul(value) for v in self.vectors.values()])

    def mul_(self, value):
        for key in self.vectors:
            self.vectors[key].mul_(value)
        return self

    def dot(self, other):
        assert self.params == other.params
        return torch.dot(self.get_flatten_vector(), other.get_flatten_vector())

    def norm(self):
        return torch.norm(self.get_flatten_vector())

    def get_vectors_by_module(self, module: nn.Module):
        params = [p for p in module.parameters()]
        return self.get_vectors_by_params(params)

    def get_vectors_by_params(self, params: List[torch.Tensor]):
        vectors = {p: self.vectors[p] for p in params if p in self.vectors}
        return ParamVector(params, vectors)

    def get_flatten_vector(self):
        flat_vecs = [v.flatten() for v in self.vectors.values()]
        return torch.cat(flat_vecs)


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
        rst = ParamVector(vectors.params, packed_tensor)
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
