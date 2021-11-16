from typing import List, Dict

import torch
import torch.nn as nn


class ParamVector:
    def __init__(self, params, vectors):
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

    def __iadd__(self, other):
        assert self.params == other.params
        for p in self.params:
            self.vectors[p] += other.vectors[p]

    def extend(self, other):
        self.params.extend(other.params)
        self.vectors.update(other.vectors)

    def scaling(self, scale):
        for key in self.vectors:
            self.vectors[key].mul_(scale)
        return self

    def get_vectors_by_module(self, module: nn.Module):
        params = [p for p in module.parameters()]
        return self.get_vectors_by_params(params)

    def get_vectors_by_params(self, params: List[torch.Tensor]):
        vectors = {p: self.vectors[p] for p in params if p in self.vectors}
        return ParamVector(params, vectors)

    def get_flatten_vector(self):
        flat_vecs = [v.flatten() for v in self.vectors.values()]
        return torch.cat(flat_vecs)
