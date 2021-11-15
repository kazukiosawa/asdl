from typing import Dict

import torch
import torch.nn as nn


class ParamVector:
    def __init__(self, params, vectors):
        self.params = params
        self.vectors: Dict[torch.Tensor, torch.Tensor] = {}

        if isinstance(vectors, torch.Tensor):
            pointer = 0
            for p in self.params:
                numel = p.numel()
                v = vectors[pointer: pointer + numel]
                assert p.shape == v.shape
                self.vectors[p] = v
                pointer += numel
            assert pointer == vectors.numel()
        else:
            assert isinstance(vectors, dict)
            for p in self.params:
                assert p in vectors
            self.vectors = vectors

    def __add__(self, other):
        assert self.params == other.params
        vectors = {}
        for p in self.params:
            vectors[p] = self.vectors[p] + other.vectors[p]

    def __iadd__(self, other):
        assert self.params == other.params
        for p in self.params:
            self.vectors[p] += other.vectors[p]

    def scaling(self, scale):
        for key in self.vectors:
            self.vectors[key].mul_(scale)
        return self

    def get_vectors_by_module(self, module: nn.Module):
        vectors = {p: self.vectors.get(p, None) for p in module.parameters()}
        return ParamVector(module, vectors)

    def get_flatten_vector(self):
        flat_vecs = [v.flatten() for v in self.vectors.values()]
        return torch.cat(flat_vecs)
