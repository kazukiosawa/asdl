from dataclasses import dataclass
from typing import Union, Tuple, Any, List
import itertools

import numpy as np

import torch
import torch.nn.parameter
from torch import Tensor
from torch.nn.parameter import Parameter
from .prec_grad_maker import PreconditionedGradientMaker, PreconditionedGradientConfig


__all__ = ['ShampooGradientMaker', 'ShampooGradientConfig']

_invalid = -1


"""
GradientMaker for Shampoo (https://arxiv.org/abs/1802.09568).

This implementation is based on
https://github.com/google-research/google-research/tree/master/scalable_shampoo/pytorch,
simplified and modified to be compatible with PreconditionedGradientMaker.

The role of Shampoo"GradientMaker" is to "make param.grad", so optimization is
performed by a torch.optim.Optimizer (e.g., torch.optim.SGD).
"""


@dataclass
class ShampooGradientConfig(PreconditionedGradientConfig):
    damping: float = 1e-12
    init_scale: float = 1e-12
    inverse_exponent: int = _invalid # fixed exponent for preconditioner (default: invalid)
    ema_decay: float = _invalid
    # Block size for large layers (default: invalid).
    # Block size = 1 ==> AdaGrad (Don't do this, extremely inefficient!)
    # Block size should be as large as feasible under memory/time constraints.
    block_size: int = _invalid
    # Automatic shape interpretation (for eg: [4, 3, 1024, 512] would result in
    # 12 x [1024, 512] L and R statistics. Disabled by default which results in
    # Shampoo constructing statistics [4, 4], [3, 3], [1024, 1024], [512, 512].
    best_effort_shape_interpretation: bool = False


class ShampooGradientMaker(PreconditionedGradientMaker):
    def __init__(self, model, config: ShampooGradientConfig):
        super().__init__(model, config)
        self.preconditioners: List[Preconditioner] = [
            Preconditioner(p, config) for p in model.parameters() if p.ndim > 1]

    def _forward_and_backward(self, *args, **kwargs) -> Union[Tuple[Any, Tensor], Any]:
        rst = self.forward()
        self.backward()

        if self.do_update_curvature():
            self.update_curvature()

        if self.do_update_preconditioner():
            self.update_preconditioner()

        self.precondition()

        return rst

    def update_curvature(self):
        for preconditioner in self.preconditioners:
            preconditioner.update_statistics()

    def update_preconditioner(self):
        for preconditioner in self.preconditioners:
            preconditioner.update_preconditioners()

    def precondition(self):
        for preconditioner in self.preconditioners:
            preconditioner.precondition()


class Preconditioner:
    def __init__(self, param: Parameter, config: ShampooGradientConfig):
        self.config = config
        self.param = param
        self._transformed_shape = param.shape
        if config.best_effort_shape_interpretation:
            self._transformed_shape = _merge_small_dims(param.shape, config.block_size)

        self._partitioner = BlockPartitioner(self._transformed_shape, config.block_size)
        shapes = self._partitioner.kronecker_factor_shapes()
        ndim = len(self._transformed_shape)
        device = param.device
        assert ndim > 1
        self.statistics = [
            config.init_scale * torch.eye(s[0], device=device) for s in shapes
        ]
        self.preconditioners = [
            torch.eye(s[0], device=device) for s in shapes
        ]

    def update_statistics(self):
        """
        Compute statistics from gradients.
        """
        reshaped_grad = torch.reshape(self.param.grad, self._transformed_shape)
        partitioned_grads = self._partitioner.partition(reshaped_grad)
        ema_decay = self.config.ema_decay
        ndim = len(self._transformed_shape)
        for j, grad in enumerate(partitioned_grads):
            for i in range(ndim):
                axes = list(range(i)) + list(range(i + 1, ndim))
                stat = torch.tensordot(grad, grad, [axes, axes])
                if ema_decay == _invalid:
                    self.statistics[j * ndim + i].add_(stat)
                else:
                    self.statistics[j * ndim + i].mul_(1 - ema_decay).add_(stat, alpha=ema_decay)

    def update_preconditioners(self):
        """Compute L^{-1/exp} for each stats matrix L."""
        exp = self.config.inverse_exponent
        if exp == _invalid:
            exp = 2 * len(self._transformed_shape)
        damping = self.config.damping
        for i, stat in enumerate(self.statistics):
            self.preconditioners[i] = ComputePower(
                stat, exp, ridge_epsilon=damping)

    def precondition(self):
        """Precondition the parameter gradient."""
        reshaped_grad = torch.reshape(self.param.grad, self._transformed_shape)
        partitioned_grads = self._partitioner.partition(reshaped_grad)
        preconditioned_partitioned_grads = []
        num_splits = self._partitioner.num_splits()
        for i, grad in enumerate(partitioned_grads):
            preconditioners_for_grad = self.preconditioners[i * num_splits:(i + 1) * num_splits]
            ndim = len(grad.shape)
            precond_grad = grad
            for j in range(ndim):
                preconditioner = preconditioners_for_grad[j]
                precond_grad = torch.tensordot(precond_grad, preconditioner, [[0], [0]])
            preconditioned_partitioned_grads.append(precond_grad)
        merged_grad = self._partitioner.merge_partitions(
            preconditioned_partitioned_grads)
        self.param.grad.data.copy_(merged_grad.resize_as_(self.param))


def _merge_small_dims(shape_to_merge, max_dim):
    """Merge small dimensions.

  If there are some small dimensions, we collapse them:
  e.g. [1, 2, 512, 1, 2048, 1, 3, 4] --> [1024, 2048, 12] if max_dim = 1024
       [1, 2, 768, 1, 2048] --> [2, 768, 2048]

  Args:
    shape_to_merge: Shape to merge small dimensions.
    max_dim: Maximal dimension of output shape used in merging.

  Returns:
    Merged shape.
  """
    resulting_shape = []
    product = 1
    for d in shape_to_merge:
        if product * d <= max_dim:
            product *= d
        else:
            if product > 1:
                resulting_shape.append(product)
            product = d
    if product > 1:
        resulting_shape.append(product)
    return resulting_shape


class BlockPartitioner:
    """Partitions a tensor into smaller tensors for preconditioning.

    For example, if a tensor has shape (4096, 512), we might split the
    4096 into 4 blocks, so we effectively have 4 tensors of size
    (1024, 512) each.
  """
    def __init__(self, shape: Tuple[int], block_size=_invalid):
        self._shape = shape
        self._splits = []
        self._split_sizes = []
        split_sizes = []
        # We split tensor into smaller blocks. Here we store the metadata to make
        # that split.
        for i, d in enumerate(shape):
            if block_size != _invalid and d > block_size:
                # d-1, otherwise split appends a 0-size array.
                nsplit = (d - 1) // block_size
                indices = (np.arange(nsplit, dtype=np.int32) + 1) * block_size
                sizes = np.ones(nsplit + 1, dtype=np.int32) * block_size
                sizes[-1] = d - indices[-1]
                self._splits.append((i, indices))
                self._split_sizes.append((i, sizes))
                split_sizes.append(sizes)
            else:
                split_sizes.append(np.array([d], dtype=np.int32))
        self._num_splits = len(split_sizes)
        self._kronecker_factor_shapes = []
        for t in itertools.product(*split_sizes):
            self._kronecker_factor_shapes.extend([[d, d] for d in t])

    def kronecker_factor_shapes(self):
        return self._kronecker_factor_shapes

    def num_splits(self):
        return self._num_splits

    def partition(self, tensor):
        """Partition tensor into blocks."""

        assert tensor.shape == self._shape
        tensors = [tensor]
        for (i, sizes) in self._split_sizes:
            tensors_local = []
            for t in tensors:
                tensors_local.extend(torch.split(t, tuple(sizes), dim=i))
            tensors = tensors_local
        return tensors

    def merge_partitions(self, partitions):
        """Merge partitions back to original shape."""

        for (i, indices) in reversed(self._splits):
            n = len(indices) + 1
            partial_merged_tensors = []
            ind = 0
            while ind < len(partitions):
                partial_merged_tensors.append(
                    torch.cat(partitions[ind:ind + n], axis=i))
                ind += n
            partitions = partial_merged_tensors
        assert len(partitions) == 1
        return partitions[0]


@torch.no_grad()
def ComputePower(mat_g,
                 p,
                 iter_count=100,
                 error_tolerance=1e-6,
                 ridge_epsilon=1e-6):
    """A method to compute G^{-1/p} using a coupled Newton iteration.

  See for example equation 3.2 on page 9 of:
  A Schur-Newton Method for the Matrix p-th Root and its Inverse
  by Chun-Hua Guo and Nicholas J. Higham
  SIAM Journal on Matrix Analysis and Applications,
  2006, Vol. 28, No. 3 : pp. 788-804
  https://pdfs.semanticscholar.org/0abe/7f77433cf5908bfe2b79aa91af881da83858.pdf

  Args:
    mat_g: A square positive semidefinite matrix
    p: a positive integer
    iter_count: Stop iterating after this many rounds.
    error_tolerance: Threshold for stopping iteration
    ridge_epsilon: We add this times I to G, to make is positive definite.
                   For scaling, we multiply it by the largest eigenvalue of G.
  Returns:
    (mat_g + rI)^{-1/p} (r = ridge_epsilon * max_eigenvalue of mat_g).
  """
    shape = list(mat_g.shape)
    if len(shape) == 1:
        return torch.pow(mat_g + ridge_epsilon, -1 / p)
    identity = torch.eye(shape[0], device=mat_g.device)
    if shape[0] == 1:
        return identity
    alpha = -1.0 / p
    max_ev, _, _ = PowerIter(mat_g)
    ridge_epsilon *= max_ev
    mat_g += ridge_epsilon * identity
    z = (1 + p) / (2 * torch.norm(mat_g))
    # The best value for z is
    # (1 + p) * (c_max^{1/p} - c_min^{1/p}) /
    #            (c_max^{1+1/p} - c_min^{1+1/p})
    # where c_max and c_min are the largest and smallest singular values of
    # mat_g.
    # The above estimate assumes that c_max > c_min * 2^p
    # Can replace above line by the one below, but it is less accurate,
    # hence needs more iterations to converge.
    # z = (1 + p) / tf.trace(mat_g)
    # If we want the method to always converge, use z = 1 / norm(mat_g)
    # or z = 1 / tf.trace(mat_g), but these can result in many
    # extra iterations.

    mat_root = identity * torch.pow(z, 1.0 / p)
    mat_m = mat_g * z
    error = torch.max(torch.abs(mat_m - identity))
    count = 0
    while error > error_tolerance and count < iter_count:
        tmp_mat_m = (1 - alpha) * identity + alpha * mat_m
        new_mat_root = torch.matmul(mat_root, tmp_mat_m)
        mat_m = torch.matmul(MatPower(tmp_mat_m, p), mat_m)
        new_error = torch.max(torch.abs(mat_m - identity))
        if new_error > error * 1.2:
            break
        mat_root = new_mat_root
        error = new_error
        count += 1
    return mat_root


@torch.no_grad()
def PowerIter(mat_g, error_tolerance=1e-6, num_iters=100):
    """Power iteration.

  Compute the maximum eigenvalue of mat, for scaling.
  v is a random vector with values in (-1, 1)

  Args:
    mat_g: the symmetric PSD matrix.
    error_tolerance: Iterative exit condition.
    num_iters: Number of iterations.

  Returns:
    eigen vector, eigen value, num_iters
  """
    v = torch.rand(list(mat_g.shape)[0], device=mat_g.device) * 2 - 1
    error = 1
    iters = 0
    singular_val = 0
    while error > error_tolerance and iters < num_iters:
        v = v / torch.norm(v)
        mat_v = torch.mv(mat_g, v)
        s_v = torch.dot(v, mat_v)
        error = torch.abs(s_v - singular_val)
        v = mat_v
        singular_val = s_v
        iters += 1
    return singular_val, v / torch.norm(v), iters


@torch.no_grad()
def MatPower(mat_m, p):
    """Computes mat_m^p, for p a positive integer.

  Args:
    mat_m: a square matrix
    p: a positive integer

  Returns:
    mat_m^p
  """
    if p in [1, 2, 4, 8, 16, 32]:
        p_done = 1
        res = mat_m
        while p_done < p:
            res = torch.matmul(res, res)
            p_done *= 2
        return res

    power = None
    while p > 0:
        if p % 2 == 1:
            power = torch.matmul(mat_m, power) if power is not None else mat_m
        p //= 2
        mat_m = torch.matmul(mat_m, mat_m)
    return power
