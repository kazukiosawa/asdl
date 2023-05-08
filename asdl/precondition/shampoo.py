from typing import Tuple, List
import itertools

import numpy as np

import torch
import torch.nn.parameter
from torch.nn.parameter import Parameter
from .prec_grad_maker import PreconditionedGradientMaker, PreconditioningConfig

from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch.distributed as dist
from torch.cuda import nvtx


__all__ = ['ShampooGradientMaker']

_invalid = -1


class ShampooGradientMaker(PreconditionedGradientMaker):
    """GradientMaker for calculating the preconditioned gradient by `Shampoo <https://arxiv.org/abs/1802.09568>`_.

    This implementation is based on
    https://github.com/google-research/google-research/tree/master/scalable_shampoo/pytorch,
    simplified and modified to be compatible with the GradientMaker interface.

    Args:
        model (Module): Target module to calculate gradient
        config (PreconditioningConfig): Configuration for gradient preconditioning
        block_size (int): defines the even smaller partition if not _invalid (see class BlockPartitioner)
    """

    def __init__(self, model: torch.nn.Module, config: PreconditioningConfig, 
                 block_size: int = _invalid, sync_group: dist.ProcessGroup = None,):
        super().__init__(model, config)
        self.sync_group = sync_group
        self.block_size = block_size
        if dist.is_initialized(): #if initialized, we do automatically distr model parallelism (atm only support layer-wise distributed (future maybe dim-wise of each layer parallelized))
            self.world_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.splits, self.partitioned_modules = self.get_distr_prec_partition()
        else:
            self.world_rank = 0
            self.world_size = 1
            self.splits, self.partitioned_modules = self.get_distr_prec_partition()

        assert self.world_size >= len(self.splits) + 1, "world_size and number of splits do not match! splits = " + str(self.splits) 

        self.preconditioners = []
        layer = 0
        for p in model.parameters():
            if p.ndim > 1 and p.requires_grad:
                if self.world_rank == self.partitioned_modules[layer]:
                    self.preconditioners.append(Preconditioner(p, config))
                layer += 1

    def get_distr_prec_partition(self):
        """
        Distributes the workload by computational cost of each layer for total number of GPUs

        TODO: multiple GPUs for one layer

        e.g.
        1 GPU for ResNet18:
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        3 GPUs for ResNet18:
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2]

        8 GPUs for ResNet18:
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 3, 4, 5, 6, 7]

        21 or more GPUs for ResNet18:
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

        2 GPUs for 3 layers MLP (if first layer is bigger than 2nd and 3rd):
        [0,1,1]
        """

        total_comp_cost = 0
        comp_cost_layers = []
        shapes_list = []
        for p in self.model.parameters():
            if p.ndim > 1 and p.requires_grad:
                _transformed_shape = _merge_small_dims(p.shape, self.block_size)
                _partitioner = BlockPartitioner(_transformed_shape, self.block_size)
                shapes = _partitioner.kronecker_factor_shapes()

                shapes_list.append(_transformed_shape)
                comp_cost = self.computational_cost(shapes)
                total_comp_cost += comp_cost
                comp_cost_layers.append(comp_cost)

        num_layers = len(comp_cost_layers)

        partitions = [0]*num_layers
        if self.world_size == 1:
            return [], partitions
        elif num_layers > self.world_size:
            split_list = np.array([0])

            for rank in range(self.world_size-1):
                if rank == 0:
                    split_list = np.append(split_list, self.next_split(comp_cost_layers))
                else:
                    sub_sums = []
                    for i in range(1, len(split_list)):
                        
                        local_comp_cost = np.sum(comp_cost_layers[split_list[i-1]:split_list[i]])
                        sub_sums.append(local_comp_cost)
                        
                        if i == len(split_list) - 1:
                            local_comp_cost = np.sum(comp_cost_layers[split_list[i]:])
                            sub_sums.append(local_comp_cost)

                    while(True):
                        i = np.argmax(sub_sums)
                        if i == len(sub_sums) - 1:
                            sub_comp_cost_layers = comp_cost_layers[split_list[i]:]
                            shift = split_list[i]
                        else:
                            sub_comp_cost_layers = comp_cost_layers[split_list[i]:split_list[i+1]]
                            shift = split_list[i]

                        if len(sub_comp_cost_layers) > 1:
                            break
                        else:
                            sub_sums[i] = -1


                    split_list = np.append(split_list, self.next_split(sub_comp_cost_layers) + shift)
                    split_list = np.sort(split_list)

            sub_sums = []
            for i in range(1, len(split_list)):
                
                local_comp_cost = np.sum(comp_cost_layers[split_list[i-1]:split_list[i]])
                sub_sums.append(local_comp_cost)
                
                if i == len(split_list) - 1:
                    local_comp_cost = np.sum(comp_cost_layers[split_list[i]:])
                    sub_sums.append(local_comp_cost)

            #if self.world_rank == 0:
            #    print(sub_sums, "\n")

            next_split = split_list[1]
            rank = 0
            for i in range(len(partitions)):
                if i == next_split:
                    rank += 1
                    if rank != self.world_size - 1:
                        next_split = split_list[rank+1]
                
                partitions[i] = rank
            return split_list[1:], partitions
        else: #atm, we do not support multiple gpus for one layer
            rank = 0
            for i in range(num_layers):
                partitions[i] = i
                
            return partitions[1:], partitions


    def computational_cost(self, shapes):
        """
        input: shape: [[x, x],[y, y],...] (Blockpartitioner.kronecker_factor_shape)

        output: returns the compuational cost of this Blockpartitioned layers
        """
        tmp_cost = 0
        for shape in shapes:
            assert len(shape) == 2
            assert shape[0] == shape[1]

            tmp_cost += shape[0]**0.4 # ATM simple O(n^3) assumption (maybe even less 0.4)

        return tmp_cost

    def next_split(self, subset_partitions):
        """
        deciding where the next split is happening
        
        input: subset_partitions: [] is a subset of comp_cost_layers

        output: index where to split (int)
        """
        assert len(subset_partitions) > 1

        x = np.array(subset_partitions)
        y = np.sum(subset_partitions)/2

        split_loc = len(x[np.cumsum(x) < y])

        split_loc += 1
        
        return split_loc
        
    
    def do_forward_and_backward(self, step=None):
        return True

    def update_curvature(self):
        # TODO: Not needed if ASDL combined with PyTorch's DDP
        if self.world_size > 1:
            with nvtx.range('reduce_scatter_grads'):
                self.reduce_scatter_grads()
        
        for preconditioner in self.preconditioners:
            preconditioner.update_statistics()

    def update_preconditioner(self):
        for preconditioner in self.preconditioners:
            preconditioner.update_preconditioners()

    def precondition(self):
        for preconditioner in self.preconditioners:
            preconditioner.precondition()

        if self.world_size > 1:
            with nvtx.range('all_gather_grads'):
                self.all_gather_grads()

    def reduce_scatter_grads(self):
        grads = [p.grad for p in self.model.parameters() if p.ndim > 1 and p.requires_grad] #this could be all done ones at __init__

        grads_list = []
        tensor_list = []
        for i in range(len(self.splits)):
            if i == 0:
                grads_split = grads[:self.splits[i]]
                grads_list.append(grads_split)
                tensor_list.append(parameters_to_vector(grads_split))
            elif len(self.splits) > 1:
                grads_split = grads[self.splits[i-1]:self.splits[i]]
                grads_list.append(grads_split)
                tensor_list.append(parameters_to_vector(grads_split))
            
            if i == len(self.splits) - 1:
                grads_split = grads[self.splits[i]:]
                grads_list.append(grads_split)
                tensor_list.append(parameters_to_vector(grads_split))

        assert len(self.splits)+1 == len(tensor_list) <= self.world_size, str(self.splits) + ', ' + len(tensor_list) + ', '  + str(self.world_size)
            
        group = self.sync_group

        #print("before scatter: ", grads, "\n")

        handler_list = []
        for i in range(len(tensor_list)):
            handler = dist.reduce(tensor_list[i], i, op=dist.ReduceOp.AVG, group=group, async_op=True)
            handler_list.append(handler)

        for handler in handler_list:
            handler.wait()
        
        if self.world_rank < len(tensor_list):  # this check is needed if there are more GPUs than layers
            vector_to_parameters(tensor_list[self.world_rank], grads_list[self.world_rank])

        #print("after scatter: ", grads, "\n")

    def all_gather_grads(self):
        grads = [p.grad for p in self.model.parameters() if p.ndim > 1 and p.requires_grad] #this could be all done ones at __init__

        grads_list = []
        tensor_list = []
        for i in range(len(self.splits)):
            if i == 0:
                grads_split = grads[:self.splits[i]]
                grads_list.append(grads_split)
                tensor_list.append(parameters_to_vector(grads_split))
            elif len(self.splits) > 1:
                grads_split = grads[self.splits[i-1]:self.splits[i]]
                grads_list.append(grads_split)
                tensor_list.append(parameters_to_vector(grads_split))
            
            if i == len(self.splits) - 1:
                grads_split = grads[self.splits[i]:]
                grads_list.append(grads_split)
                tensor_list.append(parameters_to_vector(grads_split))

        assert len(self.splits)+1 == len(tensor_list) <= self.world_size, str(self.splits) + ', ' + len(tensor_list) + ', '  + str(self.world_size)

        group = self.sync_group

        handler_list = []
        for i in range(len(tensor_list)):
            handler = dist.broadcast(tensor_list[i], i, group=group, async_op=True)
            handler_list.append(handler)

        for handler in handler_list:
            handler.wait()

        for i in range(len(tensor_list)): # all GPUs unpack the new gotten grads
            vector_to_parameters(tensor_list[i], grads_list[i])


class Preconditioner:
    def __init__(self, param: Parameter, config: PreconditioningConfig,
                 block_size: int = _invalid, inverse_exponent: int = _invalid,
                 best_effort_shape_interpretation: bool = False, init_scale: float = 1e-12):
        self.config = config
        self.param = param
        self._transformed_shape = param.shape
        if best_effort_shape_interpretation:
            self._transformed_shape = _merge_small_dims(param.shape, block_size)

        self._partitioner = BlockPartitioner(self._transformed_shape, block_size)
        shapes = self._partitioner.kronecker_factor_shapes()
        ndim = len(self._transformed_shape)
        device = param.device
        if ndim <= 1:
            raise ValueError(f'len(self._transformed_shape) has to be > 1. Got {ndim}.')
        self.statistics = [
            init_scale * torch.eye(s[0], device=device) for s in shapes
        ]
        self.preconditioners = [
            torch.eye(s[0], device=device) for s in shapes
        ]
        self.inverse_exponent = inverse_exponent

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
        exp = self.inverse_exponent
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

        if tensor.shape != self._shape:
            raise ValueError(f'tensor shape ({tensor.shape}) does not match self._shape ({self._shape}).')
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
        if len(partitions) > 1:
            raise ValueError(f'len(partitions) has to be 1. Got {len(partitions)}.')
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
