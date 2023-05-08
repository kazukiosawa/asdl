from typing import List, Union, Any

import numpy as np
import torch
from torch import nn
from torch.cuda import nvtx
import torch.distributed as dist
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from ..core import module_wise_assignments, modules_to_assign
from ..matrices import *
from ..symmatrix import SymMatrix
from ..vector import ParamVector
from ..fisher import LOSS_CROSS_ENTROPY, get_fisher_maker, FisherConfig
from .prec_grad_maker import PreconditionedGradientMaker, PreconditioningConfig

_normalizations = (nn.BatchNorm1d, nn.BatchNorm2d)
_invalid_ema_decay = -1
_invalid_data_size = -1
_module_level_shapes = [SHAPE_LAYER_WISE, SHAPE_KRON, SHAPE_SWIFT_KRON, SHAPE_KFE, SHAPE_UNIT_WISE, SHAPE_DIAG]

__all__ = [
    'NaturalGradientMaker', 'FullNaturalGradientMaker', 'LayerWiseNaturalGradientMaker',
    'KfacGradientMaker', 'EkfacGradientMaker', 'UnitWiseNaturalGradientMaker', 'DiagNaturalGradientMaker', 'EmpNaturalGradientMaker'
]


class NaturalGradientMaker(PreconditionedGradientMaker):
    """GradientMaker for calculating the `Natural Gradient <https://ieeexplore.ieee.org/document/6790500>`_.

    Args:
        model (Module): Target module to calculate gradient
        config (PreconditioningConfig): Configuration for gradient preconditioning
        fisher_type (str, optional): The type of Fisher information matrix estimation:
            *"fisher_exact"* (an exact calculation), *"fisher_mc"* (an estimate by MC samples),
            or *"fisher_emp"* (empirical Fisher). (default: *"fisher_mc"*)
        fisher_shape (Union[str, List[Any]], optional): The shape of Fisher information matrix to be used for
            gradient preconditioning: *"full"*, *"layer_wise"* (layer-wise block-diagonal),
            *"kron"* (`K-FAC <https://arxiv.org/abs/1503.05671>`_), *"unit_wise"* (unit-wise block-diagonal),
            or *"diag"* (diagonal). (default: *"full"*)
        loss_type (str, optional): The type of loss on which the Fisher calculation is based:
            *"cross_entropy"* or *"mse"*. This does not have any effect when :obj:`fisher_type = "fisher_emp"`.
            (default: *"cross_entropy"*)
        scale (float, optional): The scale of Fisher information matrix. (default: 1.)
        n_mc_samples (int, optional): The number of MC samples used when :obj:`fisher_type = "fisher_mc"`.
            (default: 1)
        var (float, optional): The variance of the Gaussian distribution used when
            :obj:`fisher_type = "fisher_mc"` and :obj:`loss_type = "mse"`. (default: 1.)
        seed (int, optional): The random seed set every time Fisher matrix is calculated.
            This is useful to make the sampling deterministic when :obj:`fisher_type = "fisher_mc"`
            (default: None)
    """
    _supported_classes = (nn.Linear, nn.Conv2d, nn.BatchNorm1d, nn.BatchNorm2d,
                          nn.LayerNorm, nn.Embedding)

    def __init__(self, model, config: PreconditioningConfig,
                 fisher_type: str = FISHER_MC, fisher_shape: Union[str, List[Any]] = SHAPE_FULL,
                 loss_type: str = LOSS_CROSS_ENTROPY, scale: float = 1, grad_scale: float = 1,
                 sync_group: dist.ProcessGroup = None, sync_group_ranks: List[int] = None,
                 module_partitions: List[List[nn.Module]] = None,
                 n_mc_samples: int = 1, var: float = 1., seed: int = None):
        from torch.nn.parallel import DistributedDataParallel as DDP
        if isinstance(model, DDP):
            raise TypeError(f'{DDP} is not supported.')
        del DDP
        super().__init__(model, config)
        if isinstance(fisher_shape, str):
            fisher_shape = [fisher_shape]
        if not self.do_accumulate:
            if config.curvature_upd_ratio is not None:
                raise ValueError('curvature_upd_ratio cannot be specified when no curvature accumulation is performed.')

        self.named_modules_for_curvature = []
        self.modules_for_curvature = []
        self.shape_for = {}
        for name, module, shapes in module_wise_assignments(self.module_dict,
                                                            *fisher_shape,
                                                            named=True):
            if len(shapes) != 1:
                raise ValueError(f'Each module has to be assigned one Fisher shape. '
                                 f'{name} is assigned {len(shapes)} shapes.')
            self.modules_for_curvature.append(module)
            self.named_modules_for_curvature.append((name, module))
            self.shape_for[module] = shapes[0]
            self.shape_for[name] = shapes[0]
        self._named_modules_for = {}

        fisher_config = FisherConfig(
            fisher_type=fisher_type,
            fisher_shapes=fisher_shape,
            loss_type=loss_type,
            n_mc_samples=n_mc_samples,
            ignore_modules=config.ignore_modules,
            var=var,
            seed=seed,
        )
        self.fisher_maker = get_fisher_maker(model, fisher_config)
        self.fisher_type = fisher_type
        self.fisher_shape = fisher_shape
        self.scale = scale
        self.grad_scale = grad_scale
        self.module_partitions = module_partitions

        if module_partitions is not None:
            if not dist.is_initialized():
                raise EnvironmentError('torch.distributed has to be initialized when module_partitions is set.')
            world_size = dist.get_world_size(sync_group)
            if len(module_partitions) != world_size:
                raise ValueError(f'Number of partitions has to be world_size. Got {len(module_partitions)}')
            if any(len(module_partitions[0]) != len(module_partitions[i]) for i in range(1, world_size)):
                raise ValueError(f'Number of members in each partition has to be the same. '
                                 f'Got {[len(module_partitions[i]) for i in range(world_size)]}')
            self.partitioned_modules = [m for partition in module_partitions for m in partition]
            self.num_modules_per_partition = len(module_partitions[0])
            assert module_partitions is None, "No longer supported! Distr training is now automatically done if available!"
        elif dist.is_initialized(): #if initialized, we do automatically distr model parallelism
            self.partitioned_modules = []
            self.num_modules_per_partition = None
            self.world_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.partitions = self.get_distr_prec_partition()
        else:
            self.partitioned_modules = []
            self.num_modules_per_partition = None
            self.world_rank = 0
            self.world_size = 1
            self.partitions = self.get_distr_prec_partition()


        if sync_group is not None:
            if sync_group_ranks is None:
                raise ValueError('sync_group_ranks is not set.')
            if sync_group.size() != len(sync_group_ranks):
                raise ValueError(f'sync_group.size() ({sync_group.size()}) does not match '
                                 f'len(sync_group_ranks) ({len(sync_group_ranks)}).')
        self.sync_group = sync_group
        self.sync_group_ranks = sync_group_ranks

        self.curvature_sync_handles = []
        self.grad_sync_handles = []
        self.grads = []
        self.packed_grads = []


    def get_fisher_from_model(self):
        """
        returns a list of all the tensors of the FIM
        """
        tensor_list = []
        for shape in _module_level_shapes:
            if shape == SHAPE_KFE: #TODO KFE atm not supported!
                continue
            keys_list = self._keys_list_from_shape(shape)
            for module in self.modules_for(shape):
                for keys in keys_list:
                    tensor = self.fisher_maker.get_fisher_tensor(module, *keys)
                    if tensor is None:
                        continue
                    tensor_list.append(tensor)
        return tensor_list

    def get_distr_prec_partition(self): 
        """
        this method distributes the workload over the rank by the computational cost of each layer.
        If there are more GPUs than layers. Some will be idle.

        workload is splitted by the type of FIM:
        [[SHAPE_LAYER_WISE], [SHAPE_KRON], [SHAPE_SWIFT_KRON], [SHAPE_KFE], [SHAPE_UNIT_WISE], [SHAPE_DIAG]]

        e.g.
        world_size 8:
        
        ResNet18:
        [[], [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6], [], [], [6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7], []]

        MLP 3 layers:
        [[], [0, 1, 2], [], [], [], []]

        world_size = 100:
        ResNet18:
        [[], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], [], [], [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40], []]

        """
        len_shapes = len(_module_level_shapes)
        num_modules = [0]*len_shapes
        num_params = []
        comp_cost_layers = []
        for enum_shape, shape in enumerate(_module_level_shapes):
            enum_module = None
            for enum_module, module in enumerate(self.modules_for(shape)):
                #if not self.is_module_for_inv_and_precondition(module):
                if module.weight.requires_grad:     
                    comp_cost_layers.append(self.computational_cost(shape, module))
                    p_ = 0
                    for p in module.parameters():
                        p_ += p.numel()
                    
                num_params.append(p_)
                    
            num_modules[enum_shape] = enum_module + 1 if enum_module is not None else 0

        partitions = []
        tot_num_modules = 0
        for shape in range(len_shapes):
            partitions.append([0]*num_modules[shape])
            tot_num_modules += num_modules[shape]

        

        if self.world_size == 1:                    
            return partitions

        elif self.world_size >= tot_num_modules:
            rank = 0
            for shape in range(len_shapes):
                module_ = None
                for module_ in range(num_modules[shape]):
                    partitions[shape][module_] = rank + module_
                rank += module_ + 1 if module_ is not None else 0
            return partitions
            
        else:
            assert tot_num_modules == len(comp_cost_layers)

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


            next_split = split_list[1]
            rank = 0
            split = next_split
            tot_module = 0
            for shape in range(len_shapes):
                module_ = None
                for module_ in range(num_modules[shape]):
                    if tot_module + module_ >= split and rank != self.world_size - 1:
                        rank  += 1
                        if rank != self.world_size - 1:
                            next_split = split_list[rank+1]
                        split = next_split

                    partitions[shape][module_] = rank
                tot_module += module_ + 1 if module_ is not None else 0

            return partitions
            
    def computational_cost(self, shape, module):
        """
        returns the estimated computational cost of a given module for computing the inv and prec
        """
        supported_layers = ['Linear', 'Conv2d', 'BatchNorm1d', 'BatchNorm2d', 'LayerNorm', 'Embedding']
        module_name = str(module).partition('(')[0]
        assert module_name in supported_layers, 'Not supported module: ' + str(module)

        p = module.weight

        if not self.is_module_for_inv_and_precondition(module):
            return 0
        else:
            return p.numel()**0.3
            
                
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
        return not self.do_update_curvature(step)

    def named_modules_for(self, shape):
        if shape not in self._named_modules_for:
            self._named_modules_for[shape] = list(modules_to_assign(self.model,
                                                                    shape,
                                                                    *self.fisher_shape,
                                                                    ignore_modules=self.config.ignore_modules,
                                                                    named=True))
        return self._named_modules_for[shape]

    def modules_for(self, shape):
        return [m for _, m in self.named_modules_for(shape)]

    def parameters_for(self, shape):
        for module in self.modules_for(shape):
            for p in module.parameters():
                if p.requires_grad:
                    yield p

    @property
    def _fisher_attr(self):
        return self.fisher_maker.config.fisher_attr

    def _get_module_fisher(self, module, postfix=None):
        if postfix is None:
            attr = self._fisher_attr
        else:
            attr = f'{self._fisher_attr}_{postfix}'
        fisher = getattr(module, attr, None)
        return fisher

    def _set_module_fisher(self, module, fisher, postfix=None):
        if postfix is None:
            attr = self._fisher_attr
        else:
            attr = f'{self._fisher_attr}_{postfix}'
        setattr(module, attr, fisher)

    def _get_full_fisher(self):
        return self._get_module_fisher(self.model)

    def _get_module_symmatrix(self, module, shape, postfix=None) -> SymMatrix:
        fisher = self._get_module_fisher(module, postfix)
        if fisher is None:
            return None
        if shape in [SHAPE_FULL, SHAPE_LAYER_WISE]:
            return fisher
        elif shape in [SHAPE_KRON, SHAPE_SWIFT_KRON]:
            return fisher.kron
        elif shape == SHAPE_KFE:
            return fisher.kfe
        elif shape == SHAPE_UNIT_WISE:
            return fisher.unit
        elif shape == SHAPE_DIAG:
            return fisher.diag
        else:
            raise ValueError(f'Invalid shape: {shape}.')

    def _scale_fisher(self, scale):
        for shape in _module_level_shapes:
            for module in self.modules_for(shape):
                matrix = self._get_module_symmatrix(module, shape)
                if matrix is not None:
                    matrix.mul_(scale)
        fisher = self._get_full_fisher()
        if fisher is not None:
            fisher.mul_(scale)

    @property
    def do_accumulate(self):
        return self.config.ema_decay != _invalid_ema_decay

    def update_curvature(self):
        config = self.config
        fisher_maker = self.fisher_maker
        scale = self.scale

        ema_decay = config.ema_decay
        if ema_decay != _invalid_ema_decay:
            scale *= ema_decay
            self._scale_fisher(1 - ema_decay)

        self.delegate_forward_and_backward(fisher_maker,
                                           data_size=self.config.data_size,
                                           scale=scale,
                                           accumulate=self.do_accumulate,
                                           calc_loss_grad=True,
                                           calc_inv=not self.do_accumulate,
                                           damping=self.config.damping
                                           )

        if self.do_accumulate and self.world_size > 1:         
            self.reduce_scatter_curvature()

    def update_preconditioner(self, damping=None, module_name=None, kron=None, zero_curvature=False, partition_aware=False):
        if not self.do_accumulate:
            return

        if kron is None:
            kron = ['A', 'B']
        if damping is None:
            damping = self.config.damping

        for enum_shape, shape in enumerate(_module_level_shapes):
            for enum_module, name_module in enumerate(self.named_modules_for(shape)):
                name, module = name_module
                if module_name is not None:
                    if name != module_name:
                        continue
                    if partition_aware and module in self.partitioned_modules:
                        partition_id = self.partitioned_modules.index(module) // self.num_modules_per_partition
                        module_id_in_partition = self.module_partitions[partition_id].index(module)
                        rank_in_group = dist.get_rank(self.sync_group)
                        modified_partition_id = (partition_id + rank_in_group) % len(self.module_partitions)
                        module = self.module_partitions[modified_partition_id][module_id_in_partition]

                if self.world_rank == self.partitions[enum_shape][enum_module]:
                    matrix = self._get_module_symmatrix(module, shape)
                    if matrix is None:
                        continue

                    event = f'inv_{shape}'
                    if shape in [SHAPE_KRON, SHAPE_SWIFT_KRON]:
                        for A_or_B in kron:
                            event += f'_{A_or_B}'

                    with nvtx.range(event + "_" + name):
                        if self.is_module_for_inv_and_precondition(module):
                            if shape in [SHAPE_KRON, SHAPE_SWIFT_KRON]:
                                matrix.update_inv(damping, calc_A_inv='A' in kron, calc_B_inv='B' in kron)
                            else:
                                matrix.update_inv(damping)

                        if zero_curvature:
                            with torch.no_grad():
                                if shape in [SHAPE_KRON, SHAPE_SWIFT_KRON]:
                                    if 'A' in kron:
                                        matrix.A.mul_(0)
                                    if 'B' in kron:
                                        matrix.B.mul_(0)
                                else:
                                    matrix.mul_(0)

                if module_name is not None:
                    break

        fisher = self._get_full_fisher()
        if fisher is not None:
            fisher.update_inv(damping)
            if zero_curvature:
                with torch.no_grad():
                    fisher.mul_(0)

    def precondition(self, vectors: ParamVector = None, grad_scale=None, use_inv=True):
        if grad_scale is None:
            grad_scale = self.grad_scale
        for enum_shape, shape in enumerate(_module_level_shapes):
            for enum_module, module in enumerate(self.modules_for(shape)):
                if self.world_rank == self.partitions[enum_shape][enum_module]:
                    if not self.is_module_for_inv_and_precondition(module):
                        continue
                    self._precondition_module(module, shape, vectors, grad_scale=grad_scale, use_inv=use_inv)

        params = [p for p in self.parameters_for(SHAPE_FULL)]
        if len(params) > 0:
            fisher = self._get_full_fisher()
            if fisher is None:
                raise ValueError(f'Fisher of shape {SHAPE_FULL} has not been calculated.')
            if vectors is None:
                vectors = ParamVector(params, [p.grad for p in params])
            if vectors is None:
                raise ValueError('gradient has not been calculated.')
            if grad_scale != 1:
                vectors.mul_(grad_scale)
            fisher.mvp(vectors=vectors, use_inv=use_inv, inplace=True)

        # all_reduce all the grads after preconditioning them. (Basic DDP. Will be changed when DP & MP)
        if self.world_size > 1:
            if self.do_accumulate:
                self.all_gather_or_reduce_grad()
            else:
                self.all_reduce_all_grad(async_op=False)

    def _precondition_module(self, module, shape=None, vectors: ParamVector = None,
                            vec_weight: torch.Tensor = None, vec_bias: torch.Tensor = None,
                            grad_scale=None, use_inv=True):
        if grad_scale is None:
            grad_scale = self.grad_scale
        if shape is None:
            for s in _module_level_shapes:
                if module in self.modules_for(s):
                    shape = s
                    break
        if vectors is not None:
            vec_weight = vectors.get_vector_by_param(module.weight, None)
            vec_bias = vectors.get_vector_by_param(module.bias, None)
        if shape is None:
            raise ValueError(f'No shape is assigned to module: {module}.')
        matrix = self._get_module_symmatrix(module, shape)
        if matrix is None:
            raise ValueError(f'Matrix of shape {shape} for module {module} has not been calculated.')
        if vec_weight is None and module.weight.requires_grad:
            vec_weight = module.weight.grad
        if vec_weight is None:
            raise ValueError(f'weight gradient for module {module} has not been calculated.')
        if _bias_requires_grad(module):
            if vec_bias is None:
                vec_bias = module.bias.grad
            if vec_bias is None:
                raise ValueError(f'bias gradient for module {module} has not been calculated.')
        if grad_scale != 1:
            vec_weight.data.mul_(grad_scale)
            if vec_bias is not None:
                vec_bias.data.mul_(grad_scale)
        if not use_inv or matrix.has_inv:
            kwargs = dict(vec_weight=vec_weight, vec_bias=vec_bias, use_inv=use_inv, inplace=True)
            if shape == SHAPE_KFE:
                kwargs['eps'] = self.config.damping
            matrix.mvp(**kwargs)

    def is_module_for_inv_and_precondition(self, module: nn.Module):
        if module not in self.modules_for_curvature:
            return False
        module_partitions = self.module_partitions
        if module_partitions is None:
            return True
        if module not in self.partitioned_modules:
            return True
        else:
            rank = dist.get_rank(self.sync_group)
            return module in module_partitions[rank]

    def sync_curvature(self, module_name=None, kron=None, diag=None, with_grad=False, enabled=True, async_op=False):
        if not enabled:
            return
        handles = []
        if self.module_partitions is not None:
            if module_name is not None:
                handles += self.reduce_curvature(module_name, kron=kron, diag=diag, with_grad=with_grad)
            else:
                handles += self.reduce_scatter_curvature_module_partitions(kron=kron, diag=diag, with_grad=with_grad)
        handles += self.all_reduce_undivided_curvature(module_name=module_name, kron=kron, diag=diag, with_grad=with_grad)
        if async_op:
            self.curvature_sync_handles += handles
        else:
            for handle in handles:
                handle.wait()

    def sync_grad_pre_precondition(self, enabled=True, async_op=False):
        if not enabled:
            return
        if self.module_partitions is not None:
            self.reduce_scatter_grad(async_op=async_op)
        self.all_reduce_undivided_grad(async_op=async_op)

    def sync_grad_post_precondition(self, enabled=True, async_op=False):
        if not enabled:
            return
        if self.module_partitions is not None:
            self.all_gather_grad(async_op=async_op)
        self.all_reduce_no_curvature_grad(async_op=async_op)

    def reduce_scatter_curvature_module_partitions(self, kron=None, diag=None, with_grad=False):
        module_partitions = self.module_partitions
        if module_partitions is None:
            raise ValueError('module_partitions is not set.')
        handles = []
        for shape in _module_level_shapes:
            keys_list = self._keys_list_from_shape(shape, kron=kron, diag=diag)
            for keys in keys_list:
                handles += self.fisher_maker.reduce_scatter_fisher(module_partitions,
                                                                   *keys,
                                                                   with_grad=with_grad,
                                                                   group=self.sync_group,
                                                                   async_op=True)
        return handles

    
    @nvtx.range('reduce_scatter_curvature')
    def reduce_scatter_curvature(self, kron=None, diag=None, with_grad=True):
        assert kron is None and diag is None
        group = self.sync_group

        rank = 0
        tensor_list = []
        for enum_shape, shape in enumerate(_module_level_shapes):
            keys_list = self._keys_list_from_shape(shape)
            for enum_module, name_module in enumerate(self.named_modules_for(shape)):

                # we will send when we reached the end of the partitioned rank
                if rank != self.partitions[enum_shape][enum_module]:
                    vector = parameters_to_vector(tensor_list)
                    dist.reduce(vector, rank, op=dist.ReduceOp.AVG, group=group)
                    if self.world_rank == rank:
                        vector_to_parameters(vector, tensor_list)
                    rank += 1
                    assert rank == self.partitions[enum_shape][enum_module]
                    tensor_list = []

                name, module = name_module

                for keys in keys_list:
                    tensor = self.fisher_maker.get_fisher_tensor(module, *keys)
                    
                    if tensor is None:
                        continue
                    assert tensor.is_cuda
                    tensor_list.append(tensor)
                if with_grad:
                    for p in module.parameters():
                        if p.requires_grad and p.grad is not None:
                            tensor_list.append(p.grad)

        #last reduce for last rank
        vector = parameters_to_vector(tensor_list)
        dist.reduce(vector, rank, op=dist.ReduceOp.AVG, group=group)
        if self.world_rank == rank:
            vector_to_parameters(vector, tensor_list)

        assert rank < self.world_size

    def reduce_curvature(self, module_name, kron=None, diag=None, with_grad=False):
        module_partitions = self.module_partitions
        if module_partitions is None:
            raise ValueError('module_partitions is not set.')
        try:
            module = next(m for name, m in self.named_modules_for_curvature if name == module_name)
            if module not in self.partitioned_modules:
                return []
            dst = next(i for i, partition in enumerate(module_partitions) if module in partition)
            if self.sync_group is not None:
                dst = self.sync_group_ranks[dst]
        except StopIteration:
            return []
        keys_list = self._keys_list_from_shape(self.shape_for[module], kron=kron, diag=diag)
        handles = []
        for keys in keys_list:
            handles += self.fisher_maker.reduce_fisher([module],
                                                       *keys,
                                                       all_reduce=False,
                                                       dst=dst,
                                                       with_grad=with_grad,
                                                       group=self.sync_group,
                                                       async_op=True)
        return handles

    def all_reduce_undivided_curvature(self, module_name=None, kron=None, diag=None, with_grad=False):
        modules = []
        for name, module in self.named_modules_for_curvature:
            if module in self.partitioned_modules:
                continue
            if module_name is not None and name != module_name:
                continue
            modules.append(module)
        handles = []
        for shape in _module_level_shapes:
            keys_list = self._keys_list_from_shape(shape, kron=kron, diag=diag)
            for keys in keys_list:
                handles += self.fisher_maker.reduce_fisher(modules,
                                                           *keys,
                                                           all_reduce=True,
                                                           with_grad=with_grad,
                                                           group=self.sync_group,
                                                           async_op=True)
        return handles

    @staticmethod
    def _keys_list_from_shape(shape, kron=None, diag=None):
        if shape == SHAPE_FULL:
            return [['data']]
        elif shape == SHAPE_LAYER_WISE:
            return [['data']]
        elif shape in [SHAPE_KRON, SHAPE_SWIFT_KRON]:
            if kron is None:
                kron = ['A', 'B']
            if any(A_or_B not in ['A', 'B'] for A_or_B in kron):
                raise ValueError(f'kron has to be a list of "A" or "B". Got {kron}')
            return [['kron', A_or_B] for A_or_B in kron]
        elif shape == SHAPE_UNIT_WISE:
            return [['unit', 'data']]
        elif shape == SHAPE_DIAG:
            if diag is None:
                diag = ['weight', 'bias']
            if any(w_or_b not in ['weight', 'bias'] for w_or_b in diag):
                raise ValueError(f'diag has to be a list of "weight" or "bias". Got {diag}')
            return [['diag', w_or_b] for w_or_b in diag]

    @nvtx.range('reduce_scatter_grad')
    def reduce_scatter_grad(self, async_op=False):
        self._scatter_or_gather_grad('scatter', async_op=async_op)

    @nvtx.range('all_gather_grad')
    def all_gather_or_reduce_grad(self):
        group = self.sync_group

        rank = 0
        tensor_list = []
        tensor_list_not_prec = []
        for enum_shape, shape in enumerate(_module_level_shapes):
            for enum_module, module in enumerate(self.modules_for(shape)):
                
                # we will broadcast when we reached the end of the partitioned rank
                if rank != self.partitions[enum_shape][enum_module]:
                    if len(tensor_list) > 0:
                        vector = parameters_to_vector(tensor_list)
                        dist.broadcast(vector, rank, group=group)
                        vector_to_parameters(vector, tensor_list)
                    rank += 1
                    assert rank == self.partitions[enum_shape][enum_module]
                    tensor_list = []

                if self.is_module_for_inv_and_precondition(module):
                    for p in module.parameters():
                        if p.requires_grad and p.grad is not None:
                            tensor_list.append(p.grad)
                else:
                    for p in module.parameters():
                        if p.requires_grad and p.grad is not None:
                            tensor_list_not_prec.append(p.grad)

        #last broadcast of last rank
        if len(tensor_list) > 0:
            vector = parameters_to_vector(tensor_list)
            dist.broadcast(vector, rank, group=group)
            vector_to_parameters(vector, tensor_list)

        assert rank < self.world_size

        # all_reduce all grads, which do not need prec
        if len(tensor_list_not_prec) > 0:
            vector = parameters_to_vector(tensor_list_not_prec)
            dist.all_reduce(vector, op=dist.ReduceOp.AVG, group=group)
            vector_to_parameters(vector, tensor_list_not_prec)

    def all_gather_grad(self, async_op=False):
        self._scatter_or_gather_grad('gather', async_op=async_op)

    def _scatter_or_gather_grad(self, scatter_or_gather, async_op=False):
        if not dist.is_initialized():
            raise EnvironmentError('torch.distributed is not initialized.')
        group = self.sync_group
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)
        module_partitions = self.module_partitions
        if module_partitions is None:
            raise ValueError('module_partitions is not set.')
        if len(module_partitions) != world_size:
            raise ValueError(f'Number of partitions has to be world_size. Got {len(module_partitions)}')
        if any(len(module_partitions[0]) != len(module_partitions[i]) for i in range(1, world_size)):
            raise ValueError(f'Number of members in each partition has to be the same. '
                             f'Got {[len(module_partitions[i]) for i in range(world_size)]}')
        num_modules_per_partition = len(module_partitions[0])
        for i in range(num_modules_per_partition):
            tensor_list = []
            grads_list = []
            for j in range(world_size):
                grads = [p.grad for p in module_partitions[j][i].parameters() if p.requires_grad and p.grad is not None]
                grads_list.append(grads)
                tensor_list.append(parameters_to_vector(grads))
            if scatter_or_gather == 'scatter':
                handle = dist.reduce_scatter(tensor_list[rank], tensor_list, group=group, async_op=async_op)
                if async_op:
                    self.grad_sync_handles.append(handle)
                    self.grads.append(grads_list[rank])
                    self.packed_grads.append(tensor_list[rank])
                else:
                    vector_to_parameters(tensor_list[rank], grads_list[rank])
            else:
                handle = dist.all_gather(tensor_list, tensor_list[rank], group=group, async_op=async_op)
                if async_op:
                    self.grad_sync_handles.append(handle)
                    self.grads.append([grads_list[j] for j in range(world_size)])
                    self.packed_grads.append([tensor_list[j] for j in range(world_size)])
                else:
                    for j in range(world_size):
                        vector_to_parameters(tensor_list[j], grads_list[j])

    def all_reduce_undivided_grad(self, async_op=False):
        if not dist.is_initialized():
            raise EnvironmentError('torch.distributed is not initialized.')
        module_list = nn.ModuleList([m for m in self.modules_for_curvature if m not in self.partitioned_modules])
        self._all_reduce_grad(module_list, async_op=async_op)

    def all_reduce_no_curvature_grad(self, async_op=False):
        module_list = nn.ModuleList([m for m in self.model.modules()
                                     if len(list(m.children())) == 0 and m not in self.modules_for_curvature])
        self._all_reduce_grad(module_list, async_op=async_op)

    @nvtx.range('all_reduce_all_grad')
    def all_reduce_all_grad(self, async_op=False):
        module_list = nn.ModuleList([m for m in self.model.modules()])
        self._all_reduce_grad(module_list, async_op=async_op, op=dist.ReduceOp.AVG)

    def _all_reduce_grad(self, module: nn.Module, async_op=False, op=dist.ReduceOp.SUM):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        if len(grads) == 0:
            return
        packed_tensor = parameters_to_vector(grads)
        handle = dist.all_reduce(packed_tensor, op=op, group=self.sync_group, async_op=async_op)
        if async_op:
            self.grad_sync_handles.append(handle)
            self.grads.append(grads)
            self.packed_grads.append(packed_tensor)
        else:
            vector_to_parameters(packed_tensor, grads)

    def wait_all_curvature_sync(self):
        for _ in range(len(self.curvature_sync_handles)):
            self.curvature_sync_handles.pop(0).wait()

    def wait_all_grad_sync(self):
        for _ in range(len(self.grad_sync_handles)):
            self.grad_sync_handles.pop(0).wait()
            grads = self.grads.pop(0)
            packed_grads = self.packed_grads.pop(0)
            if isinstance(grads, list) and isinstance(grads[0], list):
                if not isinstance(packed_grads, list):
                    raise TypeError(f'packed_grads has to be list. Got {type(packed_grads)}.')
                for p, g in zip(packed_grads, grads):
                    vector_to_parameters(p, g)
            else:
                vector_to_parameters(packed_grads, grads)


class FullNaturalGradientMaker(NaturalGradientMaker):
    def __init__(self, model, config: PreconditioningConfig, *args, **kwargs):
        super().__init__(model, config, *args, **kwargs, fisher_shape=SHAPE_FULL)


class LayerWiseNaturalGradientMaker(NaturalGradientMaker):
    def __init__(self, model, config: PreconditioningConfig, *args, **kwargs):
        super().__init__(model, config, *args, **kwargs, fisher_shape=SHAPE_LAYER_WISE)


class KfacGradientMaker(NaturalGradientMaker):
    """GradientMaker for calculating the preconditioned gradient by `K-FAC <https://arxiv.org/abs/1503.05671>`_.
    A wrapper class of :ref:`NaturalGradientMaker <natural_gradient_maker>`.

    Args:
        model (Module): Target module to calculate gradient
        config (PreconditioningConfig): Configuration for gradient preconditioning
        fisher_type (str, optional): The type of Fisher information matrix estimation:
            *"fisher_exact"* (an exact calculation), *"fisher_mc"* (an estimate by MC samples),
            or *"fisher_emp"* (empirical Fisher). (default: *"fisher_mc"*)
        loss_type (str, optional): The type of loss on which the Fisher calculation is based:
            *"cross_entropy"* or *"mse"*. This does not have any effect when :obj:`fisher_type = "fisher_emp"`.
            (default: *"cross_entropy"*)
        scale (float, optional): The scale of Fisher information matrix. (default: 1.)
        n_mc_samples (int, optional): The number of MC samples used when :obj:`fisher_type = "fisher_mc"`.
            (default: 1)
        var (float, optional): The variance of the Gaussian distribution used when
            :obj:`fisher_type = "fisher_mc"` and :obj:`loss_type = "mse"`. (default: 1.)
        seed (int, optional): The random seed set every time Fisher matrix is calculated.
            This is useful to make the sampling deterministic when :obj:`fisher_type = "fisher_mc"`
            (default: None)
        swift (bool, optional): If True, KfacGradientMaker calculates preconditioned gradients by
            `SKFAC <https://ieeexplore.ieee.org/document/9578481>`_ algorithm. (default: False)
    """
    def __init__(self, model, config: PreconditioningConfig,
                 fisher_type: str = FISHER_MC, loss_type: str = LOSS_CROSS_ENTROPY,
                 scale: float = 1, grad_scale: float = 1,
                 n_mc_samples: int = 1, var: float = 1., seed: int = None, swift=False):
        fisher_shape = [SHAPE_SWIFT_KRON if swift else SHAPE_KRON,
                        (nn.BatchNorm1d, SHAPE_UNIT_WISE),
                        (nn.BatchNorm2d, SHAPE_UNIT_WISE),
                        (nn.LayerNorm, SHAPE_UNIT_WISE)]
        super().__init__(model, config,
                         fisher_type=fisher_type,
                         fisher_shape=fisher_shape,
                         loss_type=loss_type,
                         scale=scale,
                         grad_scale=grad_scale,
                         n_mc_samples=n_mc_samples,
                         var=var,
                         seed=seed
                         )


class EkfacGradientMaker(NaturalGradientMaker):
    def __init__(self, model, config: PreconditioningConfig, *args, **kwargs):
        super().__init__(model, config, *args, **kwargs, fisher_shape=SHAPE_KFE)
        if self.fisher_type != FISHER_EMP:
            raise ValueError(f'{EkfacGradientMaker} supports only {FISHER_EMP}.')

    def _update_preconditioner(self, *args, **kwargs):
        pass

    def _precondition(self, vectors: ParamVector = None, grad_scale=None, use_inv=False):
        if use_inv:
            raise ValueError('EKFAC does not calculate the inverse matrix.')
        super().precondition(vectors=vectors, grad_scale=grad_scale, use_inv=False)


class UnitWiseNaturalGradientMaker(NaturalGradientMaker):
    def __init__(self, model, config: PreconditioningConfig, *args, **kwargs):
        super().__init__(model, config, *args, **kwargs, fisher_shape=SHAPE_UNIT_WISE)


class DiagNaturalGradientMaker(NaturalGradientMaker):
    def __init__(self, model, config: PreconditioningConfig, *args, **kwargs):
        super().__init__(model, config, *args, **kwargs, fisher_shape=SHAPE_DIAG)


class EmpNaturalGradientMaker(NaturalGradientMaker):
    def __init__(self, model, config: PreconditioningConfig, *args, **kwargs):
        super().__init__(model, config, *args, **kwargs, fisher_type=FISHER_EMP)


def _bias_requires_grad(module):
    return hasattr(module, 'bias') \
           and module.bias is not None \
           and module.bias.requires_grad
