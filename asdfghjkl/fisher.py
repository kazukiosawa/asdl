from typing import List, Union, Any, Tuple
from dataclasses import dataclass
import numpy as np

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from .core import no_centered_cov
from .operations import OperationContext
from .utils import skip_param_grad, has_reduction
from .grad_maker import GradientMaker, LOSS_CROSS_ENTROPY, LOSS_MSE
from .matrices import *
from .vector import ParamVector, reduce_vectors
from .mvp import power_method, stochastic_lanczos_quadrature, conjugate_gradient_method, quadratic_form
from .symmatrix import SymMatrix

__all__ = [
    'FisherConfig',
    'get_fisher_maker',
]

_supported_types = [FISHER_EXACT, FISHER_MC, FISHER_EMP]
_invalid_data_size = -1


@dataclass
class FisherConfig:
    fisher_type: str
    fisher_shapes: List[Any]
    loss_type: str = None
    n_mc_samples: int = 1
    var: float = 1.
    seed: int = None
    fisher_attr: str = 'fisher'
    fvp_attr: str = 'fvp'
    ignore_modules: List[Any] = None
    data_size: int = _invalid_data_size
    scale: float = 1.
    is_distributed: bool = False
    all_reduce: bool = False
    is_master: bool = True


class FisherMaker(GradientMaker):
    def __init__(self, model, config):
        super().__init__(model)
        self.config: FisherConfig = config

    def zero_fisher(self, fvp=False):
        attr = self.config.fvp_attr if fvp else self.config.fisher_attr
        for module in self.model.modules():
            if hasattr(module, attr):
                delattr(module, attr)

    @property
    def is_fisher_emp(self):
        return False

    @property
    def do_local_accumulate(self) -> bool:
        raise NotImplementedError

    def forward_and_backward(self,
                             data_size=_invalid_data_size,
                             scale=None,
                             accumulate=False,
                             calc_loss_grad=False,
                             calc_inv=False,
                             fvp=False,
                             damping=None,
                             vec: ParamVector = None) -> Union[Tuple[Any, Tensor], Any]:
        assert not (accumulate and calc_inv), 'accumulate and calc_inv cannot be True at the same time.'
        assert not (fvp and calc_inv), 'fvp and calc_inv cannot be True at the same time.'
        model = self.model
        config = self.config
        fisher_shapes = config.fisher_shapes
        if isinstance(fisher_shapes, str):
            fisher_shapes = [fisher_shapes]
        ignore_modules = config.ignore_modules
        seed = config.seed
        if data_size == _invalid_data_size:
            data_size = config.data_size  # refer config value (default: _invalid_data_size)
        assert data_size != _invalid_data_size, 'data_size is not specified.'
        if scale is None:
            scale = config.scale  # refer config value (default: 1)
        scale /= data_size

        if not accumulate:
            # set Fisher/FVP zero
            self.zero_fisher(fvp=fvp)

        if seed is not None:
            torch.random.manual_seed(seed)

        calc_loss_grad_with_fisher = calc_loss_grad and self.is_fisher_emp
        calc_loss_grad_after_fisher = calc_loss_grad and not self.is_fisher_emp
        calc_inv_with_fisher = calc_inv and not self.do_local_accumulate
        calc_inv_after_fisher = calc_inv and self.do_local_accumulate

        kwargs = dict(ignore_modules=ignore_modules, cvp=fvp, vectors=vec, calc_inv=calc_inv_with_fisher)
        with no_centered_cov(model, fisher_shapes, **kwargs) as cxt:
            if accumulate:
                self._register_fisher(cxt)
            if damping is not None:
                cxt.set_damping(damping)
            cxt.set_cov_scale(scale)

            self.call_model()
            loss = None
            if self.is_fisher_emp or calc_loss_grad:
                self.call_loss()
                loss = self._loss

            def closure(nll_expr, retain_graph=False):
                cxt.clear_batch_grads()
                with skip_param_grad(model, disable=calc_loss_grad_with_fisher):
                    nll_expr().backward(retain_graph=retain_graph or calc_loss_grad_after_fisher)
                if fvp:
                    cxt.calc_full_cvp(model, scale=scale)
                else:
                    cxt.calc_full_cov(model, scale=scale, calc_inv=calc_inv_with_fisher, damping=damping)

            if self.is_fisher_emp:
                closure(lambda: loss)
            else:
                self._fisher_loop(closure)

            self._extract_fisher(cxt)

        if calc_inv_after_fisher:
            self.replace_fisher_with_inv(damping)
        if calc_loss_grad_after_fisher:
            loss.backward()

        if calc_loss_grad:
            # divide gradients by data size
            # (every loss function returns the sum of loss, not the average)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.div_(data_size)
            loss.div_(data_size)

        return self._model_output, loss

    def _register_fisher(self, cxt: OperationContext):
        model = self.model
        attr = self.config.fisher_attr
        for module in model.modules():
            fisher = getattr(module, attr, None)
            if fisher is not None:
                cxt.register_symmatrix(module, fisher)
        fisher = getattr(model, attr, None)
        if fisher is not None:
            cxt.register_full_symmatrix(model, fisher)

    def _extract_fisher(self, cxt: OperationContext):
        model = self.model
        fisher_attr = self.config.fisher_attr
        fvp_attr = self.config.fvp_attr

        def extract_if_not_exist(_module, func, attr):
            if getattr(_module, attr, None) is None:  # not exist
                fisher_or_fvp = func(_module, pop=True)  # extract
                if fisher_or_fvp is not None:
                    setattr(_module, attr, fisher_or_fvp)

        for module in model.modules():
            extract_if_not_exist(module, cxt.cov_symmatrix, fisher_attr)
            extract_if_not_exist(module, cxt.cvp_paramvector, fvp_attr)
        extract_if_not_exist(model, cxt.full_cov_symmatrix, fisher_attr)
        extract_if_not_exist(model, cxt.full_cvp_paramvector, fvp_attr)

    def replace_fisher_with_inv(self, damping):
        model = self.model
        attr = self.config.fisher_attr
        for module in model.modules():
            fisher: SymMatrix = getattr(module, attr, None)
            if fisher is not None:
                fisher.update_inv(damping=damping, replace=True)
        fisher: SymMatrix = getattr(model, attr, None)
        if fisher is not None:
            fisher.update_inv(damping=damping, replace=True)

    def _call_loss_fn(self) -> Tensor:
        assert has_reduction(self._loss_fn), 'loss_fn has to have "reduction" option'
        if isinstance(self._loss_fn, nn.Module):
            self._loss_fn.reduction = 'sum'
        else:
            self._loss_fn_kwargs['reduction'] = 'sum'
        args, kwargs = self._get_mapped_loss_fn_args_kwargs()
        return self._loss_fn(*args, **kwargs)

    def _fisher_loop(self, closure):
        raise NotImplementedError

    def get_fisher_tensor(self, module: nn.Module, *keys) -> Union[torch.Tensor, None]:
        fisher = getattr(module, self.config.fisher_attr, None)
        if fisher is None:
            return None
        data = fisher
        for key in keys:
            data = getattr(data, key, None)
        if data is not None:
            assert isinstance(data, torch.Tensor)
        return data

    def reduce_scatter_fisher(self,
                              module_partitions: List[List[torch.nn.Module]],
                              *keys,
                              with_grad=False,
                              group: dist.ProcessGroup = None,
                              async_op=False):
        assert dist.is_initialized()
        assert torch.cuda.is_available()
        assert dist.get_backend(group) == dist.Backend.NCCL
        world_size = dist.get_world_size(group)
        assert len(module_partitions) == world_size
        assert all(len(module_partitions[0]) == len(module_partitions[i]) for i in range(1, world_size))
        tensor_partitions = []
        for module_list in module_partitions:
            tensor_list = []
            for module in module_list:
                tensor = self.get_fisher_tensor(module, *keys)
                if tensor is None:
                    continue
                assert tensor.is_cuda
                tensor_list.append(tensor)
                if with_grad:
                    for p in module.parameters():
                        if p.requires_grad and p.grad is not None:
                            tensor_list.append(p.grad)
            tensor_partitions.append(tensor_list)
        num_tensors_per_partition = len(tensor_partitions[0])
        assert all(len(tensor_partitions[i]) == num_tensors_per_partition for i in range(1, world_size))
        handles = []
        for i in range(num_tensors_per_partition):
            input_list = [tensor_list[i] for tensor_list in tensor_partitions]
            output = input_list[dist.get_rank(group)]
            handles.append(dist.reduce_scatter(output, input_list, group=group, async_op=async_op))
        return handles

    def reduce_fisher(self,
                      modules,
                      *keys,
                      all_reduce=True,
                      with_grad=False,
                      dst=0,
                      group: dist.ProcessGroup = None,
                      async_op=False):
        assert dist.is_initialized()
        tensor_list = []
        for module in modules:
            tensor = self.get_fisher_tensor(module, *keys)
            if tensor is None:
                continue
            tensor_list.append(tensor)
            if with_grad:
                for p in module.parameters():
                    if p.requires_grad and p.grad is not None:
                        tensor_list.append(p.grad)
        handles = []
        for tensor in tensor_list:
            if all_reduce:
                handles.append(dist.all_reduce(tensor, group=group, async_op=async_op))
            else:
                handles.append(dist.reduce(tensor, dst=dst, group=group, async_op=async_op))
        return handles

    def reduce_fvp(self, fisher_shape, is_master=True, all_reduce=False):
        v = self.load_fvp(fisher_shape)
        v = reduce_vectors(v, is_master, all_reduce)
        attr = self.config.fvp_attr
        if fisher_shape == SHAPE_FULL:
            setattr(self.model, attr, v)
        else:
            for module in self.model.modules():
                if hasattr(module, attr):
                    setattr(module, attr, v.get_vectors_by_module(module))

    def load_fvp(self, fisher_shape: str) -> ParamVector:
        if fisher_shape == SHAPE_FULL:
            v = getattr(self.model, self.config.fvp_attr, None)
            if v is None:
                return None
            return v.copy()
        else:
            rst = None
            for module in self.model.modules():
                if module == self.model:
                    continue
                v = getattr(module, self.config.fvp_attr, None)
                if v is not None:
                    if rst is None:
                        rst = v.copy()
                    else:
                        rst.extend(v.copy())
            return rst

    def _get_fvp_fn(self):
        def fvp_fn(vec: ParamVector) -> ParamVector:
            self.forward_and_backward(fvp=True, vec=vec)
            return self.load_fvp(self.config.fisher_shapes[0])
        return fvp_fn

    def fisher_eig(self,
                   top_n=1,
                   max_iters=100,
                   tol=1e-7,
                   is_distributed=False,
                   print_progress=False
                   ):
        # for making MC samplings at each iteration deterministic
        random_seed = torch.rand(1) * 100 if self.config.fisher_type == FISHER_MC else None

        eigvals, eigvecs = power_method(self._get_fvp_fn(),
                                        self.model,
                                        top_n=top_n,
                                        max_iters=max_iters,
                                        tol=tol,
                                        is_distributed=is_distributed,
                                        print_progress=print_progress,
                                        random_seed=random_seed
                                        )

        return eigvals, eigvecs

    def fisher_esd(self,
                   n_v=1,
                   num_iter=100,
                   num_bins=10000,
                   sigma_squared=1e-5,
                   overhead=None,
                   is_distributed=False
                   ):
        # for making MC samplings at each iteration deterministic
        random_seed = torch.rand(1) * 100 if self.config.fisher_type == FISHER_MC else None

        eigvals, weights = stochastic_lanczos_quadrature(self._get_fvp_fn(),
                                                         self.model,
                                                         n_v=n_v,
                                                         num_iter=num_iter,
                                                         is_distributed=is_distributed,
                                                         random_seed=random_seed
                                                         )
        # referenced from https://github.com/amirgholami/PyHessian/blob/master/density_plot.py
        eigvals = np.array(eigvals)
        weights = np.array(weights)

        lambda_max = np.mean(np.max(eigvals, axis=1), axis=0)
        lambda_min = np.mean(np.min(eigvals, axis=1), axis=0)

        sigma_squared = sigma_squared * max(1, (lambda_max - lambda_min))
        if overhead is None:
            overhead = np.sqrt(sigma_squared)

        range_max = lambda_max + overhead
        range_min = np.maximum(0., lambda_min - overhead)

        grids = np.linspace(range_min, range_max, num=num_bins)

        density_output = np.zeros((n_v, num_bins))

        for i in range(n_v):
            for j in range(num_bins):
                x = grids[j]
                tmp_result = np.exp(-(x - eigvals[i, :])**2 / (2.0 * sigma_squared)) / np.sqrt(2 * np.pi * sigma_squared)
                density_output[i, j] = np.sum(tmp_result * weights[i, :])
        density = np.mean(density_output, axis=0)
        normalization = np.sum(density) * (grids[1] - grids[0])
        density = density / normalization
        return density, grids

    def fisher_free(self,
                    b=None,
                    init_x=None,
                    damping=1e-3,
                    max_iters=None,
                    tol=1e-8,
                    preconditioner=None,
                    print_progress=False,
                    random_seed=None
                    ) -> ParamVector:
        if b is None:
            grads = {p: p.grad for p in self.model.parameters() if p.requires_grad}
            b = ParamVector(grads.keys(), grads.values())

        # for making MC samplings at each iteration deterministic
        if self.config.fisher_type == FISHER_MC and random_seed is None:
            random_seed = int(torch.rand(1) * 100)

        return conjugate_gradient_method(self._get_fvp_fn(),
                                         b,
                                         init_x=init_x,
                                         damping=damping,
                                         max_iters=max_iters,
                                         tol=tol,
                                         preconditioner=preconditioner,
                                         print_progress=print_progress,
                                         random_seed=random_seed)

    def fisher_quadratic_form(self, vec: ParamVector = None):
        if vec is None:
            grads = {p: p.grad for p in self.model.parameters() if p.requires_grad}
            vec = ParamVector(grads.keys(), grads.values())

        return quadratic_form(self._get_fvp_fn(), vec)


class FisherExactCrossEntropy(FisherMaker):
    @property
    def do_local_accumulate(self) -> bool:
        if self._logits is not None:
            return self._logits.shape[-1] > 1  # out_dim > 1
        return True

    def _fisher_loop(self, closure):
        logits = self._logits
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = log_probs.view(-1, log_probs.size(-1))
        n, n_classes = log_probs.shape
        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            sqrt_probs = torch.sqrt(probs)
        for i in range(n_classes):
            targets = torch.tensor([i] * n, device=logits.device)

            def nll_expr():
                nll = F.nll_loss(log_probs, targets, reduction='none', ignore_index=-1)
                return nll.mul(sqrt_probs[:, i]).sum()
            closure(nll_expr, retain_graph=i < n_classes - 1)


class FisherMCCrossEntropy(FisherMaker):
    @property
    def do_local_accumulate(self) -> bool:
        return self.config.n_mc_samples > 1

    def _fisher_loop(self, closure):
        logits = self._logits
        log_probs = F.log_softmax(logits, dim=-1)
        n_mc_samples = self.config.n_mc_samples
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        for i in range(n_mc_samples):
            with torch.no_grad():
                targets = dist.sample()
            closure(lambda: F.nll_loss(log_probs.view(-1, log_probs.size(-1)),
                                       targets.view(-1), reduction='sum', ignore_index=-1) / n_mc_samples,
                    retain_graph=i < n_mc_samples - 1)


class FisherExactMSE(FisherMaker):
    @property
    def do_local_accumulate(self) -> bool:
        if self._logits is not None:
            return self._logits.shape[-1] > 1  # out_dim > 1
        return True

    def _fisher_loop(self, closure):
        logits = self._logits
        n_dims = logits.size(-1)
        for i in range(n_dims):
            closure(lambda: logits[:, i].sum(), retain_graph=i < n_dims - 1)


class FisherMCMSE(FisherMaker):
    @property
    def do_local_accumulate(self) -> bool:
        return self.config.n_mc_samples > 1

    def _fisher_loop(self, closure):
        logits = self._logits
        n_mc_samples = self.config.n_mc_samples
        var = self.config.var
        dist = torch.distributions.normal.Normal(logits, scale=np.sqrt(var))
        for i in range(n_mc_samples):
            with torch.no_grad():
                targets = dist.sample()
            closure(lambda: 0.5 * F.mse_loss(logits, targets, reduction='sum') / n_mc_samples,
                    retain_graph=i < n_mc_samples - 1)


class FisherEmp(FisherMaker):
    @property
    def do_local_accumulate(self) -> bool:
        return False

    @property
    def is_fisher_emp(self):
        return True


def get_fisher_maker(model: nn.Module, config: FisherConfig):
    fisher_type = config.fisher_type
    loss_type = config.loss_type
    assert fisher_type in _supported_types
    if fisher_type == FISHER_EMP:
        return FisherEmp(model, config)
    assert loss_type in [LOSS_CROSS_ENTROPY, LOSS_MSE]
    if fisher_type == FISHER_EXACT:
        if loss_type == LOSS_CROSS_ENTROPY:
            return FisherExactCrossEntropy(model, config)
        else:
            return FisherExactMSE(model, config)
    else:
        if loss_type == LOSS_CROSS_ENTROPY:
            return FisherMCCrossEntropy(model, config)
        else:
            return FisherMCMSE(model, config)
