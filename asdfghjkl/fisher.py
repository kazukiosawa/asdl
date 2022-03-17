from functools import partial
from typing import List, Union
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import Stream
import torch.distributed as dist

from .core import no_centered_cov
from .utils import skip_param_grad
from .matrices import *
from .vector import ParamVector, reduce_vectors
from .mvp import power_method, stochastic_lanczos_quadrature, conjugate_gradient_method, quadratic_form

_COV_FULL = 'cov_full'
_CVP_FULL = 'cvp_full'

LOSS_CROSS_ENTROPY = 'cross_entropy'
LOSS_MSE = 'mse'

__all__ = [
    'calculate_fisher',
    'fisher_for_cross_entropy',
    'fisher_for_mse',
    'fvp_for_cross_entropy',
    'fvp_for_mse',
    'fisher_eig',
    'fisher_eig_for_cross_entropy',
    'fisher_eig_for_mse',
    'fisher_esd',
    'fisher_esd_for_cross_entropy',
    'fisher_esd_for_mse',
    'fisher_free',
    'fisher_free_for_cross_entropy',
    'fisher_free_for_mse',
    'fisher_quadratic_form',
    'fisher_quadratic_form_for_cross_entropy',
    'fisher_quadratic_form_for_mse',
    'LOSS_CROSS_ENTROPY',
    'LOSS_MSE',
    'FisherManager',
    'get_fisher_class',
]

_supported_types = [FISHER_EXACT, FISHER_MC, FISHER_EMP]
_supported_shapes = [SHAPE_FULL, SHAPE_LAYER_WISE, SHAPE_KRON, SHAPE_UNIT_WISE, SHAPE_DIAG]
_supported_shapes_for_fvp = [SHAPE_FULL, SHAPE_LAYER_WISE]


class FisherManager(MatrixManager):
    def __init__(self, model, fisher_type):
        super().__init__(model, fisher_type)
        self.fisher_type = fisher_type

    @property
    def is_fisher_emp(self):
        return self.fisher_type == FISHER_EMP

    @property
    def loss_fn(self):
        raise NotImplementedError

    @property
    def fisher_attr(self):
        return self.fisher_type

    @property
    def fvp_attr(self):
        return f'{self.fisher_type}_fvp'

    def zero_fisher(self, fvp=False):
        attr = self.fvp_attr if fvp else self.fisher_attr
        for module in self._model.modules():
            if hasattr(module, attr):
                delattr(module, attr)

    def calculate_fisher(self,
                         fisher_shapes,
                         inputs: torch.Tensor = None,
                         targets: torch.Tensor = None,
                         data_loader: torch.utils.data.DataLoader = None,
                         fvp=False,
                         vec: ParamVector = None,
                         data_average=True,
                         accumulate=False,
                         calc_emp_loss_grad=False,
                         seed=None,
                         scale=1.,
                         stream: Stream = None):
        model = self._model
        device = self._device
        if isinstance(fisher_shapes, str):
            fisher_shapes = [fisher_shapes]

        if not accumulate:
            # set Fisher/FVP zero
            self.zero_fisher(fvp=fvp)

        total_loss = 0
        calc_emp_loss_grad_with_fisher = calc_emp_loss_grad and self.is_fisher_emp
        calc_emp_loss_grad_after_fisher = calc_emp_loss_grad and not self.is_fisher_emp

        def fisher_for_one_batch(x, t=None):
            x = x.to(device)
            if t is not None:
                t = t.to(device)
            if seed:
                torch.random.manual_seed(seed)

            with no_centered_cov(model, fisher_shapes, cvp=fvp, vectors=vec, stream=stream) as cxt:
                def closure(loss_expr, retain_graph=False):
                    cxt.clear_batch_grads()
                    loss = loss_expr()
                    with skip_param_grad(model, disable=calc_emp_loss_grad_with_fisher):
                        loss.backward(retain_graph=retain_graph or calc_emp_loss_grad_after_fisher)
                    if fvp:
                        cxt.calc_full_cvp(model)
                    else:
                        cxt.calc_full_cov(model)
                    if not calc_emp_loss_grad_after_fisher:
                        nonlocal total_loss
                        total_loss += float(loss)

                y = model(x)
                self._fisher_core(closure, y, t)
                self.accumulate(cxt, scale, fvp=fvp)

            if calc_emp_loss_grad_after_fisher:
                assert t is not None
                emp_loss = self.loss_fn(y, t)
                emp_loss.backward()
                nonlocal total_loss
                total_loss += float(emp_loss)

            return y

        outputs = None
        if data_loader is not None:
            # calculate fisher/fvp for the data_loader
            data_size = len(data_loader.dataset)
            if data_average:
                scale /= data_size
            for inputs, targets in data_loader:
                outputs = fisher_for_one_batch(inputs, targets)
        else:
            # calculate fisher/fvp for a single batch
            assert inputs is not None
            data_size = inputs.shape[0]
            if data_average:
                scale /= data_size
            outputs = fisher_for_one_batch(inputs, targets)

        if calc_emp_loss_grad and data_average:
            # divide gradients by data size
            # (every loss function returns the sum of loss, not the average)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.div_(data_size)

        if data_average:
            total_loss /= data_size
        return total_loss, outputs

    def _fisher_core(self, closure, outputs, targets):
        raise NotImplementedError

    def accumulate(self, cxt, scale=1., target_module=None, target_module_name=None, fvp=False):
        model = self._model
        for name, module in model.named_modules():
            if target_module is not None and module != target_module:
                continue
            if target_module_name is not None and name != target_module_name:
                continue
            # accumulate layer-wise fisher/fvp
            if fvp:
                self._accumulate_fvp(module, cxt.cvp_paramvector(module, pop=True), scale)
            else:
                self._accumulate_fisher(module, cxt.cov_symmatrix(module, pop=True), scale)
            if target_module is not None:
                break
            if target_module_name is not None:
                target_module = module
                break

        if target_module is None or target_module == model:
            # accumulate full fisher/fvp
            if fvp:
                self._accumulate_fvp(model, cxt.full_cvp_paramvector(model, pop=True), scale)
            else:
                self._accumulate_fisher(model, cxt.full_cov_symmatrix(model, pop=True), scale)

    def _accumulate_fisher(self, module: nn.Module, new_fisher, scale=1., fvp=False):
        if new_fisher is None:
            return
        if scale != 1:
            new_fisher.mul_(scale)
        dst_attr = self.fvp_attr if fvp else self.fisher_attr
        dst_fisher = getattr(module, dst_attr, None)
        if dst_fisher is None:
            setattr(module, dst_attr, new_fisher)
        else:
            # this must be __iadd__ to preserve inv
            dst_fisher += new_fisher

    def _accumulate_fvp(self, module: nn.Module, new_fisher, scale=1.):
        self._accumulate_fisher(module, new_fisher, scale, fvp=True)

    def get_fisher_tensor(self, module: nn.Module, *keys) -> Union[torch.Tensor, None]:
        fisher = getattr(module, self.fisher_attr, None)
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
        attr = self.fvp_attr
        if fisher_shape == SHAPE_FULL:
            setattr(self._model, attr, v)
        else:
            for module in self._model.modules():
                if hasattr(module, attr):
                    setattr(module, attr, v.get_vectors_by_module(module))

    def load_fvp(self, fisher_shape: str) -> ParamVector:
        if fisher_shape == SHAPE_FULL:
            v = getattr(self._model, self.fvp_attr, None)
            if v is None:
                return None
            return v.copy()
        else:
            rst = None
            for module in self._model.modules():
                if module == self._model:
                    continue
                v = getattr(module, self.fvp_attr, None)
                if v is not None:
                    if rst is None:
                        rst = v.copy()
                    else:
                        rst.extend(v.copy())
            return rst


class _FisherCrossEntropy(FisherManager):
    @property
    def loss_fn(self):
        return partial(F.cross_entropy, reduction='sum')


class FisherExactCrossEntropy(_FisherCrossEntropy):
    def __init__(self, model):
        super().__init__(model, FISHER_EXACT)

    def _fisher_core(self, closure, outputs, unused):
        log_probs = F.log_softmax(outputs, dim=1)
        n, n_classes = log_probs.shape
        with torch.no_grad():
            probs = F.softmax(outputs, dim=1)
            sqrt_probs = torch.sqrt(probs)
        for i in range(n_classes):
            targets = torch.tensor([i] * n, device=outputs.device)

            def loss_expr():
                loss = F.nll_loss(log_probs, targets, reduction='none')
                return loss.mul(sqrt_probs[:, i]).sum()
            closure(loss_expr, retain_graph=i < n_classes - 1)


class FisherMCCrossEntropy(_FisherCrossEntropy):
    def __init__(self, model, n_mc_samples=1):
        super().__init__(model, FISHER_MC)
        self.n_mc_samples = n_mc_samples

    def _fisher_core(self, closure, outputs, unused):
        probs = F.softmax(outputs, dim=1)
        log_probs = F.log_softmax(outputs, dim=1)
        dist = torch.distributions.Categorical(probs)
        for i in range(self.n_mc_samples):
            with torch.no_grad():
                targets = dist.sample()
            closure(lambda: F.nll_loss(log_probs, targets, reduction='sum') / self.n_mc_samples,
                    retain_graph=i < self.n_mc_samples - 1)


class FisherEmpCrossEntropy(_FisherCrossEntropy):
    def __init__(self, model):
        super().__init__(model, FISHER_EMP)

    def _fisher_core(self, closure, outputs, targets):
        log_probs = F.log_softmax(outputs, dim=1)
        closure(lambda: F.nll_loss(log_probs, targets, reduction='sum'),
                retain_graph=False)


class _FisherMSE(FisherManager):
    @property
    def loss_fn(self):
        return lambda x, y: 0.5 * (x - y).norm(dim=1).sum()


class FisherExactMSE(_FisherMSE):
    def __init__(self, model):
        super().__init__(model, FISHER_EXACT)

    def _fisher_core(self, closure, outputs, unused):
        _, n_dims = outputs.shape
        for i in range(n_dims):
            closure(lambda: outputs[:, i].sum(), retain_graph=i < n_dims - 1)


class FisherMCMSE(_FisherMSE):
    def __init__(self, model, n_mc_samples=1, var=0.5):
        super().__init__(model, FISHER_MC)
        self.n_mc_samples = n_mc_samples
        self.var = var

    def _fisher_core(self, closure, outputs, unused):
        dist = torch.distributions.normal.Normal(outputs, scale=np.sqrt(self.var))
        for i in range(self.n_mc_samples):
            with torch.no_grad():
                targets = dist.sample()
            closure(lambda: 0.5 * F.mse_loss(outputs, targets, reduction='sum') / self.n_mc_samples,
                    retain_graph=i < self.n_mc_samples - 1)


class FisherEmpMSE(_FisherMSE):
    def __init__(self, model):
        super().__init__(model, FISHER_EMP)

    def _fisher_core(self, closure, outputs, targets):
        closure(lambda: 0.5 * F.mse_loss(outputs, targets, reduction='sum'),
                retain_graph=False)


def get_fisher_class(fisher_type, loss_type):
    assert fisher_type in _supported_types
    assert loss_type in [LOSS_CROSS_ENTROPY, LOSS_MSE]
    if loss_type == LOSS_CROSS_ENTROPY:
        if fisher_type == FISHER_EXACT:
            return FisherExactCrossEntropy
        elif fisher_type == FISHER_MC:
            return FisherMCCrossEntropy
        else:
            return FisherEmpCrossEntropy
    else:
        if fisher_type == FISHER_EXACT:
            return FisherExactMSE
        elif fisher_type == FISHER_MC:
            return FisherMCMSE
        else:
            return FisherEmpMSE


def calculate_fisher(
        model: nn.Module,
        fisher_type: str,
        fisher_shapes,
        loss_type: str,
        inputs: torch.Tensor = None,
        targets: torch.Tensor = None,
        data_loader: torch.utils.data.DataLoader = None,
        fvp=False,
        vec: ParamVector = None,
        is_distributed=False,
        all_reduce=False,
        is_master=True,
        accumulate=False,
        data_average=True,
        calc_emp_loss_grad=False,
        return_loss=False,
        seed=None,
        scale=1.,
        **kwargs
):
    fisher_cls = get_fisher_class(fisher_type, loss_type)
    f = fisher_cls(model, **kwargs)
    loss, outputs = f.calculate_fisher(
        fisher_shapes,
        inputs=inputs,
        targets=targets,
        data_loader=data_loader,
        fvp=fvp,
        vec=vec,
        accumulate=accumulate,
        data_average=data_average,
        calc_emp_loss_grad=calc_emp_loss_grad,
        seed=seed,
        scale=scale)
    if is_distributed:
        if fvp:
            f.reduce_fvp(is_master, all_reduce)
        else:
            f.reduce_fisher(is_master, all_reduce)
    if return_loss:
        return f, loss, outputs
    else:
        return f


fisher_for_cross_entropy = partial(calculate_fisher, loss_type=LOSS_CROSS_ENTROPY, fvp=False)
fisher_for_mse = partial(calculate_fisher, loss_type=LOSS_MSE, fvp=False)
fvp_for_cross_entropy = partial(calculate_fisher, loss_type=LOSS_CROSS_ENTROPY, fvp=True)
fvp_for_mse = partial(calculate_fisher, loss_type=LOSS_MSE, fvp=True)


def fisher_eig(
        model,
        fisher_type: str,
        fisher_shape,
        loss_type: str,
        inputs=None,
        targets=None,
        data_loader=None,
        top_n=1,
        max_iters=100,
        tol=1e-3,
        is_distributed=False,
        print_progress=False,
        **kwargs
):

    def fvp_fn(vec: ParamVector) -> ParamVector:
        f = calculate_fisher(model,
                             fisher_type,
                             fisher_shape,
                             loss_type,
                             inputs=inputs,
                             targets=targets,
                             data_loader=data_loader,
                             fvp=True,
                             vec=vec,
                             is_distributed=is_distributed,
                             all_reduce=True,
                             **kwargs)
        return f.load_fvp(fisher_shape)

    # for making MC samplings at each iteration deterministic
    random_seed = torch.rand(1) * 100 if fisher_type == FISHER_MC else None

    eigvals, eigvecs = power_method(fvp_fn,
                                    model,
                                    top_n=top_n,
                                    max_iters=max_iters,
                                    tol=tol,
                                    is_distributed=is_distributed,
                                    print_progress=print_progress,
                                    random_seed=random_seed
                                    )

    return eigvals, eigvecs


fisher_eig_for_cross_entropy = partial(fisher_eig, loss_type=LOSS_CROSS_ENTROPY)
fisher_eig_for_mse = partial(fisher_eig, loss_type=LOSS_MSE)


def fisher_esd(
        model,
        fisher_type: str,
        fisher_shape,
        loss_type: str,
        inputs=None,
        targets=None,
        data_loader=None,
        n_v=1,
        num_iter=100,
        num_bins=10000,
        sigma_squared=1e-5,
        overhead=None,
        is_distributed=False,
        **kwargs
):

    # referenced from https://github.com/amirgholami/PyHessian/blob/master/density_plot.py

    def fvp_fn(vec: ParamVector) -> ParamVector:
        f = calculate_fisher(model,
                             fisher_type,
                             fisher_shape,
                             loss_type,
                             inputs=inputs,
                             targets=targets,
                             data_loader=data_loader,
                             fvp=True,
                             vec=vec,
                             is_distributed=is_distributed,
                             all_reduce=True,
                             **kwargs)
        return f.load_fvp(fisher_shape)
    
    # for making MC samplings at each iteration deterministic
    random_seed = torch.rand(1) * 100 if fisher_type == FISHER_MC else None

    eigvals, weights = stochastic_lanczos_quadrature(fvp_fn,
                                                     model,
                                                     n_v=n_v,
                                                     num_iter=num_iter,
                                                     is_distributed=is_distributed,
                                                     random_seed=random_seed
                                                     )
    
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


fisher_esd_for_cross_entropy = partial(fisher_esd, loss_type=LOSS_CROSS_ENTROPY)
fisher_esd_for_mse = partial(fisher_esd, loss_type=LOSS_MSE)


def fisher_free(
        model,
        fisher_type: str,
        fisher_shape,
        loss_type: str,
        b=None,
        data_loader=None,
        inputs=None,
        targets=None,
        init_x=None,
        damping=1e-3,
        max_iters=None,
        tol=1e-8,
        preconditioner=None,
        is_distributed=False,
        print_progress=False,
        random_seed=None,
        **kwargs
) -> ParamVector:

    def fvp_fn(vec: ParamVector) -> ParamVector:
        f = calculate_fisher(model,
                             fisher_type,
                             fisher_shape,
                             loss_type,
                             inputs=inputs,
                             targets=targets,
                             data_loader=data_loader,
                             fvp=True,
                             vec=vec,
                             is_distributed=is_distributed,
                             all_reduce=True,
                             **kwargs)
        return f.load_fvp(fisher_shape)

    if b is None:
        grads = {p: p.grad for p in model.parameters() if p.requires_grad}
        b = ParamVector(grads.keys(), grads.values())

    # for making MC samplings at each iteration deterministic
    if fisher_type == FISHER_MC and random_seed is None:
        random_seed = int(torch.rand(1) * 100)

    return conjugate_gradient_method(fvp_fn,
                                     b,
                                     init_x=init_x,
                                     damping=damping,
                                     max_iters=max_iters,
                                     tol=tol,
                                     preconditioner=preconditioner,
                                     print_progress=print_progress,
                                     random_seed=random_seed)


fisher_free_for_cross_entropy = partial(fisher_free, loss_type=LOSS_CROSS_ENTROPY)
fisher_free_for_mse = partial(fisher_free, loss_type=LOSS_MSE)


def fisher_quadratic_form(
        model,
        fisher_type: str,
        fisher_shape,
        loss_type: str,
        v=None,
        data_loader=None,
        inputs=None,
        targets=None,
        is_distributed=False,
        **kwargs
):
    def fvp_fn(vec: ParamVector) -> ParamVector:
        f = calculate_fisher(model,
                             fisher_type,
                             fisher_shape,
                             loss_type,
                             inputs=inputs,
                             targets=targets,
                             data_loader=data_loader,
                             fvp=True,
                             vec=vec,
                             is_distributed=is_distributed,
                             all_reduce=True,
                             **kwargs)
        return f.load_fvp(fisher_shape)

    if v is None:
        grads = {p: p.grad for p in model.parameters() if p.requires_grad}
        v = ParamVector(grads.keys(), grads.values())

    return quadratic_form(fvp_fn, v, **kwargs)


fisher_quadratic_form_for_cross_entropy = partial(fisher_quadratic_form, loss_type=LOSS_CROSS_ENTROPY)
fisher_quadratic_form_for_mse = partial(fisher_quadratic_form, loss_type=LOSS_MSE)
