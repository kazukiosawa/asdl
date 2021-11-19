from functools import partial
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from .core import no_centered_cov
from .utils import skip_param_grad
from .matrices import *
from .vector import ParamVector, reduce_vectors
from .mvp import power_method, conjugate_gradient_method

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
    'fisher_free',
    'fisher_free_for_cross_entropy',
    'fisher_free_for_mse',
    'LOSS_CROSS_ENTROPY',
    'LOSS_MSE'
]

_supported_types = [FISHER_EXACT, FISHER_MC, FISHER_EMP]
_supported_shapes = [SHAPE_FULL, SHAPE_LAYER_WISE, SHAPE_KRON, SHAPE_UNIT_WISE, SHAPE_DIAG]
_supported_shapes_for_fvp = [SHAPE_FULL, SHAPE_LAYER_WISE]


class _FisherBase(MatrixManager):
    def __init__(self, model, **kwargs):
        super().__init__(model, self.fisher_type)

    @property
    def fisher_type(self):
        raise NotImplementedError

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
            f = getattr(module, attr, None)
            if f is not None:
                f.mul_(0)

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
                         scale=1.):
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

            with no_centered_cov(model, fisher_shapes, cvp=fvp, vectors=vec) as cxt:
                def closure(loss_expr):
                    cxt.clear_batch_grads()
                    loss = loss_expr()
                    with skip_param_grad(model, disable=calc_emp_loss_grad_with_fisher):
                        loss.backward(retain_graph=True)
                    if fvp:
                        cxt.calc_full_cvp(model)
                    else:
                        cxt.calc_full_cov(model)
                    if not calc_emp_loss_grad_after_fisher:
                        nonlocal total_loss
                        total_loss += loss.item()

                y = model(x)
                self._fisher_core(closure, y, t)
                for module in model.modules():
                    # accumulate layer-wise fisher/fvp
                    self._accumulate_fisher(module, cxt.cov_symmatrix(module), scale)
                    self._accumulate_fvp(module, cxt.cvp_paramvector(module), scale)
                # accumulate full fisher/fvp
                self._accumulate_fisher(model, cxt.full_cov_symmatrix(model), scale)
                self._accumulate_fvp(model, cxt.full_cvp_paramvector(model), scale)

            if calc_emp_loss_grad_after_fisher:
                assert t is not None
                emp_loss = self.loss_fn(y, t)
                emp_loss.backward()
                nonlocal total_loss
                total_loss += emp_loss.item()

        if data_loader is not None:
            # calculate fisher/fvp for the data_loader
            data_size = len(data_loader.dataset)
            if data_average:
                scale /= data_size
            for inputs, targets in data_loader:
                fisher_for_one_batch(inputs, targets)
        else:
            # calculate fisher/fvp for a single batch
            assert inputs is not None
            data_size = inputs.shape[0]
            if data_average:
                scale /= data_size
            fisher_for_one_batch(inputs, targets)

        if calc_emp_loss_grad and data_average:
            # divide gradients by data size
            # (every loss function returns the sum of loss, not the average)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.div_(data_size)

        if data_average:
            total_loss /= data_size
        return total_loss

    def _fisher_core(self, closure, outputs, targets):
        raise NotImplementedError

    def _accumulate_fisher(self, module: nn.Module, new_fisher, scale=1., fvp=False):
        if new_fisher is None:
            return
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

    def reduce_fisher(self, is_master=True, all_reduce=False):
        self.reduce_matrices(is_master=is_master, all_reduce=all_reduce)

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


class _FisherCrossEntropy(_FisherBase):
    @property
    def loss_fn(self):
        return partial(F.cross_entropy, reduction='sum')


class FisherExactCrossEntropy(_FisherCrossEntropy):
    @property
    def fisher_type(self):
        return FISHER_EXACT

    def _fisher_core(self, closure, outputs, unused):
        probs = F.softmax(outputs, dim=1)
        log_probs = F.log_softmax(outputs, dim=1)
        _, n_classes = probs.shape
        probs, _targets = torch.sort(probs, dim=1, descending=True)
        sqrt_probs = torch.sqrt(probs)
        for i in range(n_classes):
            def loss_expr():
                loss = F.nll_loss(log_probs, _targets[:, i], reduction='none')
                return loss.mul(sqrt_probs[:, i]).sum()
            closure(loss_expr)


class FisherMCCrossEntropy(_FisherCrossEntropy):
    def __init__(self, model, n_mc_samples=1):
        super().__init__(model)
        self.n_mc_samples = n_mc_samples

    @property
    def fisher_type(self):
        return FISHER_MC

    def _fisher_core(self, closure, outputs, unused):
        probs = F.softmax(outputs, dim=1)
        log_probs = F.log_softmax(outputs, dim=1)
        dist = torch.distributions.Categorical(probs)
        for i in range(self.n_mc_samples):
            with torch.no_grad():
                targets = dist.sample()
            closure(lambda: F.nll_loss(log_probs, targets, reduction='sum') / self.n_mc_samples)


class FisherEmpCrossEntropy(_FisherCrossEntropy):
    @property
    def fisher_type(self):
        return FISHER_EMP

    def _fisher_core(self, closure, outputs, targets):
        log_probs = F.log_softmax(outputs, dim=1)
        closure(lambda: F.nll_loss(log_probs, targets, reduction='sum'))


class _FisherMSE(_FisherBase):
    @property
    def loss_fn(self):
        return lambda x, y: 0.5 * (x - y).norm(dim=1).sum()


class FisherExactMSE(_FisherMSE):
    @property
    def fisher_type(self):
        return FISHER_EXACT

    def _fisher_core(self, closure, outputs, unused):
        _, n_dims = outputs.shape
        for i in range(n_dims):
            closure(lambda: outputs[:, i].sum())


class FisherMCMSE(_FisherMSE):
    def __init__(self, model, n_mc_samples=1, var=0.5):
        super().__init__(model)
        self.n_mc_samples = n_mc_samples
        self.var = var

    @property
    def fisher_type(self):
        return FISHER_MC

    def _fisher_core(self, closure, outputs, unused):
        dist = torch.distributions.normal.Normal(outputs, scale=np.sqrt(self.var))
        for i in range(self.n_mc_samples):
            with torch.no_grad():
                targets = dist.sample()
            closure(lambda: 0.5 * F.mse_loss(outputs, targets, reduction='sum') / self.n_mc_samples)


class FisherEmpMSE(_FisherMSE):
    @property
    def fisher_type(self):
        return FISHER_EMP

    def _fisher_core(self, closure, outputs, targets):
        closure(lambda: 0.5 * (outputs - targets).norm(dim=1).sum())


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
    assert fisher_type in _supported_types
    assert loss_type in [LOSS_CROSS_ENTROPY, LOSS_MSE]
    if loss_type == LOSS_CROSS_ENTROPY:
        if fisher_type == FISHER_EXACT:
            fisher_cls = FisherExactCrossEntropy
        elif fisher_type == FISHER_MC:
            fisher_cls = FisherMCCrossEntropy
        else:
            fisher_cls = FisherEmpCrossEntropy
    else:
        if fisher_type == FISHER_EXACT:
            fisher_cls = FisherExactMSE
        elif fisher_type == FISHER_MC:
            fisher_cls = FisherMCMSE
        else:
            fisher_cls = FisherEmpMSE

    f = fisher_cls(model, **kwargs)
    loss = f.calculate_fisher(
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
        return f, loss
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
        b = ParamVector(list(grads.keys()), grads)

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
