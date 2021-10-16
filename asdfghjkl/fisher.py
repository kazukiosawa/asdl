from contextlib import contextmanager
from functools import partial
import numpy as np

import torch
import torch.nn.functional as F
from .core import extend
from .utils import disable_param_grad
from .operations import *
from .symmatrix import SymMatrix, Kron, Diag, UnitWise
from .matrices import *
from .mvp import power_method, conjugate_gradient_method, reduce_params

_SHAPE_TO_OP = {
    SHAPE_FULL: OP_BATCH_GRADS,  # full
    SHAPE_BLOCK_DIAG: OP_BATCH_GRADS,  # block-diagonal
    SHAPE_KRON: OP_COV_KRON,  # Kronecker-factored
    SHAPE_DIAG: OP_COV_DIAG,  # diagonal
}

_COV_FULL = 'cov_full'
_CVP_FULL = 'cvp_full'
_COV_BLOCK_DIAG = 'cov_block_diag'
_CVP_BLOCK_DIAG = 'cvp_block_diag'

_LOSS_CROSS_ENTROPY = 'cross_entropy'
_LOSS_MSE = 'mse'

__all__ = [
    'fisher_for_cross_entropy',
    'fisher_for_mse',
    'fvp_for_cross_entropy',
    'fvp_for_mse',
    'fisher_eig_for_cross_entropy',
    'fisher_eig_for_mse',
    'fisher_free_for_cross_entropy',
    'fisher_free_for_mse',
    'woodbury_ifvp'
]

_supported_types = [FISHER_EXACT, FISHER_MC, FISHER_EMP]
_supported_shapes = [SHAPE_FULL, SHAPE_BLOCK_DIAG, SHAPE_KRON, SHAPE_DIAG]
_supported_shapes_for_fvp = [SHAPE_FULL, SHAPE_BLOCK_DIAG]


class _FisherBase(MatrixManager):
    def __init__(self, model, **kwargs):
        super().__init__(model, self.fisher_type)

    @property
    def fisher_type(self):
        raise NotImplementedError

    @property
    def fvp_attr(self):
        return _get_fvp_attr(self.fisher_type)

    def zero_fisher(self):
        ftype = self.fisher_type
        for module in self._model.modules():
            if hasattr(module, ftype):
                delattr(module, ftype)

    def zero_fvp(self):
        attr = self.fvp_attr
        for module in self._model.modules():
            if hasattr(module, attr):
                delattr(module, attr)

    def calculate_fisher(self,
                         fisher_shapes,
                         inputs=None,
                         targets=None,
                         data_loader=None,
                         fvp=False,
                         vec=None,
                         data_average=True):
        if isinstance(fisher_shapes, str):
            fisher_shapes = [fisher_shapes]
        for fshape in fisher_shapes:
            if fvp:
                assert fshape in _supported_shapes_for_fvp
            else:
                assert fshape in _supported_shapes
        if fvp:
            self.zero_fvp()
        else:
            self.zero_fisher()

        # setup operations for extend
        op_names = [_SHAPE_TO_OP[shape] for shape in fisher_shapes]
        # remove duplicates
        op_names = list(set(op_names))

        model = self._model
        base_scale = 1.

        def closure(loss_expr, scale=1., grad_scale=None):
            model.zero_grad(set_to_none=True)
            loss = loss_expr()
            with _grads_scale(model, grad_scale):
                with disable_param_grad(model):
                    loss.backward(retain_graph=True)
            if SHAPE_FULL in fisher_shapes and not fvp:
                _full_covariance(model)
            if SHAPE_FULL in fisher_shapes and fvp:
                _full_cvp(model, vec)
            if SHAPE_BLOCK_DIAG in fisher_shapes and not fvp:
                _block_diag_covariance(model)
            if SHAPE_BLOCK_DIAG in fisher_shapes and fvp:
                _block_diag_cvp(model, vec)
            _register_fisher(model, self.fisher_type, scale * base_scale)

        device = self._device
        if data_loader is not None:
            if data_average:
                base_scale = 1 / len(data_loader.dataset)
            # calculate fisher/fvp for the data_loader
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                with extend(model, op_names):
                    self._fisher_core(closure, model(inputs), targets)
        else:
            # calculate fisher/fvp for a single batch
            assert inputs is not None
            if data_average:
                base_scale = 1 / inputs.shape[0]
            inputs = inputs.to(device)
            if targets is not None:
                targets = targets.to(device)
            with extend(model, op_names):
                self._fisher_core(closure, model(inputs), targets)

    def _fisher_core(self, closure, outputs, targets):
        raise NotImplementedError

    def reduce_fisher(self, is_master=True, all_reduce=False):
        self.reduce_matrices(is_master=is_master, all_reduce=all_reduce)

    def reduce_fvp(self, fisher_shape, is_master=True, all_reduce=False):
        v = self.load_fvp(fisher_shape)
        v = reduce_params(v, is_master, all_reduce)
        attr = self.fvp_attr
        if fisher_shape == SHAPE_FULL:
            setattr(self._model, attr, v)
        else:
            idx = 0
            for module in self._model.modules():
                if hasattr(module, attr):
                    setattr(module, attr, v[idx])
                    idx += 1
            assert idx == len(v)

    def load_fvp(self, fisher_shape):
        if fisher_shape == SHAPE_FULL:
            return getattr(self._model, self.fvp_attr, None)
        else:
            rst = []
            for module in self._model.modules():
                v = getattr(module, self.fvp_attr, None)
                if v is not None:
                    rst.extend(v)
            return rst


class FisherExactCrossEntropy(_FisherBase):
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
            closure(lambda: F.nll_loss(log_probs, _targets[:, i], reduction='sum'),
                    grad_scale=sqrt_probs[:, i])


class FisherMCCrossEntropy(_FisherBase):
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
            closure(lambda: F.nll_loss(log_probs, targets, reduction='sum'),
                    scale=1/self.n_mc_samples)


class FisherEmpCrossEntropy(_FisherBase):
    @property
    def fisher_type(self):
        return FISHER_EMP

    def _fisher_core(self, closure, outputs, targets):
        log_probs = F.log_softmax(outputs, dim=1)
        closure(lambda: F.nll_loss(log_probs, targets, reduction='sum'))


class FisherExactMSE(_FisherBase):
    @property
    def fisher_type(self):
        return FISHER_EXACT

    def _fisher_core(self, closure, outputs, unused):
        _, n_dims = outputs.shape
        for i in range(n_dims):
            closure(lambda: outputs[:, i].sum())


class FisherMCMSE(_FisherBase):
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
            closure(lambda: 0.5 * (outputs - targets).norm(dim=1).sum())


class FisherEmpMSE(_FisherBase):
    @property
    def fisher_type(self):
        return FISHER_EMP

    def _fisher_core(self, closure, outputs, targets):
        closure(lambda: 0.5 * (outputs - targets).norm(dim=1).sum())


def _module_batch_grads(model):
    rst = []
    for module in model.modules():
        operation = getattr(module, 'operation', None)
        if operation is None:
            continue
        batch_grads = operation.get_op_results()[OP_BATCH_GRADS]
        rst.append((module, batch_grads))
    return rst


def _module_batch_flatten_grads(model):
    rst = []
    for module, batch_grads in _module_batch_grads(model):
        batch_flatten_grads = torch.cat(
            [g.flatten(start_dim=1) for g in batch_grads.values()],
            dim=1
        )
        rst.append((module, batch_flatten_grads))
    return rst


def _module_batch_gvp(model, vec):
    rst = []
    pointer = 0
    for module, batch_grads in _module_batch_grads(model):
        batch_gvp = None
        for b_g in batch_grads.values():
            v = vec[pointer]
            b_gvp = b_g.mul(v.unsqueeze(0)).flatten(start_dim=1).sum(1)  # n
            if batch_gvp is None:
                batch_gvp = b_gvp
            else:
                batch_gvp += b_gvp
            pointer += 1
        rst.append((module, batch_gvp))
    assert pointer == len(vec)
    return rst


def _full_covariance(model):
    batch_all_g = []
    for _, batch_g in _module_batch_flatten_grads(model):
        batch_all_g.append(batch_g)
    batch_all_g = torch.cat(batch_all_g, dim=1)  # n x p_all
    cov_full = torch.matmul(batch_all_g.T, batch_all_g)  # p_all x p_all
    setattr(model, _COV_FULL, cov_full)


def _block_diag_covariance(model):
    for module, batch_g in _module_batch_flatten_grads(model):
        cov_block = torch.matmul(batch_g.T, batch_g)  # p_all x p_all
        setattr(module, _COV_BLOCK_DIAG, cov_block)


def _full_cvp(model, vec):
    """
    g: n x p
    v: p
    c = sum[gg^t]: p x p
    cvp = sum[gg^t]v = sum[g(g^t)v]: p
    """
    # compute batched (g^t)v
    batch_all_gvp = None
    for module, batch_gvp in _module_batch_gvp(model, vec):
        if batch_all_gvp is None:
            batch_all_gvp = batch_gvp
        else:
            batch_all_gvp += batch_gvp

    # compute cvp = sum[g(g^t)v]
    cvp = []
    for module, batch_grads in _module_batch_grads(model):
        for b_g in batch_grads.values():
            cvp.append(torch.einsum('n...,n->...', b_g, batch_all_gvp))

    setattr(model, _CVP_FULL, cvp)


def _block_diag_cvp(model, vec):
    """
    g: n x p
    v: p
    c = sum[gg^t]: p x p
    cvp = sum[gg^t]v = sum[g(g^t)v]: p
    """
    batch_gvp_dict = {k: v for k, v in _module_batch_gvp(model, vec)}
    for module, batch_grads in _module_batch_grads(model):
        cvp = []
        # compute cvp = sum[g(g^t)v]
        batch_gvp = batch_gvp_dict[module]
        for b_g in batch_grads.values():
            cvp.append(torch.einsum('n...,n->...', b_g, batch_gvp))

        setattr(module, _CVP_BLOCK_DIAG, cvp)


@contextmanager
def _grads_scale(model, scale):
    for module in model.modules():
        operation = getattr(module, 'operation', None)
        if operation is None:
            continue
        operation.grads_scale = scale

    yield

    for module in model.modules():
        operation = getattr(module, 'operation', None)
        if operation is None:
            continue
        operation.grads_scale = None


def _register_fisher(model, fisher_type, scale=1.):
    """
    module.{fisher_type} = op_results
    op_results = {
        'diag': {'weight': torch.Tensor, 'bias': torch.Tensor},
        'kron': {'A': torch.Tensor, 'B': torch.Tensor},
        'block_diag': torch.Tensor,
        'unit_wise': torch.Tensor,
    }
    """
    device = next(model.parameters()).device
    for module in model.modules():
        operation = getattr(module, 'operation', None)
        if operation is None:
            continue
        op_results = operation.get_op_results()
        kron = diag = unit = None
        if OP_COV_KRON in op_results:
            rst = op_results[OP_COV_KRON]
            kron = Kron(rst['A'], rst['B'], device=device)
        if OP_COV_DIAG in op_results:
            rst = op_results[OP_COV_DIAG]
            diag = Diag(
                rst.get('weight', None), rst.get('bias', None), device=device
            )
        if OP_COV_UNIT_WISE in op_results:
            rst = op_results[OP_COV_UNIT_WISE]
            unit = UnitWise(rst, device=device)
        operation.clear_op_results()
        # move block_diag/kron/diag fisher
        _accumulate_fisher(
            module,
            _COV_BLOCK_DIAG,
            fisher_type,
            kron=kron,
            diag=diag,
            unit=unit,
            scale=scale
        )
        # move block_diag fvp
        _accumulate_fvp(module, _CVP_BLOCK_DIAG, fisher_type, scale)

    # move full fisher
    _accumulate_fisher(model, _COV_FULL, fisher_type, scale=scale)
    # move full fvp
    _accumulate_fvp(model, _CVP_FULL, fisher_type, scale)


def _accumulate_fisher(
    module,
    data_src_attr,
    dst_attr,
    kron=None,
    diag=None,
    unit=None,
    scale=1.
):
    data = getattr(module, data_src_attr, None)
    if all(v is None for v in [data, kron, diag, unit]):
        return
    dst_fisher = getattr(module, dst_attr, None)
    if dst_fisher is None:
        new_fisher = SymMatrix(data, kron, diag, unit)
        new_fisher.scaling(scale)
        setattr(module, dst_attr, new_fisher)
    else:
        dst_fisher.accumulate(data, kron, diag, unit, scale=scale)
        # TODO: not accumulate kron.A for fisher_mc, fisher_exact
    if data is not None:
        delattr(module, data_src_attr)


def _accumulate_fvp(module, src_attr, fisher_type, scale=1.):
    dst_attr = _get_fvp_attr(fisher_type)
    cvp = getattr(module, src_attr, None)
    if cvp is None:
        return
    cvp = [v * scale for v in cvp]
    dst_fvp = getattr(module, dst_attr, None)
    if dst_fvp is None:
        setattr(module, dst_attr, cvp)
    else:
        dst_fvp = [u.add(v) for u, v in zip(dst_fvp, cvp)]
        setattr(module, dst_attr, dst_fvp)

    delattr(module, src_attr)


def _get_fvp_attr(fisher_type):
    return f'{fisher_type}_fvp'


def fisher(
        model,
        loss_type,
        fisher_type,
        fisher_shapes,
        inputs=None,
        targets=None,
        data_loader=None,
        fvp=False,
        vec=None,
        is_distributed=False,
        all_reduce=False,
        is_master=True,
        data_average=True,
        **kwargs
):
    assert fisher_type in _supported_types
    assert loss_type in [_LOSS_CROSS_ENTROPY, _LOSS_MSE]
    if loss_type == _LOSS_CROSS_ENTROPY:
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
    f.calculate_fisher(
        fisher_shapes,
        inputs=inputs,
        targets=targets,
        data_loader=data_loader,
        fvp=fvp,
        vec=vec,
        data_average=data_average)
    if is_distributed:
        if fvp:
            f.reduce_fvp(is_master, all_reduce)
        else:
            f.reduce_fisher(is_master, all_reduce)
    return f


fisher_for_cross_entropy = partial(fisher, loss_type=_LOSS_CROSS_ENTROPY, fvp=False)
fisher_for_mse = partial(fisher, loss_type=_LOSS_MSE, fvp=False)
fvp_for_cross_entropy = partial(fisher, loss_type=_LOSS_CROSS_ENTROPY, fvp=True)
fvp_for_mse = partial(fisher, loss_type=_LOSS_MSE, fvp=True)


def fisher_eig(
        model,
        loss_type,
        fisher_type,
        fisher_shape,
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

    def fvp_fn(vec):
        f = fisher(model,
                   loss_type,
                   fisher_type,
                   fisher_shape,
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


fisher_eig_for_cross_entropy = partial(fisher_eig, loss_type=_LOSS_CROSS_ENTROPY)
fisher_eig_for_mse = partial(fisher_eig, loss_type=_LOSS_MSE)


def fisher_free(
        model,
        b,
        loss_type,
        fisher_type,
        fisher_shape,
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
        save_log=False,
        **kwargs
):

    def fvp_fn(vec):
        f = fisher(model,
                   loss_type,
                   fisher_type,
                   fisher_shape,
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
                                     random_seed=random_seed,
                                     save_log=save_log)


fisher_free_for_cross_entropy = partial(fisher_free, loss_type=_LOSS_CROSS_ENTROPY)
fisher_free_for_mse = partial(fisher_free, loss_type=_LOSS_MSE)


def woodbury_ifvp(
        vec,
        model,
        inputs,
        targets,
        loss_fn,
        damping=1e-5,
        data_average=True,
):
    """
    Calculate inverse-empirical Fisher vector product by using the Woodbury matrix identity
    """
    assert damping > 0, 'Damping value has to be positive.'

    with extend(model, OP_BATCH_GRADS):
        model.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets, reduction='sum')
        loss.backward()

        batch_g_all = []
        for _, batch_g in _module_batch_flatten_grads(model):
            batch_g_all.append(batch_g)
        grads = torch.cat(batch_g_all, dim=1).T  # (p, n)

    p, n = grads.shape
    if data_average:
        grads /= np.sqrt(n)
    assert vec.shape == (p,)
    gram = torch.matmul(grads.T, grads)  # (n, n)
    inv = torch.inverse(gram + torch.eye(n) * damping)  # (n, n)
    b = torch.matmul(inv, torch.matmul(grads.T, vec))

    return (vec - torch.matmul(grads, b)) / damping  # (p,)
