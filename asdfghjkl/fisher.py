from contextlib import contextmanager
from functools import partial
import numpy as np

import torch
import torch.nn.functional as F
from .core import extend
from .utils import skip_param_grad
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
_COV_LAYER_WISE = 'cov_layer_wise'
_CVP_LAYER_WISE = 'cvp_layer_wise'

LOSS_CROSS_ENTROPY = 'cross_entropy'
LOSS_MSE = 'mse'

__all__ = [
    'fisher',
    'fisher_for_cross_entropy',
    'fisher_for_mse',
    'fvp_for_cross_entropy',
    'fvp_for_mse',
    'fisher_eig_for_cross_entropy',
    'fisher_eig_for_mse',
    'fisher_free_for_cross_entropy',
    'fisher_free_for_mse',
    'woodbury_ifvp',
    'LOSS_CROSS_ENTROPY',
    'LOSS_MSE'
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

    def zero_fisher(self):
        attr = self.fisher_attr
        for module in self._model.modules():
            f = getattr(module, attr, None)
            if f is not None:
                f.scaling(0)

    def zero_fvp(self):
        attr = self.fvp_attr
        for module in self._model.modules():
            fvp = getattr(module, attr, None)
            if fvp is not None:
                setattr(module, attr, fvp * 0)

    def calculate_fisher(self,
                         fisher_shapes,
                         inputs=None,
                         targets=None,
                         data_loader=None,
                         fvp=False,
                         vec=None,
                         data_average=True,
                         accumulate=False,
                         calc_emp_loss_grad=False,
                         seed=None,
                         scale=1.):
        model = self._model
        op_names, modules_for = matrix_shapes_to_values(fisher_shapes, _SHAPE_TO_OP, list(model.modules()))

        if not accumulate:
            # set Fisher/FVP zero
            if fvp:
                self.zero_fvp()
            else:
                self.zero_fisher()

        total_loss = 0
        calc_emp_loss_grad_with_fisher = calc_emp_loss_grad and self.is_fisher_emp
        calc_emp_loss_grad_after_fisher = calc_emp_loss_grad and not self.is_fisher_emp

        def fisher_for_one_batch(x, t=None):

            def closure(loss_expr, grad_scale=None):
                self._zero_op_batch_grads(set_to_none=True)
                loss = loss_expr()
                with _grads_scale(model, grad_scale):
                    with skip_param_grad(model, disable=calc_emp_loss_grad_with_fisher):
                        loss.backward(retain_graph=True)
                if fvp:
                    _full_cvp(model, modules_for[SHAPE_FULL], vec)
                    _layer_wise_cvp(modules_for[SHAPE_BLOCK_DIAG], vec)
                else:
                    _full_covariance(model, modules_for[SHAPE_FULL])
                    _layer_wise_covariance(modules_for[SHAPE_BLOCK_DIAG])
                if not calc_emp_loss_grad_after_fisher:
                    nonlocal total_loss
                    total_loss += loss.item()

            x = x.to(device)
            if t is not None:
                t = t.to(device)
            if seed:
                torch.random.manual_seed(seed)
            with extend(model, op_names):
                y = model(x)
                self._fisher_core(closure, y, t)
                self._register_fisher(scale)
            if calc_emp_loss_grad_after_fisher:
                assert t is not None
                emp_loss = self.loss_fn(y, t)
                emp_loss.backward()
                nonlocal total_loss
                total_loss += emp_loss.item()

        device = self._device
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

    def _zero_op_batch_grads(self, set_to_none=False):
        for module in self._model.modules():
            operation = getattr(module, 'operation', None)
            if operation is None:
                continue
            op_results = operation.get_op_results()
            if op_results.get(OP_BATCH_GRADS, None):
                if set_to_none:
                    op_results.pop(OP_BATCH_GRADS, None)
                else:
                    op_results[OP_BATCH_GRADS] *= 0

    def _fisher_core(self, closure, outputs, targets):
        raise NotImplementedError

    def _register_fisher(self, scale=1.):
        """
        module.operation.get_op_results():
        {
            'diag': {'weight': torch.Tensor, 'bias': torch.Tensor},
            'kron': {'A': torch.Tensor, 'B': torch.Tensor},
            'block_diag': torch.Tensor,
            'unit_wise': torch.Tensor,
        }
        """
        model = self._model
        for module in model.modules():
            operation = getattr(module, 'operation', None)
            if operation is None:
                continue
            op_results = operation.get_op_results()
            kron = diag = unit = None
            if OP_COV_KRON in op_results:
                rst = op_results[OP_COV_KRON]
                kron = Kron(rst['A'], rst['B'])
            if OP_COV_DIAG in op_results:
                rst = op_results[OP_COV_DIAG]
                diag = Diag(rst.get('weight', None), rst.get('bias', None))
            if OP_COV_UNIT_WISE in op_results:
                rst = op_results[OP_COV_UNIT_WISE]
                unit = UnitWise(rst)
            operation.clear_op_results()
            # move layer-wise/kron/diag fisher
            self._accumulate_fisher(
                module,
                _COV_LAYER_WISE,
                kron=kron,
                diag=diag,
                unit=unit,
                scale=scale
            )
            # move block_diag fvp
            self._accumulate_fvp(module, _CVP_LAYER_WISE, scale)

        # move full fisher
        self._accumulate_fisher(model, _COV_FULL, scale=scale)
        # move full fvp
        self._accumulate_fvp(model, _CVP_FULL, scale)

    def _accumulate_fisher(
            self,
            module,
            data_src_attr,
            kron=None,
            diag=None,
            unit=None,
            scale=1.
    ):
        dst_attr = self.fisher_attr
        data = getattr(module, data_src_attr, None)
        if all(v is None for v in [data, kron, diag, unit]):
            return
        new_fisher = SymMatrix(data, kron, diag, unit).scaling(scale)
        dst_fisher = getattr(module, dst_attr, None)
        if dst_fisher is None:
            setattr(module, dst_attr, new_fisher)
        else:
            # this must be __iadd__ to preserve inv
            dst_fisher += new_fisher
        if data is not None:
            delattr(module, data_src_attr)

    def _accumulate_fvp(self, module, src_attr, scale=1.):
        dst_attr = self.fvp_attr
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
            closure(lambda: F.nll_loss(log_probs, _targets[:, i], reduction='sum'),
                    grad_scale=sqrt_probs[:, i])


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
            closure(lambda: F.nll_loss(log_probs, targets, reduction='sum'),
                    grad_scale=1 / self.n_mc_samples)


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
            closure(lambda: 0.5 * (outputs - targets).norm(dim=1).sum(),
                    grad_scale=1 / self.n_mc_samples)


class FisherEmpMSE(_FisherMSE):
    @property
    def fisher_type(self):
        return FISHER_EMP

    def _fisher_core(self, closure, outputs, targets):
        closure(lambda: 0.5 * (outputs - targets).norm(dim=1).sum())


def _module_batch_grads(modules):
    rst = []
    for module in modules:
        operation = getattr(module, 'operation', None)
        if operation is None:
            continue
        op_results = operation.get_op_results()
        batch_grads = op_results.get(OP_BATCH_GRADS, None)
        if batch_grads is None:
            continue
        rst.append((module, batch_grads))
    return rst


def _module_batch_flatten_grads(modules):
    rst = []
    for module, batch_grads in _module_batch_grads(modules):
        batch_flatten_grads = torch.cat(
            [g.flatten(start_dim=1) for g in batch_grads.values()],
            dim=1
        )
        rst.append((module, batch_flatten_grads))
    return rst


def _module_batch_gvp(modules, vec):
    rst = []
    pointer = 0
    for module, batch_grads in _module_batch_grads(modules):
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


def _full_covariance(model, modules):
    if len(modules) == 0:
        return
    batch_all_g = []
    for _, batch_g in _module_batch_flatten_grads(modules):
        batch_all_g.append(batch_g)
    batch_all_g = torch.cat(batch_all_g, dim=1)  # n x p_all
    new_cov_full = torch.matmul(batch_all_g.T, batch_all_g)  # p_all x p_all
    cov_full = getattr(model, _COV_FULL, None)
    if cov_full is not None:
        new_cov_full += cov_full
    setattr(model, _COV_FULL, new_cov_full)


def _layer_wise_covariance(modules):
    if len(modules) == 0:
        return
    for module, batch_g in _module_batch_flatten_grads(modules):
        new_cov_block = torch.matmul(batch_g.T, batch_g)  # p_all x p_all
        cov_block = getattr(module, _COV_LAYER_WISE, None)
        if cov_block is not None:
            new_cov_block += cov_block
        setattr(module, _COV_LAYER_WISE, new_cov_block)


def _full_cvp(model, modules, vec):
    """
    g: n x p
    v: p
    c = sum[gg^t]: p x p
    cvp = sum[gg^t]v = sum[g(g^t)v]: p
    """
    if len(modules) == 0:
        return
    # compute batched (g^t)v
    batch_all_gvp = None
    for module, batch_gvp in _module_batch_gvp(modules, vec):
        if batch_all_gvp is None:
            batch_all_gvp = batch_gvp
        else:
            batch_all_gvp += batch_gvp

    # compute cvp = sum[g(g^t)v]
    new_cvp = []
    for module, batch_grads in _module_batch_grads(modules):
        for b_g in batch_grads.values():
            new_cvp.append(torch.einsum('n...,n->...', b_g, batch_all_gvp))
    cvp = getattr(model, _CVP_FULL, None)
    if cvp is not None:
        new_cvp = [v1 + v2 for v1, v2 in zip(new_cvp, cvp)]
    setattr(model, _CVP_FULL, new_cvp)


def _layer_wise_cvp(modules, vec):
    """
    g: n x p
    v: p
    c = sum[gg^t]: p x p
    cvp = sum[gg^t]v = sum[g(g^t)v]: p
    """
    if len(modules) == 0:
        return
    batch_gvp_dict = {k: v for k, v in _module_batch_gvp(modules, vec)}
    for module, batch_grads in _module_batch_grads(modules):
        new_cvp = []
        # compute cvp = sum[g(g^t)v]
        batch_gvp = batch_gvp_dict[module]
        for b_g in batch_grads.values():
            new_cvp.append(torch.einsum('n...,n->...', b_g, batch_gvp))
        cvp = getattr(module, _CVP_LAYER_WISE, None)
        if cvp is not None:
            new_cvp = [v1 + v2 for v1, v2 in zip(new_cvp, cvp)]
        setattr(module, _CVP_LAYER_WISE, new_cvp)


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


fisher_for_cross_entropy = partial(fisher, loss_type=LOSS_CROSS_ENTROPY, fvp=False)
fisher_for_mse = partial(fisher, loss_type=LOSS_MSE, fvp=False)
fvp_for_cross_entropy = partial(fisher, loss_type=LOSS_CROSS_ENTROPY, fvp=True)
fvp_for_mse = partial(fisher, loss_type=LOSS_MSE, fvp=True)


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


fisher_eig_for_cross_entropy = partial(fisher_eig, loss_type=LOSS_CROSS_ENTROPY)
fisher_eig_for_mse = partial(fisher_eig, loss_type=LOSS_MSE)


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


fisher_free_for_cross_entropy = partial(fisher_free, loss_type=LOSS_CROSS_ENTROPY)
fisher_free_for_mse = partial(fisher_free, loss_type=LOSS_MSE)


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
