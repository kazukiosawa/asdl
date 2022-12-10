from typing import List, Iterable, Tuple

import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils import vector_to_parameters
from torch import Tensor
from torch.linalg import solve_triangular
from .prec_grad_maker import PreconditionedGradientMaker, PreconditioningConfig


__all__ = ['PsgdGradientMaker', 'KronPsgdGradientMaker']


def parameters_to_vector(parameters: Iterable[Tensor]) -> Tensor:
    # torch.nn.utils.parameters_to_vector uses param.view(-1) which doesn't work
    # with non-contiguous parameters
    vec = []
    for param in parameters:
        vec.append(param.reshape(-1))
    return torch.cat(vec)


class PsgdGradientMaker(PreconditionedGradientMaker):
    r"""GradientMaker for calculating the preconditioned gradient by `PSGD <https://arxiv.org/abs/1512.04202>`_
    with a full preconditioning matrix.

    .. note::

        PsgdGradientMaker constructs a :math:`P\times P` triangular matrix
        (:math:`P`: number of trainable parameters of the `model`),
        which is the Cholesky factor of the preconditioning matrix.
        Therefore, it easily causes an out-of-memory error and is not recommend to use
        unless `model` is a very small network.
        :ref:`KronPsgdGradientMaker <kron_psgd_maker>` is recommended for most practical cases.

    Args:
        model (Module): Target module to calculate gradient
        config (PreconditioningConfig): Configuration for gradient preconditioning
        precond_lr (float, optional): The learning rate for updating the Cholesky factor of
            the preconditioning matrix. (default: 0.01)
        init_scale (float, optional): The Cholesky factor of the preconditioning matrix will
            be initialized by an identity matrix multiplied by this value. (default: 1.)
        use_functorch (bool, optional): If True, the Hessian-vector product will be calculated
            by using `functorch <https://pytorch.org/functorch/stable/>`_. (default: False)
    """
    _supported_classes = (nn.Linear, nn.Conv2d)

    def __init__(self, model: nn.Module, config: PreconditioningConfig,
                 precond_lr: float = 0.01, init_scale: float = 1., use_functorch=False):
        super().__init__(model, config)
        self.precond_lr = precond_lr
        self.init_scale = init_scale
        self._init_cholesky_factors()
        self.use_functorch = use_functorch

    def _init_cholesky_factors(self):
        num_params = sum([p.numel() for p in self.module_dict.parameters()])
        init_scale = self.init_scale
        self.cholesky_factor = init_scale * torch.eye(num_params, device=self.device)

    def do_forward_and_backward(self, step=None):
        return not self.do_update_preconditioner(step)

    def update_preconditioner(self, retain_graph=False):
        if self.use_functorch:
            vs = tuple([torch.randn_like(p) for p in self.module_dict.parameters()])
            Hvs = self.loss_hvp(tangents=vs)
        else:
            self.forward()
            params = list(self.module_dict.parameters())
            grads = torch.autograd.grad(self._loss, params, create_graph=True)
            for p, g in zip(params, grads):
                p.grad = g
            vs = [torch.randn_like(p) for p in params]
            Hvs = list(torch.autograd.grad(grads, params, vs, retain_graph=retain_graph))
        self._update_preconditioner(list(vs), list(Hvs))

    @torch.no_grad()
    def _update_preconditioner(self, dxs: List[Tensor], dgs: List[Tensor], eps=1.2e-38):
        dx = parameters_to_vector(dxs)
        dg = parameters_to_vector(dgs)
        Q = self.cholesky_factor

        a = Q.mv(dg)
        b = solve_triangular(Q.T, dx.unsqueeze(-1), upper=False).squeeze()

        grad = torch.triu(torch.outer(a, a) - torch.outer(b, b))
        lr = self.precond_lr / (grad.abs().max() + eps)

        Q.sub_(grad.mm(Q), alpha=float(lr))

    @torch.no_grad()
    def precondition(self):
        grads = [p.grad for p in self.module_dict.parameters()]
        g = parameters_to_vector(grads)
        Q = self.cholesky_factor
        vector_to_parameters(Q.T.mv(Q.mv(g)), grads)

    def criterion(self, n_samples=1):
        total = 0
        for i in range(n_samples):
            vs = tuple([torch.randn_like(p) for p in self.module_dict.parameters()])
            Hvs = self.loss_hvp(tangents=vs)
            total += self._criterion(vs, Hvs)
        return total / n_samples

    @torch.no_grad()
    def _criterion(self, dxs: Tuple[Tensor], dgs: Tuple[Tensor]):
        dx = parameters_to_vector(dxs)
        dg = parameters_to_vector(dgs)
        Q = self.cholesky_factor

        res = dx.dot(solve_triangular(Q, solve_triangular(Q.T, dx.unsqueeze(-1), upper=False), upper=True).squeeze())
        res += dg.dot(Q.T.mv(Q.mv(dg)))
        return float(res)


class KronPsgdGradientMaker(PsgdGradientMaker):
    r"""GradientMaker for calculating the preconditioned gradient by `PSGD <https://arxiv.org/abs/1512.04202>`_
    with a layer-wise block-diagonal Kronecker-factored preconditioning matrix.

    Args:
        model (Module): Target module to calculate gradient
        config (PreconditioningConfig): Configuration for gradient preconditioning
        precond_lr (float, optional): The learning rate for updating the Cholesky factor of
            the preconditioning matrix. (default: 0.01)
        init_scale (float, optional): The Cholesky factor of the preconditioning matrix will
            be initialized by an identity matrix multiplied by this value. (default: 1.)
        use_functorch (bool, optional): If True, the Hessian-vector product will be calculated
            by using `functorch <https://pytorch.org/functorch/stable/>`_. (default: False)
    """
    def _init_cholesky_factors(self):
        init_scale = self.init_scale
        self.cholesky_factors = {}
        for module in self.module_dict.children():
            in_dim = int(np.prod(module.weight.shape[1:]))
            out_dim = module.weight.shape[0]
            if module.bias is not None:
                in_dim += 1
            Ql = init_scale * torch.eye(out_dim, device=self.device)
            Qr = init_scale * torch.eye(in_dim, device=self.device)
            self.cholesky_factors[module] = (Ql, Qr)

    @torch.no_grad()
    def _update_preconditioner(self, dxs: List[Tensor], dgs: List[Tensor]):
        for module in self.module_dict.children():
            dX = dxs.pop(0)
            dG = dgs.pop(0)
            if isinstance(module, nn.Conv2d):
                dX = dX.flatten(start_dim=1)
                dG = dG.flatten(start_dim=1)
            if module.bias is not None:
                dX = torch.cat([dX, dxs.pop(0).unsqueeze(-1)], dim=1)
                dG = torch.cat([dG, dgs.pop(0).unsqueeze(-1)], dim=1)
            update_precond_kron(*self.cholesky_factors[module], dX, dG, step=self.precond_lr)
            del dX, dG
        if len(dxs) != 0:
            raise ValueError('dxs are still remaining.')
        if len(dgs) != 0:
            raise ValueError('dgs are still remaining.')

    @torch.no_grad()
    def precondition(self):
        grads = [p.grad for p in self.module_dict.parameters()]
        for module in self.module_dict.children():
            G = grads.pop(0)
            if isinstance(module, nn.Conv2d):
                G = G.flatten(start_dim=1)
            if module.bias is not None:
                G = torch.cat([G, grads.pop(0).unsqueeze(-1)], dim=1)
            G = precond_grad_kron(*self.cholesky_factors[module], G)
            if module.bias is not None:
                module.weight.grad.copy_(G[:, :-1].view_as(module.weight))
                module.bias.grad.copy_(G[:, -1].view_as(module.bias))
            else:
                module.weight.grad.copy_(G.view_as(module.weight))
            del G
        if len(grads) != 0:
            raise ValueError('grads are still remaining')


"""
Pytorch functions for preconditioned SGD
@author: XILIN LI, lixilinx@gmail.com
Adopted from https://github.com/lixilinx/psgd_torch

Replaced deprecated functions to the latest ones
"""


def update_precond_kron(Ql, Qr, dX, dG, step=0.01, _tiny=1.2e-38):
    """
    Update Kronecker product preconditioner P = kron_prod(Qr^T*Qr, Ql^T*Ql)
    Either Ql or Qr can be sparse, and the code can choose the right update rule.
    dX: perturbation of (matrix) parameter
    dG: perturbation of (matrix) gradient
    step: update step size
    _tiny: an offset to avoid division by zero
    """
    m, n = Ql.shape
    p, q = Qr.shape
    if m == n:  # left is dense
        if p == q:  # (dense, dense) format
            return _update_precond_dense_dense(Ql, Qr, dX, dG, step, _tiny)
        elif p == 2:  # (dense, normalization) format
            return _update_precond_norm_dense(Qr, Ql, dX.t(), dG.t(), step, _tiny)[::-1]
        elif p == 1:  # (dense, scaling) format
            return _update_precond_dense_scale(Ql, Qr, dX, dG, step, _tiny)
        else:
            raise Exception('Unknown Kronecker product preconditioner')
    elif m == 2:  # left is normalization
        if p == q:  # (normalization, dense) format
            return _update_precond_norm_dense(Ql, Qr, dX, dG, step, _tiny)
        elif p == 1:  # (normalization, scaling) format
            return _update_precond_norm_scale(Ql, Qr, dX, dG, step, _tiny)
        else:
            raise Exception('Unknown Kronecker product preconditioner')
    elif m == 1:  # left is scaling
        if p == q:  # (scaling, dense) format
            return _update_precond_dense_scale(Qr, Ql, dX.t(), dG.t(), step, _tiny)[::-1]
        elif p == 2:  # (scaling, normalization) format
            return _update_precond_norm_scale(Qr, Ql, dX.t(), dG.t(), step, _tiny)[::-1]
        else:
            raise Exception('Unknown Kronecker product preconditioner')
    else:
        raise Exception('Unknown Kronecker product preconditioner')


def precond_grad_kron(Ql, Qr, Grad):
    """
    return preconditioned gradient using Kronecker product preconditioner P = kron_prod(Qr^T*Qr, Ql^T*Ql)
    Either Ql or Qr can be sparse, and the code can choose the right way to precondition the gradient
    Grad: (matrix) gradient
    """
    m, n = Ql.shape
    p, q = Qr.shape
    if m == n:  # left is dense
        if p == q:  # (dense, dense) format
            return _precond_grad_dense_dense(Ql, Qr, Grad)
        elif p == 2:  # (dense, normalization) format
            return _precond_grad_norm_dense(Qr, Ql, Grad.t()).t()
        elif p == 1:  # (dense, scaling) format
            return _precond_grad_dense_scale(Ql, Qr, Grad)
        else:
            raise Exception('Unknown Kronecker product preconditioner')
    elif m == 2:  # left is normalization
        if p == q:  # (normalization, dense) format
            return _precond_grad_norm_dense(Ql, Qr, Grad)
        elif p == 1:  # (normalization, scaling) format
            return _precond_grad_norm_scale(Ql, Qr, Grad)
        else:
            raise Exception('Unknown Kronecker product preconditioner')
    elif m == 1:  # left is scaling
        if p == q:  # (scaling, dense) format
            return _precond_grad_dense_scale(Qr, Ql, Grad.t()).t()
        elif p == 2:  # (scaling, normalization) format
            return _precond_grad_norm_scale(Qr, Ql, Grad.t()).t()
        else:
            raise Exception('Unknown Kronecker product preconditioner')
    else:
        raise Exception('Unknown Kronecker product preconditioner')


@torch.jit.script
def _update_precond_dense_dense(Ql, Qr, dX, dG, step=0.01, _tiny=1.2e-38):
    # type: (Tensor, Tensor, Tensor, Tensor, float, float) -> None
    """
    update Kronecker product preconditioner P = kron_prod(Qr^T*Qr, Ql^T*Ql)
    Ql: (left side) Cholesky factor of preconditioner with positive diagonal entries
    Qr: (right side) Cholesky factor of preconditioner with positive diagonal entries
    dX: perturbation of (matrix) parameter
    dG: perturbation of (matrix) gradient
    step: update step size normalized to range [0, 1]
    _tiny: an offset to avoid division by zero
    """
    max_l = torch.max(torch.diag(Ql))
    max_r = torch.max(torch.diag(Qr))

    rho = torch.sqrt(max_l / max_r)
    Ql /= rho
    Qr *= rho

    A = Ql.mm(dG.mm(Qr.t()))
    Bt = solve_triangular(Ql, solve_triangular(Qr, dX.t(), upper=True).t(), upper=True)

    grad1 = torch.triu(A.mm(A.T) - Bt.mm(Bt.T))
    grad2 = torch.triu(A.T.mm(A) - Bt.T.mm(Bt))

    step1 = step / (torch.max(torch.abs(grad1)) + _tiny)
    step2 = step / (torch.max(torch.abs(grad2)) + _tiny)

    Ql.sub_(grad1.mm(Ql), alpha=float(step1))
    Qr.sub_(grad2.mm(Qr), alpha=float(step2))


@torch.jit.script
def _precond_grad_dense_dense(Ql, Qr, Grad):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    """
    return preconditioned gradient using Kronecker product preconditioner
    Ql: (left side) Cholesky factor of preconditioner
    Qr: (right side) Cholesky factor of preconditioner
    Grad: (matrix) gradient
    """
    return torch.linalg.multi_dot([Ql.t(), Ql, Grad, Qr.t(), Qr])
    #return torch.chain_matmul(Ql.t(), Ql, Grad, Qr.t(), Qr)


###############################################################################
# (normalization, dense) format Kronecker product preconditioner
@torch.jit.script
def _update_precond_norm_dense(ql, Qr, dX, dG, step=0.01, _tiny=1.2e-38):
    # type: (Tensor, Tensor, Tensor, Tensor, float, float) -> Tuple[Tensor, Tensor]
    """
    update (normalization, dense) Kronecker product preconditioner P = kron_prod(Qr^T*Qr, Ql^T*Ql), where
    dX and dG have shape (M, N)
    ql has shape (2, M)
    Qr has shape (N, N)
    ql[0] is the diagonal part of Ql
    ql[1,0:-1] is the last column of Ql, excluding the last entry
    dX is perturbation of (matrix) parameter
    dG is perturbation of (matrix) gradient
    step: update step size normalized to range [0, 1]
    _tiny: an offset to avoid division by zero
    """
    # make sure that Ql and Qr have similar dynamic range
    max_l = torch.max(ql[0])
    max_r = torch.max(torch.diag(Qr))
    rho = torch.sqrt(max_l / max_r)
    ql /= rho
    Qr *= rho

    # refer to https://arxiv.org/abs/1512.04202 for details
    A = ql[0:1].t() * dG + ql[1:].t().mm(dG[-1:])  # Ql*dG
    A = A.mm(Qr.t())

    Bt = dX / ql[0:1].t()
    Bt[-1:] -= (ql[1:] / (ql[0:1] * ql[0, -1])).mm(dX)
    Bt = torch.triangular_solve(Bt.t(), Qr, upper=True, transpose=True)[0].t()

    grad1_diag = torch.sum(A * A, dim=1) - torch.sum(Bt * Bt, dim=1)
    grad1_bias = A[:-1].mm(A[-1:].t()) - Bt[:-1].mm(Bt[-1:].t())
    grad1_bias = torch.cat([torch.squeeze(grad1_bias), grad1_bias.new_zeros(1)])

    step1 = step / (torch.max(torch.max(torch.abs(grad1_diag)),
                              torch.max(torch.abs(grad1_bias))) + _tiny)
    new_ql0 = ql[0] - step1 * grad1_diag * ql[0]
    new_ql1 = ql[1] - step1 * (grad1_diag * ql[1] + ql[0, -1] * grad1_bias)

    grad2 = torch.triu(A.t().mm(A) - Bt.t().mm(Bt))
    step2 = step / (torch.max(torch.abs(grad2)) + _tiny)

    return torch.stack((new_ql0, new_ql1)), Qr - step2 * grad2.mm(Qr)


@torch.jit.script
def _precond_grad_norm_dense(ql, Qr, Grad):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    """
    return preconditioned gradient using (normalization, dense) Kronecker product preconditioner
    Suppose Grad has shape (M, N)
    ql[0] is the diagonal part of Ql
    ql[1, 0:-1] is the last column of Ql, excluding the last entry
    Qr: shape (N, N), Cholesky factor of right preconditioner
    Grad: (matrix) gradient
    """
    preG = ql[0:1].t() * Grad + ql[1:].t().mm(Grad[-1:])  # Ql*Grad
    preG = torch.chain_matmul(preG, Qr.t(), Qr)
    add_last_row = ql[1:].mm(preG)  # use it to modify the last row
    preG *= ql[0:1].t()
    preG[-1:] += add_last_row

    return preG


###############################################################################
# (normalization, scaling) Kronecker product preconditioner
# the left one is a normalization preconditioner; the right one is a scaling preconditioner
@torch.jit.script
def _update_precond_norm_scale(ql, qr, dX, dG, step=0.01, _tiny=1.2e-38):
    # type: (Tensor, Tensor, Tensor, Tensor, float, float) -> Tuple[Tensor, Tensor]
    """
    update (normalization, scaling) preconditioner P = kron_prod(Qr^T*Qr, Ql^T*Ql), where
    dX and dG have shape (M, N)
    ql has shape (2, M)
    qr has shape (1, N)
    ql[0] is the diagonal part of Ql
    ql[1, 0:-1] is the last column of Ql, excluding the last entry
    qr is the diagonal part of Qr
    dX is perturbation of (matrix) parameter
    dG is perturbation of (matrix) gradient
    step: update step size
    _tiny: an offset to avoid division by zero
    """
    # make sure that Ql and Qr have similar dynamic range
    max_l = torch.max(ql[0])
    max_r = torch.max(qr)  # qr always is positive
    rho = torch.sqrt(max_l / max_r)
    ql /= rho
    qr *= rho

    # refer to https://arxiv.org/abs/1512.04202 for details
    A = ql[0:1].t() * dG + ql[1:].t().mm(dG[-1:])  # Ql*dG
    A *= qr  # Ql*dG*Qr

    Bt = dX / ql[0:1].t()
    Bt[-1:] -= (ql[1:] / (ql[0:1] * ql[0, -1])).mm(dX)
    Bt /= qr  # Ql^(-T)*dX*Qr^(-1)

    grad1_diag = torch.sum(A * A, dim=1) - torch.sum(Bt * Bt, dim=1)
    grad1_bias = A[:-1].mm(A[-1:].t()) - Bt[:-1].mm(Bt[-1:].t())
    grad1_bias = torch.cat([torch.squeeze(grad1_bias), grad1_bias.new_zeros(1)])

    step1 = step / (torch.max(torch.max(torch.abs(grad1_diag)),
                              torch.max(torch.abs(grad1_bias))) + _tiny)
    new_ql0 = ql[0] - step1 * grad1_diag * ql[0]
    new_ql1 = ql[1] - step1 * (grad1_diag * ql[1] + ql[0, -1] * grad1_bias)

    grad2 = torch.sum(A * A, dim=0, keepdim=True) - torch.sum(Bt * Bt, dim=0, keepdim=True)
    step2 = step / (torch.max(torch.abs(grad2)) + _tiny)

    return torch.stack((new_ql0, new_ql1)), qr - step2 * grad2 * qr


@torch.jit.script
def _precond_grad_norm_scale(ql, qr, Grad):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    """
    return preconditioned gradient using (normalization, scaling) Kronecker product preconditioner
    Suppose Grad has shape (M, N)
    ql has shape (2, M)
    qr has shape (1, N)
    ql[0] is the diagonal part of Ql
    ql[1, 0:-1] is the last column of Ql, excluding the last entry
    qr is the diagonal part of Qr
    Grad: (matrix) gradient
    """
    preG = ql[0:1].t() * Grad + ql[1:].t().mm(Grad[-1:])  # Ql*Grad
    preG *= (qr * qr)  # Ql*Grad*Qr^T*Qr
    add_last_row = ql[1:].mm(preG)  # use it to modify the last row
    preG *= ql[0:1].t()
    preG[-1:] += add_last_row

    return preG


###############################################################################
@torch.jit.script
def _update_precond_dense_scale(Ql, qr, dX, dG, step=0.01, _tiny=1.2e-38):
    # type: (Tensor, Tensor, Tensor, Tensor, float, float) -> Tuple[Tensor, Tensor]
    """
    update (dense, scaling) preconditioner P = kron_prod(Qr^T*Qr, Ql^T*Ql), where
    dX and dG have shape (M, N)
    Ql has shape (M, M)
    qr has shape (1, N)
    qr is the diagonal part of Qr
    dX is perturbation of (matrix) parameter
    dG is perturbation of (matrix) gradient
    step: update step size
    _tiny: an offset to avoid division by zero
    """
    max_l = torch.max(torch.diag(Ql))
    max_r = torch.max(qr)

    rho = torch.sqrt(max_l / max_r)
    Ql /= rho
    qr *= rho

    A = Ql.mm(dG * qr)
    Bt = torch.triangular_solve(dX / qr, Ql, upper=True, transpose=True)[0]

    grad1 = torch.triu(A.mm(A.t()) - Bt.mm(Bt.t()))
    grad2 = torch.sum(A * A, dim=0, keepdim=True) - torch.sum(Bt * Bt, dim=0, keepdim=True)

    step1 = step / (torch.max(torch.abs(grad1)) + _tiny)
    step2 = step / (torch.max(torch.abs(grad2)) + _tiny)

    return Ql - step1 * grad1.mm(Ql), qr - step2 * grad2 * qr


@torch.jit.script
def _precond_grad_dense_scale(Ql, qr, Grad):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    """
    return preconditioned gradient using (dense, scaling) Kronecker product preconditioner
    Suppose Grad has shape (M, N)
    Ql: shape (M, M), (left side) Cholesky factor of preconditioner
    qr: shape (1, N), defines a diagonal matrix for output feature scaling
    Grad: (matrix) gradient
    """
    return torch.chain_matmul(Ql.t(), Ql, Grad * (qr * qr))


###############################################################################
@torch.jit.script
def update_precond_splu(L12, l3, U12, u3, dxs, dgs, step=0.01, _tiny=1.2e-38):
    # type: (Tensor,Tensor,Tensor,Tensor, List[Tensor],List[Tensor], float,float) -> Tuple[Tensor,Tensor,Tensor,Tensor]
    """
    update sparse LU preconditioner P = Q^T*Q, where
    Q = L*U,
    L12 = [L1; L2]
    U12 = [U1, U2]
    L = [L1, 0; L2, diag(l3)]
    U = [U1, U2; 0, diag(u3)]
    l3 and u3 are column vectors
    dxs: a list of random perturbation on parameters
    dgs: a list of resultant perturbation on gradients
    step: update step size normalized to range [0, 1]
    _tiny: an offset to avoid division by zero
    """
    # make sure that L and U have similar dynamic range
    max_l = torch.max(torch.max(torch.diag(L12)), torch.max(l3))
    max_u = torch.max(torch.max(torch.diag(U12)), torch.max(u3))
    rho = torch.sqrt(max_l / max_u)
    L12 /= rho
    l3 /= rho
    U12 *= rho
    u3 *= rho

    # extract the blocks
    r = U12.shape[0]
    L1 = L12[:r]
    L2 = L12[r:]
    U1 = U12[:, :r]
    U2 = U12[:, r:]

    dx = torch.cat([torch.reshape(x, [-1, 1]) for x in dxs])  # a tall column vector
    dg = torch.cat([torch.reshape(g, [-1, 1]) for g in dgs])  # a tall column vector

    # U*dg
    Ug1 = U1.mm(dg[:r]) + U2.mm(dg[r:])
    Ug2 = u3 * dg[r:]
    # Q*dg
    Qg1 = L1.mm(Ug1)
    Qg2 = L2.mm(Ug1) + l3 * Ug2
    # inv(U^T)*dx
    iUtx1 = torch.triangular_solve(dx[:r], U1, upper=True, transpose=True)[0]
    iUtx2 = (dx[r:] - U2.t().mm(iUtx1)) / u3
    # inv(Q^T)*dx
    iQtx2 = iUtx2 / l3
    iQtx1 = torch.triangular_solve(iUtx1 - L2.t().mm(iQtx2), L1, upper=False, transpose=True)[0]
    # L^T*Q*dg
    LtQg1 = L1.t().mm(Qg1) + L2.t().mm(Qg2)
    LtQg2 = l3 * Qg2
    # P*dg
    Pg1 = U1.t().mm(LtQg1)
    Pg2 = U2.t().mm(LtQg1) + u3 * LtQg2
    # inv(L)*inv(Q^T)*dx
    iLiQtx1 = torch.triangular_solve(iQtx1, L1, upper=False)[0]
    iLiQtx2 = (iQtx2 - L2.mm(iLiQtx1)) / l3
    # inv(P)*dx
    iPx2 = iLiQtx2 / u3
    iPx1 = torch.triangular_solve(iLiQtx1 - U2.mm(iPx2), U1, upper=True)[0]

    # update L
    grad1 = Qg1.mm(Qg1.t()) - iQtx1.mm(iQtx1.t())
    grad1 = torch.tril(grad1)
    grad2 = Qg2.mm(Qg1.t()) - iQtx2.mm(iQtx1.t())
    grad3 = Qg2 * Qg2 - iQtx2 * iQtx2
    max_abs_grad = torch.max(torch.abs(grad1))
    max_abs_grad = torch.max(max_abs_grad, torch.max(torch.abs(grad2)))
    max_abs_grad = torch.max(max_abs_grad, torch.max(torch.abs(grad3)))
    step0 = step / (max_abs_grad + _tiny)
    newL1 = L1 - step0 * grad1.mm(L1)
    newL2 = L2 - step0 * grad2.mm(L1) - step0 * grad3 * L2
    newl3 = l3 - step0 * grad3 * l3

    # update U
    grad1 = Pg1.mm(dg[:r].t()) - dx[:r].mm(iPx1.t())
    grad1 = torch.triu(grad1)
    grad2 = Pg1.mm(dg[r:].t()) - dx[:r].mm(iPx2.t())
    grad3 = Pg2 * dg[r:] - dx[r:] * iPx2
    max_abs_grad = torch.max(torch.abs(grad1))
    max_abs_grad = torch.max(max_abs_grad, torch.max(torch.abs(grad2)))
    max_abs_grad = torch.max(max_abs_grad, torch.max(torch.abs(grad3)))
    step0 = step / (max_abs_grad + _tiny)
    newU1 = U1 - U1.mm(step0 * grad1)
    newU2 = U2 - U1.mm(step0 * grad2) - step0 * grad3.t() * U2
    newu3 = u3 - step0 * grad3 * u3

    return torch.cat([newL1, newL2], dim=0), newl3, torch.cat([newU1, newU2], dim=1), newu3


@torch.jit.script
def precond_grad_splu(L12, l3, U12, u3, grads):
    # type: (Tensor,Tensor,Tensor,Tensor, List[Tensor]) -> List[Tensor]
    """
    return preconditioned gradient with sparse LU preconditioner
    where P = Q^T*Q,
    Q = L*U,
    L12 = [L1; L2]
    U12 = [U1, U2]
    L = [L1, 0; L2, diag(l3)]
    U = [U1, U2; 0, diag(u3)]
    l3 and u3 are column vectors
    grads: a list of gradients to be preconditioned
    """
    grad = [torch.reshape(g, [-1, 1]) for g in grads]  # a list of column vector
    lens = [g.shape[0] for g in grad]  # length of each column vector
    grad = torch.cat(grad)  # a tall column vector

    r = U12.shape[0]
    L1 = L12[:r]
    L2 = L12[r:]
    U1 = U12[:, :r]
    U2 = U12[:, r:]

    # U*g
    Ug1 = U1.mm(grad[:r]) + U2.mm(grad[r:])
    Ug2 = u3 * grad[r:]
    # Q*g
    Qg1 = L1.mm(Ug1)
    Qg2 = L2.mm(Ug1) + l3 * Ug2
    # L^T*Q*g
    LtQg1 = L1.t().mm(Qg1) + L2.t().mm(Qg2)
    LtQg2 = l3 * Qg2
    # P*g
    pre_grad = torch.cat([U1.t().mm(LtQg1),
                          U2.t().mm(LtQg1) + u3 * LtQg2])

    pre_grads = []  # restore pre_grad to its original shapes
    idx = 0
    for i in range(len(grads)):
        pre_grads.append(torch.reshape(pre_grad[idx: idx + lens[i]], grads[i].shape))
        idx = idx + lens[i]

    return pre_grads