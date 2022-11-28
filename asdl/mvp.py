import math
from typing import List, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from .vector import ParamVector, normalization, orthnormal

__all__ = [
    'power_method',
    'stochastic_lanczos_quadrature',
    'conjugate_gradient_method',
    'quadratic_form'
]


def power_method(mvp_fn: Callable[[ParamVector], ParamVector],
                 model: nn.Module,
                 top_n=1,
                 max_iters=100,
                 tol=1e-7,
                 is_distributed=False,
                 print_progress=False,
                 random_seed=None):
    # main logic is adopted from https://github.com/amirgholami/PyHessian/blob/master/pyhessian/hessian.py
    # modified interface and format
    # modified for various matrices and distributed memory run

    if top_n < 1:
        raise ValueError(f'top_n has to be >=1. Got {top_n}.')
    if max_iters < 1:
        raise ValueError(f'max_iters has to be >=1. Got {max_iters}.')

    params = [p for p in model.parameters() if p.requires_grad]

    def _report(message):
        if print_progress:
            print(message)

    eigvals: List[float] = []
    eigvecs: List[ParamVector] = []
    for i in range(top_n):
        _report(f'start power iteration for lambda({i+1}).')
        vec = ParamVector(params, [torch.randn_like(p) for p in params])
        if is_distributed:
            vec = vec.get_flatten_vector()
            dist.broadcast(vec, src=0)
            vec = ParamVector(params, vec)

        eigval = None
        last_eigval = None
        # power iteration
        for j in range(max_iters):
            vec = orthnormal(vec, eigvecs)
            Mv = _mvp(mvp_fn, vec, random_seed=random_seed)
            eigval = Mv.dot(vec).item()
            if j > 0:
                diff = abs(eigval - last_eigval) / (abs(last_eigval) + 1e-6)
                _report(f'{j}/{max_iters} diff={diff}')
                if diff < tol:
                    break
            last_eigval = eigval
            vec = Mv
        eigvals.append(eigval)
        eigvecs.append(vec)

    # sort both in descending order
    indices = np.argsort(eigvals)[::-1]
    eigvals = [eigvals[idx] for idx in indices]
    eigvecs = [eigvecs[idx] for idx in indices]

    return eigvals, eigvecs


def stochastic_lanczos_quadrature(mvp_fn: Callable[[ParamVector], ParamVector],
                                  model: nn.Module,
                                  n_v=1,
                                  num_iter=100,
                                  is_distributed=False,
                                  random_seed=None):
    # referenced from https://github.com/amirgholami/PyHessian/blob/master/pyhessian/hessian.py

    if n_v < 1:
        raise ValueError(f'n_v has to be >=1. Got {n_v}.')
    if num_iter < 1:
        raise ValueError(f'num_iter has to be >=1. Got {num_iter}.')

    params = [p for p in model.parameters() if p.requires_grad]
    device = next(model.parameters()).device
    eigval_list_full: List[List[float]] = []
    weight_list_full: List[List[float]] = []

    for k in range(n_v):
        vec = [torch.randint_like(p, high=2) for p in params]
        for v_i in vec:
            v_i[v_i==0] = -1
        vec = ParamVector(params, vec)
        vec = normalization(vec)
        if is_distributed:
            vec = vec.get_flatten_vector()
            dist.broadcast(vec, src=0)
            vec = ParamVector(params, vec)

        vec_list: List[ParamVector] = [vec]
        alpha_list: List[float] = []
        beta_list: List[float] = []
        for i in range(num_iter):
            if i==0:
                w_prime = _mvp(mvp_fn, vec, random_seed=random_seed)
                alpha = w_prime.dot(vec)
                alpha_list.append(alpha.item())
                w = w_prime.add(vec, alpha=-alpha)
            else:
                beta = torch.sqrt(w.dot(w))
                beta_list.append(beta.item())
                if beta.item() != 0.:
                    vec = orthnormal(w, vec_list)
                    vec_list.append(vec)
                else:
                    vec = ParamVector(params, [torch.randn_like(p) for p in params])
                    vec = orthnormal(vec, vec_list)
                    if is_distributed:
                        vec = vec.get_flatten_vector()
                        dist.broadcast(vec, src=0)
                        vec = ParamVector(params, vec)
                    vec_list.append(vec)
                w_prime = _mvp(mvp_fn, vec, random_seed=random_seed)
                alpha = w_prime.dot(vec)
                alpha_list.append(alpha.item())
                w = w_prime.add(vec, alpha=-alpha)
                w = w.add(vec_list[-2], alpha=-beta)
        
        T = torch.zeros(num_iter, num_iter).to(device)
        for i in range(num_iter):
            T[i, i] = alpha_list[i]
            if i < num_iter-1:
                T[i+1, i] = beta_list[i]
                T[i, i+1] = beta_list[i]
        eigval, eigvec = torch.linalg.eigh(T)
        weight_list = eigvec[0,:]**2
        eigval_list_full.append(eigval.tolist())
        weight_list_full.append(weight_list.tolist())
    
    return eigval_list_full, weight_list_full


def conjugate_gradient_method(mvp_fn: Callable[[ParamVector], ParamVector],
                              b: ParamVector,
                              init_x: ParamVector = None,
                              damping=1e-3,
                              max_iters=None,
                              tol=1e-8,
                              preconditioner=None,
                              print_progress=False,
                              random_seed=None) -> ParamVector:
    """
    Solve (A + d * I)x = b by conjugate gradient method.
    d: damping
    Return x when x is close enough to inv(A) * b.
    """
    if not isinstance(b, ParamVector):
        raise TypeError(f'b has to be an instance of {ParamVector}. {type(b)} is given.')

    if max_iters is None:
        max_iters = b.numel()

    def _call_mvp(v: ParamVector) -> ParamVector:
        return _mvp(mvp_fn, v, random_seed, damping)

    x = init_x
    if x is None:
        x = ParamVector(b.params(), [torch.zeros_like(p) for p in b.params()])
        r = b.copy()
    else:
        Ax = _call_mvp(x)
        r = b.add(Ax, alpha=-1)

    if preconditioner is None:
        p = r.copy()
        last_rz = r.dot(r)
    else:
        p = preconditioner.precondition(r)
        last_rz = r.dot(p)

    b_norm = b.norm()

    for i in range(max_iters):
        Ap = _call_mvp(p)
        alpha = last_rz / p.dot(Ap)
        x.add_(p, alpha)
        r.add_(Ap, -alpha)
        rr = r.dot(r)
        err = math.sqrt(rr) / b_norm
        if print_progress:
            print(f'{i+1}/{max_iters} err={err}')
        if err < tol:
            break
        if preconditioner is None:
            z = r
            rz = rr
        else:
            z = preconditioner.precondition(r)
            rz = r.dot(z)

        beta = rz / last_rz  # Fletcher-Reeves
        p = z.add(p, beta)
        last_rz = rz

    return x


def quadratic_form(mvp_fn: Callable[[ParamVector], ParamVector],
                   v: ParamVector,
                   random_seed=None,
                   damping=0):
    Av = _mvp(mvp_fn, v, random_seed, damping)
    return v.dot(Av)


def _mvp(mvp_fn: Callable[[ParamVector], ParamVector],
         vec: ParamVector,
         random_seed=None,
         damping=None) -> ParamVector:
    if random_seed:
        # for matrices that are not deterministic (e.g., fisher_mc)
        torch.manual_seed(random_seed)
    Mv = mvp_fn(vec)
    if damping:
        Mv.add_(vec, alpha=damping)
    return Mv
