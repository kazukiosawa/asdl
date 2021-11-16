import math
import copy
from typing import List, Callable

import torch
import torch.nn as nn
import torch.distributed as dist
from .vector import ParamVector, orthnormal

__all__ = [
    'power_method',
    'conjugate_gradient_method'
]


def power_method(mvp_fn: Callable[[ParamVector], ParamVector],
                 model: nn.Module,
                 top_n=1,
                 max_iters=100,
                 tol=1e-3,
                 is_distributed=False,
                 print_progress=False,
                 random_seed=None):
    # main logic is adopted from https://github.com/amirgholami/PyHessian/blob/master/pyhessian/hessian.py
    # modified interface and format
    # modified for various matrices and distributed memory run

    assert top_n >= 1
    assert max_iters >= 1

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
    eigvals, eigvecs = (list(t) for t in zip(*sorted(zip(eigvals, eigvecs))[::-1]))

    return eigvals, eigvecs


def conjugate_gradient_method(mvp_fn: Callable[[ParamVector], ParamVector],
                              b: ParamVector,
                              init_x: ParamVector = None,
                              damping=1e-3,
                              max_iters=None,
                              tol=1e-8,
                              preconditioner=None,
                              print_progress=False,
                              random_seed=None,
                              save_log=False):
    """
    Solve (A + d * I)x = b by conjugate gradient method.
    d: damping
    Return x when x is close enough to inv(A) * b.
    """
    if max_iters is None:
        n_dim = sum([_b.numel() for _b in b])
        max_iters = n_dim

    def _call_mvp(v: ParamVector) -> ParamVector:
        return _mvp(mvp_fn, v, random_seed, damping)

    x = init_x
    if x is None:
        x = ParamVector(b.params, [torch.zeros_like(p) for p in b.params])
        r = copy.deepcopy(b)
    else:
        Ax = _call_mvp(x)
        r = b.add(Ax, alpha=-1)

    if preconditioner is None:
        p = copy.deepcopy(r)
        last_rz = r.dot(r)
    else:
        p = preconditioner.precondition_vector(r)
        last_rz = r.dot(p)

    b_norm = b.norm()

    log = []
    for i in range(max_iters):
        Ap = _call_mvp(p)
        alpha = last_rz / p.dot(Ap)
        x = x.add(p, alpha)
        r = r.add(Ap, -alpha)
        rr = r.dot(r)
        err = math.sqrt(rr) / b_norm
        log.append({'step': i + 1, 'error': err})
        if print_progress:
            print(f'{i+1}/{max_iters} err={err}')
        if err < tol:
            break
        if preconditioner is None:
            z = r
            rz = rr
        else:
            z = preconditioner.precondition_vector(r)
            rz = r.dot(z)

        beta = rz / last_rz  # Fletcher-Reeves
        p = z.add(p, beta)
        last_rz = rz

    if save_log:
        return x, log
    else:
        return x


def _mvp(mvp_fn: Callable[[ParamVector], ParamVector],
         vec: ParamVector,
         random_seed=None,
         damping=None) -> ParamVector:
    if random_seed:
        # for matrices that are not deterministic (e.g., fisher_mc)
        torch.manual_seed(random_seed)
    Mv = mvp_fn(vec)
    if damping:
        Mv = Mv.add(vec, alpha=damping)
    return Mv
