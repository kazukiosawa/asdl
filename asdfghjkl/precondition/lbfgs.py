import math
import torch
from ..utils import *


__all__ = ['LBFGS']


class LBFGS:
    """
    A preconditioner class which implements L-BFGS (https://en.wikipedia.org/wiki/Limited-memory_BFGS)
    and SLIM-QN (https://openreview.net/forum?id=eo1barn2Xmd).
    The base logic is adapted from the LBFGSOptimizer class by the authors of the SLIM-QN paper.
    This implementation reduces the memory consumption of the LBFGSOptimizer by avoiding the creation
    of flatten vectors (for model parameters, gradients).
    This class is supposed to used with SGD optimizer.

    Example:
    >>> lbfgs = LBFGS(model.parameters())
    >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    >>> update_freq = 100
    >>> for i, (x, y) in enumerate(trainloader):
    >>>     optimizer.zero_grad()
    >>>     loss = F.cross_entropy(model(x), y)
    >>>     loss.backward()
    >>>     lbfgs.update_momentum()  # for SLIM-QN
    >>>     if (i + 1) % update_freq == 0:
    >>>         lbfgs.update_history()
    >>>     lbfgs.precondition()
    >>>     optimizer.step()
    """
    def __init__(self,
                 params,
                 max_hist_size=20,
                 beta1=0.,
                 beta2=0.,
                 damping=None,
                 weight_decay=None,
                 kl_clip=0.005,
                 tau_lb=0.01,
                 tau_ub=1.0,
                 rho_min=1e-4,
                 eps=1e-5):
        self.params = list(params)
        self.max_hist_size = max_hist_size
        self.history = {'params': [], 'grads': [], 'rho': []}
        self.momentum = {'params': None, 'grads': None}
        self.last_value = {'params': None, 'grads': None}
        self.beta1 = beta1
        self.beta2 = beta2
        self.damping = damping
        self.weight_decay = weight_decay
        self.kl_clip = kl_clip
        assert tau_lb < 1
        assert tau_ub >= 1
        self.tau_lb = tau_lb
        self.tau_ub = tau_ub
        self.rho_min = rho_min
        self.eps = eps

    @property
    def hist_size(self):
        return len(self.history['params'])

    def get_params(self):
        return [p for p in self.params if p.requires_grad]

    def get_grads(self):
        return [p.grad for p in self.params if p.requires_grad]

    def get_value(self, key, clone=False):
        assert key in ['params', 'grads']
        if key == 'params':
            value = self.get_params()
        else:
            value = self.get_grads()
        if clone:
            return [v.clone().detach_().requires_grad_(False) for v in value]
        else:
            return value

    def get_clone(self, key):
        return self.get_value(key, clone=True)

    @torch.no_grad()
    def update_momentum(self):
        self._update_momentum('params', self.beta1)
        self._update_momentum('grads', self.beta2)

    def _update_momentum(self, key, beta):
        m = self.momentum
        if m[key] is None:
            m[key] = self.get_clone(key)
        else:
            value = self.get_value(key)
            group_scale_(m[key], beta)
            group_add_(m[key], value, alpha=1 - beta)
            for v in m[key]:
                v.detach_().requires_grad_(False)

    def get_momentum(self, key):
        m = self.momentum
        return self.get_clone(key) if m[key] is None else m[key]

    @torch.no_grad()
    def update_history(self, upd_g_scale=None):
        p, g = self.get_momentum('params'), self.get_momentum('grads')
        last_p, last_g = self.last_value['params'], self.last_value['grads']
        if last_p is not None:
            upd_p = group_add(p, last_p, alpha=-1)
            upd_g = group_add(g, last_g, alpha=-1)
            if upd_g_scale:
                group_scale_(upd_g, upd_g_scale)
            rho = group_product(upd_p, upd_g)
            if self.damping:
                damping, tau_lb, tau_ub = self.damping, self.tau_lb, self.tau_ub
                mu = rho / (group_square(upd_p) + self.eps)
                tau = 1
                if mu <= tau_lb:
                    tau = (1 - tau_lb) / (1 - mu)
                elif mu >= tau_ub:
                    tau = (tau_ub - 1) / (mu - 1)
                tau = min(tau, 1 - damping)
                group_add_(group_scale_(upd_g, tau), upd_p, alpha=1 - tau)
                rho = group_product(upd_p, upd_g)
            if rho >= self.rho_min:
                self._update_history('params', upd_p)
                self._update_history('grads', upd_g)
                self._update_history('rho', rho)
            else:
                print('too small rho, skip')
        self.last_value['params'] = [v.clone() for v in p]
        self.last_value['grads'] = [v.clone() for v in g]

    def _update_history(self, key, value):
        hist = self.history[key]
        if len(hist) == self.max_hist_size:
            hist.pop(0)
        hist.append(value)

    def get_history(self, index):
        hist = self.history
        return hist['params'][index], hist['grads'][index], hist['rho'][index]

    @torch.no_grad()
    def precondition(self):
        kl_clip = self.kl_clip
        if kl_clip:
            g_orig = self.get_clone('grads')

        # weight decay
        g = self.get_grads()
        if self.weight_decay:
            group_add_(g, self.get_params(), alpha=self.weight_decay)

        hist_size = self.hist_size
        if hist_size == 0:
            # no history is recorded. skip preconditioning.
            return

        # precondition
        hist_alpha = [None] * hist_size
        for i in reversed(range(hist_size)):
            upd_p, upd_g, rho = self.get_history(i)
            alpha = group_product(upd_p, g).div(rho + self.eps)
            group_add_(g, upd_g, alpha=-alpha)
            hist_alpha[i] = alpha
        upd_p, upd_g, rho = self.get_history(-1)
        gamma = rho / (group_square(upd_g) + self.eps)
        group_scale_(g, gamma)
        for i in range(hist_size):
            upd_p, upd_g, rho = self.get_history(i)
            alpha = hist_alpha[i]
            beta = group_product(upd_g, g).div(rho + self.eps)
            group_add_(g, upd_p, alpha=alpha - beta)

        # KL clipping
        if kl_clip:
            scale = min(1., math.sqrt(kl_clip / group_product(g_orig, g)))
            group_scale_(g, scale)
