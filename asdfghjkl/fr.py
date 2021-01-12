from typing import List
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset
import torch.distributed as dist

from .precondition import DiagNaturalGradient
from .fisher import FISHER_EXACT, COV
from .kernel import batch, empirical_implicit_ntk, get_preconditioned_kernel_fn
from .utils import add_value_to_diagonal


__all__ = [
    'FROMP',
]


class PastTask:
    def __init__(self, memorable_points, class_ids=None):
        self.memorable_points = memorable_points
        self.kernel = None
        self.mean = None
        self.class_ids = class_ids

    def update_kernel(self, model, kernel_fn):
        memorable_points = self.memorable_points
        if isinstance(memorable_points, DataLoader):
            kernel = batch(kernel_fn, model, memorable_points)
        else:
            kernel = kernel_fn(model, memorable_points)
        n, c = kernel.shape[0], kernel.shape[-1]  # (n, n, c, c)
        self.kernel = kernel.transpose(1, 2).reshape(n * c, n * c)  # (nc, nc)

    def update_mean(self, model):
        self.mean = self._evaluate_mean(model)

    def _evaluate_mean(self, model):
        means = []
        device = next(model.parameters()).device
        memorable_points = self.memorable_points
        if isinstance(memorable_points, DataLoader):
            for inputs, _ in self.memorable_points:
                inputs = inputs.to(device)
                means.append(model(inputs))
            return torch.cat(means)  # (n, c)
        else:
            return model(memorable_points)

    def get_regularization_grad(self, model, eps=1e-5):
        assert self.kernel is not None and self.mean is not None

        current_mean = self._evaluate_mean(model)  # (n, c)
        b = current_mean - self.mean  # (n, c)
        b = b.reshape(-1, 1)  # (nc, 1)
        kernel = add_value_to_diagonal(self.kernel, eps)  # (nc, nc)
        u = torch.cholesky(kernel)
        v = torch.cholesky_solve(b, u)  # (nc, 1)
        v.detach_().resize_as_(current_mean)  # (n, c)

        grad = torch.autograd.grad(current_mean, model.parameters(), grad_outputs=v)

        return grad


class FROMP:
    """
    Implementation of a functional-regularisation method called
    Functional Regularisation of Memorable Past (FROMP):
    Pingbo Pan et al., 2020
    Continual Deep Learning by Functional Regularisation of Memorable Past
    https://arxiv.org/abs/2004.14070

    Example::

        >>> import torch
        >>> from asdfghjkl import FROMP
        >>>
        >>> model = torch.nn.Linear(5, 3)
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>> loss_fn = torch.nn.CrossEntropyLoss()
        >>> fr = FROMP(model, tau=1.)
        >>>
        >>> for data_loader in data_loader_list:
        >>>     for x, y in data_loader:
        >>>         optimizer.zero_grad()
        >>>         loss = loss_fn(model(x), y)
        >>>         loss.backward()
        >>>         if fr.is_ready:
        >>>             fr.apply_regularization_grad()
        >>>         optimizer.step()
        >>>     fr.update_regularization_info(data_loader)
    """
    def __init__(self,
                 model: torch.nn.Module,
                 tau=None,
                 n_memorable_points=10,
                 precond_class=DiagNaturalGradient,
                 ggn_type=FISHER_EXACT,
                 prior_prec=1e-5,
                 n_mc_samples=1,
                 kernel_fn=empirical_implicit_ntk,
                 ):
        from torch.nn.parallel import DistributedDataParallel as DDP
        assert not isinstance(model, DDP), \
            f'{DDP} is not supported. Use the collective communication' \
            f'methods defined in {torch.distributed} for distributed training.'
        del DDP
        assert ggn_type != COV, f'ggn_type: {COV} is not supported.'

        self.model = model
        self.tau = tau
        self.n_memorable_points = n_memorable_points
        self.precond = precond_class(model,
                                     fisher_type=ggn_type,
                                     pre_inv_postfix='all_tasks_ggn',
                                     n_mc_samples=n_mc_samples,
                                     damping=prior_prec)
        self.kernel_fn = get_preconditioned_kernel_fn(kernel_fn, self.precond)
        self.observed_tasks: List[PastTask] = []

    @property
    def is_ready(self):
        return len(self.observed_tasks) > 0

    def update_regularization_info(self,
                                   data_loader,
                                   class_ids=None,
                                   memorable_points_as_tensor=True,
                                   is_distributed=False):
        model = self.model
        model.eval()

        # update GGN and inverse for the current task
        with customize_head(model, class_ids):
            self.precond.update_curvature(data_loader=data_loader)
        if is_distributed:
            self.precond.reduce_curvature()
        self.precond.accumulate_curvature(to_pre_inv=True)
        self.precond.update_inv()

        # register the current task with the memorable points
        with customize_head(model, class_ids):
            memorable_points = collect_memorable_points(model,
                                                        data_loader,
                                                        self.n_memorable_points,
                                                        memorable_points_as_tensor,
                                                        is_distributed)
        self.observed_tasks.append(PastTask(memorable_points, class_ids))

        # update information (kernel & mean) for each observed task
        for task in self.observed_tasks:
            with customize_head(model, task.class_ids, softmax=True):
                task.update_kernel(model, self.kernel_fn)
                task.update_mean(model)

    def apply_regularization_grad(self, tau=None, eps=1e-5):
        assert self.is_ready, 'Functional regularization is not ready yet, ' \
                              'call FROMP.update_regularization_info(data_loader).'
        if tau is None:
            tau = self.tau
        model = self.model

        # calculate FROMP grads on all the observed tasks
        grads_sum = [torch.zeros_like(p.grad) for p in model.parameters()]
        for task in self.observed_tasks:
            with customize_head(model, task.class_ids, softmax=True):
                grads = task.get_regularization_grad(model, eps=eps)
            grads_sum = [gs.add_(g) for gs, g in zip(grads_sum, grads)]

        # add regularization grad to param.grad
        for p, g in zip(model.parameters(), grads_sum):
            p.grad.add_(g, alpha=tau)


def collect_memorable_points(model,
                             data_loader: DataLoader,
                             n_memorable_points,
                             as_tensor=True,
                             is_distributed=False):
    device = next(model.parameters()).device
    dataset = data_loader.dataset
    hessian_traces = []

    # create a data loader w/o shuffling so that indices in the dataset are stored
    assert data_loader.batch_size is not None, 'DataLoader w/o batch_size is not supported.'
    if is_distributed:
        indices = range(dist.get_rank(), len(dataset), dist.get_world_size())
        dataset = Subset(dataset, indices)
    no_shuffle_loader = DataLoader(dataset,
                                   batch_size=data_loader.batch_size,
                                   num_workers=data_loader.num_workers,
                                   pin_memory=True,
                                   drop_last=False,
                                   shuffle=False)
    # collect Hessian trace
    for inputs, _ in no_shuffle_loader:
        inputs = inputs.to(device)
        logits = model(inputs)
        probs = F.softmax(logits, dim=1)  # (n, c)
        diag_hessian = probs - probs * probs  # (n, c)
        hessian_traces.append(diag_hessian.sum(dim=1))  # [(n,)]
    hessian_traces = torch.cat(hessian_traces)

    # sort indices by Hessian trace
    indices = torch.argsort(hessian_traces, descending=True).cpu()
    top_indices = indices[:n_memorable_points]

    if as_tensor:
        memorable_points = [dataset[idx][0] for idx in top_indices]
        return torch.cat(memorable_points).to(device)
    else:
        # create a DataLoader for memorable points
        memorable_points = Subset(dataset, top_indices)
        batch_size = min(n_memorable_points, data_loader.batch_size)
        return DataLoader(memorable_points,
                          batch_size=batch_size,
                          pin_memory=True,
                          drop_last=False,
                          shuffle=False)


@contextmanager
def customize_head(module: torch.nn.Module, class_ids: List[int] = None, softmax=False):

    def forward_hook(module, input, output):
        if class_ids is not None:
            output = output[:, class_ids]
        if softmax:
            return F.softmax(output, dim=1)
        else:
            return output

    handle = module.register_forward_hook(forward_hook)
    yield
    handle.remove()
    del forward_hook

