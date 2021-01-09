from typing import List

import torch
from torch.utils.data.dataloader import DataLoader

from .precondition import DiagNaturalGradient
from .fisher import FISHER_EXACT
from .kernel import batch, empirical_implicit_ntk, get_preconditioned_kernel_fn
from .utils import add_value_to_diagonal


class TaskInfo:
    def __init__(self, memorable_points: torch.Tensor):
        # TODO: support DataLoader for memorable_points
        self.memorable_points = memorable_points
        self.kernel = None
        self.mean = None
        # TODO: support multi-head, class-ids

    def update_kernel(self, model, kernel_fn, batch_size=32, is_distributed=False):
        kernel = batch(kernel_fn,
                       model,
                       self.memorable_points,
                       batch_size=min(batch_size, self.memorable_points.shape[0]),
                       is_distributed=is_distributed)  # (n, n, c, c)
        n, c = kernel.shape[0], kernel.shape[-1]
        self.kernel = kernel.transpose(1, 2).reshape(n * c, -1)  # (nc, nc)

    def update_mean(self, model):
        self.mean = model(self.memorable_points)  # (n, c)

    def get_regularization_grad(self, model, eps=1e-5):
        assert self.kernel is not None
        assert self.mean is not None

        current_mean = model(self.memorable_points)  # (n, c)
        b = current_mean - self.mean  # (n, c)
        b = b.flatten()  # (nc,)
        kernel = add_value_to_diagonal(self.kernel, eps)  # (nc, nc)
        u = torch.cholesky(kernel)
        v = torch.cholesky_solve(b, u)  # (nc,)
        v = v.reshape(self.memorable_points.shape[0], -1)  # (n, c)

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
                 device: torch.DeviceObjType = None,
                 tau=None,
                 n_memorable_points=10,
                 precond_class=DiagNaturalGradient,
                 ggn_type=FISHER_EXACT,
                 prior_prec=1e-5,
                 n_mc_samples=1,
                 kernel_fn=empirical_implicit_ntk,
                 ):
        # apply softmax to model's output
        self.model_and_softmax = torch.nn.Sequential(model, torch.nn.Softmax())
        self.device = device  # TODO: manage device of each tensor appropriately
        self.tau = tau
        self.n_memorable_points = n_memorable_points
        self.precond = precond_class(model,
                                     fisher_type=ggn_type,
                                     pre_inv_postfix='all_tasks_ggn',
                                     n_mc_samples=n_mc_samples,
                                     damping=prior_prec)
        self.kernel_fn = get_preconditioned_kernel_fn(kernel_fn, self.precond)
        self.observed_tasks: List[TaskInfo] = []

    @property
    def is_ready(self):
        return len(self.observed_tasks) > 0

    def update_regularization_info(self, data_loader, is_distributed=False, batch_size=32):
        # update GGN and inverse for the current task
        self.precond.update_curvature(data_loader=data_loader)
        if is_distributed:
            self.precond.reduce_curvature()
        self.precond.accumulate_curvature(to_pre_inv=True)
        self.precond.update_inv()

        # register the current task
        memorable_points = self._collect_top_memorable_points(data_loader)
        self.observed_tasks.append(TaskInfo(memorable_points))

        # update information (kernel & prediction) for each observed task
        model = self.model_and_softmax
        for task in self.observed_tasks:
            task.update_kernel(model, self.kernel_fn, batch_size, is_distributed)
            task.update_mean(model)

    def apply_regularization_grad(self, tau=None, is_distributed=False, eps=1e-5):
        assert self.is_ready, 'Functional regularization is not ready yet, ' \
                              'call FROMP.update_regularization_info(data_loader).'
        if tau is None:
            tau = self.tau
        model = self.model_and_softmax

        # calculate FROMP grads on all the observed tasks
        grads_sum = [torch.zeros_like(p.grad) for p in model.parameters()]
        for task in self.observed_tasks:
            grads = task.get_regularization_grad(model, eps=eps)
            grads_sum = [gs.add_(g) for gs, g in zip(grads_sum, grads)]

        if is_distributed:
            # TODO: all-reduce grads_sum
            pass

        # update param.grad
        for p, g in zip(model.parameters(), grads_sum):
            p.grad.add_(g, alpha=tau)

    def _collect_top_memorable_points(self, data_loader: DataLoader, is_distributed=False):
        # TODO: support is_distributed (DistributedSampler)
        device = self.device
        dataset = data_loader.dataset
        hessian_traces = []

        # create a data loader w/o shuffling so that indices in the dataset are stored
        no_shuffle_loader = DataLoader(dataset,
                                       batch_size=data_loader.batch_size,
                                       num_workers=data_loader.num_workers,
                                       pin_memory=True,
                                       drop_last=False,
                                       shuffle=False)
        # collect Hessian trace
        for inputs, _ in no_shuffle_loader:
            inputs = inputs.to(device)
            probs = self.model_and_softmax(inputs)  # (n, c)
            diag_hessian = probs - probs * probs  # (n, c)
            hessian_traces.append(diag_hessian.sum(dim=1))  # [(n,)]
        hessian_traces = torch.cat(hessian_traces)

        # sort indices by Hessian trace
        indices = torch.argsort(hessian_traces, descending=True)
        top_indices = indices[:self.n_memorable_points]
        memorable_points = [dataset[idx][0] for idx in top_indices]

        # TODO: support DataLoader for memorable_points
        return torch.cat(memorable_points)  # (m, *)
