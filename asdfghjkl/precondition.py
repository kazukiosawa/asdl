import warnings
import torch
from torch import nn

from .matrices import FISHER_EXACT, SHAPE_FULL, SHAPE_BLOCK_DIAG, SHAPE_KRON, SHAPE_DIAG  # NOQA
from .fisher import fisher_for_cross_entropy

_supported_modules = (nn.Linear, nn.Conv2d, nn.BatchNorm1d, nn.BatchNorm2d)
_normalizations = (nn.BatchNorm1d, nn.BatchNorm2d)
_invalid_ema_decay = -1

__all__ = [
    'NaturalGradient', 'LayerWiseNaturalGradient', 'KFAC',
    'DiagNaturalGradient'
]


class NaturalGradient:
    def __init__(
        self,
        model,
        fisher_type=FISHER_EXACT,
        n_mc_samples=1,
        damping=1e-5,
        ema_decay=_invalid_ema_decay,
    ):
        from torch.nn.parallel import DistributedDataParallel as DDP
        assert not isinstance(model, DDP), f'{DDP} is not supported.'
        del DDP
        self.model = model
        self.modules = [model]
        self.fisher_type = fisher_type
        self.n_mc_samples = n_mc_samples
        self.damping = damping
        self.ema_decay = ema_decay
        self.fisher_shape = SHAPE_FULL
        self.fisher_manager = None

    def _get_fisher_attr(self, postfix=None):
        if postfix is None:
            return self.fisher_type
        else:
            return f'{self.fisher_type}_{postfix}'

    def _get_fisher(self, module, postfix=None):
        attr = self._get_fisher_attr(postfix)
        fisher = getattr(module, attr, None)
        return fisher

    def _scale_fisher(self, scale):
        for module in self.modules:
            fisher = self._get_fisher(module)
            if fisher is not None:
                fisher.scaling(scale)

    def _update_curvature(self,
                          inputs=None,
                          targets=None,
                          data_loader=None,
                          accumulate=False,
                          ema_decay=None,
                          data_average=True,
                          seed=None,
                          scale=1):
        if ema_decay is None:
            ema_decay = self.ema_decay
        if ema_decay != _invalid_ema_decay:
            assert accumulate, 'ema_decay cannot be set when accumulate=False.'
            scale *= ema_decay
            self._scale_fisher(1 - ema_decay)

        rst = fisher_for_cross_entropy(self.model,
                                       inputs=inputs,
                                       targets=targets,
                                       data_loader=data_loader,
                                       fisher_type=self.fisher_type,
                                       fisher_shapes=self.fisher_shape,
                                       accumulate=accumulate,
                                       data_average=data_average,
                                       seed=seed,
                                       scale=scale,
                                       n_mc_samples=self.n_mc_samples)
        self.fisher_manager = rst

    def accumulate_curvature(self,
                             inputs=None,
                             targets=None,
                             data_loader=None,
                             ema_decay=None,
                             data_average=True,
                             seed=None,
                             scale=1):
        self._update_curvature(inputs,
                               targets,
                               data_loader,
                               accumulate=True,
                               ema_decay=ema_decay,
                               data_average=data_average,
                               seed=seed,
                               scale=scale)

    def refresh_curvature(self,
                          inputs=None,
                          targets=None,
                          data_loader=None,
                          data_average=True,
                          seed=None,
                          scale=1):
        if self.ema_decay != _invalid_ema_decay:
            warnings.warn(f'ema_decay ({self.ema_decay}) will be ignored.')
        self._update_curvature(inputs,
                               targets,
                               data_loader,
                               accumulate=False,
                               ema_decay=_invalid_ema_decay,
                               data_average=data_average,
                               seed=seed,
                               scale=scale)

    def accumulate_pseudo_batch_curvature(self,
                                          accumulation_step,
                                          inputs=None,
                                          targets=None,
                                          data_loader=None,
                                          data_average=True,
                                          ema_decay=None,
                                          seed=None,
                                          scale=1,
                                          pseudo_batch_size=None):
        """
        Performs an action (refresh_curvature() or accumulate_curvature())
        depending on the accumulation_step to create a pseudo-batch curvature:

        The curvature calculated at each step will be divided by pseudo_batch_size
        when data_average == True.
            >>> pseudo_batch_curvature = curvature_at_step_0 / pseudo_batch_size
            >>> pseudo_batch_curvature += curvature_at_step_1 / pseudo_batch_size
            >>> pseudo_batch_curvature += curvature_at_step_2 / pseudo_batch_size

        When w/o EMA,
            At the first step, the previous pseudo-batch curvature will be overwritten by
            the curvature calculated at this step.

            >>> # 1st pseudo-batch
            >>> pseudo_batch_curvature = curvature_at_step_0  # refresh
            >>> pseudo_batch_curvature += curvature_at_step_1  # accumulate
            >>> pseudo_batch_curvature += curvature_at_step_2  # accumulate
            >>> # 2nd pseudo-batch
            >>> pseudo_batch_curvature = curvature_at_step_0  # refresh
            >>> pseudo_batch_curvature += curvature_at_step_1  # accumulate
            >>> pseudo_batch_curvature += curvature_at_step_2  # accumulate

        When w/ EMA,
            At th first step, the EMA of curvature will be scaled by (1 - ema_decay) if exists.
            At each step, the curvature will be scaled by ema_decay.

            >>> # 1st pseudo-batch
            >>> pseudo_batch_EMA = curvature_at_step_0 * ema_decay  # accumulate (to nothing)
            >>> pseudo_batch_EMA += curvature_at_step_1 * ema_decay  # accumulate
            >>> pseudo_batch_EMA += curvature_at_step_2 * ema_decay  # accumulate
            >>> # 2nd pseudo-batch
            >>> pseudo_batch_EMA *= (1 - ema_decay)
            >>> pseudo_batch_EMA += curvature_at_step_0 * ema_decay  # accumulate
            >>> pseudo_batch_EMA += curvature_at_step_1 * ema_decay  # accumulate
            >>> pseudo_batch_EMA += curvature_at_step_2 * ema_decay  # accumulate

            NOTE: It is recommend to use ema_decay == 1 for the first pseudo-batch.


        Example::
            >>> model = Net()
            >>> ng = NaturalGradient(model, ema_decay=0.9)
            >>> n_accumulations = 3
            >>> pseudo_batch_size = n_accumulations * data_loader.batch_size
            >>> for i, (x, y) in enumerate(data_loader):
            >>>     accumulation_step = i % n_accumulations
            >>>     is_first_pseudo_batch = int(i / n_accumulations) == 0
            >>>     ema_decay = 1 if is_first_pseudo_batch else None
            >>>     ng.accumulate_pseudo_batch_curvature(accumulation_step,
            >>>                                          inputs=x,
            >>>                                          targets=y,
            >>>                                          ema_decay=ema_decay,
            >>>                                          pseudo_batch_size=pseudo_batch_size)

        NOTE: When dataset size is not divisible by the data_loader.batch_size, there will be an incomplete
        batch unless data_loader.drop_last == True. Therefore, n_accumulations * data_loader.batch_size can
        be an incorrect value for the pseudo-batch size. The "correct" value depends on how a user wants to
        treat the imcomplete batch.
        """
        if data_average:
            assert pseudo_batch_size, 'pseudo_batch_size needs to be specified when data_average==True'
            data_average = False  # disable local data averaging
            scale /= pseudo_batch_size

        if ema_decay is None:
            ema_decay = self.ema_decay
        keep_ema = ema_decay != _invalid_ema_decay
        if keep_ema:
            scale *= ema_decay
            if accumulation_step == 0:
                self._scale_fisher(1 - ema_decay)

        if not keep_ema and accumulation_step == 0:
            self.refresh_curvature(inputs,
                                   targets,
                                   data_loader,
                                   data_average=data_average,
                                   seed=seed,
                                   scale=scale)
        else:
            self.accumulate_curvature(inputs,
                                      targets,
                                      data_loader,
                                      ema_decay=ema_decay,
                                      data_average=data_average,
                                      seed=seed,
                                      scale=scale)

    def reduce_curvature(self, all_reduce=True):
        self.fisher_manager.reduce_matrices(all_reduce=all_reduce)

    def update_inv(self, damping=None):
        if damping is None:
            damping = self.damping
        for module in self.modules:
            fisher = self._get_fisher(module)
            if fisher is None:
                continue
            fisher.update_inv(damping)

    def precondition(self):
        grads = []
        for p in self.model.parameters():
            if p.requires_grad and p.grad is not None:
                grads.append(p.grad.flatten())
        g = torch.cat(grads)
        fisher = self._get_fisher(self.model)
        ng = torch.mv(fisher.inv, g)

        pointer = 0
        for p in self.model.parameters():
            if p.requires_grad and p.grad is not None:
                numel = p.grad.numel()
                val = ng[pointer:pointer + numel]
                p.grad.copy_(val.reshape_as(p.grad))
                pointer += numel

        assert pointer == ng.numel()


class LayerWiseNaturalGradient(NaturalGradient):
    def __init__(self,
                 model,
                 fisher_type=FISHER_EXACT,
                 n_mc_samples=1,
                 damping=1e-5,
                 ema_decay=1.):
        super().__init__(model, fisher_type, n_mc_samples, damping, ema_decay)
        self.fisher_shape = SHAPE_BLOCK_DIAG
        self.modules = [
            m for m in model.modules() if isinstance(m, _supported_modules)
        ]

    def precondition(self):
        for module in self.modules:
            fisher = self._get_fisher(module)
            if fisher is None:
                continue
            g = module.weight.grad.flatten()
            if _bias_requires_grad(module):
                g = torch.cat([g, module.bias.grad.flatten()])
            ng = torch.mv(fisher.inv, g)

            if _bias_requires_grad(module):
                w_numel = module.weight.numel()
                grad_w = ng[:w_numel]
                module.bias.grad.copy_(ng[w_numel:])
            else:
                grad_w = ng
            module.weight.grad.copy_(grad_w.reshape_as(module.weight.grad))


class KFAC(NaturalGradient):
    def __init__(self,
                 model,
                 fisher_type=FISHER_EXACT,
                 n_mc_samples=1,
                 damping=1e-5,
                 ema_decay=1.):
        super().__init__(model, fisher_type, n_mc_samples, damping, ema_decay)
        self.fisher_shape = SHAPE_KRON
        self.modules = [
            m for m in model.modules() if isinstance(m, _supported_modules)
        ]

    def precondition(self):
        for module in self.modules:
            fisher = self._get_fisher(module)
            if fisher is None:
                continue
            if isinstance(module, _normalizations):
                inv = fisher.unit.inv  # (f, 2, 2)
                assert _bias_requires_grad(module)
                grad_w = module.weight.grad  # (f,)
                grad_b = module.bias.grad  # (f,)
                g = torch.stack([grad_w, grad_b], dim=1)  # (f, 2)
                g = g.unsqueeze(2)  # (f, 2, 1)
                ng = torch.matmul(inv, g).squeeze(2)  # (f, 2)
                module.weight.grad.copy_(ng[:, 0])
                module.bias.grad.copy_(ng[:, 1])
            else:
                A_inv = fisher.kron.A_inv
                B_inv = fisher.kron.B_inv
                grad2d = module.weight.grad.view(B_inv.shape[0], -1)
                if _bias_requires_grad(module):
                    grad2d = torch.cat(
                        [grad2d, module.bias.grad.unsqueeze(dim=1)], dim=1)
                ng = B_inv.mm(grad2d).mm(A_inv)
                if _bias_requires_grad(module):
                    grad_w = ng[:, :-1]
                    module.bias.grad.copy_(ng[:, -1])
                else:
                    grad_w = ng
                module.weight.grad.copy_(grad_w.reshape_as(module.weight.grad))

    def precondition_vector(self, vec):
        idx = 0
        for module in self.modules:
            fisher = self._get_fisher(module)
            if fisher is None:
                continue
            if isinstance(module, _normalizations):
                inv = fisher.unit.inv  # (f, 2, 2)
                assert _bias_requires_grad(module)
                vec_w = vec[idx]  # (f,)
                vec_b = vec[idx + 1]  # (f,)
                v = torch.stack([vec_w, vec_b], dim=1)  # (f, 2)
                v = v.unsqueeze(2)  # (f, 2, 1)
                ng = torch.matmul(inv, v).squeeze(2)  # (f, 2)
                vec[idx].copy_(ng[:, 0])
                vec[idx + 1].copy_(ng[:, 1])
                idx += 2
            else:
                A_inv = fisher.kron.A_inv
                B_inv = fisher.kron.B_inv
                w_idx = idx
                vec2d = vec[w_idx].view(B_inv.shape[0], -1)
                idx += 1
                if _bias_requires_grad(module):
                    vec2d = torch.cat([vec2d, vec[idx].unsqueeze(dim=1)],
                                      dim=1)
                ng = B_inv.mm(vec2d).mm(A_inv)
                if _bias_requires_grad(module):
                    vec_w = ng[:, :-1]
                    vec[idx].copy_(ng[:, -1])
                    idx += 1
                else:
                    vec_w = ng
                vec[w_idx].copy_(vec_w.reshape_as(module.weight.data))

        assert idx == len(vec)


class DiagNaturalGradient(NaturalGradient):
    def __init__(self,
                 model,
                 fisher_type=FISHER_EXACT,
                 n_mc_samples=1,
                 damping=1e-5,
                 ema_decay=1.):
        super().__init__(model, fisher_type, n_mc_samples, damping, ema_decay)
        self.fisher_shape = SHAPE_DIAG
        self.modules = [
            m for m in model.modules() if isinstance(m, _supported_modules)
        ]

    def precondition(self):
        for module in self.modules:
            fisher = self._get_fisher(module)
            if fisher is None:
                continue
            w_inv = fisher.diag.weight_inv
            module.weight.grad.mul_(w_inv)
            if _bias_requires_grad(module):
                b_inv = fisher.diag.bias_inv
                module.bias.grad.mul_(b_inv)

    def precondition_vector(self, vec):
        idx = 0
        for module in self.modules:
            fisher = self._get_fisher(module)
            if fisher is None:
                continue
            assert fisher.diag is not None, module
            vec[idx].mul_(fisher.diag.weight_inv)
            idx += 1
            if _bias_requires_grad(module):
                vec[idx].mul_(fisher.diag.bias_inv)
                idx += 1

        assert idx == len(vec)

    def precondition_vector_module(self, vec, module):
        fisher = self._get_fisher(module)
        assert fisher is not None
        assert fisher.diag is not None, module
        vec[0].mul_(fisher.diag.weight_inv)
        if _bias_requires_grad(module):
            vec[1].mul_(fisher.diag.bias_inv)


def _bias_requires_grad(module):
    return hasattr(module, 'bias') \
           and module.bias is not None \
           and module.bias.requires_grad
