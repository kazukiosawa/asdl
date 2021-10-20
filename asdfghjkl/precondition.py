import warnings
import torch
from torch import nn

from .matrices import FISHER_EXACT, SHAPE_FULL, SHAPE_LAYER_WISE, SHAPE_KRON, SHAPE_DIAG, modules_for_matrix_shapes  # NOQA
from .fisher import calculate_fisher, LOSS_CROSS_ENTROPY

_supported_modules = (nn.Linear, nn.Conv2d, nn.BatchNorm1d, nn.BatchNorm2d)
_normalizations = (nn.BatchNorm1d, nn.BatchNorm2d)
_invalid_ema_decay = -1
_module_level_shapes = [SHAPE_LAYER_WISE, SHAPE_KRON, SHAPE_DIAG]

__all__ = [
    'NaturalGradient', 'FullNaturalGradient', 'LayerWiseNaturalGradient', 'KFAC',
    'DiagNaturalGradient'
]


class NaturalGradient:
    def __init__(
        self,
        model,
        fisher_type=FISHER_EXACT,
        fisher_shape=SHAPE_FULL,
        loss_type=LOSS_CROSS_ENTROPY,
        n_mc_samples=1,
        damping=1e-5,
        ema_decay=_invalid_ema_decay,
    ):
        from torch.nn.parallel import DistributedDataParallel as DDP
        assert not isinstance(model, DDP), f'{DDP} is not supported.'
        del DDP
        self.model = model
        self.fisher_type = fisher_type
        self.loss_type = loss_type
        self.n_mc_samples = n_mc_samples
        self.damping = damping
        self.ema_decay = ema_decay
        self.fisher_manager = None
        # check if only one fisher_shape is assigned to each module
        if isinstance(fisher_shape, list):
            assert len(fisher_shape) == 1
        elif isinstance(fisher_shape, dict):
            assert all(len(v) == 1 for v in fisher_shape.values())
        self.fisher_shape = fisher_shape
        self.modules_for = modules_for_matrix_shapes(fisher_shape, list(model.modules()))

    @property
    def precondition_for(self):
        return {SHAPE_LAYER_WISE: self.precondition_layer_wise,
                SHAPE_KRON: self.precondition_kron,
                SHAPE_DIAG: self.precondition_diag}

    def parameters_for(self, shape):
        for module in self.modules_for[shape]:
            for p in module.parameters():
                yield p

    def _get_module_fisher(self, module, postfix=None):
        if postfix is None:
            attr = self.fisher_type
        else:
            attr = f'{self.fisher_type}_{postfix}'
        fisher = getattr(module, attr, None)
        return fisher

    def _get_full_fisher(self):
        return self._get_module_fisher(self.model)

    @property
    def _has_full_fisher(self):
        return self._get_full_fisher() is not None

    def _scale_fisher(self, scale):
        for shape in _module_level_shapes:
            for module in self.modules_for[shape]:
                fisher = self._get_module_fisher(module)
                if fisher is not None:
                    fisher.scaling(scale)
        if self._has_full_fisher:
            self._get_full_fisher().scaling(scale)

    def _update_curvature(self,
                          inputs=None,
                          targets=None,
                          data_loader=None,
                          accumulate=False,
                          ema_decay=None,
                          data_average=True,
                          calc_emp_loss_grad=False,
                          seed=None,
                          scale=1):
        if ema_decay is None:
            ema_decay = self.ema_decay
        if ema_decay != _invalid_ema_decay:
            assert accumulate, 'ema_decay cannot be set when accumulate=False.'
            scale *= ema_decay
            self._scale_fisher(1 - ema_decay)

        rst = calculate_fisher(self.model,
                               loss_type=self.loss_type,
                               fisher_type=self.fisher_type,
                               fisher_shapes=self.fisher_shape,
                               inputs=inputs,
                               targets=targets,
                               data_loader=data_loader,
                               accumulate=accumulate,
                               data_average=data_average,
                               calc_emp_loss_grad=calc_emp_loss_grad,
                               return_loss=True,
                               seed=seed,
                               scale=scale,
                               n_mc_samples=self.n_mc_samples)
        self.fisher_manager = rst[0]
        return rst[1]  # loss value

    def accumulate_curvature(self,
                             inputs=None,
                             targets=None,
                             data_loader=None,
                             ema_decay=None,
                             data_average=True,
                             calc_emp_loss_grad=False,
                             seed=None,
                             scale=1):
        return self._update_curvature(inputs,
                                      targets,
                                      data_loader,
                                      accumulate=True,
                                      ema_decay=ema_decay,
                                      data_average=data_average,
                                      calc_emp_loss_grad=calc_emp_loss_grad,
                                      seed=seed,
                                      scale=scale)

    def refresh_curvature(self,
                          inputs=None,
                          targets=None,
                          data_loader=None,
                          data_average=True,
                          calc_emp_loss_grad=False,
                          seed=None,
                          scale=1):
        if self.ema_decay != _invalid_ema_decay:
            warnings.warn(f'ema_decay ({self.ema_decay}) will be ignored.')
        return self._update_curvature(inputs,
                                      targets,
                                      data_loader,
                                      accumulate=False,
                                      ema_decay=_invalid_ema_decay,
                                      data_average=data_average,
                                      calc_emp_loss_grad=calc_emp_loss_grad,
                                      seed=seed,
                                      scale=scale)

    def reduce_curvature(self, all_reduce=True):
        self.fisher_manager.reduce_matrices(all_reduce=all_reduce)

    def update_inv(self, damping=None):
        if damping is None:
            damping = self.damping
        for shape in _module_level_shapes:
            for module in self.modules_for[shape]:
                fisher = self._get_module_fisher(module)
                if fisher is None:
                    continue
                fisher.update_inv(damping)
        if self._has_full_fisher:
            self._get_full_fisher().update_inv(damping)

    def precondition(self):
        for shape in _module_level_shapes:
            for module in self.modules_for[shape]:
                fisher = self._get_module_fisher(module)
                if fisher is None:
                    continue
                self.precondition_for[shape](module, fisher)
        if self._has_full_fisher:
            self.precondition_full()

    def precondition_full(self):
        grads = []
        for p in self.parameters_for(SHAPE_FULL):
            if p.requires_grad and p.grad is not None:
                grads.append(p.grad.flatten())
        g = torch.cat(grads)
        fisher = self._get_full_fisher()
        ng = torch.mv(fisher.inv, g)

        pointer = 0
        for p in self.parameters_for(SHAPE_FULL):
            if p.requires_grad and p.grad is not None:
                numel = p.grad.numel()
                val = ng[pointer:pointer + numel]
                p.grad.copy_(val.reshape_as(p.grad))
                pointer += numel

        assert pointer == ng.numel()

    @staticmethod
    def precondition_layer_wise(module, fisher):
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

    @staticmethod
    def precondition_kron(module, fisher):
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

    @staticmethod
    def precondition_diag(module, fisher):
        w_inv = fisher.diag.weight_inv
        module.weight.grad.mul_(w_inv)
        if _bias_requires_grad(module):
            b_inv = fisher.diag.bias_inv
            module.bias.grad.mul_(b_inv)


class FullNaturalGradient(NaturalGradient):
    def __init__(self,
                 model,
                 fisher_type=FISHER_EXACT,
                 loss_type=LOSS_CROSS_ENTROPY,
                 n_mc_samples=1,
                 damping=1e-5,
                 ema_decay=_invalid_ema_decay):
        super().__init__(model, fisher_type, SHAPE_FULL, loss_type, n_mc_samples, damping, ema_decay)


class LayerWiseNaturalGradient(NaturalGradient):
    def __init__(self,
                 model,
                 fisher_type=FISHER_EXACT,
                 loss_type=LOSS_CROSS_ENTROPY,
                 n_mc_samples=1,
                 damping=1e-5,
                 ema_decay=_invalid_ema_decay):
        super().__init__(model, fisher_type, SHAPE_LAYER_WISE, loss_type, n_mc_samples, damping, ema_decay)


class KFAC(NaturalGradient):
    def __init__(self,
                 model,
                 fisher_type=FISHER_EXACT,
                 loss_type=LOSS_CROSS_ENTROPY,
                 n_mc_samples=1,
                 damping=1e-5,
                 ema_decay=_invalid_ema_decay):
        super().__init__(model, fisher_type, SHAPE_KRON, loss_type, n_mc_samples, damping, ema_decay)

    def precondition_vector(self, vec):
        idx = 0
        for module in self.modules:
            fisher = self._get_module_fisher(module)
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
                 loss_type=LOSS_CROSS_ENTROPY,
                 n_mc_samples=1,
                 damping=1e-5,
                 ema_decay=_invalid_ema_decay):
        super().__init__(model, fisher_type, SHAPE_DIAG, loss_type, n_mc_samples, damping, ema_decay)

    def precondition_vector(self, vec):
        idx = 0
        for module in self.modules:
            fisher = self._get_module_fisher(module)
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
        fisher = self._get_module_fisher(module)
        assert fisher is not None
        assert fisher.diag is not None, module
        vec[0].mul_(fisher.diag.weight_inv)
        if _bias_requires_grad(module):
            vec[1].mul_(fisher.diag.bias_inv)


def _bias_requires_grad(module):
    return hasattr(module, 'bias') \
           and module.bias is not None \
           and module.bias.requires_grad
