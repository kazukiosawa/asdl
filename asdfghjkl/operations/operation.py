from typing import Dict

import torch
import torch.nn as nn
from ..utils import original_requires_grad
from ..symmatrix import *
from ..vector import ParamVector

# compute no-centered covariance
OP_FULL_COV = 'full_cov'  # full covariance
OP_FULL_CVP = 'full_cvp'  # full covariance-vector product
OP_COV = 'cov'  # layer-wise covariance
OP_CVP = 'cvp'  # layer-wise covariance-vector product
OP_COV_KRON = 'cov_kron'  # Kronecker-factored
OP_COV_DIAG = 'cov_diag'  # diagonal
OP_COV_UNIT_WISE = 'cov_unit_wise'  # unit-wise

# compute Gram matrix
OP_GRAM_DIRECT = 'gram_direct'  # direct
OP_GRAM_HADAMARD = 'gram_hada'  # Hadamard-factored

OP_BATCH_GRADS = 'batch_grads'  # compute batched gradients (per-example gradients)

ALL_OPS = [OP_FULL_COV, OP_FULL_CVP, OP_COV, OP_CVP,
           OP_COV_KRON, OP_COV_DIAG, OP_COV_UNIT_WISE,
           OP_GRAM_DIRECT, OP_GRAM_HADAMARD, OP_BATCH_GRADS]


class Operation:
    def __init__(self, module, op_names, model_for_kernel=None):
        self._module = module
        self._model_for_kernel = model_for_kernel
        if isinstance(op_names, str):
            op_names = [op_names]
        assert isinstance(op_names, list)
        # remove duplicates
        op_names = set(op_names)
        for name in op_names:
            assert name in ALL_OPS, f'Invalid operation name: {name}.'
        self._op_names = op_names
        self._op_results = {}

    def accumulate_result(self, value, *keys):
        """
        Examples:
             accumulate_result(data, OP_COV_UNIT_WISE)
             accumulate_result(data, OP_BATCH_GRADS, 'weight')
             accumulate_result(A, OP_COV_KRON, 'A')
        """
        results = self._op_results
        if len(keys) > 1:
            for key in keys[:-1]:
                if results.get(key, None) is None:
                    results[key] = {}
                results = results[key]
        key = keys[-1]
        if results.get(key, None) is None:
            results[key] = value
        else:
            results[key] += value

    def get_result(self, *keys):
        results = self._op_results
        if len(keys) > 1:
            for key in keys[:-1]:
                results = results[key]
        key = keys[-1]
        return results.get(key, None)

    def clear_result(self, *keys):
        results = self._op_results
        if len(keys) > 1:
            for key in keys[:-1]:
                results = results[key]
        key = keys[-1]
        del results[key]

    def clear_results(self):
        self._op_results = {}

    def forward_post_process(self, in_data: torch.Tensor):
        module = self._module

        if OP_COV_KRON in self._op_names or OP_GRAM_HADAMARD in self._op_names:
            if original_requires_grad(module, 'bias'):
                in_data = self.extend_in_data(in_data)

            if OP_COV_KRON in self._op_names:
                A = self.cov_kron_A(module, in_data)
                self.accumulate_result(A, OP_COV_KRON, 'A')

            if OP_GRAM_HADAMARD in self._op_names:
                assert self._model_for_kernel is not None, f'model_for_kernel needs to be set for {OP_GRAM_HADAMARD}.'
                n_data = in_data.shape[0]
                n1 = self._model_for_kernel.kernel.shape[0]
                if n_data == n1:
                    A = self.gram_A(module, in_data, in_data)
                else:
                    A = self.gram_A(module, in_data[:n1], in_data[n1:])
                self.accumulate_result(A, OP_GRAM_HADAMARD, 'A')

    def backward_pre_process(self, in_data, out_grads, vector: torch.Tensor = None):
        module = self._module
        for op_name in self._op_names:
            if op_name in [OP_COV, OP_CVP]:
                batch_g = self.batch_grads_weight(module, in_data, out_grads).flatten(start_dim=1)
                if original_requires_grad(module, 'bias'):
                    batch_g = torch.cat([batch_g, self.batch_grads_bias(module, out_grads)], dim=1)
                if op_name == OP_COV:
                    self.accumulate_result(torch.matmul(batch_g.T, batch_g), OP_COV)
                else:
                    assert vector is not None
                    assert vector.ndim == 1
                    assert batch_g.shape[1] == vector.shape[0]
                    batch_gtv = batch_g.mul(vector.unsqueeze(0)).sum(dim=1)
                    cvp = torch.einsum('ni,n->i', batch_g, batch_gtv)
                    if original_requires_grad(module, 'bias'):
                        w_numel = module.weight.numel()
                        self.accumulate_result(cvp[:w_numel].view_as(module.weight), OP_CVP, 'weight')
                        self.accumulate_result(cvp[w_numel:].view_as(module.bias), OP_CVP, 'bias')
                    else:
                        self.accumulate_result(cvp, OP_CVP, 'weight')

            elif op_name == OP_COV_KRON:
                B = self.cov_kron_B(module, out_grads)
                self.accumulate_result(B, OP_COV_KRON, 'B')

            elif op_name == OP_COV_UNIT_WISE:
                assert original_requires_grad(module, 'weight')
                assert original_requires_grad(module, 'bias')
                rst = self.cov_unit_wise(module, self.extend_in_data(in_data), out_grads)
                self.accumulate_result(rst, OP_COV_UNIT_WISE)

            elif op_name == OP_GRAM_HADAMARD:
                assert self._model_for_kernel is not None, f'model_for_kernel needs to be set for {OP_GRAM_HADAMARD}.'
                n_data = in_data.shape[0]
                n1 = self._model_for_kernel.kernel.shape[0]
                if n_data == n1:
                    B = self.gram_B(module, out_grads, out_grads)
                else:
                    B = self.gram_B(module, out_grads[:n1], out_grads[n1:])
                A = self._op_results[OP_GRAM_HADAMARD]['A']
                self._model_for_kernel.kernel += B.mul(A)

            elif op_name == OP_GRAM_DIRECT:
                assert self._model_for_kernel is not None, f'model_for_kernel needs to be set for {OP_GRAM_DIRECT}.'
                n_data = in_data.shape[0]
                n1 = self._model_for_kernel.kernel.shape[0]

                grads = self.batch_grads_weight(module, in_data, out_grads)
                v = [grads]
                if original_requires_grad(module, 'bias'):
                    grads_b = self.batch_grads_bias(module, out_grads)
                    v.append(grads_b)
                g = torch.cat([_v.flatten(start_dim=1) for _v in v], axis=1)

                precond = getattr(module, 'gram_precond', None)
                if precond is not None:
                    precond.precondition_vector_module(v, module)
                    g2 = torch.cat([_v.flatten(start_dim=1) for _v in v], axis=1)
                else:
                    g2 = g

                if n_data == n1:
                    self._model_for_kernel.kernel += torch.matmul(g, g2.T)
                else:
                    self._model_for_kernel.kernel += torch.matmul(g[:n1], g2[n1:].T)
            else:
                rst = getattr(self,
                              f'{op_name}_weight')(module, in_data, out_grads)
                self.accumulate_result(rst, op_name, 'weight')
                if original_requires_grad(module, 'bias'):
                    rst = getattr(self, f'{op_name}_bias')(module, out_grads)
                    self.accumulate_result(rst, op_name, 'bias')

    @staticmethod
    def extend_in_data(in_data):
        # Extend in_data with ones.
        # linear: n x f_in
        #      -> n x (f_in + 1)
        # conv2d: n x (c_in)(kernel_size) x out_size
        #      -> n x {(c_in)(kernel_size) + 1} x out_size
        shape = list(in_data.shape)
        shape[1] = 1
        ones = in_data.new_ones(shape)
        return torch.cat((in_data, ones), dim=1)

    @staticmethod
    def batch_grads_weight(module, in_data, out_grads):
        raise NotImplementedError

    @staticmethod
    def batch_grads_bias(module, out_grads):
        raise NotImplementedError

    @staticmethod
    def cov_diag_weight(module, in_data, out_grads):
        raise NotImplementedError

    @staticmethod
    def cov_diag_bias(module, out_grads):
        raise NotImplementedError

    @staticmethod
    def cov_kron_A(module, in_data):
        raise NotImplementedError

    @staticmethod
    def cov_kron_B(module, out_grads):
        raise NotImplementedError

    @staticmethod
    def cov_unit_wise(module, in_data, out_grads):
        raise NotImplementedError

    @staticmethod
    def gram_A(module, in_data1, in_data2):
        raise NotImplementedError

    @staticmethod
    def gram_B(module, out_grads1, out_grads2):
        raise NotImplementedError


class OperationManager:
    def __init__(self, vectors: ParamVector = None):
        self._operations: Dict[nn.Module, Operation] = {}
        self._vectors: ParamVector = vectors

    def register_operation(self, module: nn.Module, operation: Operation):
        self._operations[module] = operation

    def get_vectors_by_module(self, module: nn.Module, flatten=False):
        if self._vectors is None:
            return None
        vectors = self._vectors.get_vectors_by_module(module)
        if flatten:
            return vectors.get_flatten_vector()
        return vectors

    def get_operation(self, module: nn.Module) -> Operation:
        try:
            return self._operations[module]
        except KeyError:
            print(f'No operation is registered to {module}.')

    def clear_operation(self, module: nn.Module):
        try:
            self._operations.pop(module)
        except KeyError:
            print(f'No operation is registered to {module}.')

    def clear_operations(self):
        keys = list(self._operations.keys())
        for key in keys:
            del self._operations[key]

    def call_operations_in_forward(self, module, in_data):
        self.get_operation(module).forward_post_process(in_data)

    def call_operations_in_backward(self, module, in_data, out_grads):
        vector = self.get_vectors_by_module(module, flatten=True)
        self.get_operation(module).backward_pre_process(in_data, out_grads, vector)

    def get_result(self, module, *keys):
        return self.get_operation(module).get_result(*keys)

    def accumulate_result(self, module, value, *keys):
        return self.get_operation(module).accumulate_result(value, *keys)

    def clear_result(self, module, *keys):
        return self.get_operation(module).clear_result(*keys)

    def batch_grads(self, module, flatten=False):
        grads = self.get_result(module, OP_BATCH_GRADS)
        if grads is not None and flatten:
            return torch.cat([g.flatten(start_dim=1) for g in grads.values()], dim=1)
        else:
            return grads

    def clear_batch_grads(self):
        for operation in self._operations.values():
            if operation.get_result(OP_BATCH_GRADS) is not None:
                operation.clear_result(OP_BATCH_GRADS)

    def full_batch_grads(self, module):
        bg = [self.batch_grads(m, flatten=True) for m in module.modules()]
        bg = [_bg for _bg in bg if _bg is not None]
        if len(bg) == 0:
            return None
        return torch.cat(bg, dim=1)  # n x p

    def full_cov(self, module):
        return self.get_result(module, OP_FULL_COV)

    def calc_full_cov(self, module):
        """
        g: (p,)
        cov = sum[gg^t]: (p, p)
        """
        bg = self.full_batch_grads(module)
        if bg is None:
            return
        cov = torch.matmul(bg.T, bg)  # p x p
        self.accumulate_result(module, cov, OP_FULL_COV)

    def full_cvp(self, module):
        return self.get_result(module, OP_FULL_CVP)

    def calc_full_cvp(self, module):
        """
        g: (p,)
        c = sum[gg^t]: (p, p)
        v: (p,)
        cvp = sum[gg^t]v = sum[g(g^t)v]: (p,)
        """
        vector = self.get_vectors_by_module(module, flatten=True)
        bg = self.full_batch_grads(module)
        if bg is None:
            return
        assert bg.shape[1] == vector.shape[0]
        bgtv = bg.mul(vector.unsqueeze(0)).sum(dim=1)
        cvp = torch.einsum('ni,n->i', bg, bgtv)
        self.accumulate_result(module, cvp, OP_FULL_CVP)

    def cov(self, module):
        return self.get_result(module, OP_COV)

    def cvp(self, module):
        return self.get_result(module, OP_CVP)

    def cov_kron(self, module):
        return self.get_result(module, OP_COV_KRON)

    def cov_unit_wise(self, module):
        return self.get_result(module, OP_COV_UNIT_WISE)

    def cov_diag(self, module):
        return self.get_result(module, OP_COV_DIAG)

    def cov_symmatrix(self, module):
        cov = self.cov(module)
        cov_kron = self.cov_kron(module)
        cov_unit_wise = self.cov_unit_wise(module)
        cov_diag = self.cov_diag(module)
        if all(v is None for v in [cov, cov_kron, cov_unit_wise, cov_diag]):
            return None
        return SymMatrix(data=cov,
                         kron_A=cov_kron['A'], kron_B=cov_kron['B'],
                         diag_weight=cov_diag['weight'], diag_bias=cov_diag['bias'],
                         unit_data=cov_unit_wise)

    def full_cov_symmatrix(self, module):
        cov = self.full_cov(module)
        if cov is None:
            return
        return SymMatrix(data=cov)

    def cvp_paramvector(self, module):
        cvp = self.cvp(module)
        if cvp is None:
            return None
        params = [p for p in module.parameters() if original_requires_grad(param=p)]
        return ParamVector(params, cvp)

    def full_cvp_paramvector(self, module):
        cvp = self.full_cvp(module)
        if cvp is None:
            return None
        params = [p for p in module.parameters() if original_requires_grad(param=p)]
        return ParamVector(params, cvp)
