import torch
from ..utils import original_requires_grad

# compute no-centered covariance
OP_COV_KRON = 'cov_kron'  # Kronecker-factored
OP_COV_DIAG = 'cov_diag'  # diagonal
OP_COV_UNIT_WISE = 'cov_unit_wise'  # unit-wise

# compute Gram matrix
OP_GRAM_DIRECT = 'gram_direct'  # direct
OP_GRAM_HADAMARD = 'gram_hada'  # Hadamard-factored

OP_BATCH_GRADS = 'batch_grads'  # compute batched gradients (per-example gradients)
OP_ACCUMULATE_GRADS = 'acc_grad'  # accumulate gradients


class Operation:
    def __init__(self, module, op_names, model_for_kernel=None):
        self._module = module
        self._model_for_kernel = model_for_kernel
        if isinstance(op_names, str):
            op_names = [op_names]
        # remove duplicates
        op_names = set(op_names)
        self._op_names = op_names
        self._op_results = {}
        self._grads_scale = None

    def _set_result(self, value, *keys):
        """
        Examples:
             set_result(data, OP_COV_UNIT_WISE)
             set_result(data, OP_BATCH_GRADS, 'weight')
             set_result(A, OP_COV_KRON, 'A')
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

    def get_op_results(self):
        return self._op_results

    def clear_op_results(self):
        self._op_results = {}

    @property
    def grads_scale(self):
        return self._grads_scale

    @grads_scale.setter
    def grads_scale(self, value):
        self._grads_scale = value

    def forward_post_process(self, in_data: torch.Tensor):
        module = self._module

        if OP_COV_KRON in self._op_names or OP_GRAM_HADAMARD in self._op_names:
            if original_requires_grad(module, 'bias'):
                # Extend in_data with ones.
                # linear: n x f_in
                #      -> n x (f_in + 1)
                # conv2d: n x (c_in)(kernel_size) x out_size
                #      -> n x {(c_in)(kernel_size) + 1} x out_size
                shape = list(in_data.shape)
                shape[1] = 1
                ones = in_data.new_ones(shape)
                in_data = torch.cat((in_data, ones), dim=1)

            if OP_COV_KRON in self._op_names:
                A = self.cov_kron_A(module, in_data)
                self._set_result(A, OP_COV_KRON, 'A')

            if OP_GRAM_HADAMARD in self._op_names:
                assert self._model_for_kernel is not None, f'fmodel_for_kernel needs to be set for {OP_GRAM_HADAMARD}.'
                n_data = in_data.shape[0]
                n1 = self._model_for_kernel.kernel.shape[0]
                if n_data == n1:
                    A = self.gram_A(module, in_data, in_data)
                else:
                    A = self.gram_A(module, in_data[:n1], in_data[n1:])
                self._set_result(A, OP_GRAM_HADAMARD, 'A')

    def backward_pre_process(self, in_data, out_grads):
        gs = self._grads_scale
        if gs is not None:
            if isinstance(gs, torch.Tensor):
                assert gs.shape[0] == out_grads.shape[0]
                shape = (-1, ) + (1, ) * (out_grads.ndim - 1)
                out_grads.mul_(gs.reshape(shape))
            else:
                out_grads.mul_(gs)

        module = self._module
        for op_name in self._op_names:
            if op_name == OP_COV_KRON:
                B = self.cov_kron_B(module, out_grads)
                self._set_result(B, OP_COV_KRON, 'B')

            elif op_name == OP_COV_UNIT_WISE:
                assert original_requires_grad(module, 'weight')
                assert original_requires_grad(module, 'bias')
                rst = self.cov_unit_wise(module, in_data, out_grads)
                self._set_result(OP_COV_UNIT_WISE, rst)

            elif op_name == OP_GRAM_HADAMARD:
                assert self._model_for_kernel is not None, f'fmodel_for_kernel needs to be set for {OP_GRAM_HADAMARD}.'
                n_data = in_data.shape[0]
                n1 = self._model_for_kernel.kernel.shape[0]
                if n_data == n1:
                    B = self.gram_B(module, out_grads, out_grads)
                else:
                    B = self.gram_B(module, out_grads[:n1], out_grads[n1:])
                A = self._op_results[OP_GRAM_HADAMARD]['A']
                self._model_for_kernel.kernel += B.mul(A)

            elif op_name == OP_GRAM_DIRECT:
                assert self._model_for_kernel is not None, f'fmodel_for_kernel needs to be set for {OP_GRAM_DIRECT}.'
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
                self._set_result(rst, op_name, 'weight')
                if original_requires_grad(module, 'bias'):
                    rst = getattr(self, f'{op_name}_bias')(module, out_grads)
                    self._set_result(rst, op_name, 'bias')

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
