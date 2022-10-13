import os
from typing import Tuple, Iterable
from operator import iadd

import numpy as np
import torch
from torch import Tensor
from .utils import cholesky_inv, smw_inv
from .vector import ParamVector

__all__ = [
    'matrix_to_tril',
    'tril_to_matrix',
    'get_n_cols_by_tril',
    'SymMatrix',
    'Kron',
    'Diag',
    'UnitWise'
]

_default_damping = 1e-5


def matrix_to_tril(mat: torch.Tensor):
    """
    Convert matrix (2D array)
    to lower triangular of it (1D array, row direction)

    Example:
      [[1, x, x],
       [2, 3, x], -> [1, 2, 3, 4, 5, 6]
       [4, 5, 6]]
    """
    assert mat.ndim == 2
    tril_indices = torch.tril_indices(*mat.shape)
    return mat[tril_indices[0], tril_indices[1]]


def tril_to_matrix(tril: torch.Tensor):
    """
    Convert lower triangular of matrix (1D array)
    to full symmetric matrix (2D array)

    Example:
                            [[1, 2, 4],
      [1, 2, 3, 4, 5, 6] ->  [2, 3, 5],
                             [4, 5, 6]]
    """
    assert tril.ndim == 1
    n_cols = get_n_cols_by_tril(tril)
    rst = torch.zeros(n_cols, n_cols, device=tril.device, dtype=tril.dtype)
    tril_indices = torch.tril_indices(n_cols, n_cols)
    rst[tril_indices[0], tril_indices[1]] = tril
    rst = rst + rst.T - torch.diag(torch.diag(rst))
    return rst


def get_n_cols_by_tril(tril: torch.Tensor):
    """
    Get number of columns of original matrix
    by lower triangular (tril) of it.

    ncols^2 + ncols = 2 * tril.numel()
    """
    assert tril.ndim == 1
    numel = tril.numel()
    return int(np.sqrt(2 * numel + 0.25) - 0.5)


def symeig(A: torch.Tensor, upper=True):
    return torch.linalg.eigvalsh(A, UPLO='U' if upper else 'L')


def _save_as_numpy(path, tensor):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    np.save(path, tensor.cpu().numpy().astype('float32'))


def _load_from_numpy(path, device='cpu'):
    data = np.load(path)
    return torch.from_numpy(data).to(device)


def is_all_none(xs: Iterable):
    return all(x is None for x in xs)


class SymMatrix:
    def __init__(self, data=None, inv=None, kron=None, kfe=None, diag=None, unit=None,
                 kron_A=None, kron_B=None, kron_A_inv=None, kron_B_inv=None,
                 kfe_A=None, kfe_B=None, kfe_scale=None,
                 unit_data=None, unit_inv=None,
                 diag_weight=None, diag_bias=None, diag_weight_inv=None, diag_bias_inv=None):
        self.data: Tensor = data
        self.inv: Tensor = inv
        if not is_all_none([kron_A, kron_B, kron_A_inv, kron_B_inv]):
            self.kron = Kron(kron_A, kron_B, kron_A_inv, kron_B_inv)
        else:
            self.kron: Kron = kron
        if not is_all_none([kfe_A, kfe_B, kfe_scale]):
            self.kfe = KFE(kfe_A, kfe_B, kfe_scale)
        else:
            self.kfe: KFE = kfe
        if not is_all_none([unit_data, unit_inv]):
            self.unit = UnitWise(unit_data, unit_inv)
        else:
            self.unit: UnitWise = unit
        if not is_all_none([diag_weight, diag_bias, diag_weight_inv, diag_bias_inv]):
            self.diag = Diag(diag_weight, diag_bias, diag_weight_inv, diag_bias_inv)
        else:
            self.diag: Diag = diag

    def __repr__(self):
        text = ''
        if self.has_data:
            text += f'data {tuple(self.data.shape)}, '
        if self.has_kron:
            if self.kron.has_A:
                text += f'kron.A {tuple(self.kron.A.shape)}, '
            if self.kron.has_B:
                text += f'kron.B {tuple(self.kron.B.shape)}, '
        if self.has_kfe:
            if self.kfe.has_Ua:
                text += f'kfe.Ua {tuple(self.kfe.Ua.shape)}'
            if self.kfe.has_Ub:
                text += f'kfe.Ub {tuple(self.kfe.Ub.shape)}'
            if self.kfe.has_scale:
                for i in range(len(self.kfe.scale)):
                    text += f'kfe.scale{i} {tuple(self.kfe.scale[i].shape)}'
        if self.has_unit and self.unit.has_data:
            text += f'unit {tuple(self.unit.data.shape)}'
        if self.has_diag:
            if self.diag.has_weight:
                text += f'diag.weight {tuple(self.diag.weight.shape)}, '
            if self.diag.has_bias:
                text += f'diag.bias {tuple(self.diag.bias.shape)}, '
        if len(text) > 0:
            text = text[:-2]
        text = 'SymMatrix(' + text + ')'
        return text

    @property
    def has_data(self):
        return self.data is not None

    @property
    def has_kron(self):
        return self.kron is not None

    @property
    def has_kfe(self):
        return self.kfe is not None

    @property
    def has_diag(self):
        return self.diag is not None

    @property
    def has_unit(self):
        return self.unit is not None

    def __add__(self, other):
        # NOTE: inv will not be preserved
        values = {}
        for attr in ['data', 'kron', 'kfe', 'diag', 'unit']:
            self_value = getattr(self, attr)
            other_value = getattr(other, attr)
            if other_value is not None:
                if self_value is not None:
                    value = self_value + other_value
                else:
                    value = other_value
            else:
                value = self_value
            values[attr] = value

        return SymMatrix(**values)

    def __iadd__(self, other):
        for attr in ['data', 'kron', 'kfe', 'diag', 'unit']:
            self_value = getattr(self, attr)
            other_value = getattr(other, attr)
            if other_value is not None:
                if self_value is not None:
                    iadd(self_value, other_value)
                else:
                    setattr(self, attr, other_value)
        return self

    def mul_(self, value):
        if self.has_data:
            self.data.mul_(value)
        if self.has_kron:
            self.kron.mul_(value)
        if self.has_kfe:
            self.kfe.mul_(value)
        if self.has_diag:
            self.diag.mul_(value)
        if self.has_unit:
            self.unit.mul_(value)
        return self

    def eigenvalues(self):
        assert self.has_data
        eig = symeig(self.data)
        return torch.sort(eig, descending=True)[0]

    def top_eigenvalue(self):
        assert self.has_data
        eig = symeig(self.data)
        return eig.max().item()

    def trace(self):
        assert self.has_data
        return torch.diag(self.data).sum().item()

    def save(self, root, relative_dir):
        relative_paths = {}
        if self.has_data:
            tril = matrix_to_tril(self.data)
            relative_path = os.path.join(relative_dir, 'tril.npy')
            absolute_path = os.path.join(root, relative_path)
            _save_as_numpy(absolute_path, tril)
            relative_paths['tril'] = relative_path
        if self.has_kron:
            rst = self.kron.save(root, relative_dir)
            relative_paths['kron'] = rst
        if self.has_diag:
            rst = self.diag.save(root, relative_dir)
            relative_paths['diag'] = rst
        if self.has_unit:
            rst = self.unit.save(root, relative_dir)
            relative_paths['unit_wise'] = rst

        return relative_paths

    def load(self, path=None, kron_path=None, diag_path=None, unit_path=None, device='cpu'):
        if path:
            tril = _load_from_numpy(path, device)
            self.data = tril_to_matrix(tril)
        if kron_path is not None:
            if not self.has_kron:
                self.kron = Kron(A=None, B=None)
            self.kron.load(
                A_path=kron_path['A_tril'],
                B_path=kron_path['B_tril'],
                device=device
            )
        if diag_path is not None:
            if not self.has_diag:
                self.diag = Diag()
            self.diag.load(
                w_path=diag_path.get('weight', None),
                b_path=diag_path.get('bias', None),
                device=device
            )
        if unit_path is not None:
            if not self.has_unit:
                self.unit = UnitWise()
            self.unit.load(path=unit_path, device=device)

    def to_vector(self):
        vec = []
        if self.has_data:
            vec.append(self.data)
        if self.has_kron:
            vec.extend(self.kron.data)
        if self.has_diag:
            vec.extend(self.diag.data)
        if self.has_unit:
            vec.extend(self.unit.data)

        vec = [v.flatten() for v in vec]
        return vec

    def to_matrices(self, vec, pointer):
        def unflatten(mat, p):
            numel = mat.numel()
            mat.copy_(vec[p:p + numel].view_as(mat))
            p += numel
            return p

        if self.has_data:
            pointer = unflatten(self.data, pointer)
        if self.has_kron:
            pointer = self.kron.to_matrices(unflatten, pointer)
        if self.has_diag:
            pointer = self.diag.to_matrices(unflatten, pointer)
        if self.has_unit:
            pointer = self.unit.to_matrices(unflatten, pointer)

        return pointer

    def update_inv(self, damping=_default_damping, replace=False):
        if self.has_data and not torch.all(self.data == 0):
            self.inv = cholesky_inv(self.data, damping)
            if replace:
                del self.data
                self.data = None
        if self.has_kron:
            self.kron.update_inv(damping, replace=replace)
        if self.has_diag:
            self.diag.update_inv(damping, replace=replace)
        if self.has_unit:
            self.unit.update_inv(damping, replace=replace)

    def mvp(self, vectors: ParamVector = None,
            vec_weight: torch.Tensor = None, vec_bias: torch.Tensor = None,
            use_inv=False, inplace=False):
        mat = self.inv if use_inv else self.data

        # full
        if vectors is not None:
            v = vectors.get_flatten_vector()
            mat_v = torch.mv(mat, v)
            rst = ParamVector(vectors.params(), mat_v)
            if inplace:
                for v1, v2 in zip(vectors.values(), rst.values()):
                    v1.copy_(v2)
            return rst

        # layer-wise
        assert vec_weight is not None or vec_bias is not None
        vecs = []
        if vec_weight is not None:
            vecs.append(vec_weight.flatten())
        if vec_bias is not None:
            vecs.append(vec_bias.flatten())
        vec1d = torch.cat(vecs)
        mvp1d = torch.mv(mat, vec1d)
        if vec_weight is not None:
            if vec_bias is not None:
                w_numel = vec_weight.numel()
                mvp_w = mvp1d[:w_numel].view_as(vec_weight)
                mvp_b = mvp1d[w_numel:]
                if inplace:
                    vec_weight.copy_(mvp_w)
                    vec_bias.copy_(mvp_b)
                return mvp_w, mvp_b
            mvp_w = mvp1d.view_as(vec_weight)
            if inplace:
                vec_weight.copy_(mvp_w)
            return [mvp_w]
        else:
            mvp_b = mvp1d.view_as(vec_bias)
            if inplace:
                vec_bias.copy_(mvp_b)
            return [mvp_b]


class Kron:
    def __init__(self, A, B, A_inv=None, B_inv=None):
        self.A = A
        self.B = B
        self.A_inv = A_inv
        self.B_inv = B_inv
        self._A_dim = self._B_dim = None

    def __add__(self, other):
        # NOTE: inv will not be preserved
        if not other.has_data:
            return self
        if other.has_A:
            A = self.A.add(other.A) if self.has_A else other.A
        else:
            A = self.A
        if other.has_B:
            B = self.B.add(other.B) if self.has_B else other.B
        else:
            B = self.B
        return Kron(A, B)

    def __iadd__(self, other):
        if not other.has_data:
            return self
        if other.has_A:
            if self.has_A:
                self.A.add_(other.A)
            else:
                self.A = other.A
        if other.has_B:
            if self.has_B:
                self.B.add_(other.B)
            else:
                self.B = other.B
        return self

    @property
    def data(self):
        return [self.A, self.B]

    @property
    def has_data(self):
        return self.has_A or self.has_B

    @property
    def has_A(self):
        return self.A is not None

    @property
    def has_B(self):
        return self.B is not None

    @property
    def A_dim(self):
        if self._A_dim is None:
            if self.A is not None:
                self._A_dim = self.A.shape[-1]
            elif self.A_inv is not None:
                self._A_dim = self.A_inv.shape[-1]
        return self._A_dim

    @property
    def B_dim(self):
        if self._B_dim is None:
            if self.B is not None:
                self._B_dim = self.B.shape[-1]
            elif self.B_inv is not None:
                self._B_dim = self.B_inv.shape[-1]
        return self._B_dim

    @property
    def A_is_square(self):
        return self.A.shape[0] == self.A.shape[1]

    @property
    def B_is_square(self):
        return self.B.shape[0] == self.B.shape[1]

    def mul_(self, value):
        if self.has_A:
            self.A.mul_(value)
        if self.has_B:
            self.B.mul_(value)
        return self

    def eigenvalues(self):
        eig_A = symeig(self.A)
        eig_B = symeig(self.B)
        eig = torch.ger(eig_A, eig_B).flatten()
        return torch.sort(eig, descending=True)[0]

    def top_eigenvalue(self):
        eig_A = symeig(self.A)
        eig_B = symeig(self.B)
        return (eig_A.max() * eig_B.max()).item()

    def trace(self):
        trace_A = torch.diag(self.A).sum().item()
        trace_B = torch.diag(self.B).sum().item()
        return trace_A * trace_B

    def save(self, root, relative_dir):
        relative_paths = {}
        for name in ['A', 'B']:
            mat = getattr(self, name, None)
            if mat is None:
                continue
            tril = matrix_to_tril(mat)
            tril_name = f'{name}_tril'
            relative_path = os.path.join(
                relative_dir, 'kron', f'{tril_name}.npy'
            )
            absolute_path = os.path.join(root, relative_path)
            _save_as_numpy(absolute_path, tril)
            relative_paths[tril_name] = relative_path

        return relative_paths

    def load(self, A_path, B_path, device):
        A_tril = _load_from_numpy(A_path, device)
        self.A = tril_to_matrix(A_tril)
        B_tril = _load_from_numpy(B_path, device)
        self.B = tril_to_matrix(B_tril)

    def to_matrices(self, unflatten, pointer):
        pointer = unflatten(self.A, pointer)
        pointer = unflatten(self.B, pointer)
        return pointer

    def update_inv(self, damping=_default_damping, calc_A_inv=True, calc_B_inv=True, eps=1e-7, replace=False):
        assert self.has_data
        if self.has_A and self.has_B:
            A_eig_mean = (self.A.trace() if self.A_is_square else torch.sum(self.A ** 2)) / self.A_dim
            B_eig_mean = (self.B.trace() if self.B_is_square else torch.sum(self.B ** 2)) / self.B_dim
            pi = torch.sqrt(A_eig_mean / B_eig_mean)
            r = damping**0.5
            damping_A = max(r * pi, eps)
            damping_B = max(r / pi, eps)
        else:
            damping_A = damping_B = damping

        if calc_A_inv:
            assert self.has_A
            if not torch.all(self.A == 0):
                if self.A_is_square:
                    self.A_inv = cholesky_inv(self.A, damping_A)
                else:
                    self.A_inv = smw_inv(self.A, damping_A)
                if replace:
                    del self.A
                    self.A = None
        if calc_B_inv:
            assert self.has_B
            if not torch.all(self.B == 0):
                if self.B_is_square:
                    self.B_inv = cholesky_inv(self.B, damping_B)
                else:
                    self.B_inv = smw_inv(self.B, damping_B)
                if replace:
                    del self.B
                    self.B = None

    def mvp(self, vec_weight, vec_bias=None, use_inv=False, inplace=False):
        mat_A = self.A_inv if use_inv else self.A
        mat_B = self.B_inv if use_inv else self.B
        vec_weight_2d = vec_weight.view(self.B_dim, -1)
        mvp_w = mat_B.mm(vec_weight_2d).mm(mat_A).view_as(vec_weight)
        if inplace:
            vec_weight.copy_(mvp_w)
        if vec_bias is not None:
            mvp_b = mat_B.mv(vec_bias)
            if inplace:
                vec_bias.copy_(mvp_b)
            return mvp_w, mvp_b
        return mvp_w


class KFE:
    def __init__(self, Ua: Tensor, Ub: Tensor, scale: Tuple[Tensor]):
        self.Ua: Tensor = Ua
        self.Ub: Tensor = Ub
        self.scale: Tuple[Tensor] = scale

    @property
    def has_Ua(self):
        return self.Ua is not None

    @property
    def has_Ub(self):
        return self.Ub is not None

    @property
    def has_scale(self):
        return self.scale is not None

    def update_inv(self, *args, **kwargs):
        pass

    def __add__(self, other):
        raise NotImplementedError

    def __iadd__(self, other):
        # NOTE: iadd only scale
        if not other.has_scale:
            return self
        if self.has_scale:
            for i in range(len(self.scale)):
                self.scale[i].add_(other.scale[i])
        else:
            self.scale = other.scale
        return self

    def mul_(self, value):
        if self.has_scale:
            for i in range(len(self.scale)):
                self.scale[i].mul_(value)
        return self

    def mvp(self, vec_weight, vec_bias=None, use_inv=False, inplace=False, eps=1.e-7):
        assert not use_inv, 'KFE does not calculate the inverse matrix.'
        Ua, Ub = self.Ua, self.Ub
        vec_weight_2d = vec_weight.view(Ub.shape[0], -1)
        vec_weight_2d_kfe = Ub.mm(vec_weight_2d).mm(Ua)
        vec_weight_2d_kfe /= (self.scale[0] + eps)
        mvp_w = Ub.mm(vec_weight_2d_kfe).mm(Ua).view_as(vec_weight)
        if inplace:
            vec_weight.copy_(mvp_w)
        if vec_bias is not None:
            vec_bias_kfe = Ub.mv(vec_bias)
            vec_bias_kfe /= (self.scale[1] + eps)
            mvp_b = Ub.mv(vec_bias_kfe)
            if inplace:
                vec_bias.copy_(mvp_b)
            return mvp_w, mvp_b
        return mvp_w


class UnitWise:
    def __init__(self, data=None, inv=None):
        self.data = data
        self.inv = inv

    def __add__(self, other):
        # NOTE: inv will not be preserved
        if not other.has_data:
            return self
        if self.has_data:
            data = self.data.add(other.data)
        else:
            data = other.data
        return UnitWise(data=data)

    def __iadd__(self, other):
        if not other.has_data:
            return self
        if self.has_data:
            self.data.add_(other.data)
        else:
            self.data = other.data
        return self

    @property
    def has_data(self):
        return self.data is not None

    def mul_(self, value):
        if self.has_data:
            self.data.mul_(value)
        return self

    def eigenvalues(self):
        assert self.has_data
        eig = [symeig(block) for block in self.data]
        eig = torch.cat(eig)
        return torch.sort(eig, descending=True)[0]

    def top_eigenvalue(self):
        top = max([symeig(block).max().item() for block in self.data])
        return top

    def trace(self):
        trace = sum([torch.trace(block).item() for block in self.data])
        return trace

    def save(self, root, relative_dir):
        relative_path = os.path.join(relative_dir, 'unit_wise.npy')
        absolute_path = os.path.join(root, relative_path)
        _save_as_numpy(absolute_path, self.data)
        return relative_path

    def load(self, path=None, device='cpu'):
        if path:
            self.data = _load_from_numpy(path, device)

    def to_matrices(self, unflatten, pointer):
        if self.has_data:
            pointer = unflatten(self.data, pointer)
        return pointer

    def update_inv(self, damping=_default_damping, replace=False):
        assert self.has_data
        data = self.data
        if not torch.all(data == 0):
            diag = torch.diagonal(data, dim1=1, dim2=2)
            diag += damping
            self.inv = torch.inverse(data)
            diag -= damping
            if replace:
                del self.data
                self.data = None

    def mvp(self, vec_weight, vec_bias, use_inv=False, inplace=False):
        mat = self.inv if use_inv else self.data  # (f, 2, 2) or (f_out, f_in+1, f_in+1)
        if vec_weight.shape == vec_bias.shape and mat.ndim == 3 and mat.shape[-1] == mat.shape[-2]:
            # for BatchNormNd and LayerNorm
            v = torch.stack([vec_weight, vec_bias], dim=1)  # (f, 2)
            v = v.unsqueeze(2)  # (f, 2, 1)
            mvp_wb = torch.matmul(mat, v).squeeze(2)  # (f, 2)
            mvp_w = mvp_wb[:, 0]
            mvp_b = mvp_wb[:, 1]
        else:
            v = torch.hstack([vec_weight, vec_bias.unsqueeze(dim=1)])  # (f_out, f_in+1)
            v = v.unsqueeze(2)  # (f_out, f_in+1, 1)
            mvp_wb = torch.matmul(mat, v).squeeze(2)  # (f_out, f_in+1)
            mvp_w = mvp_wb[:, :-1]
            mvp_b = mvp_wb[:, -1]

        if inplace:
            vec_weight.copy_(mvp_w)
            vec_bias.copy_(mvp_b)
        return mvp_w, mvp_b


class Diag:
    def __init__(self, weight=None, bias=None, weight_inv=None, bias_inv=None):
        self.weight = weight
        self.bias = bias
        self.weight_inv = weight_inv
        self.bias_inv = bias_inv

    def __add__(self, other):
        # NOTE: inv will not be preserved
        if other.has_weight:
            if self.has_weight:
                weight = self.weight.add(other.weight)
            else:
                weight = other.weight
        else:
            weight = self.weight
        if other.has_bias:
            if self.has_bias:
                bias = self.bias.add(other.bias)
            else:
                bias = other.bias
        else:
            bias = self.bias
        return Diag(weight=weight, bias=bias)

    def __iadd__(self, other):
        if other.has_weight:
            if self.has_weight:
                self.weight.add_(other.weight)
            else:
                self.weight = other.weight
        if other.has_bias:
            if self.has_bias:
                self.bias.add_(other.bias)
            else:
                self.bias = other.bias
        return self

    @property
    def data(self):
        return [d for d in [self.weight, self.bias] if d is not None]

    @property
    def has_weight(self):
        return self.weight is not None

    @property
    def has_bias(self):
        return self.bias is not None

    def mul_(self, value):
        if self.has_weight:
            self.weight.mul_(value)
        if self.has_bias:
            self.bias.mul_(value)
        return self

    def eigenvalues(self):
        eig = []
        if self.has_weight:
            eig.append(self.weight.flatten())
        if self.has_bias:
            eig.append(self.bias.flatten())
        eig = torch.cat(eig)
        return torch.sort(eig, descending=True)[0]

    def top_eigenvalue(self):
        top = -1
        if self.has_weight:
            top = max(top, self.weight.max().item())
        if self.has_bias:
            top = max(top, self.bias.max().item())
        return top

    def trace(self):
        trace = 0
        if self.has_weight:
            trace += self.weight.sum().item()
        if self.has_bias:
            trace += self.bias.sum().item()
        return trace

    def save(self, root, relative_dir):
        relative_paths = {}
        for name in ['weight', 'bias']:
            mat = getattr(self, name, None)
            if mat is None:
                continue
            relative_path = os.path.join(relative_dir, 'diag', f'{name}.npy')
            absolute_path = os.path.join(root, relative_path)
            _save_as_numpy(absolute_path, mat)
            relative_paths[name] = relative_path

        return relative_paths

    def load(self, w_path=None, b_path=None, device='cpu'):
        if w_path:
            self.weight = _load_from_numpy(w_path, device)
        if b_path:
            self.bias = _load_from_numpy(b_path, device)

    def to_matrices(self, unflatten, pointer):
        if self.has_weight:
            pointer = unflatten(self.weight, pointer)
        if self.has_bias:
            pointer = unflatten(self.bias, pointer)
        return pointer

    def update_inv(self, damping=_default_damping, replace=False):
        if self.has_weight:
            if not torch.all(self.weight == 0):
                self.weight_inv = 1 / (self.weight + damping)
                if replace:
                    del self.weight
                    self.weight = None
        if self.has_bias:
            if not torch.all(self.bias == 0):
                self.bias_inv = 1 / (self.bias + damping)
                if replace:
                    del self.bias
                    self.bias = None

    def mvp(self, vec_weight=None, vec_bias=None, use_inv=False, inplace=False):
        assert vec_weight is not None or vec_bias is not None
        rst = []
        if vec_weight is not None:
            mat_w = self.weight_inv if use_inv else self.weight
            if inplace:
                mvp_w = vec_weight.mul_(mat_w)
            else:
                mvp_w = vec_weight.mul(mat_w)
            rst.append(mvp_w)
        if vec_bias is not None:
            mat_b = self.bias_inv if use_inv else self.bias
            if inplace:
                mvp_b = vec_bias.mul_(mat_b)
            else:
                mvp_b = vec_bias.mul(mat_b)
            rst.append(mvp_b)
        return rst
