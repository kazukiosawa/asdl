from typing import Any, Tuple, Dict, Union
from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

try:
    import functorch as ft
    _is_functorch_available = True
except ImportError:
    ft = None
    _is_functorch_available = False

__all__ = ['GradientMaker', 'DummyObject', 'ft', 'LOSS_CROSS_ENTROPY', 'LOSS_MSE']

LOSS_CROSS_ENTROPY = 'cross_entropy'
LOSS_MSE = 'mse'


@dataclass
class GetFirstItem:
    pass


@dataclass
class GetItem:
    item: Any


@dataclass
class GetAttr:
    item: Any


@dataclass
class Call:
    args: Tuple[Any]
    kwargs: Dict[str, Any]


class DummyObject:
    def __init__(self, operators=None):
        if operators is None:
            operators = []
        self._operators = operators

    def __getitem__(self, item):
        return DummyObject(self._operators + [GetItem(item)])

    def __getattr__(self, item):
        return DummyObject(self._operators + [GetAttr(item)])

    def __call__(self, *args, **kwargs):
        return DummyObject(self._operators + [Call(args, kwargs)])

    def eval(self, base_value):
        rst = base_value

        def mapping(value):
            if isinstance(value, DummyObject):
                return value.eval(base_value)
            return value

        for operator in self._operators:
            if isinstance(operator, GetFirstItem):
                rst = rst[0] if isinstance(rst, (tuple, list)) else rst
            elif isinstance(operator, GetItem):
                rst = rst[operator.item]
            elif isinstance(operator, GetAttr):
                rst = getattr(rst, operator.item)
            elif isinstance(operator, Call):
                args = [mapping(arg) for arg in operator.args]
                kwargs = {k: mapping(v) for k, v in operator.kwargs.items()}
                rst = rst(*args, **kwargs)
        return rst


@dataclass
class VmapInfo:
    def __init__(self, *args, **kwargs):
        assert [v is None or isinstance(v, int) for v in args]
        assert [v is None or isinstance(v, int) for v in kwargs.values()]
        self.args_batch_dims: Tuple[int] = args
        self.kwargs_batch_dims: Dict[str, int] = kwargs


class GradientMaker:
    def __init__(self, model: nn.Module):
        self.model = model
        self._model_fn = None
        self._model_args = ()
        self._model_kwargs = dict()
        self._model_output = None
        self._model_vmap_info: VmapInfo = None
        self._logits: Tensor = None
        self._loss_fn = None
        self._loss_fn_args = ()
        self._loss_fn_kwargs = dict()
        self._loss_vmap_info: VmapInfo = None
        self._loss: Tensor = None
        self._dummy_loss: DummyObject = None
        self._dummy_logits = DummyObject([GetFirstItem()])

    def setup_model_call(self, model_fn, *args, **kwargs):
        self._model_fn = model_fn
        self._model_args = args
        self._model_kwargs = kwargs
        return DummyObject()

    def setup_model_vmap_info(self, *args, **kwargs):
        self._model_vmap_info = VmapInfo(*args, **kwargs)

    def setup_loss_call(self, loss_fn, *args, **kwargs):
        self._loss_fn = loss_fn
        self._loss_fn_args = args
        self._loss_fn_kwargs = kwargs

    def setup_loss_vmap_info(self, *args, **kwargs):
        self._loss_vmap_info = VmapInfo(*args, **kwargs)

    def setup_loss_repr(self, dummy_loss: DummyObject):
        assert isinstance(dummy_loss, DummyObject), \
            f'dummy_loss has to be an {DummyObject}, not {type(dummy_loss)}.'
        self._dummy_loss = dummy_loss

    def setup_logits_repr(self, dummy_logits: DummyObject):
        assert isinstance(dummy_logits, DummyObject), \
            f'dummy_loss has to be an {DummyObject}, not {type(dummy_logits)}.'
        self._dummy_logits = dummy_logits

    @property
    def model_output(self):
        return self._model_output

    @property
    def loss(self):
        return self._loss

    def call_model(self) -> Any:
        assert self._model_fn is not None, \
            'model_fn is not set. Call setup_model_call() ' \
            'before calling forward_and_backward().'
        self._model_output = self._model_fn(*self._model_args, **self._model_kwargs)
        self._logits = self._dummy_logits.eval(self._model_output)
        return self._model_output

    def call_loss(self) -> Tensor:
        if self._loss_fn is None:
            assert self._dummy_loss is not None, 'Neither loss_fn nor loss_repr is not set. ' \
                                                 'Call setup_loss_call() or setup_loss_repr() ' \
                                                 'before calling forward_and_backward().'
            self._loss = self._dummy_loss.eval(self._model_output)
        else:
            self._loss = self._call_loss_fn()
        return self._loss

    def forward(self) -> Union[Tuple[Any, Tensor], Any]:
        self.call_model()
        self.call_loss()
        return self._model_output, self._loss

    def backward(self):
        self._loss.backward()

    def forward_and_backward(self) -> Union[Tuple[Any, Tensor], Any]:
        # Performs a forward pass (model and loss evaluations) and a backward pass (gradient calculation).
        # A child class should override this function.
        rst = self.forward()
        self.backward()
        return rst

    def delegate_forward_and_backward(self, other, *args, **kwargs):
        other.setup_model_call(self._model_fn, *self._model_args, **self._model_kwargs)
        if self._loss_fn is None:
            other.setup_loss_repr(self._dummy_loss)
        else:
            other.setup_loss_call(self._loss_fn, *self._loss_fn_args, **self._loss_fn_kwargs)
        other.setup_logits_repr(self._dummy_logits)
        rst = other.forward_and_backward(*args, **kwargs)
        self._model_output = other.model_output
        self._loss = other.loss
        return rst

    def _get_mapped_loss_fn_args_kwargs(self):
        def mapping(value):
            if isinstance(value, DummyObject):
                return value.eval(self._model_output)
            return value
        args = []
        for arg in self._loss_fn_args:
            args.append(mapping(arg))
        kwargs = {}
        for key, arg in self._loss_fn_kwargs.items():
            kwargs[key] = mapping(arg)
        return args, kwargs

    def _call_loss_fn(self) -> Tensor:
        args, kwargs = self._get_mapped_loss_fn_args_kwargs()
        return self._loss_fn(*args, **kwargs)

    def _get_stateless_model_fn(self):
        assert _is_functorch_available, \
            'functorch is not available. Follow the installation guide in https://pytorch.org/functorch/stable/.'
        assert self._model_fn is not None, \
            'model_fn is not set. Call setup_model_call().'
        assert isinstance(self._model_fn, nn.Module), \
            'model_fn has to be an object of torch.nn.Module.'
        model_fn, params, buffers = ft.make_functional_with_buffers(self._model_fn)
        return model_fn, params, buffers

    def _get_stateless_model_fn_params_only(self):
        model_fn, params, buffers = self._get_stateless_model_fn()

        def model_fn_params_only(params):
            return model_fn(params, buffers, *self._model_args, **self._model_kwargs)

        return model_fn_params_only, params

    def _get_stateless_logit_fn_params_only(self):
        model_fn, params = self._get_stateless_model_fn_params_only()

        def logit_fn_params_only(params):
            y = model_fn(params)
            return self._dummy_logits.eval(y)

        return logit_fn_params_only, params

    def _get_stateless_model_loss_fn_params_only(self, return_output=False):
        model_fn_params_only, params = self._get_stateless_model_fn_params_only()

        def model_loss_fn_params_only(params):
            self._model_output = model_fn_params_only(params)
            self.call_loss()
            if self._loss_fn is None:
                rst = self._model_output
            else:
                rst = self._model_output, self._loss
            if return_output:
                return self._loss, rst
            else:
                return self._loss

        return model_loss_fn_params_only, params

    def _get_stateless_grad_fn_params_only(self, return_output=False):
        model_loss_fn, params = self._get_stateless_model_loss_fn_params_only(return_output=return_output)
        grad_fn = ft.grad(model_loss_fn, has_aux=return_output)
        return grad_fn, params

    def _get_random_tangents(self):
        assert isinstance(self._model_fn, nn.Module)
        return tuple([torch.randn_like(p) for p in self._model_fn.parameters()])

    @torch.no_grad()
    def loss_grad(self, return_output=False) -> Tuple[Tensor, ...]:
        grad_fn, params = self._get_stateless_grad_fn_params_only(return_output=return_output)
        return grad_fn(params)

    @torch.no_grad()
    def per_example_loss_grads(self) -> Tensor:
        model_fn, params, buffers = self._get_stateless_model_fn()
        ref_model_args = self._model_args
        ref_model_kwargs = OrderedDict(self._model_kwargs)  # order needs to be kept to restore kwargs
        ref_loss_args = self._loss_fn_args
        ref_loss_kwargs = OrderedDict(self._loss_fn_kwargs)  # order needs to be kept to restore kwargs

        # arguments to the gradient function that will be vectorized along the batch dimension
        grad_args = ()
        # the batch dimension (int or None) for each argument
        batch_dims = ()

        def extend_batch_dims(vmap_info: VmapInfo, ref_args: Tuple, ref_kwargs: OrderedDict[str, Any]):
            nonlocal batch_dims
            if vmap_info is None:
                for v in [*ref_args, *ref_kwargs.values()]:
                    if isinstance(v, Tensor):
                        batch_dims += (0,)
                    else:
                        batch_dims += (None,)
            else:
                assert len(ref_args) == len(vmap_info.args_batch_dims)
                assert len(ref_kwargs) == len(vmap_info.kwargs_batch_dims)
                assert all(k in vmap_info.kwargs_batch_dims for k in ref_kwargs.keys())
                batch_dims += vmap_info.args_batch_dims
                for key in ref_kwargs.keys():
                    batch_dims += (vmap_info.kwargs_batch_dims[key],)

        # extend grad_args and batch_dims for model call
        grad_args += tuple([*ref_model_args, *ref_model_kwargs.values()])
        num_model_args = len(grad_args)
        extend_batch_dims(self._model_vmap_info, ref_model_args, ref_model_kwargs)
        if self._loss_fn is not None:
            # extend grad_args and batch_dims for loss call
            grad_args += tuple([*ref_loss_args, *ref_loss_kwargs.values()])
            extend_batch_dims(self._loss_vmap_info, ref_loss_args, ref_loss_kwargs)

        def split_args(src_args: Tuple, ref_args: Tuple, ref_kwargs: OrderedDict[str, Any]):
            # split src_args into (args, kwargs) with the same structure as (ref_args, ref_kwargs)
            assert len(src_args) == len(ref_args) + len(ref_kwargs.values())
            args = tuple([src_args[i] for i in range(len(ref_args))])
            kwargs = {}
            for i, key in enumerate(ref_kwargs.keys()):
                kwargs[key] = src_args[len(ref_args) + i]
            return args, kwargs

        def model_loss_fn(params, *args):
            # unsqueeze Tensor for the vectorized (batch) dimension
            args = [v if dim is None else v.unsqueeze(dim) for v, dim in zip(args, batch_dims)]

            # split args into model and loss args
            model_all_args = tuple(args[:num_model_args])
            loss_all_args = tuple(args[num_model_args:])

            model_args, model_kwargs = split_args(model_all_args, ref_model_args, ref_model_kwargs)
            model_output = model_fn(params, buffers, *model_args, **model_kwargs)
            if self._loss_fn is None:
                return self._dummy_loss.eval(model_output)
            else:
                loss_all_args = tuple([v.eval(model_output) if isinstance(v, DummyObject)
                                       else v for v in loss_all_args])
                loss_args, loss_kwargs = split_args(loss_all_args, ref_loss_args, ref_loss_kwargs)
                return self._loss_fn(*loss_args, **loss_kwargs)

        grad_fn = ft.grad(model_loss_fn, argnums=0)
        return ft.vmap(grad_fn, in_dims=(None,) + batch_dims)(params, *grad_args)

    @torch.no_grad()
    def loss_hessian(self) -> Tuple[Tuple[Tensor, ...], ...]:
        model_loss_fn, params = self._get_stateless_model_loss_fn_params_only()
        return ft.hessian(model_loss_fn)(params)

    @torch.no_grad()
    def loss_hvp(self, tangents=None, return_grad=False, return_output=False) \
            -> Union[Tuple[Tensor, ...], Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]]]:
        grad_fn, params = self._get_stateless_grad_fn_params_only(return_output=return_output)
        if tangents is None:
            tangents = self._get_random_tangents()
        out = ft.jvp(grad_fn, (params,), (tangents,), has_aux=return_output)
        # out = (grad, hvp) or (grad, hvp, output) if return_output
        rst = (out[1],)
        if return_grad:
            rst += (out[0],)
        if return_output:
            rst += (out[2],)
        return rst[0] if len(rst) == 1 else rst

    @torch.no_grad()
    def logit_jacobian(self) -> Tuple[Tensor, ...]:
        logit_fn, params = self._get_stateless_logit_fn_params_only()
        return ft.jacrev(logit_fn, argnums=0)(params)

    @torch.no_grad()
    def logit_jvp(self, tangents=None, return_model_output=False) \
            -> Union[Tuple[Tensor, ...], Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]]]:
        logit_fn, params = self._get_stateless_logit_fn_params_only()
        if tangents is None:
            tangents = self._get_random_tangents()
        logits, jvp = ft.jvp(logit_fn, (params,), (tangents,))
        if return_model_output:
            return jvp, logits
        else:
            return jvp

    def fvp(self, loss_type, data_size=None, tangents=None, return_model_output=False) \
            -> Union[Tuple[Tensor, ...], Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]]]:
        assert loss_type in [LOSS_MSE, LOSS_CROSS_ENTROPY]
        logit_fn, params = self._get_stateless_logit_fn_params_only()
        if tangents is None:
            tangents = self._get_random_tangents()
        y, jvp = ft.jvp(logit_fn, (params,), (tangents,))
        assert y.ndim == 2  # n x c
        if data_size is None:
            data_size = y.shape[0]
        if loss_type == LOSS_CROSS_ENTROPY:
            with torch.no_grad():
                probs = F.softmax(y, dim=1)
                diag_p = torch.stack([torch.diag(p) for p in probs], dim=0)  # n x c x c
                ppt = torch.bmm(probs.unsqueeze(2), probs.unsqueeze(1))  # n x c x c
                loss_hessian_wrt_logit = diag_p - ppt  # n x c x c
                grad_outputs = torch.einsum('bij,bj->bi', loss_hessian_wrt_logit, jvp)
        else:
            grad_outputs = jvp
        fvp = torch.autograd.grad(y, params, grad_outputs=grad_outputs / data_size)
        if return_model_output:
            return fvp, y
        else:
            return fvp

    @torch.no_grad()
    def nvp(self, cotangents=None, return_model_output=False) \
            -> Union[Tensor, Tuple[Tensor, Tuple[Tensor, ...]]]:
        logit_fn, params = self._get_stateless_logit_fn_params_only()
        y, vjp_fn = ft.vjp(logit_fn, params)
        if cotangents is None:
            cotangents = (torch.randn_like(y),)
        vjp = vjp_fn(*cotangents)
        nvp = ft.jvp(logit_fn, (params,), vjp)[1]
        if return_model_output:
            return nvp, y
        else:
            return nvp
