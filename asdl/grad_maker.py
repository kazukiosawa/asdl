from typing import Any, Tuple, Dict, Union, Callable
from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .utils import has_reduction

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
    """A container of a sequence of operations
    (:obj:`__getitem__`, :obj:`__getattr__`, or :obj:`__call__`).

    When an operation is applied to a DummyObject,
    a new DummyObject is returned with the history of the operations and their arguments.

    Example:
        >>> dummy_obj = DummyObject()
        >>> # operations = []
        >>>
        >>> dummy_obj = dummy_obj[0]
        >>> # operations = [(__getitem__, 0)]
        >>>
        >>> dummy_obj = dummy_obj.attr_name
        >>> # operations = [(__getitem__, 0), (__getattr__, attr_name)]
        >>>
        >>> dummy_obj = dummy_obj(arg1, arg2, key1=value1, key2=value2)
        >>> # operations = [(__getitem__, 0), (__getattr__, attr_name), (__call__, arg1, arg2, key1=value1, key2=value2)]
    """
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
        """Applies the history of operations to :obj:`base_value`.

        Args:
            base_value (Any): A base value to be applied the history of operations.

        Returns:
            The evaluated result.

        Example:
            >>> module = torch.nn.Linear(5, 3)
            >>>
            >>> # direct evaluation
            >>> print(module.weight[0].numel())
            >>> # 5
            >>>
            >>> # delayed evaluation via DummyObject
            >>> dummy_numel = DummyObject().weight[0].numel()
            >>> print(dummy_numel.eval(module))
            >>> # 5
        """
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
    """A container of batch dimension information for :obj:`functorch.vmap`.

    Every argument in `*args` and `**kwargs` has to be :obj:`int` for :obj:`None`.
    """
    def __init__(self, *args, **kwargs):
        if any(v is not None and not isinstance(v, int) for v in args):
            raise TypeError('Every argument in args has to be int or None.')
        if any(v is not None and not isinstance(v, int) for v in kwargs.values()):
            raise TypeError('Every argument in kwargs has to be int or None.')
        self.args_batch_dims: Tuple[int] = args
        self.kwargs_batch_dims: Dict[str, int] = kwargs


class GradientMaker:
    """The base class of all XXXGradientMaker classes.

    Args:
        model (Module): target module to calculate gradient
    """
    _loss_reduction = None

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

    def setup_model_call(self, model_fn: Callable, *args, **kwargs) -> DummyObject:
        """Setups a function to be called in :obj:`call_model()`.
        `*args` and `**kwargs` will be passed when `model_fn` is called.

        Args:
            model_fn (Callable): e.g., a Module.

        Returns:
            A DummyObject, which can be manipulated and passed to :obj:`setup_loss_call()` or :obj:`setup_loss_repr()`.

        Example:
            >>> model = YourModule()
            >>> for x, t in data_loader:
            >>>     # if YourModule takes only inputs
            >>>     grad_maker.setup_model_call(model, x)
            >>>
            >>>     # if YourModule takes both inputs and targets
            >>>     grad_maker.setup_model_call(model, x, target=t)
        """
        self._model_fn = model_fn
        self._model_args = args
        self._model_kwargs = kwargs
        return DummyObject()

    def setup_model_vmap_info(self, *args, **kwargs):
        """Setups VmapInfo for :obj:`model_fn`."""
        self._model_vmap_info = VmapInfo(*args, **kwargs)

    def setup_loss_call(self, loss_fn: Callable, *args, **kwargs):
        """Setups a function to be called in :obj:`call_loss()`.
        `*args` and `**kwargs` will be passed when `loss_fn` is called.
        An alternative of this method is :obj:`setup_loss_repr()`.

        Args:
            loss_fn (Callable): e.g., :obj:`torch.nn.functional.cross_entropy`

        Example:
            >>> dummy_y = grad_maker.setup_model_call(model, x)
            >>> grad_maker.setup_loss_call(F.cross_entropy, dummy_y, t, label_smoothing=0.1, ignore_index=-1)
        """
        self._loss_fn = loss_fn
        self._loss_fn_args = args
        self._loss_fn_kwargs = kwargs

    def setup_loss_vmap_info(self, *args, **kwargs):
        """Setups a VmapInfo for :obj:`loss_fn`."""
        self._loss_vmap_info = VmapInfo(*args, **kwargs)

    def setup_loss_repr(self, dummy_loss: DummyObject):
        """Setups a representation (DummyObject) to be evaluated
        in :obj:`call_loss()`.
        An alternative of this method is :obj:`setup_loss_call()`.

        Args:
            dummy_loss (DummyObject): e.g., output of :obj:`setup_model_call()` or its manipulation.

        Example:
            >>> dummy_y = grad_maker.setup_model_call(model, x)
            >>> grad_maker.setup_loss_repr(dummy_y[1].sum())
        """
        if not isinstance(dummy_loss, DummyObject):
            raise TypeError(f'dummy_loss has to be an {DummyObject}, not {type(dummy_loss)}.')
        self._dummy_loss = dummy_loss

    def setup_logits_repr(self, dummy_logits: DummyObject):
        """Setups a representation (DummyObject) to be evaluated
        in :obj:`call_model()` as *logits*.

        Args:
            dummy_logits (DummyObject): e.g., output of :obj:`setup_model_call()` or its manipulation.

        Example:
            >>> dummy_y = grad_maker.setup_model_call(model, x)
            >>> grad_maker.setup_logits_repr(dummy_y[0])
        """
        if not isinstance(dummy_logits, DummyObject):
            raise TypeError(f'dummy_loss has to be an {DummyObject}, not {type(dummy_logits)}.')
        self._dummy_logits = dummy_logits

    @property
    def model_output(self):
        """Contains the output of :obj:`call_model()`"""
        return self._model_output

    @property
    def loss(self):
        """Contains the output of :obj:`call_loss()`"""
        return self._loss

    def call_model(self) -> Any:
        """Calls the function with the arguments registered by :obj:`setup_model_call()`

        Returns:
            The evaluation of the model function.
        """
        if self._model_fn is None:
            raise ValueError('model_fn is not set. Call setup_model_call() before calling forward_and_backward().')
        self._model_output = self._model_fn(*self._model_args, **self._model_kwargs)
        self._logits = self._dummy_logits.eval(self._model_output)
        return self._model_output

    def call_loss(self) -> Tensor:
        """Calls the function with the arguments registered by :obj:`setup_loss_call()`
        or evaluates the representation registered by :obj:`setup_loss_repr()`.

        Returns:
            The evaluation of the loss function or representation.
        """
        if self._loss_fn is None:
            if self._dummy_loss is None:
                raise ValueError('Neither loss_fn nor loss_repr is not set. '
                                 'Call setup_loss_call() or setup_loss_repr() before calling forward_and_backward().')
            self._loss = self._dummy_loss.eval(self._model_output)
        else:
            self._loss = self._call_loss_fn()
        return self._loss

    def forward(self):
        self.call_model()
        self.call_loss()

    def backward(self):
        self._loss.backward()

    def forward_and_backward(self) -> Tuple[Any, Tensor]:
        """Performs a forward pass (:obj:`call_model()` and :obj:`call_loss()`)
        and a backward pass (:obj:`loss.backward()`).

        A child class should override this method.

        Returns:
            - Output of the model function (Any)
            - Output of the loss function or loss evaluation (torch.Tensor)

        Example:
            >>> for x, t in data_loader:
            >>>     # Standard gradient calculation
            >>>     y = model(x)
            >>>     loss = F.mse_loss(y, t)
            >>>     loss.backward()
            >>>
            >>>     # Gradient calculation by GradientMaker
            >>>     dummy_y = grad_maker.setup_model_call(model, x)
            >>>     grad_maker.setup_loss_call(F.mse_loss, dummy_y, t)
            >>>     y, loss = grad_maker.forward_and_backward()
        """
        self.forward()
        self.backward()
        return self._model_output, self._loss

    def delegate_forward_and_backward(self, other, *args, **kwargs):
        """Delegates :obj:`forward_and_backward()` to another GradientMaker.
        `*args` and `**kwargs` will be passed to the call of :obj:`other.forward_and_backward()`.

        The results (:obj:`model_output` and :obj:`loss`) will also be stored to :obj:`self` object
        as if it calls :obj:`self.forward_and_backward()`.

        Args:
            other (GradientMaker): Another GradientMaker to perform :obj:`forward_and_backward()`.

        Example:
            >>> grad_maker1 = GradientMaker(model)
            >>> grad_maker2 = GradientMaker(model)
            >>> grad_maker1.delegate_forward_and_backward(grad_maker2)
        """
        other.setup_model_call(self._model_fn, *self._model_args, **self._model_kwargs)
        if self._loss_fn is None:
            other.setup_loss_repr(self._dummy_loss)
        else:
            other.setup_loss_call(self._loss_fn, *self._loss_fn_args, **self._loss_fn_kwargs)
        other.setup_logits_repr(self._dummy_logits)
        other.forward_and_backward(*args, **kwargs)
        self._model_output = other.model_output
        self._loss = other.loss
        return self._model_output, self._loss

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
        def call():
            args, kwargs = self._get_mapped_loss_fn_args_kwargs()
            return self._loss_fn(*args, **kwargs)

        reduction = self._loss_reduction
        if reduction is not None:
            if not has_reduction(self._loss_fn):
                raise AttributeError('loss_fn has to have "reduction" option')
            if isinstance(self._loss_fn, nn.Module):
                original_reduction = self._loss_fn.reduction
                self._loss_fn.reduction = reduction
                rst = call()
                self._loss_fn.reduction = original_reduction
                return rst
            else:
                self._loss_fn_kwargs['reduction'] = reduction
                return call()
        return call()

    def _get_stateless_model_fn(self):
        if not _is_functorch_available:
            raise EnvironmentError('functorch is not available. Follow the installation guide '
                                   'in https://pytorch.org/functorch/stable/.')
        if self._model_fn is None:
            raise ValueError('model_fn is not set. Call setup_model_call().')
        if not isinstance(self._model_fn, nn.Module):
            raise TypeError('model_fn has to be an object of torch.nn.Module.')
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
        if self._loss_reduction == 'none':
            raise ValueError('Stateless loss function is not available when _loss_reduction == "none".')

        model_fn_params_only, params = self._get_stateless_model_fn_params_only()

        def model_loss_fn_params_only(params):
            self._model_output = model_fn_params_only(params)
            self.call_loss()
            if return_output:
                return self._loss, self._model_output
            else:
                return self._loss

        return model_loss_fn_params_only, params

    def _get_stateless_grad_fn_params_only(self, return_output=False):
        model_loss_fn, params = self._get_stateless_model_loss_fn_params_only(return_output=return_output)
        grad_fn = ft.grad(model_loss_fn, has_aux=return_output)
        return grad_fn, params

    def _get_random_tangents(self):
        if not isinstance(self._model_fn, nn.Module):
            raise TypeError(f'_model_fn has to be {nn.Module}. Got {type(self._model_fn)}.')
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
                if len(ref_args) != len(vmap_info.args_batch_dims):
                    raise ValueError(f'len(ref_args) ({len(ref_args)}) does not match '
                                     f'args_batch_dims ({len(vmap_info.args_batch_dims)}).')
                if len(ref_kwargs) != len(vmap_info.kwargs_batch_dims):
                    raise ValueError(f'len(ref_kwargs) ({len(ref_kwargs)}) does not match '
                                     f'kwargs_batch_dims ({len(vmap_info.kwargs_batch_dims)}).')
                for k in ref_kwargs.keys():
                    if k not in vmap_info.kwargs_batch_dims:
                        raise ValueError(f'Key {k} is not in kwargs_batch_dims')
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
            if len(src_args) != len(ref_args) + len(ref_kwargs):
                raise ValueError(f'len(src_args) ({len(src_args)}) does not match '
                                 f'len(ref_args)+len(ref_kwargs) ({len(ref_args) + len(ref_kwargs)})')
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
    def loss_hvp(self, tangents=None, accumulate_grad=True) -> Tuple[Tensor, ...]:
        grad_fn, params = self._get_stateless_grad_fn_params_only()
        if tangents is None:
            tangents = self._get_random_tangents()
        grads, hvps = ft.jvp(grad_fn, (params,), (tangents,))
        if accumulate_grad:
            params = [p for p in self.model.parameters() if p.requires_grad]
            for param, grad in zip(params, grads):
                if param.grad is None:
                    param.grad = grad
                else:
                    param.grad.add_(grad)
        return hvps

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
        if loss_type not in [LOSS_MSE, LOSS_CROSS_ENTROPY]:
            raise ValueError(f'Invalid loss type: {loss_type}. {[LOSS_CROSS_ENTROPY, LOSS_MSE]} are supported.')
        logit_fn, params = self._get_stateless_logit_fn_params_only()
        if tangents is None:
            tangents = self._get_random_tangents()
        y, jvp = ft.jvp(logit_fn, (params,), (tangents,))
        if y.ndim != 2:  # n x c
            raise ValueError(f'Number of output dimensions has to be 2. Got {y.ndim}.')
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
