from typing import Any, Tuple, Dict, Union
from dataclasses import dataclass

from torch import Tensor
import torch.nn as nn

__all__ = ['GradientMaker', 'DummyObject']


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


class GradientMaker:
    def __init__(self, model: nn.Module):
        self.model = model
        self._model_fn = None
        self._model_args = ()
        self._model_kwargs = dict()
        self._model_output = None
        self._logits: Tensor = None
        self._loss_fn = None
        self._loss_fn_args = ()
        self._loss_fn_kwargs = dict()
        self._loss: Tensor = None
        self._dummy_loss: DummyObject = None
        self._dummy_logits = DummyObject([GetFirstItem()])

    def setup_model_call(self, model_fn, *args, **kwargs):
        self._model_fn = model_fn
        self._model_args = args
        self._model_kwargs = kwargs
        return DummyObject()

    def setup_loss_call(self, loss_fn, *args, **kwargs):
        self._loss_fn = loss_fn
        self._loss_fn_args = args
        self._loss_fn_kwargs = kwargs

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
        if self._loss_fn is None:
            return self._model_output
        else:
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
