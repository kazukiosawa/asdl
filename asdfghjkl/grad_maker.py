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
        self._model_output = None
        self._model_fn = None
        self._model_args = ()
        self._model_kwargs = dict()
        self._loss_fn = None
        self._loss_fn_args = ()
        self._loss_fn_kwargs = dict()
        self._dummy_loss = DummyObject([GetItem(1)])  # default: logits, loss = model_fn(*args, **kwargs)

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

    def extract_loss_from_model_output(self):
        return self._dummy_loss.eval(self._model_output)

    def forward_and_backward(self) -> Union[Tuple[Any, Tensor], Any]:
        # Performs a simple gradient calculation.
        # A child class should override this function.
        rst = self._forward()
        if self._loss_fn is None:
            loss = self.extract_loss_from_model_output()
        else:
            _, loss = rst
        loss.backward()
        return rst

    def _forward(self) -> Union[Tuple[Any, Tensor], Any]:
        self._call_model_fn()
        model_output = self._model_output
        if self._loss_fn is None:
            return model_output
        loss = self._call_loss_fn()
        return model_output, loss

    def _call_model_fn(self):
        assert self._model_fn is not None, \
            f'model_fn is not specified. Call {GradientMaker.setup_model_call} ' \
            f'before calling {GradientMaker.forward_and_backward}.'
        self._model_output = self._model_fn(*self._model_args, **self._model_kwargs)

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
