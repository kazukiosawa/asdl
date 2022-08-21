from typing import Tuple

from torch import Tensor
import torch.nn as nn

from ..utils import DummyObject, GetFirstItem, GetItem

__all__ = ['GradientMaker']


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
        self._dummy_logits = DummyObject([GetFirstItem()])
        self._dummy_loss = DummyObject([GetItem(1)])

    def setup_model_call(self, model_fn, *args, **kwargs):
        self._model_fn = model_fn
        self._model_args = args
        self._model_kwargs = kwargs
        return DummyObject()

    def setup_loss_call(self, loss_fn, *args, **kwargs):
        self._loss_fn = loss_fn
        self._loss_fn_args = args
        self._loss_fn_kwargs = kwargs

    def setup_logits_repr(self, dummy_logits: DummyObject):
        assert isinstance(dummy_logits, DummyObject), \
            f'dummy_logits has to be an {DummyObject}, not {type(dummy_logits)}.'
        self._dummy_logits = dummy_logits

    def setup_loss_repr(self, dummy_loss: DummyObject):
        assert isinstance(dummy_loss, DummyObject), \
            f'dummy_loss has to be an {DummyObject}, not {type(dummy_loss)}.'
        self._dummy_loss = dummy_loss

    def forward_and_backward(self) -> Tuple[Tensor, Tensor]:
        # Performs a simple gradient calculation.
        # A child class should override this function.
        logits, loss = self._forward()
        loss.backward()
        return logits, loss

    def _forward(self) -> Tuple[Tensor, Tensor]:
        self._call_model_fn()
        output = self._model_output
        logits = self._dummy_logits.eval(output)
        if self._loss_fn is None:
            loss = self._dummy_loss.eval(output)
        else:
            loss = self._call_loss_fn()
        return logits, loss

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
