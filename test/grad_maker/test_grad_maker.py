import pytest

import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector
from asdl import GradientMaker
from asdl import LOSS_CROSS_ENTROPY, LOSS_MSE


@pytest.mark.parametrize('in_dim', [5])
@pytest.mark.parametrize('hid_dim', [4])
@pytest.mark.parametrize('out_dim', [3])
@pytest.mark.parametrize('batch_size', [2])
class TestGradientMaker:

    @pytest.mark.parametrize('loss_type', [LOSS_CROSS_ENTROPY, LOSS_MSE])
    @pytest.mark.parametrize('network_type', ['mlp'])
    def test_logits(self, model, loss_fn, multi_data):
        """
        the model returns logits only
        """
        x, t = multi_data
        grad_maker = GradientMaker(model)

        # standard forward and backward
        model.zero_grad()
        y_true = model(x)
        loss_true = loss_fn(y_true, t)
        loss_true.backward()
        g_true = parameters_to_vector([p.grad for p in model.parameters()])

        # forward and backward by GradientMaker
        model.zero_grad()
        dummy_y = grad_maker.setup_model_call(model, x)
        grad_maker.setup_loss_call(loss_fn, dummy_y, t)
        y_test, loss_test = grad_maker.forward_and_backward()
        g_test = parameters_to_vector([p.grad for p in model.parameters()])

        torch.testing.assert_close(y_true, y_test)
        torch.testing.assert_close(loss_true, loss_test)
        torch.testing.assert_close(g_true, g_test)

    @pytest.mark.parametrize('loss_type', [LOSS_CROSS_ENTROPY, LOSS_MSE])
    @pytest.mark.parametrize('network_type', ['mlp'])
    def test_tuple(self, model, loss_fn, multi_data):
        """
        the model returns a tuple (logits, loss)
        """
        x, t = multi_data
        grad_maker = GradientMaker(model)

        # standard forward and backward
        model.zero_grad()
        y_true = model(x, t)
        logits_true, loss_true = y_true
        loss_true.backward()
        g_true = parameters_to_vector([p.grad for p in model.parameters()])

        # forward and backward by GradientMaker
        model.zero_grad()
        dummy_y = grad_maker.setup_model_call(model, x, t)
        grad_maker.setup_loss_repr(dummy_y[1])
        y, loss_test = grad_maker.forward_and_backward()
        logits_test, _ = y
        g_test = parameters_to_vector([p.grad for p in model.parameters()])

        torch.testing.assert_close(logits_true, logits_test)
        torch.testing.assert_close(loss_true, loss_test)
        torch.testing.assert_close(g_true, g_test)

    @pytest.mark.parametrize('seq_len', [8])
    def test_sequence_output(self, sequence_model, sequence_data):
        """
        the model returns Output class instance
        """
        x, t = sequence_data
        model = sequence_model
        grad_maker = GradientMaker(model)

        # standard forward and backward
        model.zero_grad()
        y_true = model(x)
        logits_true = y_true.logits
        loss_true = F.cross_entropy(logits_true.view(-1, logits_true.size(-1)), t.view(-1), ignore_index=-1)
        loss_true.backward()
        g_true = parameters_to_vector([p.grad for p in model.parameters()])

        # forward and backward by GradientMaker
        model.zero_grad()
        dummy_y = grad_maker.setup_model_call(model, x)
        grad_maker.setup_loss_call(F.cross_entropy, dummy_y.logits.view(-1, dummy_y.logits.size(-1)), t.view(-1), ignore_index=-1)
        y_test, loss_test = grad_maker.forward_and_backward()
        logits_test = y_test.logits
        g_test = parameters_to_vector([p.grad for p in model.parameters()])

        torch.testing.assert_close(logits_true, logits_test)
        torch.testing.assert_close(loss_true, loss_test)
        torch.testing.assert_close(g_true, g_test)

    @pytest.mark.parametrize('seq_len', [8])
    def test_sequence_tuple(self, sequence_model, sequence_data):
        """
        the model returns a tuple (loss, logits)
        """
        x, t = sequence_data
        model = sequence_model
        grad_maker = GradientMaker(model)

        # standard forward and backward
        model.zero_grad()
        y_true = model(x, t, return_dict=False)
        loss_true, logits_true = y_true
        loss_true.backward()
        g_true = parameters_to_vector([p.grad for p in model.parameters()])

        # forward and backward by GradientMaker
        model.zero_grad()
        dummy_y = grad_maker.setup_model_call(model, x, t, return_dict=False)
        grad_maker.setup_loss_repr(dummy_y[0])
        y, loss_test = grad_maker.forward_and_backward()
        _, logits_test = y
        g_test = parameters_to_vector([p.grad for p in model.parameters()])

        torch.testing.assert_close(logits_true, logits_test)
        torch.testing.assert_close(loss_true, loss_test)
        torch.testing.assert_close(g_true, g_test)
