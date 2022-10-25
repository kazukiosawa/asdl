import unittest

import torch
from torch import nn
import torch.nn.functional as F

from asdfghjkl import FISHER_EMP, SHAPE_FULL, SHAPE_LAYER_WISE
from asdfghjkl import NaturalGradientMaker, LayerWiseNaturalGradientMaker, woodbury_ifvp


def _relative_error(v1: torch.tensor, v2: torch.tensor):
    err = v1 - v2
    return (err.norm() / v1.norm()).item()


def _cosine_similarity(vec1, vec2):
    inp = vec1.mul(vec2).sum()
    return (inp / vec1.norm() / vec2.norm()).item()


def net(n_dim, n_classes):
    model = nn.Sequential(
        nn.Linear(n_dim, n_dim),
        nn.ReLU(),
        nn.Linear(n_dim, n_dim),
        nn.ReLU(),
        nn.Linear(n_dim, n_classes),
    )
    return model


def convnet(n_dim, n_channels, n_classes, kernel_size=2):
    n_features = int(n_dim / (kernel_size ** 4)) ** 2 * n_channels
    model = nn.Sequential(
        nn.Conv2d(n_channels, n_channels, kernel_size, kernel_size),
        nn.MaxPool2d(kernel_size),
        nn.ReLU(),
        nn.Conv2d(n_channels, n_channels, kernel_size, kernel_size),
        nn.MaxPool2d(kernel_size),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(n_features, n_classes),
    )
    return model


class TestIFVP(unittest.TestCase):

    def setUp(self):
        self.n_data = 50
        self.n_classes = 10
        self.places = 2
        self.damping = 1e-3
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def test_linear(self):
        """
        Compare empirical natural gradient for LinearNet.
        """
        n_dim = 32
        inputs = torch.randn(self.n_data, n_dim)
        model = net(n_dim, self.n_classes)
        self._test_empirical_natural_gradient(model, inputs)

    def test_conv(self):
        """
        Compare empirical natural gradient for ConvNet.
        """
        n_dim = 16
        n_channels = 3
        inputs = torch.randn(self.n_data, n_channels, n_dim, n_dim)
        model = convnet(n_dim, n_channels, self.n_classes, kernel_size=2)
        self._test_empirical_natural_gradient(model, inputs)

    def _test_empirical_natural_gradient(self, model, inputs):
        model = model.to(self.device)
        inputs = inputs.to(self.device)
        targets = torch.randint(self.n_classes, (self.n_data,), device=self.device)
        damping = self.damping

        def _get_ng_by_precondition(ng_fn):
            precond = ng_fn(model, fisher_type=FISHER_EMP, damping=damping)
            precond.accumulate_curvature(inputs, targets)

            model.zero_grad()
            loss = F.cross_entropy(model(inputs), targets)
            loss.backward()

            precond.update_preconditioner()
            precond.precondition()

            grads = []
            for p in model.parameters():
                if p.requires_grad:
                    grads.append(p.grad.flatten())
            return torch.cat(grads)

        def _get_ng_by_woodbury(fisher_shape):
            model.zero_grad()
            loss = F.cross_entropy(model(inputs), targets)
            loss.backward()
            grads = []
            for p in model.parameters():
                if p.requires_grad:
                    grads.append(p.grad.flatten())
            g = torch.cat(grads)
            return woodbury_ifvp(g, model, inputs, targets, F.cross_entropy,
                                 damping=damping, data_average=False)

        def _test(ng_fn, fisher_shape):
            ng = _get_ng_by_precondition(ng_fn)
            ng2 = _get_ng_by_woodbury(fisher_shape)
            self._assert_almost_equal(ng, ng2)

        _test(NaturalGradientMaker, SHAPE_FULL)

    def _assert_almost_equal(self, v1, v2):
        cos = _cosine_similarity(v1, v2)
        self.assertAlmostEqual(cos, 1, places=self.places)
        err = _relative_error(v1, v2)
        self.assertAlmostEqual(err, 0, places=self.places)


if __name__ == '__main__':
    unittest.main()
