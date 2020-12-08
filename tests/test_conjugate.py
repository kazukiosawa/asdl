import unittest

import torch
from torch import nn
import torch.nn.functional as F

from asdfghjkl import FISHER_EXACT, FISHER_MC, COV
from asdfghjkl import SHAPE_FULL, SHAPE_BLOCK_DIAG
from asdfghjkl import fisher_free_for_cross_entropy
from asdfghjkl import NaturalGradient, LayerWiseNaturalGradient


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


class TestCG(unittest.TestCase):

    def setUp(self):
        self.n_data = 40
        self.batch_size = 10
        self.n_classes = 10
        self.places = 3
        self.damping = 1e-2
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def test_linear(self):
        """
        Compare kernels for LinearNet.
        """
        n_dim = 32
        inputs = torch.randn(self.n_data, n_dim)
        model = net(n_dim, self.n_classes)
        self._test_fisher_free(model, inputs)

    def test_conv(self):
        """
        Compare kernels for ConvNet.
        """
        n_dim = 16
        n_channels = 3
        inputs = torch.randn(self.n_data, n_channels, n_dim, n_dim)
        model = convnet(n_dim, n_channels, self.n_classes, kernel_size=2)
        self._test_fisher_free(model, inputs)

    def _test_fisher_free(self, model, inputs):
        model = model.to(self.device)
        inputs = inputs.to(self.device)
        targets = torch.randint(self.n_classes, (self.n_data,), device=self.device)
        damping = self.damping

        def _get_ng_by_precondition(ng_fn, fisher_type):
            precond = ng_fn(model, fisher_type=fisher_type, damping=damping)
            precond.update_curvature(inputs, targets)

            model.zero_grad()
            loss = F.cross_entropy(model(inputs), targets)
            loss.backward()

            precond.update_inv()
            precond.precondition()

            grads = []
            for p in model.parameters():
                if p.requires_grad:
                    grads.append(p.grad.flatten())
            return torch.cat(grads)

        def _get_ng_by_fisher_free(fisher_type, fisher_shape, random_seed):
            model.zero_grad()
            loss = F.cross_entropy(model(inputs), targets)
            loss.backward()

            grad = [p.grad for p in model.parameters() if p.requires_grad]

            ng = fisher_free_for_cross_entropy(model,
                                               grad,
                                               fisher_type=fisher_type,
                                               fisher_shape=fisher_shape,
                                               inputs=inputs,
                                               targets=targets,
                                               damping=damping,
                                               max_iters=None,
                                               random_seed=random_seed,
                                               tol=1e-8)
            return torch.cat([_ng.flatten() for _ng in ng])

        def _test(ng_fn, fisher_type):
            seed = int(torch.rand(1) * 100)  # for MC sampling

            torch.manual_seed(seed)
            ng = _get_ng_by_precondition(ng_fn, fisher_type)

            if ng_fn == NaturalGradient:
                fisher_shape = SHAPE_FULL
            elif ng_fn == LayerWiseNaturalGradient:
                fisher_shape = SHAPE_BLOCK_DIAG
            else:
                return

            ng2 = _get_ng_by_fisher_free(fisher_type, fisher_shape, seed)

            self._assert_almost_equal(ng, ng2)

        for ng_class in [NaturalGradient, LayerWiseNaturalGradient]:
            for ftype in [FISHER_EXACT, FISHER_MC, COV]:
                _test(ng_class, ftype)

    def _assert_almost_equal(self, v1, v2):
        cos = _cosine_similarity(v1, v2)
        self.assertAlmostEqual(cos, 1, places=self.places)
        err = _relative_error(v1, v2)
        self.assertAlmostEqual(err, 0, places=self.places)


if __name__ == '__main__':
    unittest.main()
