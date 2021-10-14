import unittest

import torch
import torch.nn as nn

import asdfghjkl as asdl


class TestKFAC(unittest.TestCase):
    def test_linear(self):
        layer = nn.Linear(2, 2, bias=True)
        layer.weight.data = torch.tensor([[1., 2.], [4., 3.]])
        layer.bias.data = torch.tensor([0.1, -0.1])
        asdl.fisher_for_cross_entropy(
            layer,
            fisher_type=asdl.FISHER_EMP,
            fisher_shapes=[asdl.SHAPE_UNITWISE_BLOCK_DIAG],
            inputs=torch.tensor([[-0.5, 0.6]]),
            targets=torch.tensor([0], dtype=torch.int64))
        torch.isclose(
            layer.fisher_emp.unit.data,
            torch.tensor([[[0.0156, -0.0187, -0.0312],
                           [-0.0187, 0.0225, 0.0374],
                           [-0.0312, 0.0374, 0.0624]],
                          [[0.0156, -0.0187, -0.0312],
                           [-0.0187, 0.0225, 0.0374],
                           [-0.0312, 0.0374, 0.0624]]]))

    def test_conv(self):
        pass


if __name__ == '__main__':
    unittest.main()
