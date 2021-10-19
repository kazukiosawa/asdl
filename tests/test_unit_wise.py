import unittest

import torch
import torch.nn as nn

import asdfghjkl as asdl


class TestUnitWise(unittest.TestCase):
    def extract_unit_wise_from_layer_wise(self, layer_wise, n, m):
        blocks = []
        for i in range(n):
            w = layer_wise[i * m:i * m + m, i * m:i * m + m]
            b1 = layer_wise[i * m:i * m + m, i - n].unsqueeze(1)
            b2 = layer_wise[i - n, i * m:i * m + m].unsqueeze(0)
            b3 = layer_wise[i - n, i - n].unsqueeze(0).unsqueeze(0)
            block = torch.cat((b2, b3), dim=1)
            block = torch.cat((torch.cat((w, b1), dim=1), block))
            blocks.append(block.unsqueeze(0))
        return torch.cat(blocks)

    def test_linear(self):
        layer = nn.Linear(2, 2, bias=True)
        layer.weight.data = torch.tensor([[1., 2.], [4., 3.]])
        layer.bias.data = torch.tensor([0.1, -0.1])
        asdl.fisher_for_cross_entropy(layer,
                                      fisher_type=asdl.FISHER_EMP,
                                      fisher_shapes=[asdl.SHAPE_UNIT_WISE],
                                      inputs=torch.tensor([[-0.5, 0.6],
                                                           [-0.5, 0.6]]),
                                      targets=torch.tensor([0, 0],
                                                           dtype=torch.int64))

        torch.testing.assert_close(layer.fisher_emp.unit.data,
                                   torch.tensor([[[0.0156, -0.0187, -0.0312],
                                                  [-0.0187, 0.0225, 0.0374],
                                                  [-0.0312, 0.0374, 0.0624]],
                                                 [[0.0156, -0.0187, -0.0312],
                                                  [-0.0187, 0.0225, 0.0374],
                                                  [-0.0312, 0.0374, 0.0624]]]),
                                   rtol=1e-4,
                                   atol=1e-4)

    def test_random_linear(self):
        layer = nn.Linear(128, 128, bias=True)
        asdl.fisher_for_cross_entropy(
            layer,
            fisher_type=asdl.FISHER_EMP,
            fisher_shapes=[asdl.SHAPE_UNIT_WISE, asdl.SHAPE_LAYER_WISE],
            inputs=torch.randn(2, 128),
            targets=torch.tensor([0, 0], dtype=torch.int64))

        torch.testing.assert_close(
            self.extract_unit_wise_from_layer_wise(
                layer.fisher_emp.data,
                layer.fisher_emp.unit.data.shape[0],
                layer.fisher_emp.unit.data.shape[1] - 1,
            ), layer.fisher_emp.unit.data)

    def test_conv(self):
        layer = nn.Conv2d(1, 1, 2, bias=True)
        layer.weight.data = torch.tensor([[[
            [1.0, 0.0],
            [0.0, 1.0],
        ]]])
        layer.bias.data = torch.tensor([1.0])
        inputs = torch.tensor([[[
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]]])
        model = nn.Sequential(layer, nn.Flatten())

        asdl.fisher_for_cross_entropy(model,
                                      fisher_type=asdl.FISHER_EMP,
                                      fisher_shapes=[asdl.SHAPE_UNIT_WISE],
                                      inputs=inputs.repeat(2, 1, 1, 1),
                                      targets=torch.tensor([0, 0],
                                                           dtype=torch.int64))

        torch.testing.assert_close(
            layer.fisher_emp.unit.data,
            torch.tensor([[[3.8574, 3.8574, 3.8574, 3.8574, 0.0000],
                           [3.8574, 3.8574, 3.8574, 3.8574, 0.0000],
                           [3.8574, 3.8574, 3.8574, 3.8574, 0.0000],
                           [3.8574, 3.8574, 3.8574, 3.8574, 0.0000],
                           [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]]))

    def test_random_conv(self):
        layer = nn.Conv2d(2, 2, 2, bias=True)
        inputs = torch.randn(2, 2, 3, 3)
        model = nn.Sequential(layer, nn.Flatten())

        asdl.fisher_for_cross_entropy(
            model,
            fisher_type=asdl.FISHER_EMP,
            fisher_shapes=[asdl.SHAPE_UNIT_WISE, asdl.SHAPE_LAYER_WISE],
            inputs=inputs,
            targets=torch.tensor([0, 0], dtype=torch.int64))

        torch.testing.assert_close(
            self.extract_unit_wise_from_layer_wise(
                layer.fisher_emp.data,
                layer.fisher_emp.unit.data.shape[0],
                layer.fisher_emp.unit.data.shape[1] - 1,
            ), layer.fisher_emp.unit.data)


if __name__ == '__main__':
    unittest.main()
