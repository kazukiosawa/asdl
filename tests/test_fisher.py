import numpy as np
import pytest
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import asdfghjkl as asdl
from asdfghjkl import fisher_esd, calculate_fisher, Bias, Scale, LOSS_CROSS_ENTROPY, save_inputs_outgrads
from asdfghjkl import SHAPE_LAYER_WISE, SHAPE_KRON, SHAPE_UNIT_WISE, SHAPE_DIAG, OP_SAVE_INPUTS, OP_SAVE_OUTGRADS

class Net(nn.Module):
    def __init__(self, drop_rate=0.2, last_drop_rate=0.4, padding=1, bias=True, data_size=32):
        super().__init__()
        self.padding = padding
        self.data_size = data_size
        self.conv1 = nn.Conv2d(3, 3, 3, 1, padding=padding, bias=bias)
        self.grp_norm1 = nn.GroupNorm(1, 3)
        self.conv2 = nn.Conv2d(3, 6, 3, 1, padding=padding, bias=bias)
        self.grp_norm2 = nn.GroupNorm(2, 6)
        self.dropout1 = nn.Dropout(drop_rate)
        self.conv3 = nn.Conv2d(6, 6, 3, 1, padding=padding, bias=bias)
        self.grp_norm3 = nn.GroupNorm(2, 6)
        self.conv4 = nn.Conv2d(6, 6, 3, 1, padding=padding, bias=bias)
        self.grp_norm4 = nn.GroupNorm(2, 6)
        self.dropout2 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(6*(data_size//4-3+3*padding)**2, 50)
        self.dropout3 = nn.Dropout(last_drop_rate)
        self.fc2 = nn.Linear(50, 10)
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
    def forward(self, x):
        x = F.relu(self.grp_norm1(self.conv1(x)))
        x = F.relu(self.grp_norm2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout1(x)
        x = F.relu(self.grp_norm3(self.conv3(x)))
        x = F.relu(self.grp_norm4(self.conv4(x)))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout2(x)
        x = x.view(-1, (self.data_size//4-3+3*self.padding)**2 * 6)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x

def test_lw_hess_linear(loss_type, cuda):
    device = torch.device('cuda' if cuda else 'cpu')
    model = nn.Linear(20, 10).to(device)
    x = torch.randn(1, 20).to(device)
    if loss_type == LOSS_CROSS_ENTROPY:
        y = torch.tensor([0]).to(device)
    else:
        y = torch.randn(1, 10).to(device)
    z = model(x)
    loss = F.cross_entropy(z, y)
    loss.backward(inputs=z, retain_graph=True)
    out_grads = z.grad
    f = calculate_fisher(model, asdl.FISHER_EXACT, SHAPE_LAYER_WISE, loss_type, inputs=x)
    if loss_type == LOSS_CROSS_ENTROPY:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss() # To be fixed
    asdl.hessian(model, criterion, SHAPE_LAYER_WISE, inputs=x, targets=y)
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            lw = f.get_fisher_tensor(module, 'data')
            hess = getattr(module, 'hessian').data
            print((lw-hess).norm())
            for k in range(out_grads.shape[-1]):
                targets = torch.tensor([0], requires_grad=False)
                if isinstance(module, nn.Conv2d):
                    inputs = torch.randn(1, module.in_channels, module.kernel_size[0]+1, module.kernel_size[1]+1,
                                         requires_grad=False).to(device)
                else:
                    inputs = torch.randn(1, module.in_features, requires_grad=False).to(device)
                asdl.hessian(module, lambda outputs, targets: z[0, k], asdl.SHAPE_FULL, inputs=inputs, targets=targets)
                hess_out = getattr(module, 'hessian').data
                lw += hess_out * out_grads[0, k]
            torch.testing.assert_close(lw.norm(), hess.norm())

def test_lw_hess(loss_type, data_size, padding, cuda):
    device = torch.device('cuda' if cuda else 'cpu')
    model = Net(padding=padding, bias=True, data_size=data_size).to(device)
    x = torch.randn(1, 3, data_size, data_size).to(device)
    if loss_type == LOSS_CROSS_ENTROPY:
        y = torch.tensor([0]).to(device)
    else:
        y = torch.randn(1, 10).to(device)
    z = model(x)
    loss = F.cross_entropy(z, y)
    loss.backward(inputs=z, retain_graph=True)
    out_grads = z.grad
    f = calculate_fisher(model, asdl.FISHER_EXACT, SHAPE_LAYER_WISE, loss_type, inputs=x)
    if loss_type == LOSS_CROSS_ENTROPY:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    asdl.hessian(model, criterion, SHAPE_LAYER_WISE, inputs=x, targets=y)
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            lw = f.get_fisher_tensor(module, 'data')
            hess = getattr(module, 'hessian').data
            print((lw-hess).norm())
            for k in range(out_grads.shape[-1]):
                targets = torch.tensor([0], requires_grad=False)
                if isinstance(module, nn.Conv2d):
                    inputs = torch.randn(1, module.in_channels, module.kernel_size[0]+1, module.kernel_size[1]+1,
                                         requires_grad=False).to(device)
                else:
                    inputs = torch.randn(1, module.in_features, requires_grad=False).to(device)
                asdl.hessian(module, lambda outputs, targets: z[0, k], asdl.SHAPE_FULL, inputs=inputs, targets=targets)
                hess_out = getattr(module, 'hessian').data
                lw += hess_out * out_grads[0, k]
            torch.testing.assert_close(lw, hess)

def test_kron_lw(model, data, fisher_type, loss_type, data_size, padding, bias, cuda):
    device = torch.device('cuda' if cuda else 'cpu')
    model = Net(padding=padding, bias=bias, data_size=data_size).to(device)
    kron_modules = (nn.Linear, nn.Conv2d, nn.Embedding, Bias, Scale)
    x = torch.randn(1, 3, data_size, data_size)
    if loss_type == LOSS_CROSS_ENTROPY:
        y = torch.tensor([0])
    else:
        y = torch.randn(1, 10)
    # x, y = data
    with save_inputs_outgrads(model) as cxt:
        f = calculate_fisher(model, fisher_type, [SHAPE_LAYER_WISE, SHAPE_KRON], loss_type, inputs=x, targets=y)
        for name, module in model.named_modules():
            if isinstance(module, kron_modules):
                print(name)
                kron_A = f.get_fisher_tensor(module, 'kron', 'A')
                kron_B = f.get_fisher_tensor(module, 'kron', 'B')
                kron = torch.kron(kron_B, kron_A)
                lw = f.get_fisher_tensor(module, 'data')
                if isinstance(module, nn.Conv2d):
                    in_data = cxt.get_result(module, OP_SAVE_INPUTS)[0]
                    kron *= in_data.shape[-1]
                    if asdl.original_requires_grad(module, 'bias'):
                        in_data = cxt.get_operation(module).extend_in_data(in_data)
                    out_grads_list = cxt.get_result(module, OP_SAVE_OUTGRADS)
                    for out_grads in out_grads_list:
                        for i in range(in_data.shape[-1]):
                            for j in range(in_data.shape[-1]):
                                if i == j:
                                    continue
                                ggt = torch.outer(out_grads[0,:,i], out_grads[0,:,j])
                                aat = torch.outer(in_data[0,:,i], in_data[0,:,j])
                                gakron = torch.kron(ggt, aat)
                                ggt2 = torch.outer(out_grads[0,:,i], out_grads[0,:,i])
                                aat2 = torch.outer(in_data[0,:,j], in_data[0,:,j])
                                gakron2 = torch.kron(ggt2, aat2)
                                kron += gakron - gakron2
                torch.testing.assert_close(kron, lw)

def test_kron_linear(fisher_type, loss_type, bias, cuda):
    device = torch.device('cuda' if cuda else 'cpu')
    model = nn.Linear(20, 10, bias=bias).to(device)
    kron_modules = (nn.Linear, nn.Conv2d, nn.Embedding, Bias, Scale)
    x = torch.randn(1, 20)
    if loss_type == LOSS_CROSS_ENTROPY:
        y = torch.tensor([0], dtype=torch.long)
    else:
        y = torch.randn(1, 10)
    f = calculate_fisher(model, fisher_type, [SHAPE_LAYER_WISE, SHAPE_KRON], loss_type, inputs=x, targets=y, seed=1)
    kron_A = f.get_fisher_tensor(model, 'kron', 'A')
    kron_B = f.get_fisher_tensor(model, 'kron', 'B')
    if asdl.original_requires_grad(model, 'bias'):
        kron_A_weight = kron_A[:-1, :-1]
        kron_A_bias = kron_A[-1, :-1]
        kron_weight = torch.kron(kron_B, kron_A_weight)
        kron_bias = torch.kron(kron_B, kron_A_bias)
        kron = torch.cat([kron_weight, kron_bias], dim=0)
        right = torch.cat([torch.kron(kron_B, kron_A_bias.view(-1, 1)), kron_B], dim=0)
        kron = torch.cat([kron, right], dim=1)
    else:
        kron = torch.kron(kron_B, kron_A)
    lw = f.get_fisher_tensor(model, 'data')
    torch.testing.assert_close(lw, kron)

def test_kron_conv(fisher_type, loss_type, data_size, padding, bias, cuda):
    device = torch.device('cuda' if cuda else 'cpu')
    model = nn.Sequential(nn.Conv2d(3, 2, 3, padding=padding, bias=bias),nn.Flatten()).to(device)
    x = torch.randn(1, 3, data_size, data_size)
    if loss_type == LOSS_CROSS_ENTROPY:
        y = torch.tensor([0], dtype=torch.long)
    else:
        y = torch.randn(1, 2*(data_size-2+2*padding)**2)
    with save_inputs_outgrads(model) as cxt:
        f = calculate_fisher(model, fisher_type, [SHAPE_LAYER_WISE, SHAPE_KRON], loss_type, inputs=x, targets=y)
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                in_data = cxt.get_result(module, OP_SAVE_INPUTS)[0]
                kron_A = f.get_fisher_tensor(module, 'kron', 'A')
                kron_B = f.get_fisher_tensor(module, 'kron', 'B')
                kron = torch.kron(kron_B, kron_A)*in_data.shape[-1]
                if asdl.original_requires_grad(module, 'bias'):
                    in_data = cxt.get_operation(module).extend_in_data(in_data)
                out_grads_list = cxt.get_result(module, OP_SAVE_OUTGRADS)
                for out_grads in out_grads_list:
                    for i in range(in_data.shape[-1]):
                        for j in range(in_data.shape[-1]):
                            if i == j:
                                continue
                            ggt = torch.outer(out_grads[0,:,i], out_grads[0,:,j])
                            aat = torch.outer(in_data[0,:,i], in_data[0,:,j])
                            gakron = torch.kron(ggt, aat)
                            ggt2 = torch.outer(out_grads[0,:,i], out_grads[0,:,i])
                            aat2 = torch.outer(in_data[0,:,j], in_data[0,:,j])
                            gakron2 = torch.kron(ggt2, aat2)
                            kron += gakron - gakron2
                lw = f.get_fisher_tensor(module, 'data')
                torch.testing.assert_close(kron, lw)

def test_kron_mvp(fisher_type, loss_type, data_size, padding, cuda):
    pass

def test_kronA_linear(fisher_type, loss_type, cuda):
    device = torch.device('cuda' if cuda else 'cpu')
    bs = 4
    model = nn.Linear(20, 10).to(device)
    in_data = torch.randn(bs, 20).to(device)
    if loss_type == LOSS_CROSS_ENTROPY:
        y = torch.tensor([0]*bs, dtype=torch.long).to(device)
    else:
        y = torch.randn(bs, 10).to(device)
    f = calculate_fisher(model, fisher_type, asdl.SHAPE_KRON, loss_type, inputs=in_data, targets=y)
    kron_A = f.get_fisher_tensor(model, 'kron', 'A')
    shape = list(in_data.shape)
    shape[1] = 1
    ones = in_data.new_ones(shape)
    in_data_ext = torch.cat((in_data, ones), dim=1)
    torch.testing.assert_close(kron_A, torch.matmul(in_data_ext.T, in_data_ext).div(bs), rtol=0., atol=0.)

def test_kronA_conv(fisher_type, loss_type, cuda):
    device = torch.device('cuda' if cuda else 'cpu')
    bs = 4
    model = nn.Sequential(nn.Conv2d(3, 4, 5), nn.Flatten()).to(device)
    in_data = torch.randn(bs, 3, 6, 6).to(device)
    if loss_type == LOSS_CROSS_ENTROPY:
        y = torch.tensor([0]*bs, dtype=torch.long).to(device)
    else:
        y = torch.randn(bs, 16).to(device)
    f = calculate_fisher(model, fisher_type, asdl.SHAPE_KRON, loss_type, inputs=in_data, targets=y)
    for module in model.modules():
        kron_A = f.get_fisher_tensor(module, 'kron', 'A')
        if kron_A is None:
            continue
        in_data = asdl.im2col_2d(in_data, module)
        shape = list(in_data.shape)
        shape[1] = 1
        ones = in_data.new_ones(shape)
        in_data_ext = torch.cat((in_data, ones), dim=1)
        out_size = in_data.shape[-1]
        m = in_data_ext.transpose(0, 1).flatten(start_dim=1)  # (c_in)(kernel_size) x n(out_size)
        kron_A_other = torch.matmul(m, m.T).div(out_size*bs)
        torch.testing.assert_close(kron_A, kron_A_other, rtol=0., atol=0.)

def test_fisher_esd(model, fisher_type, fisher_shape, loss_type, data):
    torch.random.manual_seed(1)
    x, y = data
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    if fisher_shape in [SHAPE_KRON, SHAPE_DIAG, SHAPE_UNIT_WISE]:
        pytest.xfail(reason="fisher_shape not supported for fvp and unit_wise not implemented in Bias and Scale")
        fisher_esd(model, fisher_type, fisher_shape, loss_type, inputs=x, targets=y, n_v=1, num_iter=num_params, seed=1)
    else:
        density_slq, grids_slq = fisher_esd(
            model, fisher_type, fisher_shape, loss_type, inputs=x, targets=y, n_v=1, num_iter=num_params, seed=1)
        f = calculate_fisher(model, fisher_type, fisher_shape, loss_type, inputs=x, targets=y, seed=1)
        eigvals = f.get_eigenvalues(fisher_type, fisher_shape).tolist()
        lambda_max = np.max(eigvals)
        lambda_min = np.min(eigvals)
        sigma_squared = 1e-5 * max(1, (lambda_max - lambda_min))
        density_exact = np.zeros(10000)
        for j in range(10000):
            x = grids_slq[j]
            tmp_result = np.exp(-(x - eigvals)**2 /
                                    (2.0 * sigma_squared)) / np.sqrt(2 * np.pi * sigma_squared)
            density_exact[j] = np.sum(tmp_result)
        normalization = np.sum(density_exact) * (grids_slq[1] - grids_slq[0])
        density_exact = density_exact / normalization
        comparr = np.isclose(density_slq, density_exact)
        assert comparr.sum() / comparr.size >= 0.7
