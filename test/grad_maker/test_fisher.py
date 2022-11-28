import copy

import pytest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector
from asdl import FisherConfig, get_fisher_maker
from asdl import FISHER_EXACT, FISHER_MC, FISHER_EMP
from asdl import SHAPE_FULL, SHAPE_LAYER_WISE, SHAPE_KRON, SHAPE_SWIFT_KRON, SHAPE_UNIT_WISE, SHAPE_DIAG
from asdl import LOSS_CROSS_ENTROPY, LOSS_MSE
from asdl import ParamVector


_target_modules = (nn.Linear, nn.Conv2d)


def init_fisher_maker(fisher_type, fisher_shape, loss_type, model, loss_fn, data):
    config = FisherConfig(fisher_type=fisher_type,
                          fisher_shapes=[fisher_shape],
                          data_size=1,
                          loss_type=loss_type,
                          seed=1)
    fisher_maker = get_fisher_maker(model, config)
    x, t = data
    dummy_y = fisher_maker.setup_model_call(model, x)
    fisher_maker.setup_loss_call(loss_fn, dummy_y, t)
    return fisher_maker


@pytest.fixture
def fisher_maker_single_data(fisher_type, fisher_shape, loss_type, model, loss_fn, single_data):
    return init_fisher_maker(fisher_type, fisher_shape, loss_type, model, loss_fn, single_data)


@pytest.fixture
def fisher_maker_single_data_copy(fisher_type, fisher_shape, loss_type, model, loss_fn, single_data_copy):
    return init_fisher_maker(fisher_type, fisher_shape, loss_type, model, loss_fn, single_data_copy)


@pytest.fixture
def full_fisher_maker_single_data(fisher_type, loss_type, model, loss_fn, single_data):
    return init_fisher_maker(fisher_type, SHAPE_FULL, loss_type, model, loss_fn, single_data)


@pytest.fixture
def fisher_maker(fisher_type, fisher_shape, loss_type, model, loss_fn, multi_data):
    return init_fisher_maker(fisher_type, fisher_shape, loss_type, model, loss_fn, multi_data)


@pytest.mark.parametrize('network_type', ['mlp', 'cnn'])
@pytest.mark.parametrize('fisher_type', [FISHER_EXACT, FISHER_MC, FISHER_EMP])
@pytest.mark.parametrize('in_dim', [5])
@pytest.mark.parametrize('hid_dim', [4])
@pytest.mark.parametrize('out_dim', [4])
@pytest.mark.parametrize('batch_size', [4])
@pytest.mark.parametrize('fisher_shape', [SHAPE_FULL, SHAPE_LAYER_WISE, SHAPE_KRON, SHAPE_UNIT_WISE, SHAPE_DIAG])
@pytest.mark.parametrize('loss_type', [LOSS_CROSS_ENTROPY, LOSS_MSE])
class TestFisherMaker:

    @staticmethod
    def first_call(model, fisher_maker, fisher_shape, *args, **kwargs):
        fisher_maker.forward_and_backward(*args, **kwargs)
        for module in model.modules():
            if isinstance(module, _target_modules):
                if fisher_shape in [SHAPE_LAYER_WISE, SHAPE_KRON, SHAPE_SWIFT_KRON, SHAPE_UNIT_WISE, SHAPE_DIAG]:
                    setattr(module, 'fisher_1st', copy.deepcopy(module.fisher))
        if fisher_shape == SHAPE_FULL:
            setattr(model, 'fisher_1st', copy.deepcopy(model.fisher))

    @staticmethod
    def compare_vs_first_call(model, fisher_shape, scale=1., inv=False):
        for module in model.modules():
            if isinstance(module, _target_modules):
                if fisher_shape == SHAPE_LAYER_WISE:
                    if inv:
                        torch.testing.assert_close(module.fisher.inv, module.fisher_1st.inv * scale)
                    else:
                        torch.testing.assert_close(module.fisher.data, module.fisher_1st.data * scale)
                elif fisher_shape == SHAPE_KRON:
                    if inv:
                        torch.testing.assert_close(module.fisher.kron.A_inv, module.fisher_1st.kron.A_inv * scale)
                        torch.testing.assert_close(module.fisher.kron.B_inv, module.fisher_1st.kron.B_inv * scale)
                    else:
                        torch.testing.assert_close(module.fisher.kron.A, module.fisher_1st.kron.A * scale)
                        torch.testing.assert_close(module.fisher.kron.B, module.fisher_1st.kron.B * scale)
                elif fisher_shape == SHAPE_UNIT_WISE:
                    if inv:
                        torch.testing.assert_close(module.fisher.unit.inv, module.fisher_1st.unit.inv * scale)
                    else:
                        torch.testing.assert_close(module.fisher.unit.data, module.fisher_1st.unit.data * scale)
                elif fisher_shape == SHAPE_DIAG:
                    if inv:
                        torch.testing.assert_close(module.fisher.diag.weight_inv, module.fisher_1st.diag.weight_inv * scale)
                        torch.testing.assert_close(module.fisher.diag.bias_inv, module.fisher_1st.diag.bias_inv * scale)
                    else:
                        torch.testing.assert_close(module.fisher.diag.weight, module.fisher_1st.diag.weight * scale)
                        torch.testing.assert_close(module.fisher.diag.bias, module.fisher_1st.diag.bias * scale)
        if fisher_shape == SHAPE_FULL:
            if inv:
                torch.testing.assert_close(model.fisher.inv, model.fisher_1st.inv * scale)
            else:
                torch.testing.assert_close(model.fisher.data, model.fisher_1st.data * scale)

    def test_shape_and_value(self, model, fisher_maker_single_data, fisher_maker_single_data_copy, full_fisher_maker_single_data, fisher_shape, fisher_type, batch_size, loss_type):
        self.first_call(model, full_fisher_maker_single_data, SHAPE_FULL)  # calculate full fisher for a single data
        full_fisher = model.fisher_1st.data.clone()
        self.first_call(model, fisher_maker_single_data_copy, fisher_shape)  # calculate {fisher_shape} fisher for singe data copies
        fisher_maker_single_data.forward_and_backward()  # calculate {fisher_shape} fisher for a single data

        if fisher_type != FISHER_MC:  # fisher_mc is skipped as different MC samples are used for each data
            # check if Fisher for multiple (copies of) data is {batch_size} times larger than Fisher for a single data
            self.compare_vs_first_call(model, fisher_shape, scale=1/batch_size)

        pointer = 0
        for module in model.modules():
            if isinstance(module, _target_modules):
                n_params = sum([p.numel() for p in module.parameters()])
                if isinstance(module, nn.Linear):
                    in_dim = module.in_features
                    out_dim = module.out_features
                    assert n_params == (in_dim + 1) * out_dim
                elif isinstance(module, nn.Conv2d):
                    in_channel = module.in_channels
                    kh, kw = module.kernel_size
                    in_dim = in_channel * kh * kw
                    out_dim = module.out_channels
                    assert n_params == (in_dim + 1) * out_dim
                # layer-wise Fisher extracted from full Fisher
                layer_fisher = full_fisher[pointer:pointer+n_params, pointer:pointer+n_params]
                if fisher_shape == SHAPE_LAYER_WISE:
                    # shape check
                    assert tuple(module.fisher.data.shape) == (n_params, n_params)

                    # symmetry check
                    torch.testing.assert_close(module.fisher.data, module.fisher.data.T)

                    # value check
                    torch.testing.assert_close(layer_fisher, module.fisher.data)

                elif fisher_shape == SHAPE_KRON:
                    # shape check
                    assert tuple(module.fisher.kron.A.shape) == (in_dim, in_dim)
                    assert tuple(module.fisher.kron.B.shape) == (out_dim, out_dim)

                    # symmetry check
                    torch.testing.assert_close(module.fisher.kron.A, module.fisher.kron.A.T)
                    torch.testing.assert_close(module.fisher.kron.B, module.fisher.kron.B.T)

                    # value check (for a single data, Kronecker factorization is no longer an approximation)
                    if isinstance(module, nn.Linear):
                        weight_fisher = layer_fisher[:in_dim*out_dim, :in_dim*out_dim]
                        torch.testing.assert_close(weight_fisher, torch.kron(module.fisher.kron.B, module.fisher.kron.A))
                        bias_fisher = layer_fisher[-out_dim:, -out_dim:]
                        torch.testing.assert_close(bias_fisher, module.fisher.kron.B)

                elif fisher_shape == SHAPE_UNIT_WISE:
                    # shape check
                    assert tuple(module.fisher.unit.data.shape) == (out_dim, in_dim+1, in_dim+1)

                    local_pointer = 0
                    for i in range(out_dim):  # for each unit
                        # symmetry check
                        torch.testing.assert_close(module.fisher.unit.data[i], module.fisher.unit.data[i].T)

                        # value check
                        unit_fisher = torch.zeros(in_dim+1, in_dim+1)
                        unit_fisher[:in_dim, :in_dim] = layer_fisher[local_pointer:local_pointer+in_dim, local_pointer:local_pointer+in_dim]
                        unit_fisher[-1, :in_dim] = layer_fisher[in_dim*out_dim+i, local_pointer:local_pointer+in_dim]
                        unit_fisher[:in_dim, -1] = layer_fisher[local_pointer:local_pointer+in_dim, in_dim*out_dim+i]
                        unit_fisher[-1, -1] = layer_fisher[in_dim*out_dim+i, in_dim*out_dim+i]
                        torch.testing.assert_close(unit_fisher, module.fisher.unit.data[i])
                        local_pointer += in_dim

                elif fisher_shape == SHAPE_DIAG:
                    # shape check
                    assert module.fisher.diag.weight.shape == module.weight.shape
                    assert module.fisher.diag.bias.shape == module.bias.shape

                    # value check
                    torch.testing.assert_close(torch.diag(layer_fisher)[:out_dim * in_dim], module.fisher.diag.weight.flatten())
                    torch.testing.assert_close(torch.diag(layer_fisher)[out_dim * in_dim:], module.fisher.diag.bias)

                pointer += n_params

        if fisher_shape == SHAPE_FULL and fisher_type == FISHER_EXACT:
            # shape check
            n_params = sum(p.numel() for p in model.parameters())
            assert tuple(model.fisher.data.shape) == (n_params, n_params)

            # symmetry check
            torch.testing.assert_close(model.fisher.data, model.fisher.data.T)

            # value check (fisher_exact = generalized Gauss-Newton (GGN))
            logits = fisher_maker_single_data.call_model()  # 1 x c
            assert logits.shape[0] == 1
            assert logits.ndim == 2
            jacobian = logits.new_zeros(logits.shape[1], n_params)  # c x p
            for i in range(logits.shape[1]):
                model.zero_grad(set_to_none=True)
                logits[:, i].backward(retain_graph=True)
                g = parameters_to_vector([p.grad for p in model.parameters()])  # p
                jacobian[i, :] = g
            if loss_type == LOSS_CROSS_ENTROPY:
                # GGN = J^t @ H @ J
                prob = F.softmax(logits, dim=1).flatten()  # c
                hess = torch.diag(prob) - torch.outer(prob, prob)  # c x c
                full_ggn = jacobian.T @ hess @ jacobian  # p x p
            else:
                # GGN = J^t @ J
                full_ggn = jacobian.T @ jacobian  # p x p
            torch.testing.assert_close(full_fisher, full_ggn)

    def test_accumulate(self, model, fisher_maker, fisher_shape):
        self.first_call(model, fisher_maker, fisher_shape)

        # 2nd, 3rd, 4th, and 5th call (w/ accumulation)
        for num_acc in [2, 3, 4, 5]:
            fisher_maker.forward_and_backward(accumulate=True)
            self.compare_vs_first_call(model, fisher_shape, scale=num_acc)

    def test_scale(self, model, fisher_maker, fisher_shape):
        self.first_call(model, fisher_maker, fisher_shape)

        # 2nd call w/ scale
        scale = 0.1
        fisher_maker.forward_and_backward(scale=scale)
        self.compare_vs_first_call(model, fisher_shape, scale=scale)

    def test_data_size(self, model, fisher_maker, fisher_shape):
        self.first_call(model, fisher_maker, fisher_shape)

        # 2nd call w/ data_size
        data_size = 32
        fisher_maker.forward_and_backward(data_size=data_size)
        self.compare_vs_first_call(model, fisher_shape, scale=1/data_size)

    def test_inv(self, model, fisher_maker, fisher_shape):
        damping = 1
        self.first_call(model, fisher_maker, fisher_shape)
        fisher_maker.forward_and_backward(calc_inv=True, damping=damping)
        for module in model.modules():
            if isinstance(module, _target_modules):
                if fisher_shape == SHAPE_LAYER_WISE:
                    # when calc_inv=True, only inv should be stored
                    assert module.fisher.data is None
                    assert module.fisher.inv is not None
                    data = module.fisher_1st.data

                    # damping
                    diag = torch.diagonal(data)
                    diag += damping

                    # inv check
                    torch.testing.assert_close(torch.eye(data.shape[0]), data @ module.fisher.inv)

                elif fisher_shape == SHAPE_KRON:
                    # when calc_inv=True, only inv should be stored
                    assert module.fisher.kron.A is None
                    assert module.fisher.kron.A_inv is not None
                    assert module.fisher.kron.B is None
                    assert module.fisher.kron.B_inv is not None
                    A = module.fisher_1st.kron.A
                    B = module.fisher_1st.kron.B

                    # calculate damping for A and B
                    A_eig_mean = A.trace() / A.shape[0]
                    B_eig_mean = B.trace() / B.shape[0]
                    pi = torch.sqrt(A_eig_mean / B_eig_mean)
                    r = damping ** 0.5
                    damping_A = max(r * pi, 1e-7)
                    damping_B = max(r / pi, 1e-7)

                    # damping
                    diag = torch.diagonal(A)
                    diag += damping_A
                    diag = torch.diagonal(B)
                    diag += damping_B

                    # inv check
                    torch.testing.assert_close(torch.eye(A.shape[0]), A @ module.fisher.kron.A_inv)
                    torch.testing.assert_close(torch.eye(B.shape[0]), B @ module.fisher.kron.B_inv)

                elif fisher_shape == SHAPE_UNIT_WISE:
                    # when calc_inv=True, only inv should be stored
                    assert module.fisher.unit.data is None
                    assert module.fisher.unit.inv is not None
                    data = module.fisher_1st.unit.data

                    # damping
                    diag = torch.diagonal(data, dim1=1, dim2=2)
                    diag += damping

                    inv = module.fisher.unit.inv
                    for i in range(data.shape[0]):  # for each unit
                        # inv check
                        torch.testing.assert_close(torch.eye(data[i].shape[0]), data[i] @ inv[i])

                elif fisher_shape == SHAPE_DIAG:
                    # when calc_inv=True, only inv should be stored
                    assert module.fisher.diag.weight is None
                    assert module.fisher.diag.weight_inv is not None
                    assert module.fisher.diag.bias is None
                    assert module.fisher.diag.bias_inv is not None

                    # inv check for weight
                    w = module.fisher_1st.diag.weight
                    w_inv = module.fisher.diag.weight_inv
                    torch.testing.assert_close(torch.ones_like(w), (w+damping).mul(w_inv))

                    # inv check for bias
                    b = module.fisher_1st.diag.bias
                    b_inv = module.fisher.diag.bias_inv
                    torch.testing.assert_close(torch.ones_like(b), (b+damping).mul(b_inv))

        if fisher_shape == SHAPE_FULL:
            # when calc_inv=True, only inv should be stored
            assert model.fisher.data is None
            assert model.fisher.inv is not None
            data = model.fisher_1st.data

            # damping
            diag = torch.diagonal(data)
            diag += damping

            # inv check
            torch.testing.assert_close(torch.eye(data.shape[0]), data @ model.fisher.inv)

    def test_inv_data_size(self, model, fisher_maker, fisher_shape):
        damping = 10
        N = 10

        scaled_damping = damping * N * N if fisher_shape == SHAPE_KRON else damping * N
        self.first_call(model, fisher_maker, fisher_shape, calc_inv=True, damping=scaled_damping)

        # 2nd call w/ data_scale
        fisher_maker.forward_and_backward(data_size=N, calc_inv=True, damping=damping)
        self.compare_vs_first_call(model, fisher_shape, scale=N, inv=True)

    def test_fvp(self, model, fisher_maker, fisher_shape):
        if fisher_shape not in [SHAPE_FULL, SHAPE_LAYER_WISE]:
            return
        params = list(model.parameters())
        vectors = [torch.randn_like(p) for p in params]
        pv = ParamVector(params, vectors)
        fisher_maker.forward_and_backward(fvp=True, vec=pv)  # compute fvp
        fisher_maker.forward_and_backward()  # compute fisher

        pointer = 0
        full_v = pv.get_flatten_vector()

        # check if fisher.mv(v) is equal to fvp
        if fisher_shape == SHAPE_LAYER_WISE:
            for module in model.modules():
                if isinstance(module, _target_modules):
                    n_params = sum([p.numel() for p in module.parameters()])
                    layer_v = full_v[pointer:pointer+n_params]
                    torch.testing.assert_close(module.fisher.data.mv(layer_v), module.fvp.get_flatten_vector())
                    pointer += n_params

        if fisher_shape == SHAPE_FULL:
            torch.testing.assert_close(model.fisher.data.mv(full_v), model.fvp.get_flatten_vector())

    def test_fisher_eig(self, model, fisher_maker, fisher_shape):
        if fisher_shape not in [SHAPE_FULL, SHAPE_LAYER_WISE]:
            return
        top_n = 1  # top eigenvalue
        fisher_maker.forward_and_backward()  # compute fisher
        eigvals_test, _ = fisher_maker.fisher_eig(top_n=top_n, tol=1e-7)  # compute fisher eigenvalue
        assert len(eigvals_test) == top_n

        def compare_eig(matrix):
            eigvals = torch.linalg.eigvalsh(matrix)
            eigvals = torch.sort(eigvals, descending=True)[0]
            for i in range(top_n):
                eigvals_test_i = torch.tensor(eigvals_test[i], dtype=eigvals[i].dtype)
                torch.testing.assert_close(eigvals[i], eigvals_test_i)

        if fisher_shape == SHAPE_LAYER_WISE:
            blocks = []
            for module in model.modules():
                if isinstance(module, _target_modules):
                    blocks.append(module.fisher.data)
            compare_eig(torch.block_diag(*blocks))

        if fisher_shape == SHAPE_FULL:
            compare_eig(model.fisher.data)
