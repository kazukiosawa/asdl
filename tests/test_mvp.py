import numpy as np
import pytest
from torch import torch, nn
from asdfghjkl import stochastic_lanczos_quadrature as slq
from asdfghjkl.vector import ParamVector

@pytest.fixture
def model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return nn.Linear(10, 10, bias=False).to(device)

@pytest.fixture
def matrix(model):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    device = next(model.parameters()).device
    m = torch.randn(num_params, num_params, device=device)
    return m + m.T # make symmetric matrix

def test_slq(matrix, model):
    # vec_list in slq is needed to evaluate weight_list
    eigvals_exact = torch.linalg.eigvalsh(matrix)

    def mvp(vec):
        mv = torch.matmul(matrix, vec.get_flatten_vector())
        return ParamVector(vec.params(), mv)
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    eigvals_slq, _ = slq(mvp, model, n_v=5, num_iter=num_params)
    eigvals_slq = np.mean(eigvals_slq, axis=0)
    np.testing.assert_allclose(eigvals_exact.tolist(), eigvals_slq, rtol=1e-04)
