import numpy as np
import pytest
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from asdfghjkl import fisher_esd, calculate_fisher, Bias, Scale, LOSS_CROSS_ENTROPY
from asdfghjkl import SHAPE_KRON, SHAPE_UNIT_WISE, SHAPE_DIAG


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
