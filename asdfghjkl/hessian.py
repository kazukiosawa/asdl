import torch
from .symmatrix import SymMatrix, Diag
from .matrices import SHAPE_FULL, SHAPE_LAYER_WISE, SHAPE_DIAG, HESSIAN, MatrixManager
from .mvp import power_method, conjugate_gradient_method, quadratic_form
from .vector import ParamVector, reduce_vectors

__all__ = [
    'hessian',
    'hvp',
    'hessian_eig',
    'hessian_free',
    'hessian_quadratic_form'
]
_supported_shapes = [SHAPE_FULL, SHAPE_LAYER_WISE, SHAPE_DIAG]


class Hessian(MatrixManager):
    """
    This class manages the calculation of hessian matrix.
    Args:
        model: NN model to calculate Hessian for.
    """
    def __init__(self, model):
        super().__init__(model, HESSIAN)

    def calculate_hessian(self,
                          loss_fn,
                          hessian_shapes,
                          inputs=None,
                          targets=None,
                          data_loader=None,
                          data_average=False):
        model = self._model
        device = next(model.parameters()).device
        if data_loader is not None:
            scale = 1 / len(data_loader.dataset) if data_average else 1
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                _hessian_for_loss(model, loss_fn, hessian_shapes, inputs, targets)
                self.accumulate_matrices(scale=scale)
        else:
            assert inputs is not None and targets is not None
            inputs, targets = inputs.to(device), targets.to(device)
            _hessian_for_loss(model, loss_fn, hessian_shapes, inputs, targets)
            scale = 1 / inputs.shape[0] if data_average else 1
            self.accumulate_matrices(scale=scale)


def hessian(model,
            loss_fn,
            hessian_shapes,
            inputs=None,
            targets=None,
            data_loader=None,
            is_distributed=False,
            all_reduce=False,
            is_master=True,
            data_average=False):
    if isinstance(hessian_shapes, str):
        hessian_shapes = [hessian_shapes]
    # remove duplicates
    hessian_shapes = set(hessian_shapes)
    for hshape in hessian_shapes:
        assert hshape in _supported_shapes, f'Invalid hessian_shape: {hshape}. hessian_shape must be in {_supported_shapes}.'

    h = Hessian(model)
    h.calculate_hessian(loss_fn,
                        hessian_shapes,
                        inputs=inputs,
                        targets=targets,
                        data_loader=data_loader,
                        data_average=data_average)

    if is_distributed:
        h.reduce_matrices(is_master=is_master, all_reduce=all_reduce)


def hvp(model,
        loss_fn,
        vec: ParamVector,
        inputs=None,
        targets=None,
        data_loader=None,
        is_distributed=False,
        all_reduce=False,
        is_master=True,
        data_average=False):
    """
    Calculates Hessian Vector Product.
    Args:
        model: NN model of interest.
        loss_fn: Loss function. Reduction method is expected to be summation.
        vec: ParamVector instance used to compute HVP.
        inputs, targets: Single batch of inputs and targets to feed in to the model.
        data_loader: torch.utils.data.DataLoader instance to feed in to the model.
                     Either inputs and targets or data_loader must be specified.
        is_distributed: If True, distributed computation is supported.
        all_reduce: If this and is_distributed is True, all-reduces the computed HVP across
                    all processes. Otherwise, the computed HVP will be reduced only to master process.
        is_master: This flag indicates whether the current process is the master.
        data_average: If True, HVP is averaged over all data. Otherwise, HVP is summed over all data.
    Returns:
        HVP and gradient of the loss with respect to the model parameters.
    Example::
        >>> model = torch.nn.Linear(100, 10)
        >>> x = torch.randn(32, 100)
        >>> y = torch.tensor([0]*32, dtype=torch.long)
        >>> loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        >>> vec = {p: torch.randn_like(p) for p in model.parameters()}
        >>> vec = asdl.ParamVector(vec.keys(), vec.values())
        >>> hvp, grad = asdl.hvp(model, loss_fn, vec, x, y)
        >>> hvp.get_flatten_vector()
        tensor([ 18.2251, -13.8554,   7.7957,  ...,  -7.8146,  -1.7201,   2.8145])
    """
    device = next(model.parameters()).device
    if data_loader is not None:
        scale = 1 / len(data_loader.dataset) if data_average else 1
        hvp_all = None
        g_all = None
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            hvp_new, g_new = _hvp(model, loss_fn, inputs, targets, vec)
            if hvp_all is None:
                hvp_all = hvp_new
                g_all = g_new
            else:
                hvp_all += hvp_new
                g_all += g_new
    else:
        assert inputs is not None and targets is not None
        scale = 1 / inputs.shape[0] if data_average else 1
        inputs, targets = inputs.to(device), targets.to(device)
        hvp_all, g_all = _hvp(model, loss_fn, inputs, targets, vec)

    if is_distributed:
        hvp_all = reduce_vectors(hvp_all, is_master, all_reduce)
        g_all = reduce_vectors(g_all, is_master, all_reduce)

    return hvp_all.mul_(scale), g_all.mul_(scale)


def _hvp(model, loss_fn, inputs, targets, vec: ParamVector):
    loss = loss_fn(model(inputs), targets)
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, inputs=params, create_graph=True)
    v = torch.autograd.grad(grads, inputs=params, grad_outputs=tuple(vec.values()))
    return ParamVector(params, v), ParamVector(params, grads)


def hessian_eig(
    model,
    loss_fn,
    inputs=None,
    targets=None,
    data_loader=None,
    top_n=1,
    max_iters=100,
    tol=1e-3,
    is_distributed=False,
    print_progress=False,
    data_average=False
):
    """
    Calculates top_n greatest (in absolute value) eigenvalues and their eigenvectors using power method.
    Eigenvalues are sorted in descending order and eigenvectors are sorted correspondingly.
    Args:
        model: NN model of interest.
        loss_fn: Loss function. Reduction method is expected to be summation.
        inputs, targets: Single batch of inputs and targets to feed in to the model.
        data_loader: torch.utils.data.DataLoader instance to feed in to the model.
                     Either inputs and targets or data_loader must be specified.
        top_n: The number of greatest eigenvalues to be computed and returned.
        max_iters: The maximum number of iterations for power method.
        tol: The tolerance value for power method to check convergence.
        is_distributed: If True, distributed computation is supported.
        print_progress: If True, progress is printed out during power method.
        data_average: If True, the average over all data is taken for Hessian, otherwise summation over all data
                      is taken.
    Returns:
        The top_n greatest (in absolute value) eigenvalues and their corresponding eigenvectors. Eigenvalues is
        sorted in descending oreder and eigenvectors is sorted correspondingly.
    Example::
        >>> model = torch.nn.Linear(100, 10)
        >>> x = torch.randn(32, 100)
        >>> y = torch.tensor([0]*32, dtype=torch.long)
        >>> loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        >>> eigvals, eigvecs = hessian_eig(model, loss_fn, x, y, top_n=5)
        >>> eigvals
        [42.4885139465332, 41.401275634765625, 33.121273040771484, 32.43955993652344, 31.223182678222656]
    """
    def hvp_fn(vec: ParamVector) -> ParamVector:
        return hvp(model,
                   loss_fn,
                   vec,
                   inputs=inputs,
                   targets=targets,
                   data_loader=data_loader,
                   data_average=data_average,
                   is_distributed=is_distributed)[0]

    eigvals, eigvecs = power_method(hvp_fn,
                                    model,
                                    top_n=top_n,
                                    max_iters=max_iters,
                                    tol=tol,
                                    is_distributed=is_distributed,
                                    print_progress=print_progress)

    return eigvals, eigvecs


def hessian_free(
        model,
        loss_fn,
        b,
        data_loader=None,
        inputs=None,
        targets=None,
        init_x=None,
        damping=1e-3,
        max_iters=None,
        tol=1e-8,
        is_distributed=False,
        print_progress=False,
        data_average=False
):
    """
    Solves (H + d * I)x = b with respect to x using conjugate gradient method.
    Args:
        model: NN model of interest.
        loss_fn: Loss function. Reduction method is expected to be summation.
        b: ParamVector instance representing the right-hand side of the equation to be solved.
        data_loader: torch.utils.data.DataLoader instance to feed in to the model.
        inputs, targets: Single batch of inputs and targets to feed in to the model.
                         Either inputs and targets or data_loader must be specified.
        init_x: ParamVector instance representing the initial value for x in the equation. If None,
                x will be initialized as a zero vector.
        damping: The damping value for d in the equation to be solved.
        max_iters: The maximum number of iterations.
        tol: The tolerance value for conjugate gradient method to check convergence.
        is_distributed: If True, distributed computation is supported.
        print_progress: If True, progress is printed out during conjugate gradient method.
        data_average: If True, the average over all data is taken for Hessian, otherwise summation over all data
                      is taken.
    Returns:
        ParamVector instance which is the solution of the equation (F + d * I)x = b obtained by conjugate
        gradient method.
    Example::
        >>> model = torch.nn.Linear(100, 10)
        >>> inputs = torch.randn(32, 100)
        >>> targets = torch.tensor([0]*32, dtype=torch.long)
        >>> loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        >>> b = {p: torch.randn_like(p) for p in model.parameters()}
        >>> b = ParamVector(b.keys(), b.values())
        >>> x = hessian_free(model, loss_fn, b, inputs=inputs, targets=targets)
        >>> x.get_flatten_vector()
        tensor([  43.3498,  161.1647, -113.9113,  ..., -138.1220,  710.3869,
                -392.6661])
    """
    def hvp_fn(vec: ParamVector) -> ParamVector:
        return hvp(model,
                   loss_fn,
                   vec,
                   inputs=inputs,
                   targets=targets,
                   data_loader=data_loader,
                   data_average=data_average,
                   is_distributed=is_distributed)[0]

    return conjugate_gradient_method(hvp_fn,
                                     b,
                                     init_x=init_x,
                                     damping=damping,
                                     max_iters=max_iters,
                                     tol=tol,
                                     print_progress=print_progress)


def hessian_quadratic_form(
        model,
        loss_fn,
        v=None,
        data_loader=None,
        inputs=None,
        targets=None,
        is_distributed=False,
        data_average=True,
        damping=None,
):
    """
    Calculates quadratic form, v.T * (H + d * I) * v of Hessian Matrix H given a vector v and a damping value d.
    Also calculates dot product of v and the gradient of loss with respect to model parameters.
    Args:
        model: NN model of interest.
        loss_fn: Loss function. Reduction method is expected to be summation.
        v: ParamVector instance for computing the quadratic form. If None, gradients of model parameters is used,
           which means the model parameter's gradient must be calculated beforehand.
        data_loader: torch.utils.data.DataLoader instance to feed in to the model.
        inputs, targets: Single batch of inputs and targets to feed in to the model.
                         Either inputs and targets or data_loader must be specified.
        is_distributed: If True, distributed computation is supported.
        data_average: If True, the average over all data is taken for Hessian, otherwise summation over all data
                      is taken.
        damping: The damping value.
    Returns:
        The dot product of v and the gradient of loss with respect to model parameters, and quadratic form of
        Hessian given a vector.
    Example::
        >>> model = torch.nn.Linear(100, 10)
        >>> inputs = torch.randn(32, 100)
        >>> targets = torch.tensor([0]*32, dtype=torch.long)
        >>> loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        >>> v = {p: torch.randn_like(p) for p in model.parameters()}
        >>> v = ParamVector(v.keys(), v.values())
        >>> hessian_quadratic_form(model, loss_fn, v=v, inputs=inputs, targets=targets)
        (tensor(1.6246, grad_fn=<SumBackward0>), tensor(99.0703))
    """
    if v is None:
        grads = {p: p.grad for p in model.parameters() if p.requires_grad}
        v = ParamVector(grads.keys(), grads.values())

    hv, g = hvp(model,
                loss_fn,
                v,
                inputs=inputs,
                targets=targets,
                data_loader=data_loader,
                data_average=data_average,
                all_reduce=True,
                is_distributed=is_distributed)

    if damping:
        hv.add_(v, alpha=damping)

    return v.dot(g), v.dot(hv)


def _hessian_for_loss(model, loss_fn, hessian_shapes, inputs, targets):
    model.zero_grad()
    loss = loss_fn(model(inputs), targets)
    device = next(model.parameters()).device
    params = [p for p in model.parameters() if p.requires_grad]

    # full
    if SHAPE_FULL in hessian_shapes:
        full_hess = _hessian(loss, params)
        setattr(model, 'hessian', SymMatrix(data=full_hess))
    else:
        full_hess = None

    if SHAPE_LAYER_WISE not in hessian_shapes \
            and SHAPE_DIAG not in hessian_shapes:
        return

    idx = 0
    for module in model.modules():
        w = getattr(module, 'weight', None)
        b = getattr(module, 'bias', None)
        params = [p for p in [w, b] if p is not None and p.requires_grad]
        if len(params) == 0:
            continue

        # module hessian
        if full_hess is None:
            m_hess = _hessian(loss, params)
        else:
            m_numel = sum([p.numel() for p in params])
            m_hess = full_hess[idx:idx + m_numel, idx:idx + m_numel]
            idx += m_numel

        # block-diagonal
        if SHAPE_LAYER_WISE in hessian_shapes:
            setattr(module, 'hessian', SymMatrix(data=m_hess))

        # diagonal
        if SHAPE_DIAG in hessian_shapes:
            m_hess = torch.diag(m_hess)
            _idx = 0
            w_hess = b_hess = None
            if w is not None and w.requires_grad:
                w_numel = w.numel()
                w_hess = m_hess[_idx:_idx + w_numel].view_as(w)
                _idx += w_numel
            if b is not None and b.requires_grad:
                b_numel = b.numel()
                b_hess = m_hess[_idx:_idx + b_numel].view_as(b)
                _idx += b_numel
            diag = Diag(weight=w_hess, bias=b_hess)
            if hasattr(module, 'hessian'):
                module._hessian.diag = diag
            else:
                setattr(module, 'hessian', SymMatrix(diag=diag))


# adopted from https://github.com/mariogeiger/hessian/blob/master/hessian/hessian.py
def _hessian(output, inputs, out=None, allow_unused=False, create_graph=False):
    '''
    Compute the Hessian of `output` with respect to `inputs`
    hessian((x * y).sum(), [x, y])
    '''
    assert output.ndimension() == 0

    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)

    n = sum(p.numel() for p in inputs)
    if out is None:
        out = output.new_zeros(n, n)

    ai = 0
    for i, inp in enumerate(inputs):
        [grad] = torch.autograd.grad(
            output, inp, create_graph=True, allow_unused=allow_unused
        )
        grad = torch.zeros_like(inp) if grad is None else grad
        grad = grad.contiguous().view(-1)

        for j in range(inp.numel()):
            if grad[j].requires_grad:
                row = _gradient(
                    grad[j],
                    inputs[i:],
                    retain_graph=True,
                    create_graph=create_graph
                )[j:]
            else:
                row = grad[j].new_zeros(sum(x.numel() for x in inputs[i:]) - j)

            out[ai, ai:].add_(row.type_as(out))  # ai's row
            if ai + 1 < n:
                out[ai + 1:, ai].add_(row[1:].type_as(out))  # ai's column
            del row
            ai += 1
        del grad

    return out


# adopted from https://github.com/mariogeiger/hessian/blob/master/hessian/gradient.py
def _gradient(
    outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False
):
    '''
    Compute the gradient of `outputs` with respect to `inputs`
    gradient(x.sum(), x)
    gradient((x * y).sum(), [x, y])
    '''
    if torch.is_tensor(inputs):
        inputs = [inputs]
    else:
        inputs = list(inputs)
    grads = torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs,
        allow_unused=True,
        retain_graph=retain_graph,
        create_graph=create_graph
    )
    grads = [
        x if x is not None else torch.zeros_like(y) for x,
        y in zip(grads, inputs)
    ]
    return torch.cat([x.contiguous().view(-1) for x in grads])
