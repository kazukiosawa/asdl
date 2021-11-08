import torch
import torch.distributed as dist
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from .core import extend
from .operations import OP_BATCH_GRADS

__all__ = ['data_loader_gradient', 'batch_gradient', 'jacobian']


def data_loader_gradient(
    model,
    loss_fn,
    data_loader,
    is_distributed=False,
    all_reduce=False,
    is_master=True,
    data_average=True
):
    # NOTE: loss_fn is supposed be defined with reduction='sum'

    # accumulate gradient for data_loader
    device = next(model.parameters()).device
    total_loss = 0
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        loss = loss_fn(model(inputs), targets)
        loss.backward()
        total_loss += loss.item()

    # take average of accumulated gradient
    if data_average:
        data_size = len(data_loader.dataset)
        for param in model.parameters():
            if param.grad is not None:
                param.grad.div_(data_size)
        total_loss /= data_size

    # reduce gradient and total_loss
    if is_distributed:
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        # pack
        packed_tensor = torch.cat([parameters_to_vector(grads),
                                   torch.tensor(total_loss, device=device)])
        # reduce
        if all_reduce:
            dist.all_reduce(packed_tensor)
        else:
            dist.reduce(packed_tensor, dst=0)
        # unpack
        if is_master or all_reduce:
            total_loss = packed_tensor[-1].item()
            packed_tensor = packed_tensor[:-1]
            vector_to_parameters(
                packed_tensor.div_(dist.get_world_size()), grads
            )

        dist.barrier()

    return total_loss


def batch_gradient(model, loss_fn, inputs, targets):
    n = len(inputs)
    with extend(model, OP_BATCH_GRADS):
        model.zero_grad()
        f = model(inputs)
        loss = loss_fn(f, targets)
        loss.backward()
        batch_grad_list = _get_batch_grad_list(model)
        grads = torch.cat([g.view(n, -1) for g in batch_grad_list], dim=1)  # (n, p)
    return grads, f

    
def _get_batch_grad_list(model):
    batch_grad_list = list()
    for module in model.modules():
        if hasattr(module, 'operation'):
            res = module.operation._op_results['batch_grads']
            if 'weight' in res:
                batch_grad_list.append(res['weight'])
            if 'bias' in res:
                batch_grad_list.append(res['bias'])
            if len(set(res.keys()) - {'weight', 'bias'}) > 0:
                raise ValueError(f'Invalid parameter keys {res.keys()}')
    return batch_grad_list


def jacobian(model, x):
    f = model(x)
    assert f.ndim == 2  # (n, c)
    n, c = f.shape
    rst = []
    for i in range(c):
        with extend(model, OP_BATCH_GRADS):
            model.zero_grad()
            loss = f[:, i].sum()
            loss.backward()
        grads = [p.batch_grads for p in model.parameters() if p.requires_grad]
        grads = torch.hstack([g.view(n, -1) for g in grads])  # (n, p)
        rst.append(grads)
    return torch.stack(rst).transpose(0, 1)  # (n, c, p)
