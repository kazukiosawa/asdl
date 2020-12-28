import time
import torch

bs = 32
n_layers = 20
d_in = 28 * 28
d_hid = 128
d_out = 10
seed = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'
loop = 100

print('bs', bs)
print('n_layers', n_layers)
print('d_in', d_in)
print('d_hid', d_hid)
print('d_out', d_out)
print('device', device)
print('-------------')

torch.random.manual_seed(seed)

modules = [torch.nn.Linear(d_in, d_hid), torch.nn.ReLU()]
for i in range(n_layers - 2):
    modules.extend([
        torch.nn.Linear(d_hid, d_hid),
        torch.nn.ReLU()
    ])
modules.append(torch.nn.Linear(d_hid, d_out, bias=False))
model = torch.nn.Sequential(*modules).to(device)
x = torch.randn(bs, d_in).to(device)
targets = torch.zeros(bs).long()

y = model(x)
loss = y.sum()


def ntk_vp(v):
    v.requires_grad_(True)
    vjp = torch.autograd.grad(y, list(model.parameters()), v, create_graph=True)
    return torch.autograd.grad(vjp, v, grad_outputs=vjp, retain_graph=True)[0]


dummy_v = torch.randn_like(y).requires_grad_(True)


def ggn_vp(v):
    for _v in v:
        _v.requires_grad_(True)
    vjp = torch.autograd.grad(y, list(model.parameters()), dummy_v, create_graph=True)
    jvp = torch.autograd.grad(vjp, dummy_v, grad_outputs=v)
    return torch.autograd.grad(y, list(model.parameters()), grad_outputs=jvp, retain_graph=True)


def hvp(v):
    grad = torch.autograd.grad(loss, list(model.parameters()), create_graph=True)
    return torch.autograd.grad(grad, list(model.parameters()), grad_outputs=v, retain_graph=True)


def timeit(func, v):
    func(v)  # dummy run
    start = time.time()
    for _ in range(loop):
        v = func(v)
    elapsed = time.time() - start
    f_name = func.__name__
    print(f'{f_name}: {elapsed / loop * 1000:.3f}ms')


v = torch.randn_like(y)
timeit(ntk_vp, v)

v = [torch.randn_like(p) for p in model.parameters()]
timeit(ggn_vp, v)

v = [torch.randn_like(p) for p in model.parameters()]
timeit(hvp, v)
