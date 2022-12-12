# ASDL: Automatic Second-order Differentiation Library

NOTE: this branch `dev-grad-maker` is under development and will be merged to `master` branch soon.

ASDL is an extension library of PyTorch to easily perform **gradient preconditioning** using **second-order information** (e.g., Hessian, Fisher information) for deep neural networks.

<p align="center">
  <img src="https://user-images.githubusercontent.com/7961228/207084513-d696f459-1b6e-48cb-b597-00ec6c4bffe2.png" width="400">
</p>

ASDL provides various implementations and **a unified interface** (GradientMaker) for gradient preconditioning for deep neural networks. For example, to train your model with gradient preconditioning by [K-FAC](https://arxiv.org/abs/1503.05671) algorithm, you can replace a `<Standard>` gradient calculation procedure (i.e., a forward pass followed by a backward pass) with one by `<ASDL>` with KfacGradientMaker like the following:

```python
from asdl.precondition import PreconditioningConfig, KfacGradientMaker

# Initialize model
model = Net()

# Initialize optimizer (SGD is recommended)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Initialize KfacGradientMaker
config = PreconditioningConfig(data_size=batch_size, damping=0.01)
gm = KfacGradientMaker(model, config)

# Training loop
for x, t in data_loader:
  optimizer.zero_grad()
  
  # <Standard> (gradient calculation)
  # y = model(x)
  # loss = loss_fn(y, t)
  # loss.backward()

  # <ASDL> ('preconditioned' gradient calculation)
  dummy_y = gm.setup_model_call(model, x)
  gm.setup_loss_call(loss_fn, dummy_y, t)
  y, loss = gm.forward_and_backward()

  optimizer.step()
```

You can apply a different gradient preconditioning algorithm by replacing `gm` with another `XXXGradientMaker(model, config)` (*XXX*: algorithm name, e.g., ShampooGradientMaker for [Shampoo](https://arxiv.org/abs/1802.09568) algorithm) **with the same interface**. 
This enables a *flexible switching/comparison* of a range of gradient preconditioning algorithms.

## Installation

You can install the latest version of ASDL by running:
```shell
$ pip install git+https://github.com/kazukiosawa/asdl
```
Alternatively, you can install via PyPI:
```shell
$ pip install asdl
```

ASDL is tested with Python 3.7 and is compatible with PyTorch 1.13.

## Resource

- [ASDL poster](./ASDL_HOOML2022_poster.pdf) @ [HOOML2022 workshop](https://order-up-ml.github.io/) at NeurIPS 2022
