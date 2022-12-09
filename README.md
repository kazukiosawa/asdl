# ASDL: Automatic Second-order Differentiation Library

NOTE: this branch `dev-grad-maker` is under development and will be merged to `master` branch soon.

ASDL is an extension library of PyTorch to easily 
calculate **Second-Order Matrices**
and apply **Gradient Preconditioning** for deep neural networks.

[ASDL poster](./ASDL_HOOML2022_poster.pdf) @ [HOOML2022 workshop](https://order-up-ml.github.io/) at NeurIPS 2022

## Gradient Preconditioning by ASDL
<img src="https://user-images.githubusercontent.com/7961228/206696993-65712b6c-d2c9-412d-b3d9-42d6a8654629.png" width="400">

Example: training with gradient preconditioning by [K-FAC](https://arxiv.org/abs/1503.05671) algorithm
```diff
import torch
+from asdl.precondition import PreconditioningConfig, KfacGradientMaker

# Initialize model
model = Net()

# Initialize optimizer (SGD is recommended)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

+config = PreconditioningConfig(data_size=batch_size, damping=0.01)
+gm = KfacGradientMaker(model, config)

# Training loop
for x, t in data_loader:
-    y = model(x)
-    loss = loss_fn(y, t)
-    loss.backward()

+    dummy_y = gm.setup_model_call(model, x)
+    gm.setup_loss_call(loss_fn, dummy_y, t)
+    y, loss = gm.forward_and_backward()

```

## Installation

```shell
git clone https://github.com/kazukiosawa/asdl.git
cd asdl
pip install -e .
```
(`asdl` will be available in PyPI soon.)
