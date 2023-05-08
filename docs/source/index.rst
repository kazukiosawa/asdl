ASDL Documentation
==================

**ASDL** (*Automatic Second-order Differentiation Library*) is an extension library of `PyTorch <https://pytorch.org/>`_
to easily perform **gradient preconditioning**
using **second-order information** (e.g., Hessian, Fisher information) for deep neural networks.


.. image:: ./_figures/precond_eq.png
  :align: center
  :width: 400px

ASDL provides various implementations
and **a unified interface** (*GradientMaker*) for gradient preconditioning for deep neural networks.
For example, to train your model with gradient preconditioning
by `K-FAC <https://arxiv.org/abs/1503.05671>`_ algorithm,
you can replace a `<Standard>` gradient calculation procedure
(i.e., a forward pass followed by a backward pass)
with one by `<ASDL>` with :ref:`KfacGradientMaker <kfac_maker>` like the following:

.. code-block:: python

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
        # loss = F.mse_loss(y, t)
        # loss.backward()

        # <ASDL> ('preconditioned' gradient calculation)
        dummy_y = gm.setup_model_call(model, x)
        gm.setup_loss_call(F.mse_loss, dummy_y, t)
        y, loss = gm.forward_and_backward()

        optimizer.step()

You can apply a different gradient preconditioning algorithm by
replacing :obj:`gm` with another :obj:`XXXGradientMaker(model, config)`
(*XXX*: algorithm name, e.g., :ref:`ShampooGradientMaker <shampoo_maker>`
for `Shampoo <https://arxiv.org/abs/1802.09568>`_ algorithm)
**with the same interface**.
This enables a *flexible switching/comparison* of a range of gradient preconditioning algorithms.

See :ref:`PreconditionedGradientMakers <prec_grad_maker>` for a list of the supported gradient preconditioning algorithms
(`XXXGradientMakers`) in ASDL.

.. note::
    For training without gradient preconditioning,
    you can use :obj:`gm = asdl.GradientMaker(model)` with *the same interface*
    (i.e., no need to switch the script from `<ASDL>` to `<Standard>`).

See :ref:`Unified Interface for Gradient Preconditioning <interface>`
for detailed instructions on how to use `XXXGradientMakers` in different situations.

Installation
------------

You can install the latest version of ASDL by running:

.. code-block:: shell

    $ pip install git+https://github.com/kazukiosawa/asdl

Alternatively, you can install via PyPI:

.. code-block:: shell

    $ pip install asdl


ASDL is tested with Python 3.7 and is compatible with PyTorch 1.13.


.. toctree::
   :caption: Notes
   :maxdepth: 1

   notes/unified_interface

.. toctree::
   :caption: API Reference
   :maxdepth: 1

   modules/grad_maker
   modules/precondition
