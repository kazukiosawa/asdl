Installation & Quick Start
==========================

``pip install asdl``

ASDL is tested with Python 3.7 and is compatible with PyTorch 1.13.


To train your model with *gradient preconditioning*
by `K-FAC <https://arxiv.org/abs/1503.05671>`_ algorithm,
you can replace a standard gradient calculation
(i.e., a forward pass followed by a backward pass)
with one by :ref:`KfacGradientMaker <kfac>` like the following:

.. code-block:: python

    import torch
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
        # <Standard>
        # y = model(x)
        # loss = loss_fn(y, t)
        # loss.backward()

        # <ASDL> (gradient preconditioning by K-FAC)
        dummy_y = gm.setup_model_call(model, x)
        gm.setup_loss_call(loss_fn, dummy_y, t)
        y, loss = gm.forward_and_backward()


See :ref:`asdl.precondition <precondition>` for the list of the available :obj:`XXXGradientMaker` classes.

.. note::
    For training without gradient preconditioning,
    you can use :obj:`gm = asdl.GradientMaker(model)` with *the same interface*
    (i.e., :obj:`<ASDL>` is equivalent to :obj:`<Standard>`).

See :ref:`Unified Interface for Gradient Preconditioning <interface>`
for detailed instructions on how to use :obj:`XXXGradientMaker` in different situations.
