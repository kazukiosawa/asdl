Quick Start
===========

``pip install asdl``

.. code-block:: python

    import asdl

    gm = asdl.GradientMaker()

    # training loop
    for x, t in data_loader:
        # standard
        # y = model(x)
        # loss = loss_fn(y, t)
        # loss.backward()

        # GradientMaker
        dummy_y = gm.setup_model_call(model, x)
        gm.setup_loss_call(loss_fn, dummy_y, t)
        y, loss = gm.forward_and_backward()