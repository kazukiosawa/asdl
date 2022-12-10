asdl.precondition
===========================

.. currentmodule:: asdl.precondition

.. _prec_config:
.. autoclass:: PreconditioningConfig
   :members:

.. _prec_grad_maker:

PreconditionedGradientMakers
----------------------------

.. autosummary::

    NaturalGradientMaker
    KfacGradientMaker
    KronBfgsGradientMaker
    PsgdGradientMaker
    KronPsgdGradientMaker
    SengGradientMaker
    ShampooGradientMaker

Every PreconditionedGradientMaker has to be initialized
with :ref:`PreconditioningConfig <prec_config>`, which defines the behavior of gradient preconditioning:

.. code-block:: python

    config = PreconditioningConfig(damping=0.01, data_size=batch_size)
    gm = XXXGradientMaker(model, config, *args, **kwargs)  # XXX: algorithm name

And every PreconditionedGradientMaker works with *the unified interface* (:ref:`GradientMaker <grad_maker>`):

.. code-block:: python

    # preconditioned gradient calculation
    dummy_y = gm.setup_model_call(model, x)
    gm.setup_loss_call(F.mse_loss, dummy_y, t)
    y, loss = gm.forward_and_backward()

.. _natural_gradient_maker:
.. autoclass:: NaturalGradientMaker
   :members:

.. _kfac_maker:
.. autoclass:: KfacGradientMaker
   :members:

.. autoclass:: KronBfgsGradientMaker
   :members:

.. autoclass:: PsgdGradientMaker
   :members:

.. _kron_psgd_maker:
.. autoclass:: KronPsgdGradientMaker
   :members:

.. autoclass:: SengGradientMaker
   :members:

.. _shampoo_maker:
.. autoclass:: ShampooGradientMaker
   :members:
