ASDL: Automatic Second-order Differentiation Library
====================================================

**ASDL** is an extension library of `PyTorch <https://pytorch.org/>`_
to easily calculate :ref:`second-order matrices <matrices>`
and apply :ref:`gradient preconditioning <grad_precond>`
for deep neural networks.


.. toctree::
   :caption: Notes
   :maxdepth: 1

   notes/overview
   notes/installation
   notes/unified_interface
   notes/define_prec_grad_maker
   notes/matrix

.. toctree::
   :caption: Package Reference
   :maxdepth: 1

   modules/grad_maker
   modules/precondition
   modules/fisher
   modules/hessian
   modules/kernel
   modules/gradient
   modules/core
   modules/utils
