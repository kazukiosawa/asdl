Unified Interface for Gradient Preconditioning
==============================================

.. _interface:

Overview
--------

The unified interface (:ref:`GradientMaker <grad_maker>`) in ASDL enables
an easy integration of gradient preconditioning by hiding
*algorithm-specific* and *complex* operations.
The behavior of the gradient preconditioning is defined by
:ref:`PreconditionedGradientMakers <prec_grad_maker>` and
is configured by :ref:`PreconditioningConfig <prec_config>` object.
To perform the (preconditioned) gradient calculation in a unified way,
:ref:`GradientMaker <grad_maker>` and :ref:`PreconditionedGradientMakers <prec_grad_maker>`
have the following *common* APIs:

.. currentmodule:: asdl.GradientMaker

.. autosummary::

    setup_model_call
    setup_loss_call
    setup_loss_repr
    forward_and_backward

The idea behind the design of these APIs is to do only
"setup" outside and hide the *evaluation* inside
:obj:`forward_and_backward()`, since the proper timing/context
of the model and loss evaluations depends on the gradient
preconditioning algorithm.

However, defining an interface in this way that is compatible
with a wide  range of training pipelines is not simple.
This is because:

#. the format of the output of the :obj:`model_fn` depends on the training pipeline,
#. it is even possible that :obj:`model_fn` includes both the model and loss evaluations, and
#. the :obj:`loss_fn` usually takes (a part of) *evaluated value* of :obj:`model_fn` (or the result of manipulating it) as an argument, which we have to tell :obj:`forward_and_backward` before calling it, i.e., *before evaluating* :obj:`model_fn`.

To address these challenges, :ref:`DummyObject <dummy_obj>` plays
a key role in the APIs.

Examples
--------

Below are some common training pipeline examples in PyTorch
to demonstrate the versatility of the interface.
For each case, we assume that the :obj:`model` is defined as
a simple linear MLP with a certain output format:

.. code-block::

    @dataclass
    class Output:
        loss: torch.Tensor
        logits: torch.Tensor
        hid_state: torch.Tensor

    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(5, 4)
            self.fc2 = torch.nn.Linear(4, 3)

        def forward(self, x: torch.Tensor, t: torch.Tensor = None):
            h = self.fc1(x)
            logits = self.fc2(h)

            if t is None:
                # Case 1 & 5
                return logits

            loss = loss_fn(logits, t)

            # Case 2
            return logits, loss
            # Case 3
            return dict(loss=loss, logits=logits, hid_state=h)
            # Case 4
            return Output(loss=loss, logits=logits, hid_state=h)


    model = Network()


Case1: *torch.Tensor* output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

    # <Standard>
    y = model(x)
    loss = loss_fn(y, t)
    loss.backward()

    # <ASDL>
    dummy_y = gm.setup_model_call(model, x)
    gm.setup_loss_call(loss_fn, dummy_y, t)
    y, loss = gm.forward_and_backward()

The first case is probably the most typical one.
The :obj:`model` receives an input :obj:`x` (*torch.Tensor*),
which represents a batch of input examples (e.g., images)
and returns the :obj:`y = logits` (*torch.Tensor*),
which represents a batch of logits.
The output :obj:`y` and the target :obj:`t` (*torch.Tensor*) are passed
to :obj:`loss_fn` to evaluate the loss value.
Finally, the mini-batch gradient is calculated by performing :obj:`loss.backward()`.

In ASDL, the same procedures can be written with a similar logical structure.
:obj:`setup_model_call()` returns a *DummyObject* (:obj:`dummy_y`).
:obj:`dummy_y` can be directly passed to :obj:`setup_loss_call()`
in the same way that :obj:`y` is passed to :obj:`loss_fn()`.
When :obj:`forward_and_backward` is called, :obj:`dummy_y` is
replaced with the evaluated value and is passed to :obj:`loss_fn()`,
which is registered by `setup_loss_call()`.

Case2: Sequence (e.g., tuple, list) output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

    # <Standard>
    y = model(x, t)
    _, loss = y
    loss.backward()

    # <ASDL>
    dummy_y = gm.setup_model_call(model, x, t)
    gm.setup_loss_repr(loss_fn, dummy_y[1])
    y, loss = gm.forward_and_backward()

Next, we consider the case where the loss evaluation
is included in the :obj:`model` and it returns a tuple :obj:`(logits, loss)`.
Note that both input :obj:`x` and target :obj:`t` are passed
to the :obj:`model` this time.
In this case, instead of :obj:`setup_loss_call`,
we call :obj:`setup_loss_repr` to let the *GradientMaker* know how the :obj:`loss` value should be evaluated.
:obj:`dummy_y` behaves as if it were the actual value (tuple)
and we know that the :obj:`loss` value would be stored in the second element of the tuple,
so we can specify :obj:`dummy_y[1]` as the argument of :obj:`setup_loss_repr`.


Case 3: Mapping (e.g., dict) output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

    # <Standard>
    y = model(x, t)
    loss = y["loss"]
    loss.backward()

    # <ASDL>
    dummy_y = gm.setup_model_call(model, x, t)
    gm.setup_loss_repr(dummy_y["loss"])
    y, loss = gm.forward_and_backward()


Similarly, the case where output :obj:`y` is
a dictionary (or an arbitrary mapping object)
is also supported in ASDL.


Case 4: *dataclass* output
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # <Standard>
    y = model(x, t)
    loss = y.loss
    loss.backward()

    # <ASDL>
    dummy_y = gm.setup_model_call(model, x, t)
    gm.setup_loss_repr(dummy_y.loss)
    y, loss = gm.forward_and_backward()

It is also common for the output :obj:`y` to be
an object of the Python `dataclass <https://docs.python.org/3/library/ dataclasses.html>`_
(or a some class for stroing data).
This case can be seen, for example, `Hugging Face's Transformers <https://huggingface.co/docs/transformers/index>`_.
We can *pseudo-access* the :obj:`loss` attribute (or an arbitrary attribute)
through :obj:`dummy_y.loss` (or `dummy_hy.attr_name`).


Case 5: Complex operations on output
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # <Standard>
    y = model(x, t)
    loss = F.cross_entropy(
            y.view(-1, y.size(-1)),
            t.view(-1),
            ignore_index=-1)
    loss.backward()

    # <ASDL>
    dummy_y = gm.setup_model_call(model, x)
    gm.setup_loss_call(F.cross_entropy,
                       dummy_y.view(-1, dummy_y.size(-1)),
                       t.view(-1),
                       ignore_index=-1)
    y, loss = gm.forward_and_backward()

Finally, we consider the case in a language modeling task,
where the input :obj:`x` is a *torch.Tensor* of shape
:obj:`(batch_size, sequence_length, embedding_dimension)`
while the target :obj:`t` is a *torch.Tensor* of shape
:obj:`(batch_size, sequence_length)` containing word ids in the vocabulary.
Here, the output :obj:`y` of the :obj:`model` have the shape
:obj:`(batch_size, sequence_length, embedding_dimension)`,
and we wish to flatten :obj:`y` along the :obj:`batch_size` and :obj:`sequence_length` dimensions
before evaluating the cross-entropy loss (:obj:`F.cross_entropy`)
by :obj:`y.view(-1, y.size(-1))`.

In ASDL, these operations can be expressed in the same way, i.e.,
:obj:`dummy_y.view(-1, dummy_y.size(-1))`.
It is possible to not only *pseudo-access* the attribute of
:obj:`dummy_y` (e.g., :obj:`.view`), but also to *pseudo-call* it
(e.g., :obj:`.view()`).
Furthermore, we can pass :obj:`dummy_y` itself or the result of the
pseudo-call to the pseudo-call.
Note once again that :obj:`dummy_y` does not contain the actual evaluation value at this point.
How can the *GradientMaker* know the actual size of :obj:`y` before evaluating it?
When :obj:`forward_and_backward` is called,
the *GradientMaker* evaluates the sequence of the operations on the :ref:`DummyObject <dummy_obj>` (if any)
*recursively*.
This enables as complex operations on the output as this example.