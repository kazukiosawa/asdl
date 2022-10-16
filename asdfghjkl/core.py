from contextlib import contextmanager, nullcontext

import torch.cuda
import torch.nn as nn
from torch.cuda import Stream

from .utils import record_original_requires_grad
from .operations import *
from .matrices import *
from .vector import ParamVector

_supported_module_classes = (nn.Linear, nn.Conv2d, nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.Embedding, Bias, Scale)


__all__ = ['extend', 'no_centered_cov', 'save_inputs_outgrads', 'save_inputs', 'save_outgrads',
           'module_wise_assignments', 'modules_to_assign']


@contextmanager
def extend(model,
           *op_names,
           ignore_modules=None,
           map_rule=None,
           vectors: ParamVector = None,
           stream: Stream = None) -> OperationContext:
    handles = []
    cxt = OperationContext(vectors=vectors)
    stream_cxt = torch.cuda.stream(stream) if torch.cuda.is_available() and stream is not None else nullcontext()

    try:
        for module, _op_names in module_wise_assignments(model, *op_names, ignore_modules=ignore_modules, map_rule=map_rule):
            if len(_op_names) == 0:
                # no operation is assigned
                continue
            op_class = get_op_class(module)
            if op_class is None:
                continue
            cxt.register_operation(module, op_class(module, _op_names, model_for_kernel=model))
            has_fwd_op = any(op_name in FWD_OPS for op_name in _op_names)
            has_bwd_op = any(op_name in BWD_OPS for op_name in _op_names)
            has_bwd_op_with_inputs = any(op_name in BWD_OPS_WITH_INPUTS for op_name in _op_names)

            # register hooks and operations for child modules
            if has_fwd_op or has_bwd_op_with_inputs:
                def forward_hook(_module, in_data, out_data):
                    with stream_cxt:
                        cxt.call_operations_in_forward(_module, in_data[0].detach(), out_data.detach())
                handles.append(module.register_forward_hook(forward_hook))
            if has_bwd_op or has_bwd_op_with_inputs:
                def backward_hook(_module, unused, out_grads):
                    with stream_cxt:
                        try:
                            cxt.call_operations_in_backward(_module, out_grads[0].detach())
                        except NameError:
                            # context resource is already released.
                            pass
                handles.append(module.register_backward_hook(backward_hook))
        if not cxt.is_operation_registered(model):
            # register empty operation for parent model
            cxt.register_operation(model, Operation(model, []))

        yield cxt

    finally:
        # remove hooks and operations from modules
        for handle in handles:
            handle.remove()
        cxt.clear_operations()
        del cxt


def no_centered_cov(model: nn.Module, shapes, ignore_modules=None, cvp=False, vectors: ParamVector = None, stream: Stream = None, calc_inv=False) -> OperationContext:
    assert not (cvp and calc_inv), 'cvp and calc_inv cannot be True at the same time.'
    shape_to_op = {
        SHAPE_FULL: OP_BATCH_GRADS,  # full
        SHAPE_LAYER_WISE: OP_COV_INV if calc_inv else OP_COV,  # layer-wise block-diagonal
        SHAPE_KRON: OP_COV_KRON_INV if calc_inv else OP_COV_KRON,  # Kronecker-factored
        SHAPE_SWIFT_KRON: OP_COV_SWIFT_KRON_INV if calc_inv else OP_COV_SWIFT_KRON,  # swift Kronecker-factored
        SHAPE_KFE: OP_COV_KFE,  # Kronecker-factored eigenbasis
        SHAPE_UNIT_WISE: OP_COV_UNIT_WISE_INV if calc_inv else OP_COV_UNIT_WISE,  # unit-wise block-diagonal
        SHAPE_DIAG: OP_COV_DIAG_INV if calc_inv else OP_COV_DIAG,  # diagonal
    }
    if cvp:
        shape_to_op[SHAPE_LAYER_WISE] = OP_CVP
    return extend(model, *shapes, ignore_modules=ignore_modules, map_rule=lambda s: shape_to_op[s], vectors=vectors, stream=stream)


def save_inputs_outgrads(model: nn.Module, targets=None, ignore_modules=None) -> OperationContext:
    if targets is not None:
        assign_rules = [(t, OP_SAVE_INPUTS, OP_SAVE_OUTGRADS) for t in targets]
    else:
        assign_rules = [OP_SAVE_INPUTS, OP_SAVE_OUTGRADS]
    return extend(model, *assign_rules, ignore_modules=ignore_modules)


def save_inputs(model: nn.Module, targets=None, ignore_modules=None) -> OperationContext:
    if targets is not None:
        assign_rules = [(t, OP_SAVE_INPUTS) for t in targets]
    else:
        assign_rules = [OP_SAVE_INPUTS]
    return extend(model, *assign_rules, ignore_modules=ignore_modules)


def save_outgrads(model: nn.Module, targets=None, ignore_modules=None) -> OperationContext:
    if targets is not None:
        assign_rules = [(t, OP_SAVE_OUTGRADS) for t in targets]
    else:
        assign_rules = [OP_SAVE_OUTGRADS]
    return extend(model, *assign_rules, ignore_modules=ignore_modules)


def supported_modules(model):
    for module in model.modules():
        if isinstance(module, _supported_module_classes):
            yield module


def named_supported_modules(model):
    for name, module in model.named_modules():
        if isinstance(module, _supported_module_classes):
            yield name, module


def module_wise_assignments(model, *assign_rules, ignore_modules=None, map_rule=None, named=False):
    """
    Assign certain values to each module based on assign_rules.

    Args:
        model: base module in which the underlying modules will be assigned values
        assign_rules: assignment rules
            - Corresponding module(s) to each rule will be assigned certain values:
            - Each rule has to be one of the following format:
                1. Tuple(key, value1, value2, ...)
                    1-1. Tuple(<an instance of torch.nn.Module>, str, str, ...)
                        - for the module that is equivalent to the key
                    1-2. Tuple(str, str, str, ...)
                        - for modules that contain the key in its name
                    1-3. Tuple(<a subclass of torch.nn.Module>, str, str, ...)
                        - for modules that are instances of the key
                2. str (represents a value)
                    - for modules that havn't been assigned any value
            - Tuple rules (format 1) cannot have the same key to each others.
            - All str rules (format 2) are considered together as one Tuple rule.
            - If more than one Tuple rules are applicable to a module,
                - the rules are prioritized by their formats in the above order
                - only the first rule is applied
            - Each assigned value is mapped to another by map_rule (if specified).
        map_rule: callable str -> str
        named: if True, yields module name along with module and assigned values

    Example:
    >>> model = nn.Sequential()
    >>> model.add_module('conv1', nn.Conv2d(1, 1, (1, 1)))
    >>> model.add_module('conv2', nn.Conv2d(1, 1, (1, 1)))
    >>> model.add_module('fc1', nn.Linear(1, 1))
    >>> model.add_module('fc2', nn.Linear(1, 1))
    >>> model.add_module('bn1', nn.BatchNorm2d(1, 1))
    >>> model.add_module('bn2', nn.BatchNorm2d(1, 1))
    >>>
    >>> def map_rule(x: str):
    >>>     return x.replace('value', 'mapped')
    >>>
    >>> asgmts = module_wise_assignments(model,
    >>>                                  'value1',
    >>>                                  ('conv', 'value2'),
    >>>                                  (nn.BatchNorm2d, 'value3'),
    >>>                                  (model.bn1, 'value4'),
    >>>                                  'value5',
    >>>                                  map_rule=map_rule,
    >>>                                  named=True):
    >>> for name, module, values in asgmts:
    >>>     print(name, values)

    Outputs:
        conv1 ['mapped2']
        conv2 ['mapped2']
        fc1 ['mapped1', 'mapped5']
        fc2 ['mapped1', 'mapped5']
        bn1 ['mapped4']
        bn2 ['mapped3']
    """
    assert all(isinstance(rule, (str, tuple)) for rule in assign_rules), \
        f'every assign rule has to be {str} or {tuple}.'

    if ignore_modules is None:
        ignore_modules = []

    if map_rule is None:
        def identical(x): return x
        map_rule = identical

    common_asgmts = []
    specified_asgmts = {}
    for rule in assign_rules:
        if isinstance(rule, str):
            value = rule
            common_asgmts.append(map_rule(value))
        else:
            assert len(rule) >= 2, f'Tuple length has to be >= 2. Given: {rule}.'
            key, values = rule[0], rule[1:]
            assert all(isinstance(value, str) for value in values), \
                f'All values have to be {str}. Given: {values}.'
            assert key not in specified_asgmts, \
                f'({key}, _) is already assigned.'
            specified_asgmts[key] = [map_rule(value) for value in values]

    for name, module in named_supported_modules(model):
        if module in ignore_modules:
            continue
        if any(isinstance(module, cls) for cls in ignore_modules if isinstance(cls, type)):
            continue
        if any(keyword in name for keyword in ignore_modules if isinstance(keyword, str)):
            continue
        module_info = (name, module) if named else (module,)

        requires_grad = False
        for attr in ['weight', 'bias']:
            param = getattr(module, attr, None)
            if param is not None:
                requires_grad = requires_grad or param.requires_grad
                record_original_requires_grad(param)
        if not requires_grad:
            # no assignment for a module that do not have params that require grad
            continue

        if module in specified_asgmts:
            yield *module_info, specified_asgmts[module]
        elif any(isinstance(key, str) and key in name for key in specified_asgmts):
            key = next(key for key in specified_asgmts if isinstance(key, str) and key in name)
            yield *module_info, specified_asgmts[key]
        elif module.__class__ in specified_asgmts:
            yield *module_info, specified_asgmts[module.__class__]
        else:
            if len(common_asgmts) == 0:
                continue
            yield *module_info, common_asgmts.copy()


def modules_to_assign(model, value, *assign_rules, ignore_modules=None, named=False):
    for assign_info in module_wise_assignments(model, *assign_rules, ignore_modules=ignore_modules, named=named):
        values = assign_info[-1]
        if value in values:
            if named:
                name, module = assign_info[:2]
                yield name, module
            else:
                module = assign_info[0]
                yield module


def named_modules_to_assign(value, *assign_rules):
    return modules_to_assign(value, *assign_rules, named=True)
