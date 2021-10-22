from typing import List
from contextlib import contextmanager

import torch.nn as nn
from .utils import im2col_2d, record_original_requires_grad
from .operations import get_op_class

_supported_module_classes = (nn.Linear, nn.Conv2d, nn.BatchNorm1d, nn.BatchNorm2d)


@contextmanager
def extend(model, *op_names, map_rule=None):
    handles = []

    try:
        def forward_hook(module, in_data, out_data):
            in_data = in_data[0].clone().detach()
            in_data = _preprocess_in_data(module, in_data, out_data)
            _call_operations_in_forward(module, in_data)

            def backward_hook(out_grads):
                out_grads = out_grads.clone().detach()
                out_grads = _preprocess_out_grads(module, out_grads)
                _call_operations_in_backward(module, in_data, out_grads)

            if out_data.requires_grad:
                handles.append(out_data.register_hook(backward_hook))

        for module, op_names in module_wise_assignments(model, *op_names, map_rule=map_rule):
            requires_grad = False
            for attr in ['weight', 'bias']:
                param = getattr(module, attr, None)
                if param is not None:
                    requires_grad = requires_grad or param.requires_grad
                    record_original_requires_grad(param)
            if not requires_grad:
                continue
            # register hooks and operations in modules
            handles.append(module.register_forward_hook(forward_hook))
            _register_operations(model, module, op_names)
        yield
    finally:
        # remove hooks and operations from modules
        for handle in handles:
            handle.remove()
        for module in supported_modules(model):
            _remove_operations(module)


def supported_modules(model):
    for module in model.modules():
        if isinstance(module, _supported_module_classes):
            yield module


def named_supported_modules(model):
    for name, module in model.named_modules():
        if isinstance(module, _supported_module_classes):
            yield name, module


def module_wise_assignments(model, *assign_rules, map_rule=None, named=False):
    """
    Assign certain values to each module based on assign_rules.

    Args:
        model: base module in which the underlying modules will be assigned values
        assign_rules: assignment rules
            - Corresponding module(s) to each rule will be assigned certain values:
            - Each rule has to be one of the following format:
                1. Tuple(key, value1, value2, ...)
                    1-1. Tuple(<an instance of torch.nn.Module>, str, str, ...)
                        - for the module which is equivalent to the key
                    1-2. Tuple(str, str, str, ...)
                        - for a module(s) which contains the key in its name
                    1-3. Tuple(<a subclass of torch.nn.Module>, str, str, ...)
                        - for a module(s) which is an instance of the key
                2. str (represents a value)
                    - for a module(s) which hasn't been assigned any value
            - Tuple rules cannot have the same key to each others.
            - If more than one rules are applicable to a module,
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
        module_info = (name, module) if named else (module,)
        if module in specified_asgmts:
            yield *module_info, specified_asgmts[module]
        elif any(isinstance(key, str) and key in name for key in specified_asgmts):
            key = next(isinstance(key, str) and key in name for key in specified_asgmts)
            yield *module_info, specified_asgmts[key]
        elif module.__class__ in specified_asgmts:
            yield *module_info, specified_asgmts[module.__class__]
        else:
            yield *module_info, common_asgmts


def modules_to_assign(model, value, *assign_rules, named=False):
    for assign_info in module_wise_assignments(model, *assign_rules, named=named):
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


def _preprocess_in_data(module, in_data, out_data):
    if isinstance(module, nn.Conv2d):
        in_data = im2col_2d(in_data, module)

    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        bnorm = module
        f = bnorm.num_features
        if isinstance(module, nn.BatchNorm1d):
            shape = (1, f)
        elif isinstance(module, nn.BatchNorm2d):
            shape = (1, f, 1, 1)
        else:
            shape = (1, f, 1, 1, 1)
        # restore normalized input
        in_data_norm = (out_data -
                        bnorm.bias.view(shape)).div(bnorm.weight.view(shape))
        in_data = in_data_norm

    if isinstance(module, nn.LayerNorm):
        layernorm = module
        # restore normalized input
        in_data_norm = (out_data - layernorm.bias).div(layernorm.weight)
        in_data = in_data_norm

    return in_data


def _preprocess_out_grads(module, out_grads):
    if isinstance(module, nn.Conv2d):
        out_grads = out_grads.flatten(start_dim=2)

    return out_grads


def _register_operations(model: nn.Module, module: nn.Module, op_names: List):
    op_class = get_op_class(module)
    if op_class is not None:
        setattr(module, 'operation', op_class(module, op_names, model))


def _call_operations_in_forward(module, in_data):
    if hasattr(module, 'operation'):
        module.operation.forward_post_process(in_data)


def _call_operations_in_backward(module, in_data, out_grads):
    if hasattr(module, 'operation'):
        module.operation.backward_pre_process(in_data, out_grads)


def _remove_operations(module):
    if hasattr(module, 'operation'):
        delattr(module, 'operation')
