import math
import warnings
from typing import Union, Tuple, Any
from dataclasses import dataclass

import torch.nn as nn
from torch import Tensor
from asdfghjkl import GradientMaker


__all__ = ['PreconditionedGradientMaker',
           'get_update_schedule', 'INTERVAL_STEP', 'INTERVAL_LINEAR', 'INTERVAL_TYPES']

INTERVAL_CONSTANT = 'constant'
INTERVAL_STEP = 'step'
INTERVAL_LINEAR = 'linear'
INTERVAL_TYPES = [INTERVAL_CONSTANT, INTERVAL_STEP, INTERVAL_LINEAR]


@dataclass
class PreconditionedGradientConfig:
    num_total_steps: int
    preconditioner_upd_ratio: float = 1.
    preconditioner_warmup_ratio: float = 0.
    preconditioner_interval_type = INTERVAL_CONSTANT
    curvature_upd_ratio: float = None
    curvature_warmup_ratio: float = 0.
    curvature_interval_type = INTERVAL_CONSTANT


class PreconditionedGradientMaker(GradientMaker):
    def __init__(self, model: nn.Module, config: PreconditionedGradientConfig):
        super().__init__(model)
        self.config = config
        self.preconditioner_upd_schedule = get_update_schedule(num_total_steps=config.num_total_steps,
                                                               update_ratio=config.preconditioner_upd_ratio,
                                                               warmup_ratio=config.preconditioner_warmup_ratio)
        if config.curvature_upd_ratio is not None:
            self.curvature_upd_schedule = get_update_schedule(num_total_steps=config.num_total_steps,
                                                              update_ratio=config.curvature_upd_ratio,
                                                              warmup_ratio=config.curvature_warmup_ratio)
        else:
            self.curvature_upd_schedule = None
        self._step = 0

    def forward_and_backward(self, *args, **kwargs) -> Union[Tuple[Any, Tensor], Any]:
        rst = self._forward_and_backward(*args, **kwargs)
        self._step += 1
        return rst

    def _forward_and_backward(self, *args, **kwargs) -> Union[Tuple[Any, Tensor], Any]:
        raise NotImplementedError

    def _do_update(self, upd_schedule, step=None):
        if step is None:
            step = self._step
        try:
            return upd_schedule[step]
        except IndexError:
            warnings.warn(f'Given step+1={step+1} is larger than the total number of steps ({self.config.num_total_steps})')
            return False

    def do_update_preconditioner(self, step=None):
        return self._do_update(self.preconditioner_upd_schedule, step)

    def do_update_curvature(self, step=None):
        if self.curvature_upd_schedule is None:
            return self.do_update_preconditioner()
        return self._do_update(self.curvature_upd_schedule, step)


def get_update_schedule(num_total_steps: int,
                        update_ratio: float = 1.,
                        warmup_ratio: float = 0.,
                        interval_type: str = INTERVAL_CONSTANT,
                        reverse=False):
    assert num_total_steps > 0
    assert 0 <= update_ratio <= 1
    num_total_updates = int(num_total_steps * update_ratio)
    assert 0 <= warmup_ratio <= 1
    num_warmup_steps = int(num_total_steps * warmup_ratio)
    assert num_warmup_steps <= num_total_updates
    assert interval_type in INTERVAL_TYPES

    update_schedule = [True] * num_warmup_steps
    num_remaining_steps = num_total_steps - num_warmup_steps
    num_remaining_updates = num_total_updates - num_warmup_steps

    if num_remaining_updates > 0:
        if interval_type == INTERVAL_CONSTANT:
            # constant interval
            interval = math.floor(num_remaining_steps / num_remaining_updates)
            for i in range(num_remaining_steps):
                update_schedule.append(i % interval == 0 and update_schedule.count(True) < num_total_updates)
        elif interval_type == INTERVAL_STEP:
            # step interval (one step)
            update_schedule.extend([True] * num_remaining_updates)
        else:
            # linear interval
            n = num_remaining_updates - 1
            interval = 1
            d_interval = math.floor(2 * (num_remaining_steps - n) / n / (n - 1))
            update_schedule.append(True)
            for i in range(n):
                update_schedule.extend([False] * (interval + d_interval * i - 1))
                update_schedule.append(True)

    assert len(update_schedule) <= num_total_steps
    # padding with False
    update_schedule.extend([False] * (num_total_steps - len(update_schedule)))

    if reverse:
        return update_schedule[::-1]
    return update_schedule
