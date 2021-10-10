from contextlib import contextmanager
from torch.cuda import nvtx, profiler


@contextmanager
def nvtx_range(msg):
    try:
        nvtx.range_push(msg)
        yield
    finally:
        nvtx.range_pop()


# the main logic is adopted from https://github.com/spcl/substation/blob/master/pytorch_profiling/profiling.py
# modified to use contextmanagers and NVIDIA Nsight Systems CLI
def time_funcs(funcs, name='', func_names=None, num_iters=100, num_warmups=5):
    if func_names is None:
        func_names = [f.__name__ for f in funcs]
    with profiler.profile():
        with nvtx_range(name + 'Warmup'):
            for _ in range(num_warmups):
                for f in funcs:
                    f()
        for _ in range(num_iters):
            with nvtx_range(name + 'Iter'):
                for f, fname in zip(funcs, func_names):
                    with nvtx_range(fname):
                        f()
