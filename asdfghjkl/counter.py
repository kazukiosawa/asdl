from contextlib import contextmanager
from typing import Tuple, List, Dict, Any, Optional
import math

from torch import Tensor
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

__all__ = ['Counter', 'set_counter', 'Forward', 'BackwardWeight', 'BackwardInput',
           'AdamStats', 'AdamPrecond', 'KFACStats', 'KFACInv', 'KFACPrecond',
           'ShampooStats', 'ShampooInv', 'ShampooPrecond', 'LBFGSPrecond',
           'SMWNGStats', 'SMWNGPrecond']


class Counter:
    supported_types = (nn.Linear, nn.Conv2d, nn.MultiheadAttention)

    def count_flop_numel(self,
                         module: nn.Module,
                         inputs: Optional[Tensor] = None,
                         outputs: Optional[Tensor] = None,
                         batch_size: int = 32,
                         seq_len: int = 128) -> Tuple[int, int]:
        if type(module) == nn.Linear:
            return self.linear(batch_size, *module.weight.shape)
        elif type(module) == nn.Conv2d:
            h_in, w_in = inputs.shape[2:]
            h_out, w_out = outputs.shape[2:]
            return self.conv2d(batch_size, *module.weight.shape, h_in * w_in, h_out * w_out)
        elif type(module) == nn.MultiheadAttention:
            return self.attn(seq_len, batch_size, module.embed_dim, module.num_heads)
        else:
            raise TypeError(f'Invalid module type: {type(module)}')

    def linear(self, batch_size, f_out, f_in) -> Tuple[int, int]:
        raise NotImplementedError

    def conv2d(self, batch_size, c_out, c_in, kh, kw, f_in, f_out) -> Tuple[int, int]:
        raise NotImplementedError

    def attn(self, seq_len, batch_size, emb_dim, num_heads) -> Tuple[int, int]:
        flop, numel = self.linear(seq_len * batch_size, emb_dim, emb_dim)
        return flop * 4, numel * 4  # q, k, v, out_proj


@contextmanager
def set_counter(model: nn.Module, counters: List[Counter], batch_size=32, seq_len=128, keys=None):
    handles: List[RemovableHandle] = []
    counts: Dict[str, Any] = {}
    if keys is None:
        keys = [counter.__class__.__name__ for counter in counters]
    for key in keys:
        counts[key] = {'flop': 0, 'numel': 0}
    counts['params'] = 0
    try:
        def forward_hook(module, inputs, outputs):
            inputs = inputs[0]
            for key, counter in zip(keys, counters):
                flop, numel = counter.count_flop_numel(module, inputs, outputs, batch_size, seq_len)
                counts[key]['flop'] += flop
                counts[key]['numel'] += numel

        for module in model.modules():
            if type(module) in Counter.supported_types:
                if type(module) == nn.Conv2d:
                    handles.append(module.register_forward_hook(forward_hook))
                else:
                    for key, counter in zip(keys, counters):
                        flop, numel = counter.count_flop_numel(module, batch_size=batch_size, seq_len=seq_len)
                        counts[key]['flop'] += flop
                        counts[key]['numel'] += numel
                counts['params'] += sum(p.numel() for p in module.parameters())

        yield counts

    finally:
        for handle in handles:
            handle.remove()


class Forward(Counter):

    def linear(self, batch_size, f_out, f_in) -> Tuple[int, int]:
        return batch_size * f_in * f_out * 2, batch_size * f_out

    def conv2d(self, batch_size, c_out, c_in, kh, kw, f_in, f_out) -> Tuple[int, int]:
        return batch_size * c_in * c_out * f_out * kh * kw, batch_size * c_out * f_out

    def attn(self, seq_len, batch_size, emb_dim, num_heads) -> Tuple[int, int]:
        head_dim = int(emb_dim / num_heads)

        flop = seq_len * batch_size * emb_dim * 3 * emb_dim * 2  # proj q, k, v
        flop += batch_size * num_heads * seq_len * head_dim * seq_len * 2  # attn
        flop += batch_size * num_heads * seq_len * seq_len * 2  # softmax
        flop += batch_size * num_heads * seq_len * seq_len * 3  # dropout
        flop += batch_size * num_heads * seq_len * seq_len * head_dim * 2  # attnv
        flop += seq_len * batch_size * emb_dim * emb_dim * 2  # out_proj

        numel = 3 * seq_len * batch_size * emb_dim  # q, k, v
        numel += batch_size * num_heads * seq_len * seq_len  # attn
        numel += batch_size * num_heads * seq_len * seq_len  # softmax
        numel += batch_size * num_heads * seq_len * seq_len  # dropout
        numel += batch_size * num_heads * seq_len * head_dim  # attnv
        numel += seq_len * batch_size * emb_dim  # out_proj

        return flop, numel


class BackwardWeight(Counter):

    def linear(self, batch_size, f_out, f_in) -> Tuple[int, int]:
        return batch_size * f_in * f_out * 2, f_out * f_in

    def conv2d(self, batch_size, c_out, c_in, kh, kw, f_in, f_out) -> Tuple[int, int]:
        return batch_size * c_in * c_out * f_out * kh * kw, c_out * c_in * kh * kw

    def attn(self, seq_len, batch_size, emb_dim, num_heads) -> Tuple[int, int]:
        flop = seq_len * batch_size * emb_dim * emb_dim * 2  # out_proj
        flop += seq_len * batch_size * emb_dim * 3 * emb_dim * 2  # proj q, k, v

        numel = emb_dim * emb_dim  # out_proj
        numel += 3 * emb_dim * emb_dim  # proj q, k, v

        return flop, numel


class BackwardInput(Counter):

    def linear(self, batch_size, f_out, f_in) -> Tuple[int, int]:
        return batch_size * f_in * f_out * 2, batch_size * f_in

    def conv2d(self, batch_size, c_out, c_in, kh, kw, f_in, f_out) -> Tuple[int, int]:
        return batch_size * c_in * c_out * f_out * kh * kw, batch_size * c_in * f_in

    def attn(self, seq_len, batch_size, emb_dim, num_heads) -> Tuple[int, int]:
        head_dim = emb_dim / num_heads
        flop = seq_len * batch_size * emb_dim * emb_dim * 2  # out_proj (dattnv)
        flop += batch_size * num_heads * seq_len * seq_len * head_dim * 2  # attnv (dv)
        flop += batch_size * num_heads * seq_len * head_dim * seq_len * 2  # attnv (dattn_sm_drop)
        flop += batch_size * num_heads * seq_len * seq_len * 2  # dropout
        flop += batch_size * num_heads * seq_len * seq_len * 4  # softmax
        flop += batch_size * num_heads * head_dim * seq_len * seq_len * 2  # attn (dq)
        flop += batch_size * num_heads * head_dim * seq_len * seq_len * 2  # attn (dk)
        flop += seq_len * batch_size * emb_dim * 3 * emb_dim * 2  # proj q, k, v (dx)

        numel = batch_size * num_heads * seq_len * head_dim  # out_proj (dattnv)
        numel += batch_size * num_heads * seq_len * head_dim  # attnv (dv)
        numel += batch_size * num_heads * seq_len * seq_len  # attnv (dattn_sm_drop)
        numel += batch_size * num_heads * seq_len * seq_len  # dropout
        numel += batch_size * num_heads * seq_len * seq_len  # softmax
        numel += batch_size * num_heads * seq_len * head_dim  # attn (dq)
        numel += batch_size * num_heads * seq_len * head_dim  # attn (dk)
        numel += seq_len * batch_size * emb_dim  # proj q, k, v (dx)

        return flop, numel


class AdamStats(Counter):
    # square (mul), moving average (mul x 2, add)

    def linear(self, batch_size, f_out, f_in) -> Tuple[int, int]:
        numel = f_out * f_in
        return numel * 4, numel

    def conv2d(self, batch_size, c_out, c_in, kh, kw, f_in, f_out) -> Tuple[int, int]:
        numel = c_out * c_in * kh * kw
        return numel * 4, numel

    def attn(self, seq_len, batch_size, emb_dim, num_heads) -> Tuple[int, int]:
        numel = 3 * emb_dim * emb_dim + emb_dim * emb_dim
        return numel * 4, numel


class AdamPrecond(Counter):
    # sqrt, div

    def linear(self, batch_size, f_out, f_in) -> Tuple[int, int]:
        numel = f_out * f_in
        return numel * 2, 0

    def conv2d(self, batch_size, c_out, c_in, kh, kw, f_in, f_out) -> Tuple[int, int]:
        numel = c_out * c_in * kh * kw
        return numel * 2, 0


class KFACStats(Counter):

    def linear(self, batch_size, f_out, f_in) -> Tuple[int, int]:
        numel_A = f_in ** 2
        numel_B = f_out ** 2
        flop_A = batch_size * numel_A * 2
        flop_B = batch_size * numel_B * 2
        return flop_A + flop_B, numel_A + numel_B

    def conv2d(self, batch_size, c_out, c_in, kh, kw, f_in, f_out) -> Tuple[int, int]:
        flop_im2col = batch_size * c_in * c_out * f_out
        numel_A = (c_in * kh * kw) ** 2
        numel_B = c_out ** 2
        flop_A = batch_size * numel_A * 2
        flop_B = batch_size * numel_B * 2
        return flop_im2col + flop_A + flop_B, numel_A + numel_B


class KFACInv(Counter):

    def linear(self, batch_size, f_out, f_in) -> Tuple[int, int]:
        return f_in ** 3 + f_out ** 3, f_in ** 2 + f_out ** 2

    def conv2d(self, batch_size, c_out, c_in, kh, kw, f_in, f_out) -> Tuple[int, int]:
        return (c_in * kh * kw) ** 3 + c_out ** 3, (c_in * kh * kw) ** 2 + c_out ** 2


class KFACPrecond(Counter):

    def linear(self, batch_size, f_out, f_in) -> Tuple[int, int]:
        flop = f_in * f_in * f_out * 2  # A @ G
        flop += f_in * f_out * f_out * 2  # A @ G @ B
        return flop, 0

    def conv2d(self, batch_size, c_out, c_in, kh, kw, f_in, f_out) -> Tuple[int, int]:
        flop = (c_in * kh * kw) ** 2 * c_out * 2  # A @ G
        flop += c_in * kh * kw * c_out * c_out * 2  # A @ G @ B
        return flop, 0


class ShampooStats(Counter):

    @staticmethod
    def flop(*shape):
        prod = 1
        for s in shape:
            prod *= s
        return sum([prod * s * 2 for s in shape])

    @staticmethod
    def numel(*shape):
        return sum([s ** 2 for s in shape])

    def linear(self, batch_size, f_out, f_in) -> Tuple[int, int]:
        return self.flop(f_out, f_in), self.numel(f_out, f_in)

    def conv2d(self, batch_size, c_out, c_in, kh, kw, f_in, f_out) -> Tuple[int, int]:
        return self.flop(c_out, c_in, kh, kw), self.numel(c_out, c_in, kh, kw)


class ShampooInv(Counter):
    # eigen decomposition

    @staticmethod
    def flop(*shape):
        return math.ceil(sum([(9 + 1/3) * s ** 3 for s in shape]))

    @staticmethod
    def numel(*shape):
        return sum([s ** 2 * 2 + s for s in shape])

    def linear(self, batch_size, f_out, f_in) -> Tuple[int, int]:
        return self.flop(f_out, f_in), self.numel(f_out, f_in)

    def conv2d(self, batch_size, c_out, c_in, kh, kw, f_in, f_out) -> Tuple[int, int]:
        return self.flop(c_out, c_in, kh, kw), self.numel(c_out, c_in, kh, kw)


class ShampooPrecond(Counter):

    @staticmethod
    def flop(*shape):
        prod = 1
        for s in shape:
            prod *= s
        return sum([prod * s * 2 for s in shape])

    def linear(self, batch_size, f_out, f_in) -> Tuple[int, int]:
        return self.flop(f_out, f_in), 0

    def conv2d(self, batch_size, c_out, c_in, kh, kw, f_in, f_out) -> Tuple[int, int]:
        return self.flop(c_out, c_in, kh, kw), 0


class LBFGSPrecond(Counter):
    def __init__(self, hist_size=20):
        self.hist_size = hist_size

    def flop(self, nparams):
        flop = nparams * 2  # rho
        flop += nparams * 2  # alpha
        flop += nparams * 2  # g -= alpha * y
        flop += nparams * 2  # beta
        flop += nparams * 2  # g += (alpha - beta) * s
        return flop * self.hist_size

    def numel(self, nparams):
        return nparams * 2 * self.hist_size

    def linear(self, batch_size, f_out, f_in) -> Tuple[int, int]:
        nparams = f_out * f_in
        return self.flop(nparams), self.numel(nparams)

    def conv2d(self, batch_size, c_out, c_in, kh, kw, f_in, f_out) -> Tuple[int, int]:
        nparams = c_out * c_in * kh * kw
        return self.flop(nparams), self.numel(nparams)


class SMWNGStats(Counter):

    def linear(self, batch_size, f_out, f_in) -> Tuple[int, int]:
        flop_A = batch_size ** 2 * f_in * 2
        flop_B = batch_size ** 2 * f_out * 2
        return flop_A + flop_B, batch_size ** 2

    def conv2d(self, batch_size, c_out, c_in, kh, kw, f_in, f_out) -> Tuple[int, int]:
        flop_im2col = batch_size * c_in * c_out * f_out
        flop_A = batch_size ** 2 * c_in * kh * kw
        flop_B = batch_size ** 2 * c_out
        return flop_im2col + flop_A + flop_B, batch_size ** 2


class SMWNGPrecond(Counter):

    @staticmethod
    def flop(n):
        flop = n ** 2  # reduce Gram to a vector
        flop += 1/3 * n ** 3 + n ** 2  # Cholesky solve
        return flop

    @staticmethod
    def numel(n):
        return n  # Gram^{-1}v

    def linear(self, batch_size, f_out, f_in) -> Tuple[int, int]:
        return self.flop(batch_size), self.numel(batch_size)

    def conv2d(self, batch_size, c_out, c_in, kh, kw, f_in, f_out) -> Tuple[int, int]:
        return self.flop(batch_size), self.numel(batch_size)
