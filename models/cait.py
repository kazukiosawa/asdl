from random import randrange
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn import Module
from .SPT import ShiftedPatchTokenization

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(Module):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

# helpers
def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

def exists(val):
    return val is not None

def dropout_layers(layers, dropout):
    if dropout == 0:
        return layers

    num_layers = len(layers)
    to_drop = torch.zeros(num_layers).uniform_(0., 1.) < dropout

    # make sure at least one layer makes it
    if all(to_drop):
        rand_index = randrange(num_layers)
        to_drop[rand_index] = False

    layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
    return layers

# classes

class LayerScale(nn.Module):
    def __init__(self, dim, fn, depth):
        super().__init__()
        if depth <= 18:  # epsilon detailed in section 2 of paper
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, num_patches, heads = 8, dim_head = 64, dropout = 0., if_patch_attn=False, is_LSA=False):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.attend = nn.Softmax(dim = -1)

        self.mix_heads_pre_attn = nn.Parameter(torch.randn(heads, heads))
        self.mix_heads_post_attn = nn.Parameter(torch.randn(heads, heads))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        
        self.is_LSA = is_LSA
        if is_LSA:
            self.scale = nn.Parameter(self.scale*torch.ones(heads))
            self.mask = torch.eye(num_patches, num_patches)
            self.mask = torch.nonzero((self.mask == 1), as_tuple=False)
        self.if_patch_attn = if_patch_attn

    def forward(self, x, context = None):
        b, n, _, h = *x.shape, self.heads

        context = x if not exists(context) else context

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        if not self.is_LSA:
            dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale        
        else:
            """ LSA """
            scale = self.scale
            dots = torch.mul(einsum('b h i d, b h j d -> b h i j', q, k), scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand((x.size(0), self.heads, 1, 1)))            
            if self.if_patch_attn:
                dots[:, :, self.mask[:, 0], self.mask[:, 1]] = -1e-9
        
        dots = einsum('b h i j, h g -> b g i j', dots, self.mix_heads_pre_attn)    # talking heads, pre-softmax
        attn = self.attend(dots)        
        attn = einsum('b h i j, h g -> b g i j', attn, self.mix_heads_post_attn)   # talking heads, post-softmax

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim, dropout = 0., layer_dropout = 0., stochastic_depth=0., if_patch_attn=False, is_LSA=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.layer_dropout = layer_dropout

        for ind in range(depth):
            self.layers.append(nn.ModuleList([
                LayerScale(dim, PreNorm(dim, Attention(dim, num_patches, heads = heads, dim_head = dim_head, dropout = dropout, if_patch_attn=if_patch_attn, is_LSA=is_LSA)), depth = ind + 1),
                LayerScale(dim, PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)), depth = ind + 1)
            ]))
        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()
    def forward(self, x, context = None):
        layers = dropout_layers(self.layers, dropout = self.layer_dropout)

        for attn, ff in layers:
            
            x = self.drop_path(attn(x, context = context)) + x
            x = self.drop_path(ff(x)) + x
        return x
    
class CaiT(nn.Module):
    def __init__(
        self,
        *,
        img_size,
        patch_size,
        num_classes,
        dim=192,
        depth=24,
        cls_depth=2,
        heads=4,
        mlp_dim=384,
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.,
        layer_dropout = 0.,
        stochastic_depth = 0.,
        is_LSA=False,
        is_SPT=False
    ):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        
        if not is_SPT:
            patch_dim = 3 * patch_size ** 2
            
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
                nn.Linear(patch_dim, dim),
            )
        
        else:
            self.to_patch_embedding = nn.Sequential(
                ShiftedPatchTokenization(3, dim, patch_size, is_pe=True),
            )
        
        
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)
        num_patches = (image_height // patch_height) * (image_width // patch_width)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.patch_transformer = Transformer(dim, num_patches, depth, heads, dim_head, mlp_dim, dropout, layer_dropout, stochastic_depth=stochastic_depth, if_patch_attn=True, is_LSA=is_LSA)
        self.cls_transformer = Transformer(dim, num_patches, cls_depth, heads, dim_head, mlp_dim, dropout, layer_dropout, stochastic_depth=stochastic_depth, is_LSA=is_LSA)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        
        self.apply(init_weights)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.patch_transformer(x)

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = self.cls_transformer(cls_tokens, context = x)

        return self.mlp_head(x[:, 0])