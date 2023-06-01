
from functools import partial

import torch
import torch.nn as nn

import tensorflow.keras
from tensorflow.keras import layers

from einops import rearrange

exists = lambda x: x is not None

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class CustomPadding1D(layers.Layer):
    def __init__(self, padding, padding_mode):
        super().__init__()
        self.padding = padding
        self.padding_mode = padding_mode
    def call(self, inputs):
        return tf.pad(inputs, [[0, 0], [self.padding, self.padding], [0, 0]], mode=self.padding_mode)

def Upsample(dim):
    return layers.Conv1DTranspose(dim, 4, strides=2, padding="same")

def Downsample(dim):
    # TODO: implement relection padding
    return layers.Conv1D(dim, 4, strides=2, padding="same")

class PreNorm(layers.Layer):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        # I'm not sure if the num_channels from pytorch does anything
        self.norm = layers.GroupNormalization(1)

    def call(self, inputs):
        x = self.norm(inputs)
        return self.fn(x)

class SinusoidalPositionEmbeddings(layers.Layer):
    def __init__(self, dim, max_timescale=10000):
        super().__init__()
        assert dim % 2 == 0
        self.rate = tf.pow(max_timescale, tf.linspace(0, -1, dim // 2))

    def call(self, inputs):
        inputs = tf.expand_dims(inputs, -1)
        fs = tf.constant(self.rate, dtype=tf.float32)
        embs = inputs * fs
        embs_sin_cos = tf.concat([tf.sin(embs), tf.cos(embs)], axis=-1)
        return embs_sin_cos

class Attention(layers.Layer):
    def __init__(self, dim, heads=8, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        h_dim = dim_head * heads
        self.dim_head = dim_head
        
        self.to_qkv = layers.Conv1D(h_dim*3, 1, use_basis=False) # Maybe add input_shape for convolutonal layers here
        self.to_out = layers.Conv1D(dim, 1)

    def call(self, inputs):
        qkv = self.to_qkv(inputs)
        qkv = tf.split(qkv, 3, axis=1) # Why use 1 for the axis? Everything else uses -1
        out = self.attn(*(t for t in qkv))
        return self.to_out(out)

    def attn(self, q, k, v):
        q = q * self.scale
        sim = tf.einsum("b h d i, b h d j -> b h i j", q, k) # I have no idea what this is. Or the other functions that take an input like this
        sim = sim - tf.reduce_max(sim, axis=-1, keepdims=True)
        attn = tf.nn.softmax(sim, axis=-1)

        out = tf.einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h l c -> b (h c) l") # Not sure if this works with tensorflow
        return out

class LinearAttention(Attention):
    """https://arxiv.org/abs/1812.01243"""

    def attn:
        q = tf.nn.softmax(q, axis=-2) * self.scale
        k = tf.nn.softmax(k, dim=-1)

        ctx = tf.einsum("b h d n, b h e n -> b h d e", k, v) # Again I have no idea what this is
        out = tf.einsum("b h d e, b h d n -> b h e n", ctx, q)
        out = rearrange(out, "b h c l -> b (h c) l")
        return out
    
# class WaveBlock(nn.Module):
#     """context is acquired from num_stacks*2**stack_depth neighborhood"""
    
#     def __init__(self, dim, stack_depth, num_stacks, mult=1, h_dim_groups=1, up=False):
#         super().__init__()

#         self.in_net = nn.Conv1d(dim, dim * mult, 1)
#         self.out_net = nn.Conv1d(dim * mult, dim, 1)
        
#         self.nets = nn.ModuleList([
#             nn.Sequential(
#                 (nn.ConvTranspose1d if up else nn.Conv1d)(
#                     in_channels=dim * mult, 
#                     out_channels=2 * dim * mult,
#                     kernel_size=2,
#                     padding=2**i,
#                     dilation=2**(i+1),
#                     groups=h_dim_groups,
#                     **({} if up else dict(padding_mode='replicate')),
#                 ),
#                 nn.GLU(dim=1),
#             )
#             for _ in range(num_stacks)
#             for i in range(stack_depth)
#         ])
        
#     def forward(self, x: "N,C,L") -> "N,C,L":
#         x = self.in_net(x)
#         h = x
#         for net in self.nets:
#             h = net(h)
#             x = x + h
#         return self.out_net(x)

class GLU(layers.Layer):
    """https://github.com/Rishit-dagli/GLU/blob/main/glu_tf/glu.py"""
    def __init__(self, bias=True, dim=-1, **kwargs):
        super(GLU, self).__init__(**kwargs)
        self.bias = bias
        self.dim = dim
        self.dense = layers.Dense(2, use_bias=bias)

    def call(self, x):
        out, gate = tf.split(x, num_split=2, axis=self.dim)
        gate = tf.sigmoid(gate)
        x = tf.multiply(out, gate)
        return x

class WaveBlock(layers.Layer):
    """context is acquired from num_stacks*2**stack_depth neighborhood"""

    def __init__(self, dim, stack_depth, num_stacks, mult=1, h_dim_groups=1, up=False):
        super().__init__()

        self.in_net = layers.Conv1D(dim * mult, 1) # Maybe add input_shape for convolutonal layers here too
        self.out_net = layers.Conv1D(dim, 1)
        self.nets = []
        for _ in range(num_stacks):
            for i in range(stack_depth)
                net = tf.keras.Sequential()
                if up:
                    net.add(layers.ZeroPadding1D(2**i))
                else:
                    net.add(CustomPadding1D(2**i, "SYMMETRIC"))
                net.add((layers.Conv1DTranspose if up else layers.Conv1D)(
                    filters=2 * dim * mult,
                    kernel_size=2,
                    padding="valid",
                    dilation_rate=2**(i+1),
                    groups=h_dim_groups,
                ))
                net.add(GLU(dim=1))
                self.nets.append(net)
        def call(self, inputs):
            x = self.in_net(inputs)
            h = x
            for net in self.nets:
                h = net(h)
                x = x + h
            return self.out_net(x)

class ConvNextBlock(nn.Module):
    """https://arxiv.org/abs/2201.03545"""

    def __init__(self, dim, dim_out, *, emb_dim=None, mult=2, norm=True, groups=1):
        super().__init__()
        
        self.mlp = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(emb_dim, dim),
            )
            if exists(emb_dim)
            else None
        )

        self.ds_conv = nn.Conv1d(dim, dim, 7, padding=3, groups=dim, padding_mode='reflect')

        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv1d(dim, dim_out * mult, 7,1,3, padding_mode='reflect', groups=groups),
            nn.SiLU(),

            nn.GroupNorm(1, dim_out * mult),
            nn.Conv1d(dim_out * mult, dim_out, 7,1,3, padding_mode='reflect', groups=groups),
        )

        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: "N,C,L", time_emb: "N,T" = None) -> "N,D,L":
        h: "N,C,L" = self.ds_conv(x)

        if exists(self.mlp) and exists(time_emb):
            condition: "N,C" = self.mlp(time_emb)
            h = h + condition.unsqueeze(-1)

        h: "N,D,L" = self.net(h)
        return h + self.res_conv(x)

class ConvNextBlock(layers.Layer):
    """https://arxiv.org/abs/2201.03545"""

    def __init__(self, dim, dim_out, *, emb_dim=None, mult=2, norm=True, groups=1):
        super().__init__()

        self.mlp = (
            tf.keras.Sequential([
                layers.Lambda(tf.nn.silu),
                layers.Dense(dim)
            ])
            if exsits(emb_dim)
            else None
        )

        self.padding = CustomPadding1D(3, "REFLECT")
        self.ds_conv = layers.Conv1d(dim, 7, padding="valid", groups=dim)

        self.net = tf.keras.Sequential([
            layers.GroupNormalization(1) if norm else layers.Identity(),
            CustomPadding1D(3, "REFLECT"),
            layers.Conv1D(dim_out * mult, 7, strides=1, padding="valid", groups=groups),
            layers.Lambda(tf.nn.silu),
            layers.GroupNormalization(1),
            CustomPadding1D(3, "REFLECT"),
            layers.Conv1D(dim_out, 7, strides=1, padding="valid", groups=groups)
        ])

        self.res_conv = layers.Conv1D(dim_out, 1) if dim != dim_out else layers.Identity()

    def call(self, inputs, time_emb=None):
        h = self.ds_conv(inputs)

        if exists(self.mlp) and exists(time_emb):
            condition = self.mlp(time_emb)
            h = h + tf.expand_dims(condition, -1)
        
        h = self.net(h)
        return h + self.res_conv(inputs)
    
class UNet(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        h_dims,
        h_dim_groups,
        convnext_mult,
        wave_stack_depth,
        wave_num_stacks,
        blocks_per_depth,
        attn_heads,
        attn_dim,
    ):
        super().__init__()
        
        block = partial(ConvNextBlock, mult=convnext_mult, groups=h_dim_groups)

        in_out = list(zip(h_dims[:-1], h_dims[1:]))
        num_layers = len(in_out)
        
        self.init_conv = nn.Sequential(
            nn.Conv1d(in_dim, h_dims[0], 7, padding=3),
            WaveBlock(h_dims[0], wave_stack_depth, wave_num_stacks),
        )
        
        # time embeddings
        emb_dim = h_dims[0] * 4
        self.time_mlp: "N, -> N,T" = nn.Sequential(
            SinusoidalPositionEmbeddings(h_dims[0]),
            nn.Linear(h_dims[0], emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        # layers
        
        self.downs = nn.ModuleList([
            nn.ModuleList([
                nn.ModuleList([
                    block(dim_in if i==0 else dim_out, dim_out, emb_dim=emb_dim)
                    for i in range(blocks_per_depth)
                ]),
                nn.ModuleList([
                    Residual(PreNorm(dim_out, LinearAttention(dim_out, heads=attn_heads, dim_head=attn_dim)))
                    for _ in range(blocks_per_depth)
                ]),
                Downsample(dim_out) if ind < (num_layers - 1) else nn.Identity(),
            ])
            for ind, (dim_in, dim_out) in enumerate(in_out)
        ])

        mid_dim = h_dims[-1]
        self.mid_block1 = block(mid_dim, mid_dim, emb_dim=emb_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, heads=attn_heads, dim_head=attn_dim)))
        self.mid_block2 = block(mid_dim, mid_dim, emb_dim=emb_dim)
        
        self.ups = nn.ModuleList([
            nn.ModuleList([
                nn.ModuleList([
                    block(dim_out * 2 if i==0 else dim_in, dim_in, emb_dim=emb_dim)
                    for i in range(blocks_per_depth)
                ]),
                nn.ModuleList([
                    Residual(PreNorm(dim_in, LinearAttention(dim_in, heads=attn_heads, dim_head=attn_dim)))
                    for _ in range(blocks_per_depth)
                ]),
                Upsample(dim_in) if ind < (num_layers - 1) else nn.Identity(),
            ])
            for ind, (dim_in, dim_out) in enumerate(in_out[::-1])
        ])

        self.final_conv = nn.Sequential(
            *(
                block(h_dims[0], h_dims[0])
                for _ in range(blocks_per_depth)
            ),
            zero_module(nn.Conv1d(h_dims[0], out_dim, 1)),
        )
        

    def forward(self, x: "N,X,L", a: "N,A,L", ts: "N,") -> "N,X,L":
        
        x: "N,X+A,L" = torch.cat([x,a], dim=1)
        
        x: "N,h_dim,L" = self.init_conv(x)

        h = []
        emb: "N,T" = self.time_mlp(ts)

        # downsample
        for blocks, attns, downsample in self.downs:
            for block, attn in zip(blocks, attns):
                x = attn(block(x, emb))
            h.append(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x, emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, emb)

        # upsample
        for blocks, attns, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            for block, attn in zip(blocks, attns):
                x = attn(block(x, emb))
            x = upsample(x)

        return self.final_conv(x)
