from functools import partial
from typing import Callable, Optional, Sequence, Tuple

import jax.numpy as jnp
from flax import linen as nn

from .common import ConvBlock, ModuleDef


class ResNetStem(nn.Module):
    conv_block_cls: ModuleDef = ConvBlock

    @nn.compact
    def __call__(self, x):
        return self.conv_block_cls(
            64, kernel_size=(7, 7), strides=(2, 2), padding=[(3, 3), (3, 3)]
        )(x)


class ResNetSkipConnection(nn.Module):
    strides: Tuple[int, int]
    conv_block_cls: ModuleDef = ConvBlock

    @nn.compact
    def __call__(self, x, out_shape):
        if x.shape != out_shape:
            x = self.conv_block_cls(
                out_shape[-1],
                kernel_size=(1, 1),
                strides=self.strides,
                activation=lambda y: y,
            )(x)
        return x


class ResNetBlock(nn.Module):
    n_hidden: int
    strides: Tuple[int, int] = (1, 1)

    activation: Callable = nn.relu
    conv_block_cls: ModuleDef = ConvBlock
    skip_cls: ModuleDef = ResNetSkipConnection

    @nn.compact
    def __call__(self, x):
        skip_cls = partial(self.skip_cls, conv_block_cls=self.conv_block_cls)
        y = self.conv_block_cls(
            self.n_hidden, padding=[(1, 1), (1, 1)], strides=self.strides
        )(x)
        y = self.conv_block_cls(self.n_hidden, padding=[(1, 1), (1, 1)], is_last=True)(
            y
        )
        return self.activation(y + skip_cls(self.strides)(x, y.shape))


def ResNet(
    block_cls: ModuleDef,
    *,
    stage_sizes: Sequence[int],
    n_classes: int,
    hidden_sizes: Sequence[int] = (64, 128, 256, 512),
    conv_cls: ModuleDef = nn.Conv,
    norm_cls: Optional[ModuleDef] = partial(nn.BatchNorm, momentum=0.9),
    conv_block_cls: ModuleDef = ConvBlock,
    stem_cls: ModuleDef = ResNetStem,
    pool_fn: Callable = partial(
        nn.max_pool, window_shape=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1))
    )
):
    conv_block_cls = partial(conv_block_cls, conv_cls=conv_cls, norm_cls=norm_cls)
    stem_cls = partial(stem_cls, conv_block_cls=conv_block_cls)
    block_cls = partial(block_cls, conv_block_cls=conv_block_cls)

    client_layers = [stem_cls(), pool_fn]
    server_layers = []

    for i, (hsize, n_blocks) in enumerate(zip(hidden_sizes, stage_sizes)):
        if i < 2:
            for b in range(n_blocks):
                strides = (1, 1) if i == 0 or b != 0 else (2, 2)
                client_layers.append(block_cls(n_hidden=hsize, strides=strides))
        else:
            for b in range(n_blocks):
                strides = (1, 1) if i == 0 or b != 0 else (2, 2)
                server_layers.append(block_cls(n_hidden=hsize, strides=strides))

    server_layers.append(partial(jnp.mean, axis=(1, 2)))  # global average pool
    server_layers.append(nn.Dense(n_classes))
    return nn.Sequential(client_layers), nn.Sequential(server_layers)


# yapf: disable
ResNet18 = partial(ResNet, stage_sizes=(2, 2, 2, 2), n_classes=1000, stem_cls=ResNetStem, block_cls=ResNetBlock)
