# Copyright 2024 The swirl_fem Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transformer backbone for encoder-decoder of the NN closure model."""

from collections.abc import Sequence
from functools import partial
from typing import Any, Callable

from absl import logging
import distrax
import flax.linen as nn
import jax
from jax import lax
import jax.numpy as jnp
import ml_collections
import numpy as np
from swirl_fem.sde.flax_nn_sde import nn_sdeint
from swirl_fem.sde.sdeint import brownian_path


def attention_pooling(x: jnp.ndarray,
                      pooling_window: tuple[int, ...],
                      strides: tuple[int, ...],
                      pool_mode: str,
                      num_heads: int) -> jnp.ndarray:
  """Applies pooling to an input array before splitting it into attention heads.

  Args:
    x: Input array of shape (..., features).
    pooling_window: Pooling window size.  If empty or (1,...,1), no pooling is performed.
    strides: Strides for pooling operation.
    pool_mode: Pooling type ('avg' or 'max'). 'none' disables pooling.
    num_heads: Number of attention heads.

  Returns:
    Array of shape (..., num_heads, head_dim), where head_dim = features // num_heads.
  """
  head_dim = x.shape[-1] // num_heads
  if pool_mode == 'none' or not pooling_window or np.prod(pooling_window) == 1:
    x = x.reshape(x.shape[:-1] + (num_heads, head_dim))
    return x

  if pool_mode == 'avg':
    x = nn.avg_pool(x, pooling_window, strides, padding='same')
  elif pool_mode == 'max':
    x = nn.max_pool(x, pooling_window, strides, padding='same')
  else:
    raise ValueError(f'Unknown PoolMode: {pool_mode}')

  x = x.reshape(x.shape[:-1] + (num_heads, head_dim))
  return x


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""

  mlp_dim: int
  out_dim: int | None = None
  use_bias: bool = True
  kernel_init = nn.initializers.xavier_uniform()
  bias_init = nn.initializers.normal(stddev=1e-6)
  activation_fn = nn.gelu
  precision: jax.lax.Precision | None = None
  dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, inputs: jnp.ndarray):
    """Applies Transformer MlpBlock module."""
    actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
    x = nn.Dense(
        self.mlp_dim,
        dtype=self.dtype,
        use_bias=self.use_bias,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        precision=self.precision)(inputs)
    x = self.activation_fn(x)
    output = nn.Dense(
        actual_out_dim,
        dtype=self.dtype,
        use_bias=self.use_bias,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        precision=self.precision)(x)
    return output


class MultiScaleAttentionDownSample(nn.Module):
  num_heads: int
  pool_q: tuple[int, ...]
  pool_kv: tuple[int, ...]
  stride_q: tuple[int, ...]
  stride_kv: tuple[int, ...]
  pool_mode: str
  use_residual_q_pooling: bool = True
  use_bias: bool = False
  dtype: jnp.dtype = jnp.float32
  precision: lax.Precision | None = None

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    num_features = x.shape[-1]
    dense = partial(
        nn.Dense,
        dtype=self.dtype,
        features=num_features,
        use_bias=self.use_bias,
        precision=self.precision)

    pool_fn = partial(
        attention_pooling,
        pool_mode=self.pool_mode,
        num_heads=self.num_heads)

    q = dense(name='query')(x)
    q = pool_fn(q, self.pool_q, self.stride_q)
    k, v = dense(name='key')(x), dense(name='value')(x)
    k = pool_fn(k, self.pool_kv, self.stride_kv)
    v = pool_fn(v, self.pool_kv, self.stride_kv)
    x = nn.dot_product_attention(
        q, k, v, dtype=self.dtype, precision=self.precision)

    if self.use_residual_q_pooling:
      x += q

    # DenseGeneral goes over head_dim and num_heads.
    out = nn.DenseGeneral(
        features=num_features,
        axis=(-2, -1),
        use_bias=self.use_bias,
        dtype=self.dtype,
        precision=self.precision,
        name='out')(x)
    return out


class MultiScaleAttentionUpsample(nn.Module):
  num_heads: int
  qkv_tile_reps: tuple[int, ...]
  use_residual_q_pooling: bool = True
  use_bias: bool = False
  dtype: jnp.dtype = jnp.float32
  precision: lax.Precision | None = None

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Applies scaled multi-head dot product attention on the input data.

    Args:
      x: input array of shape `[batch_sizes..., T*W*H, features]`.
    Returns:
      output of shape `[batch_sizes..., T, W, H, features]`
    """

    # project inputs to multi-headed q/k/v
    # dimensions are then [batch..., length, n_heads * head_dim]
    # keeping the last 2 dims together will make pooling easier, we will reshape
    # after that.
    num_features = x.shape[-1]
    dense = partial(
        nn.Dense,
        dtype=self.dtype,
        features=num_features,
        use_bias=self.use_bias,
        precision=self.precision)

    tile_fn = partial(jnp.tile, reps=self.qkv_tile_reps)
    q = dense(name='query')(x)
    q = tile_fn(q)
    k, v = dense(name='key')(x), dense(name='value')(x)
    k = tile_fn(k)
    v = tile_fn(v)
    q = q.reshape(q.shape[:2] + (self.num_heads, -1))
    k = k.reshape(k.shape[:2] + (self.num_heads, -1))
    v = v.reshape(v.shape[:2] + (self.num_heads, -1))

    x = nn.dot_product_attention(
        q, k, v, dtype=self.dtype, precision=self.precision)

    if self.use_residual_q_pooling:
      x += q  # Residual Pooling from MViTv2 paper.

    # DenseGeneral goes over head_dim and num_heads.
    out = nn.DenseGeneral(
        features=num_features,
        axis=(-2, -1),
        use_bias=self.use_bias,
        dtype=self.dtype,
        precision=self.precision,
        name='out')(x)
    return out


class EncoderBlock(nn.Module):
  """Encoder block for multiscale transformer."""
  out_dim: int
  num_heads: int
  pooling_kernel_q: tuple[int, ...]
  pooling_kernel_kv: tuple[int, ...]
  pooling_stride_q: tuple[int, ...]
  pooling_stride_kv: tuple[int, ...]
  use_bias: bool = False
  dtype: jnp.dtype = jnp.float32
  precision: lax.Precision | None = None

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    """Applies Encoder1DBlock module.

    Args:
      inputs: Input data of shape [batch..., T*W*H, features].

    Returns:
      Output after transformer encoder block.
    """
    x = nn.LayerNorm(use_bias=self.use_bias, dtype=self.dtype)(inputs)
    x = MultiScaleAttentionDownSample(
        num_heads=self.num_heads,
        pool_q=self.pooling_kernel_q,
        pool_kv=self.pooling_kernel_kv,
        stride_q=self.pooling_stride_q,
        stride_kv=self.pooling_stride_kv,
        use_bias=self.use_bias,
        dtype=self.dtype,
        pool_mode='avg',
        precision=self.precision)(x)  # pylint: disable=not-a-mapping

    if self.pooling_stride_q and np.prod(self.pooling_stride_q) > 1:
      inputs_res = attention_pooling(
          inputs,
          tuple([s + 1 if s > 1 else s for s in self.pooling_stride_q]),
          strides=self.pooling_stride_q,
          pool_mode='max',  # Skip connections always use max-pooling.
          num_heads=self.num_heads)
      inputs_res = inputs_res.reshape(inputs_res.shape[:-2] + (-1,))
    else:
      inputs_res = inputs

    x = x + inputs_res

    # MLP block.
    x_norm = nn.LayerNorm(dtype=self.dtype, use_bias=self.use_bias)(x)
    y = MlpBlock(
        mlp_dim=inputs.shape[-1] * 4,
        out_dim=self.out_dim,
        use_bias=self.use_bias,
        dtype=self.dtype,
        precision=self.precision)(x_norm)

    if self.out_dim == inputs.shape[-1]:
      x_res = x
    else:
      x_res = nn.Dense(
          y.shape[-1],
          name='project_skip',
          dtype=self.dtype,
          precision=self.precision,
          use_bias=self.use_bias)(x_norm)  # Reference uses x_norm here!
    return y + x_res  # pytype: disable=bad-return-type  # jax-ndarray


class DecoderBlock(nn.Module):
  out_dim: int
  num_heads: int
  qkv_tile_reps: tuple[int, ...]
  use_bias: bool = False
  dtype: jnp.dtype = jnp.float32
  precision: lax.Precision | None = None

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    x = nn.LayerNorm(use_bias=self.use_bias, dtype=self.dtype)(inputs)
    x = MultiScaleAttentionUpsample(
        num_heads=self.num_heads,
        qkv_tile_reps=self.qkv_tile_reps,
        use_bias=self.use_bias,
        dtype=self.dtype,
        precision=self.precision)(x)  # pylint: disable=not-a-mapping

    if np.prod(list(self.qkv_tile_reps)) > 1:
      input_res = jnp.tile(inputs, self.qkv_tile_reps)
    else:
      input_res = inputs
    x = x + input_res

    # MLP block.
    x_norm = nn.LayerNorm(dtype=self.dtype, use_bias=self.use_bias)(x)
    y = MlpBlock(
        mlp_dim=inputs.shape[-1] * 4,
        out_dim=self.out_dim,
        use_bias=self.use_bias,
        dtype=self.dtype,
        precision=self.precision)(x_norm)

    if self.out_dim == inputs.shape[-1]:
      x_res = x
    else:
      x_res = nn.Dense(
          y.shape[-1],
          name='project_skip',
          dtype=self.dtype,
          precision=self.precision,
          use_bias=self.use_bias)(x_norm)  # Reference uses x_norm here!
    return y + x_res


class MultiscaleEncoder(nn.Module):
  """Encoder for multiscale transformer."""
  depth: int
  width: int
  pooling_layers: Sequence[int]
  pooling_kernel: tuple[int, ...]
  pooling_strides_q: tuple[int, ...]
  initial_kv_pooling_strides: tuple[int, ...]
  num_initial_heads: int = 1
  use_bias: bool = False
  pool_q_every_layer: bool = True
  dtype: jnp.dtype = jnp.float32
  precision: lax.Precision | None = None

  @nn.compact
  def __call__(self, x: jnp.ndarray):
    """Applies Transformer model on the input x."""

    assert x.ndim == 3, f'Wrong dims: {x.shape}'  # Shape is [N, T*W*H, C].
    skip_outs = {}
    num_heads = self.num_initial_heads
    stride_kv = self.initial_kv_pooling_strides[:]
    for layer_idx in range(self.depth):
      layer_input_shape = x.shape
      out_dim = x.shape[-1]
      pool_q = self.pooling_kernel
      if layer_idx in self.pooling_layers:
        num_heads *= 2
        stride_kv = [(x // 2) if x > 1 else 1 for x in stride_kv]
        stride_q = self.pooling_strides_q
        skip_outs[layer_idx] = x
      else:
        stride_q = [1 for _ in self.pooling_strides_q]
        if not self.pool_q_every_layer:
          pool_q = [1 for _ in self.pooling_strides_q]

      if layer_idx + 1 in self.pooling_layers:
        out_dim = x.shape[-1] * 2

      x = EncoderBlock(
          num_heads=num_heads,
          out_dim=out_dim,
          pooling_kernel_q=pool_q,
          pooling_kernel_kv=self.pooling_kernel,
          pooling_stride_q=stride_q,
          pooling_stride_kv=stride_kv,
          name=f'block_{layer_idx}',
          use_bias=self.use_bias,
          dtype=self.dtype,
          precision=self.precision,
          )(x)
      logging.info('encoder block %d: input shape %s; output shape %s',
                   layer_idx, layer_input_shape, x.shape)

    x_norm = nn.LayerNorm(use_bias=self.use_bias, name='encoder_norm')(x)
    return x_norm, skip_outs


class MultiscaleDecoder(nn.Module):
  """Decoder for multiscale transformer."""
  depth: int
  width: int
  pooling_layers: Sequence[int]
  qkv_tile_reps: tuple[int, ...]
  num_initial_heads: int
  use_bias: bool = False
  use_residuals: bool = True
  dtype: jnp.dtype = jnp.float32
  precision: lax.Precision | None = None

  @nn.compact
  def __call__(self, x: jnp.ndarray, skips: Any):
    assert x.ndim == 3, f'Wrong dims: {x.shape}'  # Shape is [N, T*W*H, C].
    num_heads = self.num_initial_heads
    for layer_idx in reversed(range(self.depth)):
      layer_input_shape = x.shape
      out_dim = x.shape[-1]
      if layer_idx in self.pooling_layers:
        tile_reps = self.qkv_tile_reps
      else:
        tile_reps = (1 for _ in self.qkv_tile_reps)

      if layer_idx + 1 in self.pooling_layers:
        out_dim = x.shape[-1] // 2

      num_heads = x.shape[-1] // self.width

      x = DecoderBlock(
          num_heads=num_heads,
          out_dim=out_dim,
          qkv_tile_reps=tile_reps,
          name=f'decoder_block_{layer_idx}',
          use_bias=self.use_bias,
          dtype=self.dtype,
          precision=self.precision,
          )(x)

      if layer_idx in self.pooling_layers and self.use_residuals:
        x = x + skips[layer_idx]
        logging.info('Adding residual connection in layer %d with shape %s',
                     layer_idx, x.shape)
        assert x.shape == skips[layer_idx].shape, skips[layer_idx].shape

      if layer_idx - 1 in self.pooling_layers:
        num_heads //= 2
      logging.info('decoder block %d: input shape %s; output shape %s',
                   layer_idx, layer_input_shape, x.shape)

    x_norm = nn.LayerNorm(use_bias=self.use_bias, name='decoder_norm')(x)
    return x_norm


class AddPosEmbs(nn.Module):
  """Adds learned positional embeddings to the inputs.

  Attributes:
    posemb_init: Positional embedding initializer.

  Returns:
    Output in shape `[bs, timesteps, in_dim]`.
  """
  posemb_init: Any = nn.initializers.normal(stddev=0.02)  # From BERT.

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    # Inputs.shape is (batch_size, seq_len, emb_dim).
    assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                              ' but it is: %d' % inputs.ndim)
    posemb_shape = (1, inputs.shape[1], inputs.shape[2])
    posemb = self.param(
        'pos_embedding', self.posemb_init, posemb_shape, inputs.dtype
    )
    return inputs + posemb


class Model(nn.Module):
  """The multiscale transformer model."""
  num_layers: int
  num_heads: int
  depth: int
  width: int
  pooling_layers: tuple[int, ...]
  pooling_kernel: tuple[int, ...]
  pooling_strides_q: tuple[int, ...]
  initial_kv_pooling_strides: tuple[int, ...]
  qkv_tile_reps: tuple[int, ...]
  processor_config: ml_collections.FrozenConfigDict
  num_initial_heads: int = 1
  use_residuals: bool = True
  use_bias: bool = False
  pool_q_every_layer: bool = True
  dtype: jnp.dtype = jnp.float32
  precision: lax.Precision | None = None
  freeze_encoder: bool = False
  mean_after_decoder: bool = False

  @nn.compact
  def __call__(self,
               inputs: jnp.ndarray,
               z_rng: jax.Array,
               *,
               debug: bool = False):
    """Applies Transformer model on the input x."""
    dtype = jax.dtypes.canonicalize_dtype(self.dtype)
    # input shape is (batch_size, seq_len, patch_dimension) which is
    # projected to (batch_size, seq_len, width)
    x = inputs
    aux = {}
    assert x.ndim == 3, x.shape
    assert x.dtype == self.dtype, x.dtype
    x = nn.Dense(self.width, name='embedding')(x)

    # use the regular transformer encoder
    x = AddPosEmbs(
        posemb_init=nn.initializers.normal(stddev=0.02),
        name='encoder_posembed')(x)
    encoded = x
    assert encoded.shape == inputs.shape[:2] + (
        self.width,), encoded.shape
    logging.info('Shape after encoder: %s', encoded.shape)

    # do the multiscale transformer now.
    if self.depth > 0:
      x = encoded
      x, skips = MultiscaleEncoder(
          depth=self.depth,
          width=self.width,
          pooling_layers=self.pooling_layers,
          pooling_strides_q=self.pooling_strides_q,
          pooling_kernel=self.pooling_kernel,
          initial_kv_pooling_strides=self.initial_kv_pooling_strides,
          num_initial_heads=self.num_initial_heads,
          use_bias=self.use_bias,
          pool_q_every_layer=self.pool_q_every_layer,
          dtype=dtype,
          precision=self.precision,
          name='multiscale_encoder')(x)
      logging.info('Shape after multiscale encoder: %s', x.shape)

      if self.processor_config.num_samples > 0:
        # processor_input = x
        if not self.processor_config.use_transformer:
          x = x.reshape((inputs.shape[0], -1))
        logging.info('Beginning processor: %s', x.shape)
        x, aux = LatentSDE(
            model_config=self.processor_config)(x, z_rng)
        if not self.mean_after_decoder:
          x = x.mean(axis=1)
        logging.info('After processor: %s', x.shape)
        # x = x.reshape(processor_input.shape)  # only works for num_samples = 1
        # shaped (batch_size, num_samples, seq_len, emb) take a mean of the
        # num samples
      decoder_fn = lambda y: MultiscaleDecoder(
          depth=self.depth,
          width=self.width,
          num_initial_heads=(
              self.num_initial_heads * (2 ** len(self.pooling_layers))),
          qkv_tile_reps=self.qkv_tile_reps,
          pooling_layers=self.pooling_layers,
          use_bias=self.use_bias,
          use_residuals=self.use_residuals,
          dtype=dtype,
          precision=self.precision,
          name='multiscale_decoder')(y, skips=skips)
      if not self.mean_after_decoder:
        x = decoder_fn(x)
      else:
        x = jax.vmap(decoder_fn, in_axes=1, out_axes=1)(x)
        x = x.mean(axis=1)
      logging.info('Shape after multiscale decoder: %s', x.shape)
      decoder_input = encoded + x
    else:
      decoder_input = encoded

    x = nn.Dense(
        inputs.shape[-1],
        name='decoded_patches',
        kernel_init=nn.initializers.zeros,
        bias_init=nn.initializers.normal(stddev=1e-6),
    )(x)

    logging.info('Shape after decoder: %s', x.shape)

    assert x.dtype == self.dtype, x.dtype
    batch_size = inputs.shape[0]
    for key in ['kl_path', 'kl_q0', 'z0_means', 'z1_means', 'z1_stds']:
      if key not in aux:
        aux[key] = jnp.zeros(batch_size, dtype=jnp.float32)

    return x, aux


def _divide_no_nan(x, y):
  is_zero = jnp.isclose(y, jnp.zeros_like(y))
  return jnp.where(is_zero, jnp.zeros_like(x), x / y)


class MLP(nn.Module):
  """MLP module."""
  features: tuple[int, ...]
  activation_fn: Callable[[jax.Array], jax.Array] = nn.gelu
  final_activation_fn: Callable[[jax.Array], jax.Array] = lambda x: x
  final_kernel_init_scale: float | None = 0.1
  bias_stddev: float = 1e-6

  @nn.compact
  def __call__(self, x):
    for feat in self.features[:-1]:
      x = nn.Dense(
          feat,
          use_bias=True,
          kernel_init=nn.initializers.zeros,
          bias_init=nn.initializers.normal(stddev=self.bias_stddev),
      )(x)
      x = self.activation_fn(x)

    # Set the final kernel init to be closer to zero so as to prevent blowups.
    final_kernel_init = (
        nn.initializers.zeros
        # if self.final_kernel_init_scale is None
        # else nn.initializers.normal(stddev=self.final_kernel_init_scale)
    )
    x = nn.Dense(
        self.features[-1],
        use_bias=True,
        kernel_init=final_kernel_init,
        bias_init=nn.initializers.normal(stddev=self.bias_stddev),
    )(x)
    return self.final_activation_fn(x)


class Drift(nn.Module):
  """Parameterizes the drift of a latent SDE."""
  features: tuple[int, ...]

  def setup(self):
    # use tanh as the activation function as gelu causes instability
    self.mlp = MLP(self.features, final_activation_fn=nn.tanh)

  @nn.compact
  def __call__(self, x, context=None):
    assert x.shape[-1] == self.features[-1], x.shape
    # drift function doesn't take any batch axis, instead uses an outer vmap
    assert x.ndim == 1, x.shape

    if context is not None:
      assert context.ndim == 1, context.shape
      x = jnp.concatenate([x, context], axis=0)
    return self.mlp(x)


class Diffusion(nn.Module):
  """Parameterizes the diagonal diffusion of a latent SDE."""
  features: tuple[int, ...]
  ndim: int

  def setup(self):
    # `ndim` independent MLPs where the ith MLP depends only on the ith
    # coordinate of x. This is requirement for strongly diagonal noise.
    # Ensure positivity of the output diffusion value, using jnp.exp as the
    # final activation. Also use tanh in other activations to make the SDE
    # training more stable.
    self.mlps = [
        MLP(self.features, activation_fn=nn.gelu, final_activation_fn=jnp.exp,
            final_kernel_init_scale=0.01)
        for _ in range(self.ndim)
    ]

  @nn.compact
  def __call__(self, x):
    assert x.ndim == 1, f'Got array of rank {x.ndim} instead of 1'
    assert len(x) == self.ndim, (
        f'Expected len(x) to be equal to ndim. Got {len(x)} != {self.ndim}')
    ys = [self.mlps[i](x[i][jnp.newaxis]) for i in range(self.ndim)]
    y = jnp.concatenate(ys, axis=0)
    return y


class VariationalDriftDiffusion(nn.Module):
  """Combined Drift and Diffusion parameterization along with KL term."""
  prior_drift_features: tuple[int, ...]
  post_drift_features: tuple[int, ...]
  diffusion_features: tuple[int, ...]

  def setup(self):
    assert self.diffusion_features[-1] == 1, (
        'For diagonal diffusion, `diffusion_features` must map each '
        'coordinate independently and should have output size 1; got '
        f'f{self.diffusion_features}.')
    assert self.post_drift_features[-1] == self.prior_drift_features[-1]
    self.latent_size = self.post_drift_features[-1]
    self.prior_drift_mlp = Drift(features=self.prior_drift_features)
    self.post_drift_mlp = Drift(features=self.post_drift_features)
    self.diffusion_mlp = Diffusion(
        features=self.diffusion_features, ndim=self.latent_size)

  @nn.compact
  def __call__(self, state, ts, dw, context):
    del ts
    assert dw.shape[-1] == self.latent_size, (
        f'Unexpected last dim in {dw.shape}. Expected {self.latent_size}')
    assert state.shape == (self.latent_size + 1,), (
        f'Got {state.shape} vs {(self.latent_size + 1,)}')

    # state is made up of [z; logqp].
    z = state[:self.latent_size]

    # Only the posterior drift receives the `context` as input.
    post_drift = self.post_drift_mlp(z, context)
    prior_drift = self.prior_drift_mlp(z)
    diffusion = self.diffusion_mlp(z)

    # KL term.
    logqp = .5 * jnp.square(_divide_no_nan(
        post_drift - prior_drift, diffusion)).sum(axis=0)[jnp.newaxis]
    # logqp should be scalar.
    assert logqp.shape == (1,)

    aug_drift = jnp.concatenate([post_drift, logqp], axis=-1)
    aug_diffusion = jnp.concatenate([
        dw * diffusion, jnp.zeros((1,), dtype=jnp.float32)], axis=-1)
    aug_diffusion = jnp.zeros_like(aug_diffusion)
    return aug_drift, aug_diffusion


class Dynamics(nn.Module):
  """Combined Drift and Diffusion parameterization along with KL term."""
  num_layers: int
  latent_size: int
  hidden_size: int = 32

  def transformer_block(self, x, name):
    for lyr in range(self.num_layers):
      x = Encoder1DBlock(
          mlp_dim=self.latent_size,
          num_heads=2,
          activation_fn=nn.gelu, # nn.tanh,
          name=f'{name}_dynamics_block_{lyr}',
          dtype=jnp.float32)(x)
    x = nn.LayerNorm(name=f'{name}_dynamics_norm')(x)
    x = nn.Dense(self.latent_size, name=f'{name}_decoded_patches',
                 kernel_init=nn.initializers.zeros,
                 bias_init=nn.initializers.normal(stddev=1e-6))(x)
    return x

  @nn.compact
  def __call__(self, state, ts, dw, ctx):
    assert state.ndim == 1, state.shape

    # the state is made up of (seq_len, latent_size) and 1 extra dim for the KL
    # term
    assert state.shape[0] % self.latent_size == 1, state.shape
    latent_x = state[:-1]
    # transformer block requires a 'batch' dimension so just prepend a dim
    x = latent_x.reshape((1, -1, self.latent_size))
    seq_len = x.shape[1]
    expanded_latent_size = seq_len * self.latent_size

    logging.info('dynamics ts: %s', ts.shape)
    logging.info('Before dynamics attention layers: %s', state.shape)
    ts = jnp.tile(ts[None, None, None], [1, 1, self.latent_size])
    x = jnp.concatenate([x, ts], axis=-2)

    # for posterior drift, also tack on the context to get a longer sequence
    ctx = ctx.reshape((1, seq_len, self.latent_size))
    x_with_ctx = jnp.concatenate([x, ctx], axis=-2)

    logging.info('dynamics transformer block input: %s', x_with_ctx.shape)
    post_drift = self.transformer_block(x_with_ctx, 'posterior_drift')
    prior_drift = self.transformer_block(x, 'prior_drift')
    logging.info('dynamics transformer post_drift: %s', post_drift.shape)
    logging.info('dynamics transformer prior_drift: %s', prior_drift.shape)

    # flatten, remove the time token and context
    post_drift = post_drift[0, :seq_len, :].reshape(-1)
    prior_drift = prior_drift[0, :seq_len, :].reshape(-1)
    assert post_drift.shape == (expanded_latent_size,), post_drift.shape
    assert prior_drift.shape == (expanded_latent_size,), prior_drift.shape
    features = (self.hidden_size,) * self.num_layers + (1,)
    diffusion = MLP(
        features,
        activation_fn=nn.tanh,
        final_activation_fn=jnp.exp,
        name='diffusion_mlp')(latent_x[:, None]).reshape(-1)
    assert diffusion.shape == (expanded_latent_size,), diffusion.shape

    logqp = .5 * jnp.square(_divide_no_nan(
        post_drift - prior_drift, diffusion)).sum(axis=0)[jnp.newaxis]
    logging.info('dynamics logqp: %s', logqp.shape)

    # drift, diffusion
    logging.info('dynamics diffusion: %s', diffusion.shape)

    aug_drift = jnp.concatenate([post_drift, logqp], axis=-1)
    aug_diffusion = jnp.concatenate([
        dw * diffusion, jnp.zeros((1,), dtype=jnp.float32)], axis=-1)

    assert aug_drift.shape == state.shape, aug_drift.shape
    assert aug_diffusion.shape == state.shape, aug_diffusion.shape
    return aug_drift, aug_diffusion


class LatentSDE(nn.Module):
  """Latent SDE model."""
  model_config: ml_collections.ConfigDict

  def setup(self):
    self.num_gridpoints = self.model_config.num_gridpoints
    self.latent_size = self.model_config.latent_size
    self.context_size = self.model_config.context_size
    self.state_size = self.model_config.data_size
    self.post_input_size = 2 * self.latent_size + self.context_size
    self.prior_scale = self.model_config.prior_scale
    self.use_transformer = self.model_config.use_transformer

    num_layers = self.model_config.num_layers
    # Make these transformers?
    self.encoder_mlp = MLP(  # pytype: disable=wrong-arg-types
        (self.model_config.hidden_size,) * num_layers + (self.post_input_size,),
        activation_fn=nn.gelu,
        final_activation_fn=nn.gelu, name='sde_encoder_mlp')
    self.decoder_mlp = MLP(  # pytype: disable=wrong-arg-types
        (self.model_config.hidden_size,) * num_layers + (self.state_size,),
        activation_fn=nn.gelu,
        final_activation_fn=nn.gelu, name='sde_decoder_mlp')
    self.num_samples = self.model_config.num_samples

    num_layers = self.model_config.num_sde_layers
    prior_drift_features = (self.model_config.hidden_size,) * num_layers + (
        self.latent_size,)
    post_drift_features = [self.model_config.hidden_size, self.latent_size]
    diffusion_features = (self.model_config.hidden_size,) * num_layers + (1,)
    self.sde_net = nn_sdeint(VariationalDriftDiffusion)(
        prior_drift_features, post_drift_features, diffusion_features)

  def encode(self, inputs):
    """Encodes the first `enc_length` sequence elements into a latent state."""
    batch_size = inputs.shape[0]
    assert inputs.shape == (batch_size, self.state_size)

    hidden = self.encoder_mlp(inputs)
    assert hidden.shape == (batch_size, self.post_input_size)

    q0_mean, q0_logstd, context = jnp.split(
        hidden, [self.latent_size, 2 * self.latent_size], axis=-1)
    q0_dist = distrax.MultivariateNormalDiag(q0_mean, jnp.exp(q0_logstd))
    return q0_dist, context

  def sample(self, z0, context, rng):
    batch_size = z0.shape[0]
    assert z0.shape == (
        batch_size, self.num_samples, self.latent_size), z0.shape
    assert context.shape == (batch_size, self.context_size), context.shape

    # squash the batch shape to (batch_size * num_samples)
    expanded_batch_size = batch_size * self.num_samples
    z0 = jnp.reshape(z0, (expanded_batch_size, self.latent_size))
    context = jnp.broadcast_to(
        context[:, jnp.newaxis, :],
        (batch_size, self.num_samples, self.context_size))
    context = jnp.reshape(context, (expanded_batch_size, self.context_size))

    # concatenate one scalar for every SDE trajectory that is used to compute
    # the KL divergence of each trajectory.
    init_state = jnp.concatenate(
        [z0, jnp.zeros((expanded_batch_size, 1), dtype=jnp.float32)], axis=-1)

    # obtain the brownian path to be fed into the SDE solver
    dw_rngs = jax.random.split(rng, num=expanded_batch_size * self.latent_size)
    del rng
    dw = jax.vmap(
        partial(brownian_path, n=self.num_gridpoints), out_axes=-1)(dw_rngs)
    dw = dw.reshape(
        (expanded_batch_size, self.num_gridpoints, self.latent_size))

    # time points (0, 1) indicate that we want the output at state 1 only.
    ts = jnp.broadcast_to(
        jnp.array([0.0, 1.0], dtype=jnp.float32)[jnp.newaxis, :],
        (expanded_batch_size, 2),
    )

    states = jax.vmap(self.sde_net)(init_state, ts, dw, context)

    z1, kl_path = jnp.split(states, [self.latent_size], axis=-1)
    assert z1.shape == (expanded_batch_size, 1, self.latent_size), z1.shape
    assert kl_path.shape == (expanded_batch_size, 1, 1), kl_path.shape

    z1 = z1[:, 0, :].reshape((batch_size, self.num_samples, self.latent_size))
    kl_path = kl_path[:, 0, 0].reshape((batch_size, self.num_samples))
    return z1, kl_path.sum(axis=-1)

  def sample_transformer(self, z0):
    batch_size = z0.shape[0]
    latent_size = z0.shape[-1]
    dw = jnp.zeros(
        (batch_size, self.num_gridpoints, latent_size), dtype=jnp.float32)
    ts = jnp.broadcast_to(
        jnp.array([0., 1.], dtype=jnp.float32)[jnp.newaxis, :], (batch_size, 2))
    sde_nn = nn_sdeint(Dynamics)(num_layers=self.model_config.num_sde_layers,
                                 latent_size=self.latent_size)
    context = z0
    z1 = jax.vmap(sde_nn)(z0, ts, dw, context)
    return z1

  def sample_sde(self, z0, rng):
    assert z0.ndim == 2, z0.shape
    batch_size = z0.shape[0]
    # expanded latent size = latent_size * seq_len
    # expanded batch size = batch_size * num_samples
    expanded_latent_size = z0.shape[-1]
    expanded_batch_size = batch_size * self.num_samples
    # z0 = jnp.tile(z0[:, None, :], [1, self.num_samples, 1]).reshape(
    #     (expanded_batch_size, expanded_latent_size))
    dw_rng, z0_rng = jax.random.split(rng)
    # z0 is num_samples independent samples of a Gaussian, centered at z0
    # with scale
    q0_dist = distrax.MultivariateNormalDiag(
        z0, self.prior_scale * jnp.ones_like(z0))
    p0_dist = distrax.MultivariateNormalDiag(
        jnp.zeros_like(z0), self.prior_scale * jnp.ones_like(z0))
    z0 = q0_dist.sample(seed=z0_rng, sample_shape=self.num_samples)
    # distrax prepends the `num_samples` dimension in front of the batch
    # dimension; so transpose from (num_samples, batch_size, latent_size) to
    # (batch_size, num_samples, latent_size)
    z0 = jnp.transpose(z0, axes=(1, 0, 2))
    assert z0.shape == (
        batch_size, self.num_samples, expanded_latent_size), z0.shape
    z0 = z0.reshape((expanded_batch_size, expanded_latent_size))
    kl_q0 = q0_dist.kl_divergence(p0_dist)

    dw_rngs = jax.random.split(
        dw_rng, num=expanded_batch_size * expanded_latent_size)
    dw = jax.vmap(
        partial(brownian_path, n=self.num_gridpoints), out_axes=-1)(dw_rngs)
    dw = dw.reshape((expanded_batch_size, self.num_gridpoints,
                     expanded_latent_size))
    ts = jnp.broadcast_to(
        jnp.array([0., 1.], dtype=jnp.float32)[jnp.newaxis, :],
        (expanded_batch_size, 2))
    sde_nn = nn_sdeint(Dynamics)(num_layers=self.model_config.num_sde_layers,
                                 latent_size=self.latent_size)
    context = z0
    # append another dim to z0 to track the KL term
    aug_z0 = jnp.concatenate(
        [z0, jnp.zeros((expanded_batch_size, 1), dtype=jnp.float32)], axis=-1)
    aug_z1 = jax.vmap(sde_nn)(aug_z0, ts, dw, context)
    # the middle 1 dimension is due to len(ts) - 1 time points.
    assert aug_z1.shape == (expanded_batch_size, 1, expanded_latent_size + 1), (
        aug_z1.shape)
    z1 = aug_z1[:, 0, :expanded_latent_size]
    kl_path = aug_z1[:, 0, -1]

    assert z1.shape == z0.shape, f'{z0.shape} != {z0.shape}'
    z1 = z1.reshape((batch_size, self.num_samples, expanded_latent_size))
    kl_path = kl_path.reshape((batch_size, self.num_samples))
    kl_q0 = kl_q0.reshape((batch_size,))
    logging.info('kl path shape: %s', kl_path.shape)
    logging.info('kl q0 shape: %s', kl_q0.shape)
    return z1, kl_path, kl_q0

  @nn.compact
  def __call__(self, inputs, rng):
    batch_size = inputs.shape[0]
    if self.use_transformer:
      # don't encode, just use the latent state as is.
      seq_len = inputs.shape[1]
      # squash the seq_len and latent size together
      assert inputs.shape == (
          batch_size, seq_len, self.latent_size), inputs.shape
      z0 = inputs.reshape((batch_size, -1))
      kl = None
      if self.num_samples > 1:
        logging.info('using SDE sampler for num samples: %d', self.num_samples)
        z1, kl_path, kl_q0 = self.sample_sde(z0, rng)
        assert kl_path.shape == (batch_size, self.num_samples,), kl_path.shape
        assert kl_q0.shape == (batch_size,), kl_q0.shape
        kl = kl_path.mean(axis=-1) + kl_q0  # mean across samples
      else:
        logging.info('using ODE sampler for num samples: %d', self.num_samples)
        z1 = self.sample_transformer(z0)

      assert z1.shape == (
          batch_size, self.num_samples, seq_len * self.latent_size), z1.shape
      z1 = z1.reshape((batch_size, self.num_samples, seq_len, self.latent_size))

      z0_means = z0.reshape((batch_size, -1)).mean(axis=-1)
      z1_means = z1.reshape((batch_size, -1)).mean(axis=-1)
      # z1_stds = jnp.std(z1, axis=1).reshape((batch_size, -1)).mean(axis=-1)

      # kl_q0 = jnp.zeros((batch_size,), dtype=jnp.float32)
      # kl_path = jnp.zeros((batch_size,), dtype=jnp.float32)
      aux = {
          'kl_q0': kl_q0,
          'kl_path': kl_path.mean(axis=-1),
          'z0_means': z0_means,
          'z1_means': z1_means,
          # 'z1_stds': z1_stds,
      }

      return z1, aux

    assert inputs.shape == (batch_size, self.state_size), inputs.shape
    q0_dist, context = self.encode(inputs)

    q0_rng, dw_rng = jax.random.split(rng)
    # z0 = q0_dist.sample(seed=q0_rng, sample_shape=self.num_samples)
    # make it deterministic
    z0 = q0_dist.mean()[jnp.newaxis, :, :]
    # distrax prepends the `num_samples` dimension in front of the batch
    # dimension; so transpose from (num_samples, batch_size, latent_size) to
    # (batch_size, num_samples, latent_size)
    z0 = jnp.transpose(z0, axes=(1, 0, 2))
    assert z0.shape == (
        batch_size, self.num_samples, self.latent_size), z0.shape

    p0_dist = distrax.MultivariateNormalDiag(
        jnp.zeros(self.latent_size, dtype=jnp.float32),
        self.prior_scale * jnp.ones(self.latent_size, dtype=jnp.float32),
    )
    kl_q0 = q0_dist.kl_divergence(p0_dist)
    assert kl_q0.shape == (batch_size,), kl_q0.shape

    z1, kl_path = self.sample(z0, context, dw_rng)
    assert z1.shape == (
        batch_size, self.num_samples, self.latent_size), z1.shape
    assert kl_path.shape == (batch_size,), kl_path.shape

    y = self.decoder_mlp(z1)
    assert y.shape == (batch_size, self.num_samples, self.state_size), y.shape
    aux = {'kl_q0': kl_q0, 'kl_path': kl_path}
    return y, aux


class Encoder1DBlock(nn.Module):
  """Transformer encoder layer.

  Attributes:
    mlp_dim: Dimension of the mlp on top of attention block.
    num_heads: Number of self-attention heads.
    dtype: The dtype of the computation (default: float32).

  Returns:
    output after transformer encoder block.
  """
  mlp_dim: int
  num_heads: int
  dtype: Any = jnp.float32
  activation_fn: Any = nn.gelu

  @nn.compact
  def __call__(self, inputs: jax.Array) -> jax.Array:
    """Applies Encoder1DBlock module.

    Args:
      inputs: Input data.

    Returns:
      Output after transformer encoder block.
    """
    # Attention block.
    assert inputs.ndim == 3
    x = nn.LayerNorm(dtype=self.dtype)(inputs)
    x = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform())(x, x)
    x = x + inputs

    # MLP block.
    y = nn.LayerNorm(dtype=self.dtype)(x)
    y = MlpBlock(mlp_dim=self.mlp_dim)(y)
    return y + x

