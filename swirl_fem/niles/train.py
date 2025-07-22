# Copyright 2025 The swirl_fem Authors.
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

"""Trainer module."""

from collections.abc import Callable
from functools import partial  # pylint: disable=g-importing-member
import itertools
import math
import time
from typing import Any

from absl import logging
from clu import metric_writers
from clu import parameter_overview
from clu import periodic_actions
from flax import jax_utils
from flax.training import checkpoints
from flax.training import common_utils
from flax.training import train_state
import jax
from jax import lax
from jax import random
from jax import vmap
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import scipy.spatial
from swirl_fem.common import premesh_commons
from swirl_fem.core.interpolation import Nodes1D
from swirl_fem.core.interpolation import NodeType
from swirl_fem.core.mesh_refiner import refine_premesh
from swirl_fem.navier_stokes import navier_stokes
from swirl_fem.niles import input_pipeline
from swirl_fem.niles import transformer
import tensorflow_datasets as tfds


def transfer_perm(source_mesh, target_mesh):
  """Computes permutation of nodes to transfer from source to target mesh."""
  kdtree = scipy.spatial.KDTree(source_mesh.node_coords)
  perm = []

  for target_node_idx in range(target_mesh.num_nodes):
    _, source_node_idx = kdtree.query(target_mesh.node_coords[target_node_idx])
    perm.append(source_node_idx)
  return np.array(perm, dtype=np.int32)


def get_tke(u, sem, uniform_mesh, first_order_perm, config):
  """Computes the turbulent kinetic energy (TKE) of a velocity field."""
  n = int(math.sqrt(config.num_nodes))
  u_flat = jax.vmap(partial(uniform_mesh.interpolate,
                            source_mesh=sem.velocity.mesh),
                    in_axes=-1, out_axes=-1)(u)
  u_flat = u_flat[first_order_perm]
  u_flat = u_flat.reshape((n, n, 2))

  def _get_fft(x):
    return jnp.abs(jnp.fft.fftshift(jnp.fft.fftn(x)))

  u_hat = jax.vmap(_get_fft, in_axes=-1, out_axes=-1)(u_flat)
  tke = .5 * jnp.square(u_hat).sum(axis=-1)
  return tke

def get_energy_spectrum(tke, config):
  """Compute the energy spectrum of a TKE field."""
  n = int(math.sqrt(config.num_nodes))
  dx = 1 / n
  freqs = np.fft.fftshift(np.fft.fftfreq(n, dx))
  kx, ky = np.meshgrid(freqs, freqs)
  k = np.sqrt(kx**2 + ky**2)
  num_bins = 20
  bins = np.linspace(0, np.max(k), num=num_bins)
  indices = np.digitize(k, bins)

  spectrum = []
  for i in range(1, num_bins - 1):
    mask = indices == i
    spectrum.append((tke * mask).sum())
  return bins[1:-1], jnp.stack(spectrum)


def create_model(model_name: str,
                 model_config: ml_collections.FrozenConfigDict):
  """Create a model based on the model name and config."""
  if model_name == 'multiscale_transformer':
    return transformer.Model(
        num_layers=model_config.num_layers,
        num_heads=model_config.num_heads,
        depth=model_config.depth,
        width=model_config.width,
        use_residuals=model_config.use_residuals,
        freeze_encoder=model_config.freeze_encoder,
        mean_after_decoder=model_config.mean_after_decoder,
        processor_config=model_config.processor_config,
        num_initial_heads=model_config.num_initial_heads,
        pooling_layers=model_config.pooling_layers,
        pooling_kernel=model_config.pooling_kernel,
        pooling_strides_q=model_config.pooling_strides_q,
        initial_kv_pooling_strides=model_config.initial_kv_pooling_strides,
        qkv_tile_reps=model_config.qkv_tile_reps)
  raise ValueError('Unsupported model')


def initialized(key, num_elements, num_channels, model):
  input_shape = (1, num_elements, num_channels)
  init_key, z_rng = jax.random.split(key)
  @jax.jit
  def init(*args):
    return model.init(*args)
  variables = init(
      {'params': init_key}, jnp.ones(input_shape, dtype=model.dtype), z_rng
  )
  return variables['params']


def _kolmogorov_forcing(config, x, u):
  """Forcing term for the Kolmogorov flow."""
  assert x.shape == (2,)
  assert u.shape == (2,)
  k = 4.
  f0 = jnp.sin(2 * np.pi * k * x[1])
  f1 = 0.
  return jnp.array([f0, f1]) - config.drag_coeff * u


def _solve_one_step(us: tuple[jax.Array, ...],
                    ps: tuple[jax.Array, ...], cus: tuple[jax.Array, ...],
                    f: jax.Array,
                    sem: navier_stokes.StokesSEM,
                    config: ml_collections.FrozenConfigDict):
  """Move forward Navier Stokes by one step.

  Args:
    us: The current (and previous `K - 1`) velocity fields, where `K` is the
      time order. Each field is shaped `(num_nodes, ndim)`.
    ps: The current (and previous `K - 1`) pressure fields, where `K` is the
      time order. Each field  is shaped `(num_nodes,)`.
    cus: Advection operator applied to each element of `us`.
    f: The forcing term. Should be shaped `(num_nodes, ndim)`
    sem: The spectral element data structure used to solve the associated Stokes
      system.
    config: The config options for the solver.

  Returns:
    A tuple of the next velocity, pressure and the advection operator applied to
    the velocity.
  """
  # Extrapolate advection term
  ext_coeffs = navier_stokes.extk_coeffs(k=config.time_order - 1)
  cu = sum(ext_coeffs[-i] * cus[-i] for i in range(1, len(ext_coeffs) + 1))

  # Solve the stokes system with extrapolated advection term
  f = f + vmap(partial(_kolmogorov_forcing, config))(
      sem.velocity.mesh.node_coords, us[-1])
  f = -cu + sem.B(f)
  u, p, _ = sem.stokes_one_step(
      us, ps, f, mu=1 / config.reynolds_number, dt=config.dt,
      alpha=config.alpha, time_order=config.time_order, tol=0, atol=1e-7)
  return u, p, sem.C(u)


def make_multiscale_perm(size=12, patch_sizes=(2, 3), factors=(2, 4)):
  """Permute elements so that nearby elements are placed together."""
  def _lex(n):
    return np.array(list(itertools.product(range(n), repeat=2)), dtype=np.int32)

  def _shifted(p, n, shift_factor):
    shifts = _lex(n)
    ps = []
    for shift in shifts:
      ps.append(p + shift_factor * shift)
    return np.concatenate(ps)

  p = _lex(int(size / np.prod(patch_sizes)))
  for ps, factor in zip(patch_sizes, factors):
    p = _shifted(p, n=ps, shift_factor=factor)

  ravel_p = np.array([size * i + j for i, j in list(p)], dtype=np.int32)
  return ravel_p


def compute_mse_loss(batch, params, model_apply_fn: Callable[..., Any],
                     step_rng: jax.Array, kl_penalty: float,
                     sem: navier_stokes.StokesSEM,
                     uniform_mesh: Any | None,
                     first_order_perm: Any | None,
                     config: ml_collections.FrozenConfigDict,
                     train: bool):
  """Compute MSE loss of modeled closure term by rolling out the solver."""
  # separate out the first `time_order` out of stride length which becomes the
  # input to the model and the initial state for the solver.
  # batch is shaped (batch_size, window_size, num_nodes, ...) and we take
  # the first `time_order` dimensions out of the window_size axis.
  us = tuple(
      lax.index_in_dim(batch['u'], i, axis=1, keepdims=False)
      for i in range(config.time_order)
  )
  ps = tuple(
      lax.index_in_dim(batch['p'], i, axis=1, keepdims=False)
      for i in range(config.time_order)
  )
  # apply the non-linear advection operator to the initial states.
  cus = tuple(map(vmap(sem.C), us))
  # cus = tuple(map(partial(jnp.asarray, dtype=jnp.float32), us))
  dropout_rng, z_rng = jax.random.split(step_rng)
  batch_size = us[-1].shape[0]
  kl_q0 = jnp.zeros(batch_size, dtype=jnp.float32)
  kl_path = jnp.zeros(batch_size, dtype=jnp.float32)
  perm = make_multiscale_perm()
  invperm = np.argsort(perm)

  # Consider switching to lax.scan if memory or compile time becomes an issue.
  def body_fn(carry, _):
    i, us, ps, cus, prev_aux = carry
    z_key = jax.random.fold_in(z_rng, i)
    dropout_train_key = jax.random.fold_in(dropout_rng, i)

    inputs = jax.vmap(sem.velocity.gather)(us[-1])
    inputs = inputs.astype(jnp.float32)
    inputs = inputs.reshape(
        (inputs.shape[0],
         sem.velocity.mesh.num_elements,
         sem.velocity.mesh.num_nodes_per_element * sem.velocity.mesh.ndim))
    if config.permute_elements:
      inputs = inputs[:, perm, :]
    assert inputs.shape[1:] == (config.num_elements, config.num_channels)

    if train:
      forcing_term, aux = model_apply_fn(
          {'params': params}, inputs, rngs={'dropout': dropout_train_key},
          z_rng=z_key, train=True)
      if config.num_pushforward_steps > 0:
        forcing_term = jax.lax.cond(
            i < config.num_pushforward_steps, jax.lax.stop_gradient,
            lambda x: x, forcing_term)
    else:
      forcing_term, aux = model_apply_fn(
          {'params': params}, inputs, z_rng=z_key, train=False)

    # 'ungather' from the per-element view to the nodal view.
    if config.permute_elements:
      forcing_term = forcing_term[:, invperm, :]

    forcing_term = forcing_term.reshape((
        forcing_term.shape[0],
        sem.velocity.mesh.num_elements,
        sem.velocity.mesh.num_nodes_per_element,
        sem.velocity.mesh.ndim,
    ))
    forcing_term = jax.vmap(sem.velocity.scatter)(forcing_term)
    u, p, cu = vmap(partial(_solve_one_step, sem=sem, config=config))(
        us, ps, cus, forcing_term)

    # sum up the KL term from each step
    updated_aux = {}
    for key in aux:
      if key == 'kl_path' or key == 'kl_q0':
        updated_aux[key] = aux[key] + prev_aux[key]
      else:
        updated_aux[key] = aux[key]

    us = us[1:] + (u,)
    ps = ps[1:] + (p,)
    cus = cus[1:] + (cu,)
    carry = (i + 1, us, ps, cus, updated_aux)
    return carry, u

  num_solver_steps = config.num_steps if train else config.eval_num_steps
  logging.info('num_solver_steps: %d. train: %d', num_solver_steps, train)
  aux = {
      'kl_q0': jnp.zeros(batch_size, dtype=jnp.float32),
      'kl_path': jnp.zeros(batch_size, dtype=jnp.float32),
      'z0_means': jnp.zeros(batch_size, dtype=jnp.float32),
      'z1_means': jnp.zeros(batch_size, dtype=jnp.float32),
      'z1_stds': jnp.zeros(batch_size, dtype=jnp.float32),
  }
  init_carry = (jnp.array(0, dtype=jnp.int32), us, ps, cus, aux)
  carry, preds = lax.scan(
      body_fn, init_carry, jnp.arange(num_solver_steps, dtype=jnp.int32))
  aux = carry[-1]
  kl_path = aux['kl_path']
  kl_q0 = aux['kl_q0']
  logging.info('aux: %s', jax.tree_map(jnp.shape, aux))  # pytype: disable=module-attr
  preds = jnp.moveaxis(preds, source=0, destination=1)
  logging.info('preds.shape: %s', preds.shape)

  # preds = jnp.stack(preds, axis=1)
  targets = lax.slice_in_dim(
      batch['u'], start_index=config.time_order,
      limit_index=config.time_order + num_solver_steps,
      axis=1)

  # (batch_size, num_steps, num_nodes, ndim)
  mse = optax.l2_loss(predictions=preds, targets=targets)
  mse = mse.sum(axis=(-1, -2))  # sum across num_nodes, ndim
  mse = mse.mean(axis=0)  # mean across batch
  kl_q0 = kl_q0.mean(axis=0)
  kl_path = kl_path.mean(axis=0)
  kl = kl_q0 + kl_path
  # take sum across num steps
  loss = mse.sum(axis=0) + kl_penalty * kl

  if not train:
    # compute TKE
    pred_tkes = vmap(vmap(
        partial(get_tke, sem=sem, uniform_mesh=uniform_mesh,
                first_order_perm=first_order_perm, config=config)))(preds)
    target_tkes = vmap(vmap(
        partial(get_tke, sem=sem, uniform_mesh=uniform_mesh,
                first_order_perm=first_order_perm, config=config)))(targets)
    logging.info('pred_tkes.shape: %s', pred_tkes.shape)
    logging.info('target_tkes.shape: %s', target_tkes.shape)
    # mean across last half of num_steps
    pred_tke = pred_tkes[:, num_solver_steps // 2:].mean(axis=1)
    target_tke = target_tkes[:, num_solver_steps // 2:].mean(axis=1)
    _, pred_spectrum = vmap(partial(get_energy_spectrum, config=config))(
        pred_tke)
    _, target_spectrum = vmap(partial(get_energy_spectrum, config=config))(
        target_tke)
    tke_error = jnp.square(
        jnp.log(pred_spectrum) - jnp.log(target_spectrum)).sum(axis=-1)
  else:
    tke_error = jnp.zeros(batch_size, dtype=jnp.float32)
  aux = {
      'kl_q0': kl_q0,
      'kl_path': kl_path,
      'mse': mse,
      'kl': kl_penalty * kl,
      'z0_means': jnp.abs(aux['z0_means']).mean(axis=0),
      'z1_means': jnp.abs(aux['z1_means']).mean(axis=0),
      'z1_stds': jnp.abs(aux['z1_stds']).mean(axis=0),
      'tke_err': tke_error.mean(axis=0),
  }
  logging.info('aux: %s', jax.tree_map(jnp.shape, aux))  # pytype: disable=module-attr
  return loss, aux


def compute_metrics(loss, aux, train: bool):
  """Compute metrics from loss and aux data structures."""
  metrics = {
      'loss': loss,
      'kl_q0': aux['kl_q0'],
      'kl_path': aux['kl_path'],
      'kl': aux['kl'],
      'mse': aux['mse'].mean(axis=0),
      'z0_means': aux['z0_means'],
      'z1_means': aux['z1_means'],
      'z1_stds': aux['z1_stds'],
  }
  if not train:
    metrics['tke_err'] = aux['tke_err']
    metrics['mse@1to8'] = aux['mse'][:8].mean()
    metrics['mse@8'] = aux['mse'][8 - 1]
    metrics['mse@16'] = aux['mse'][16 - 1]
    metrics['mse@32'] = aux['mse'][32 - 1]
    metrics['mse@64'] = aux['mse'][64 - 1]
    metrics['mse@128'] = aux['mse'][128 - 1]

  logging.info('metrics: %s', jax.tree_map(jnp.shape, metrics))  # pytype: disable=module-attr
  metrics = lax.pmean(metrics, axis_name='batch')
  return metrics


def create_updown_learning_rate_fn(
    config: ml_collections.ConfigDict,
    base_learning_rate: float,
    steps_per_epoch: int):
  """Create learning rate schedule."""
  warmup_fn = optax.linear_schedule(
      init_value=0., end_value=base_learning_rate,
      transition_steps=config.warmup_epochs * steps_per_epoch)
  warmdown_fn = optax.linear_schedule(
      init_value=base_learning_rate, end_value=0,
      transition_steps=config.warmup_epochs * steps_per_epoch)
  schedule_fn = optax.join_schedules(
      schedules=[warmup_fn, warmdown_fn],
      boundaries=[config.warmup_epochs * steps_per_epoch])
  return schedule_fn


def create_learning_rate_fn(
    config: ml_collections.ConfigDict,
    base_learning_rate: float,
    steps_per_epoch: int):
  """Create learning rate schedule."""
  warmup_fn = optax.linear_schedule(
      init_value=0., end_value=base_learning_rate,
      transition_steps=config.warmup_epochs * steps_per_epoch)
  cosine_epochs = max(config.num_epochs - config.warmup_epochs, 1)
  cosine_fn = optax.cosine_decay_schedule(
      init_value=base_learning_rate,
      decay_steps=cosine_epochs * steps_per_epoch)
  schedule_fn = optax.join_schedules(
      schedules=[warmup_fn, cosine_fn],
      boundaries=[config.warmup_epochs * steps_per_epoch])
  return schedule_fn


def create_kl_penalty_fn(
    config: ml_collections.ConfigDict,
    steps_per_epoch: int) -> optax.Schedule:
  """Obtain the KL schedule from the config."""
  kl_scheduler = optax.linear_schedule(
      init_value=0.0,
      end_value=config.kl_penalty,
      transition_steps=config.kl_transition_epochs * steps_per_epoch,
  )
  kl_scheduler = optax.join_schedules(
      schedules=[optax.constant_schedule(value=0.), kl_scheduler],
      boundaries=[config.kl_zero_epochs * steps_per_epoch],
  )
  return kl_scheduler


def train_step(state, batch, learning_rate_fn, kl_penalty_fn,
               sem: navier_stokes.StokesSEM,
               train_rng: jax.Array,
               config: ml_collections.FrozenConfigDict):
  """Perform a single training step."""
  step_rng = jax.random.fold_in(train_rng, state.step)
  kl_penalty = kl_penalty_fn(state.step)

  def loss_fn(params):
    """loss function used for training."""
    loss, aux = compute_mse_loss(
        batch, params, model_apply_fn=state.apply_fn,
        step_rng=step_rng,
        kl_penalty=kl_penalty,
        sem=sem, uniform_mesh=None,
        first_order_perm=None,
        config=config, train=True)
    return loss, aux

  lr = learning_rate_fn(state.step)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  loss_and_aux, grads = grad_fn(state.params)
  # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
  grads = lax.pmean(grads, axis_name='batch')
  loss, aux = loss_and_aux
  metrics = compute_metrics(loss, aux, train=True)
  metrics['learning_rate'] = lr
  metrics['kl_penalty'] = kl_penalty

  new_state = state.apply_gradients(grads=grads)
  return new_state, metrics


def eval_step(state, batch, kl_penalty_fn, sem: navier_stokes.StokesSEM,
              uniform_mesh: Any, first_order_perm: np.ndarray,
              eval_rng: jax.Array,
              config: ml_collections.FrozenConfigDict):
  """Perform a single evaluation step."""
  step_rng = jax.random.fold_in(eval_rng, state.step)
  kl_penalty = kl_penalty_fn(state.step)
  loss, aux = compute_mse_loss(
      batch, params=state.params, model_apply_fn=state.apply_fn,
      step_rng=step_rng,
      kl_penalty=kl_penalty,
      sem=sem, uniform_mesh=uniform_mesh, first_order_perm=first_order_perm,
      config=config, train=False)
  return compute_metrics(loss, aux, train=False)


def prepare_tf_data(xs):
  """Convert a input batch from tf Tensors to numpy arrays."""
  local_device_count = jax.local_device_count()
  def _prepare(x):
    # Use _numpy() for zero-copy conversion between TF and NumPy.
    x = x._numpy()  # pylint: disable=protected-access

    # reshape (host_batch_size, ...) to (local_devices, device_batch_size, ...)
    return x.reshape((local_device_count, -1) + x.shape[1:])

  return jax.tree_util.tree_map(_prepare, xs)


def create_input_iter(batch_size, train, config):
  ds = input_pipeline.create_split(
      batch_size=batch_size, train=train, config=config)
  it = map(prepare_tf_data, ds)
  it = jax_utils.prefetch_to_device(it, 2)
  return it


def restore_checkpoint(state, workdir):
  return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state, workdir):
  state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
  step = int(state.step)
  logging.info('Saving checkpoint step %d.', step)
  checkpoints.save_checkpoint_multiprocess(workdir, state, step, keep=50)


# pmean only works inside pmap because it needs an axis name.
# This function will average the inputs across all devices.
cross_replica_mean = jax.pmap(lambda x: lax.pmean(x, 'x'), 'x')


def create_train_state(rng, model, learning_rate_fn,
                       updown_learning_rate_fn,
                       config: ml_collections.FrozenConfigDict):
  """Create initial training state."""
  init_key, rng = jax.random.split(rng)
  del rng
  params = initialized(init_key, num_elements=config.num_elements,
                       num_channels=config.num_channels, model=model)
  parameter_overview.log_parameter_overview(params)

  tx = optax.adamw(learning_rate=learning_rate_fn, b1=0.9, b2=0.95, eps=1e-6,
                   weight_decay=config.weight_decay)
  if config.grad_clip_norm is not None:
    tx = optax.chain(optax.clip_by_global_norm(config.grad_clip_norm), tx)

  state = train_state.TrainState.create(
      apply_fn=model.apply,
      params=params,
      tx=tx)
  return state


def train_and_evaluate(config: ml_collections.FrozenConfigDict,
                       workdir: str) -> train_state.TrainState:
  """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.

  Returns:
    Final TrainState.
  """

  writer = metric_writers.create_default_writer(
      logdir=workdir, just_logging=jax.process_index() != 0)

  rng = random.PRNGKey(0)

  if config.batch_size % jax.device_count() > 0:
    raise ValueError('Batch size must be divisible by the number of devices')
  local_batch_size = config.batch_size // jax.process_count()

  train_iter = create_input_iter(local_batch_size, train=True, config=config)
  eval_iter = create_input_iter(local_batch_size, train=False, config=config)

  steps_per_epoch = (
      input_pipeline.get_num_examples(
          tfds.Split.TRAIN, window_size=config.train_window_size,
          window_stride=config.train_window_stride, debug=config.debug
      )
      // config.batch_size
  )
  if config.num_train_steps <= 0:
    num_steps = int(steps_per_epoch * config.num_epochs)
  else:
    num_steps = config.num_train_steps

  if config.steps_per_eval == -1:
    num_validation_examples = (
        input_pipeline.NUM_SPLIT_EXAMPLES[tfds.Split.VALIDATION]
        // config.batch_size
    )
    steps_per_eval = num_validation_examples // config.batch_size
  else:
    steps_per_eval = config.steps_per_eval

  steps_per_checkpoint = int(steps_per_epoch * config.checkpoint_epochs)
  logging.info('Steps per checkpoint: %d', steps_per_checkpoint)
  eval_every_steps = int(steps_per_epoch * config.eval_every_epochs)
  logging.info('Steps per eval: %d', eval_every_steps)

  base_learning_rate = config.learning_rate * config.batch_size / 256.
  logging.info('base_learning_rate: %f', base_learning_rate)

  model = create_model(model_name=config.model_name, model_config=config.model)

  learning_rate_fn = create_learning_rate_fn(
      config, base_learning_rate, steps_per_epoch)
  updown_learning_rate_fn = create_updown_learning_rate_fn(
      config, base_learning_rate, steps_per_epoch)
  kl_penalty_fn = create_kl_penalty_fn(config, steps_per_epoch)

  state = create_train_state(rng=rng, model=model,
                             learning_rate_fn=learning_rate_fn,
                             updown_learning_rate_fn=updown_learning_rate_fn,
                             config=config)
  state = restore_checkpoint(state, workdir)
  # step_offset > 0 if restarting from checkpoint
  step_offset = int(state.step)
  state = jax_utils.replicate(state)

  # We init the first set of dropout PRNG keys, but update it afterwards inside
  # the main pmap"d training update for performance.
  train_rng, eval_rng = jax.random.split(rng)
  train_rng = jax.random.split(train_rng, jax.local_device_count())
  eval_rng = jax.random.split(eval_rng, jax.local_device_count())

  sem = navier_stokes.StokesSEM.create(
      premesh_commons.unit_cube_mesh(
          config.element_grid_size, periodic_dims=(0, 1)),
      boundary_conditions={}, order=config.order)
  uniform_mesh = refine_premesh(
      premesh_commons.unit_cube_mesh(
          config.element_grid_size, periodic_dims=(0, 1)
      ),
      gridpoints_1d=Nodes1D.create(
          num_points=config.order + 1, node_type=NodeType.NEWTON_COTES
      ),
  ).finalize()
  first_order_mesh = premesh_commons.unit_cube_mesh(
      config.element_grid_size * config.order
  ).finalize()
  first_order_perm = transfer_perm(uniform_mesh, first_order_mesh)

  p_train_step = jax.pmap(partial(
      train_step, sem=sem, config=config, learning_rate_fn=learning_rate_fn,
      kl_penalty_fn=kl_penalty_fn), axis_name='batch')
  p_eval_step = jax.pmap(
      partial(eval_step, kl_penalty_fn=kl_penalty_fn,
              sem=sem, uniform_mesh=uniform_mesh,
              first_order_perm=first_order_perm,
              config=config), axis_name='batch')

  train_metrics = []
  hooks = []
  if jax.process_index() == 0:
    hooks += [periodic_actions.Profile(num_profile_steps=5, logdir=workdir)]
  train_metrics_last_t = time.time()
  logging.info('Initial compilation, this might take some minutes...')
  for step, batch in zip(range(step_offset, num_steps), train_iter):
    state, metrics = p_train_step(state, batch, train_rng=train_rng)
    for h in hooks:
      h(step)
    if step == step_offset:
      logging.info('Initial compilation completed.')

    if config.get('log_every_steps'):
      train_metrics.append(metrics)
      if (step + 1) % config.log_every_steps == 0:
        train_metrics = common_utils.get_metrics(train_metrics)
        summary = {
            f'train_{k}': v
            for k, v in jax.tree_util.tree_map(
                lambda x: x.mean(), train_metrics
            ).items()
        }
        summary['steps_per_second'] = config.log_every_steps / (
            time.time() - train_metrics_last_t)
        writer.write_scalars(step + 1, summary)
        train_metrics = []
        train_metrics_last_t = time.time()

    if (step + 1) % eval_every_steps == 0:
      epoch = step // eval_every_steps
      eval_metrics = []

      # sync batch statistics across replicas
      for _ in range(steps_per_eval):
        eval_batch = next(eval_iter)
        metrics = p_eval_step(state, eval_batch, eval_rng=eval_rng)
        eval_metrics.append(metrics)
      eval_metrics = common_utils.get_metrics(eval_metrics)
      summary = jax.tree_util.tree_map(lambda x: x.mean(), eval_metrics)
      logging.info('eval epoch: %d, loss: %.4f', epoch, summary['loss'])
      writer.write_scalars(
          step + 1, {f'eval_{key}': val for key, val in summary.items()})
      writer.flush()
    if (step + 1) % steps_per_checkpoint == 0 or step + 1 == num_steps:
      save_checkpoint(state, workdir)

  # Wait until computations are done before exiting
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

  return state
