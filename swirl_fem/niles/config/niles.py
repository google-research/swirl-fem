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

"""Hyperparameter configuration to NiLES training pipeline."""

import ml_collections


def get_config():
  """Get the hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.batch_size = 128

  config.debug = False  # if true, uses dummy dataset to make iteration quicker

  # used by the solver for unrolling during training
  config.num_steps = 8
  config.eval_num_steps = 125
  config.permute_elements = True
  config.num_pushforward_steps = config.num_steps - 1

  config.model_name = 'multiscale_transformer'
  config.model = ml_collections.ConfigDict()
  config.model.width = 48

  # configure the multiscale encoder. set depth = 0 to disable
  config.model.use_residuals = True
  config.model.depth = 6
  config.model.num_initial_heads = 1
  config.model.pooling_layers = (config.model.depth - 4, config.model.depth - 2)
  config.model.pooling_kernel = (1, 5)
  config.model.initial_kv_pooling_strides = (1, 4)
  config.model.pooling_strides_q = (1, 4)
  config.model.qkv_tile_reps = (4, 1)

  # SDE config. num_samples = 0 disables the processor
  config.model.processor_config = ml_collections.ConfigDict()

  ############
  # ODE/SDE:
  ############
  num_samples = 4
  config.model.mean_after_decoder = True if num_samples > 0 else False
  config.model.processor_config.num_samples = num_samples

  # enable new transformer based processor
  config.model.processor_config.use_transformer = True
  config.model.processor_config.data_size = config.model.width * 4 * 9
  config.model.processor_config.latent_size = config.model.width * 4
  config.model.processor_config.num_gridpoints = 16
  # below are unused currently in the transformer version
  config.model.processor_config.num_sde_layers = 4
  config.model.processor_config.context_size = 32
  config.model.processor_config.hidden_size = 32
  config.model.processor_config.prior_scale = 0.1

  config.window_step = 1

  # Currently unused, but may be used to switch between datasets.
  config.dataset = 'kolmogorov_flow'
  config.ndim = 2
  config.element_grid_size = 12
  config.order = 4
  config.resolution = config.element_grid_size * config.order
  config.time_order = 3
  config.drag_coeff = 0.04
  config.reynolds_number = 20000
  config.dt = 1e-3 * config.window_step
  config.alpha = 0.
  config.num_nodes = (config.resolution + 1) ** 2
  config.num_elements = 144
  config.num_channels = (config.order + 1) ** 2 * config.ndim

  config.train_window_size = (config.num_steps + 3) * config.window_step
  config.train_window_stride = 1
  config.eval_window_size = (config.eval_num_steps + 3) * config.window_step
  config.eval_window_stride = 4

  # Training and learning rate
  config.num_epochs = 15.
  config.learning_rate = 0.0075
  config.grad_clip_norm = 0.01
  config.weight_decay = 0.05
  config.warmup_epochs = 1.
  config.kl_penalty = 0.01
  config.kl_transition_epochs = 10.
  config.kl_zero_epochs = 0

  config.log_every_steps = 100
  config.checkpoint_epochs = 1  # checkpoint after every this many epochs
  config.eval_every_epochs = 0.1

  config.cache = True

  # If num_train_steps==-1 then the number of training steps is calculated from
  # num_epochs using the entire dataset. Similarly for steps_per_eval.
  config.num_train_steps = -1
  config.steps_per_eval = 10

  return config


def metrics() -> list[str]:
  return [
      'steps_per_second',
      'train_learning_rate',
      'train_kl_penalty',
      'train_loss',
      'train_mse',
      'train_z0_means',
      'train_z1_means',
      'train_z1_stds',
      'eval_loss',
      'eval_mse',
      'eval_mse@1t08',
      'eval_mse@8',
      'eval_mse@16',
      'eval_mse@32',
      'eval_z0_means',
      'eval_z1_means',
      'eval_z1_stds',
      'eval_tke_err',
  ]


def sweep(add):
  """Hyperparameter search."""
  for depth in [36, 40, 48]:
    add(**{
        'model.depth': depth,
        'model.pooling_layers': (depth - 4, depth - 2),
    })
