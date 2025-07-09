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

"""Config for Navier-Stokes simulation."""

import ml_collections


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.model_name = 'kolmogorov'

  # Used to initialize the velocity field randomly
  config.seed = 0

  # Write file to disk after this many steps
  config.num_steps_per_cycle = 500
  # Number of cycles to run the simulation for
  config.num_cycles = 500
  # snapshots are added to an output every this many steps
  config.sample_rate = 10

  # Constant timestep for the solver
  config.dt = 1e-4
  # Order of the spectral element mesh
  config.order = 8
  config.reynolds_number = 20000  # 1000  # 20000

  # Size of the element grid. The total number of elements will be the square
  # of this number
  config.element_grid_size = 48

  # Drag value for the Kolmogorov flow to stabilize the computation
  config.drag_coeff = 0.05  # 0.1  # 5e-2 looks good for Re = 20000

  # Wavenumber for the Kolmogorov flow forcing
  config.forcing_wavenumber = 4.

  return config


def sweep(add):
  for seed in range(32):
    add(seed=seed)
