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

"""Binary for generating a Kolmogorov flow dataset."""

import io
import os
import time

from absl import app
from absl import flags
from absl import logging
from clu import platform
from etils import epath
import h5py
import jax
import jax.numpy as jnp
from ml_collections import config_flags
import numpy as np

from swirl_fem.common import premesh_commons
from swirl_fem.navier_stokes import navier_stokes


# pylint: disable=invalid-name

FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data.')
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True)


RESOLUTION = 64  # 32  # 48
ORDER = 8
TIME_ORDER = 3
REYNOLDS_NUMBER = 20000
NUM_CYCLES = 500
NUM_STEPS_PER_CYCLE = 500
DT = 1e-4  # 2e-4  # 2.5e-4
DRAG_COEFF = 0.1


def u_init_fn(x):
  """Initial velocity field of the Kolmogorov flow."""
  assert x.shape == (2,)
  l = 2.
  u0 = jnp.cos(2 * l * np.pi * x[0]) * jnp.sin(2 * l * np.pi * x[1])
  u1 = -jnp.sin(2 * l * np.pi * x[0]) * jnp.cos(2 * l * np.pi * x[1])
  return jnp.array([u0, u1])


def forcing(x, u):
  """Kolmogorov flow forcing function."""
  assert x.shape == (2,)
  assert u.shape == (2,)
  k = 4.
  f0 = jnp.sin(2 * np.pi * k * x[1])
  f1 = 0.
  return jnp.array([f0, f1]) - DRAG_COEFF * u


def compute_dx(mesh):
  """Compute the minimum distance between mesh nodes."""
  dx = np.inf
  elem_coords = np.asarray(mesh.element_coords(), dtype=np.float32)
  for i in range(mesh.num_elements):
    x = elem_coords[i, ...]
    pairwise = np.linalg.norm(x[:, np.newaxis, :] - x[np.newaxis, :, :],
                              axis=-1)
    pairwise += np.diag(np.inf * np.ones(mesh.num_nodes_per_element))
    dx = min(dx, pairwise.min())
  return dx


@jax.jit
def _solve_one_step(sem, us, ps, Cus):
  """Solve one step of the Navier-Stokes system."""
  # Extrapolate advection term
  ext_coeffs = navier_stokes.extk_coeffs(k=TIME_ORDER - 1)
  Cu = sum(ext_coeffs[-i] * Cus[-i] for i in range(1, len(ext_coeffs) + 1))

  # Solve the stokes system with extrapolated advection term
  f = jax.vmap(forcing)(sem.velocity.mesh.node_coords, us[-1])
  f = -Cu + sem.B(f)
  u, p = sem.stokes_one_step(us, ps, f, mu=1 / REYNOLDS_NUMBER, dt=DT,
                             time_order=TIME_ORDER, tol=1e-5, atol=1e-4)
  return u, p, sem.C(u)


def one_cycle(sem: navier_stokes.StokesSEM,
              start_step: int, num_steps: int, us: tuple[jax.Array, ...],
              ps: tuple[jax.Array, ...]):
  """Performs a single simulation cycle and writes the results to an HDF5 file.

  Args:
    sem: A StokesSEM object representing the simulation setup.
    start_step: The starting step number of this cycle.  Used for file naming.
    num_steps: The number of steps to simulate in this cycle.
    us: A tuple of jax.Array representing the history of velocity fields.
       The last element is the initial state for this cycle.  Must contain at
       least one element.
    ps: A tuple of jax.Array representing the history of pressure fields.
       The last element is the initial state for this cycle. Must contain at
       least one element and have the same length as `us`.

  Returns:
    A tuple containing:
      - us: Updated tuple of velocity fields after completing the cycle.
      - ps: Updated tuple of pressure fields after completing the cycle.
  """
  t = start_step * DT
  dataset: dict[str, list[jax.typing.ArrayLike]] = {
      't': [t],
      'u': list(us[-1:]),
      'p': list(ps[-1:]),
  }

  start_time = time.time()
  Cus = tuple(map(sem.C, us))
  for step_idx in range(1, num_steps + 1):
    t += DT
    u, p, Cu = _solve_one_step(sem, us, ps, Cus)
    us = us[1:] + (u,)
    ps = ps[1:] + (p,)
    Cus = Cus[1:] + (Cu,)

    # update dataset with the new state.
    if step_idx % 10 == 0:
      dataset['t'].append(t)
      dataset['u'].append(u)
      dataset['p'].append(p)

  logging.info('one cycle walltime %f seconds', time.time() - start_time)

  # Write to file
  for key in ['u', 'p', 't']:
    dataset[key] = np.stack(dataset[key])

  file_path = os.path.join(FLAGS.workdir, (
      f'kolmogorov_flow_grid_{RESOLUTION}_order_{ORDER}'
      f'_step_{start_step}_{start_step + num_steps}.hdf5'))
  logging.info('writing output to file %s', file_path)

  bio = io.BytesIO()
  with h5py.File(bio, 'w') as f:
    for k, v in dataset.items():
      f[k] = v

  output_file = epath.Path(file_path)
  output_file.write_bytes(bio.getvalue())

  return us, ps


def run_simulation():
  """Run simulation and write the dataset."""
  # setup
  unit_square_premesh = premesh_commons.unit_cube_mesh(
      RESOLUTION, ndim=2, periodic_dims=(0, 1))
  sem = navier_stokes.StokesSEM.create(
      unit_square_premesh, boundary_conditions={}, order=ORDER)
  mesh_dx = compute_dx(sem.velocity.mesh)

  logging.info('Created mesh with nodes %d and %d elements:',
               sem.velocity.mesh.num_nodes, sem.velocity.mesh.num_elements)
  logging.info('Mesh dx: %f', mesh_dx)

  u_init = jax.vmap(u_init_fn)(sem.velocity.mesh.node_coords)

  # Initialize velocities and pressures.
  p_init = jnp.zeros(sem.pressure.pspace.mesh.num_nodes)
  us, ps = zip(*[(u_init, p_init) for _ in range(TIME_ORDER)])

  for cycle_idx in range(NUM_CYCLES):
    assert len(us) == len(ps)
    assert len(us) == TIME_ORDER
    us, ps = one_cycle(sem=sem, start_step=cycle_idx * NUM_STEPS_PER_CYCLE,
                       num_steps=NUM_STEPS_PER_CYCLE, us=us, ps=ps)
    cfl = (np.float32(us[-1].max()) * DT) / mesh_dx
    logging.info('At cycle %d, CFL number: %f', cycle_idx, cfl)

  logging.info('Done writing everything!')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
  logging.info('JAX local devices: %r', jax.local_devices())

  # Add a note so that we can tell which task is which JAX host.
  # (Depending on the platform task 0 is not guaranteed to be host 0)
  platform.work_unit().set_task_status(f'process_index: {jax.process_index()}, '
                                       f'process_count: {jax.process_count()}')
  platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                       FLAGS.workdir, 'workdir')

  run_simulation()


if __name__ == '__main__':
  flags.mark_flags_as_required(['config', 'workdir'])
  jax.config.update('jax_enable_x64', True)
  app.run(main)
