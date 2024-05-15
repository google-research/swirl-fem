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

"""Tests for pscan."""

from functools import partial  # pylint: disable=g-importing-member

from absl.testing import absltest
import jax
from jax import sharding
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
import numpy as np
from swirl_fem.communication.pscan import preduce
from swirl_fem.communication.pscan import pscan
from swirl_fem.communication.set_test_device_count import set_host_platform_device_count


NDEVICES = 12


def jit_pscan(mesh_shape, axis_names, partition_spec):
  devices = mesh_utils.create_device_mesh((jax.device_count(),))
  mesh = sharding.Mesh(devices.reshape(mesh_shape), axis_names=axis_names)
  shmap = partial(shard_map, mesh=mesh, in_specs=partition_spec,
                  out_specs=partition_spec, check_rep=False)

  @partial(jax.jit, static_argnames=['op', 'axis_name', 'reduction'])
  def do_scan(x, op, axis_name, reduction=False):
    scan = lambda x: pscan(x, op=op, axis_name=axis_name, reduction=reduction)
    return shmap(scan)(x)

  @partial(jax.jit, static_argnames=['op', 'axis_name'])
  def do_reduce(x, op, axis_name):
    reduce = lambda x: preduce(x, op=op, axis_name=axis_name)
    return shmap(reduce)(x)

  return do_scan, do_reduce


class PscanTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    set_host_platform_device_count(NDEVICES)

  def test_pscan(self):
    assert jax.device_count() == NDEVICES
    n = NDEVICES

    do_pscan, do_preduce = jit_pscan(
        mesh_shape=(n,),
        axis_names='i',
        partition_spec=sharding.PartitionSpec('i'),
    )

    for x in [jnp.arange(n), jnp.flip(jnp.arange(n))]:
      for op_name in [
          'add',
          'multiply',
          'maximum',
          'minimum',
          'bitwise_and',
          'bitwise_or',
          'bitwise_xor',
      ]:
        jnp_op = getattr(jnp, op_name)
        np_op = getattr(np, op_name)
        exclusive = do_pscan(x, jnp_op, axis_name='i')
        inclusive = np_op.accumulate(x)
        np.testing.assert_array_equal(jnp_op(exclusive, x), inclusive)

        exclusive2, allreduce = do_pscan(
            x, jnp_op, axis_name='i', reduction=True
        )
        np.testing.assert_array_equal(exclusive, exclusive2)
        np.testing.assert_array_equal(allreduce, np_op.reduce(x))

        allreduce2 = do_preduce(x, jnp_op, axis_name='i')
        np.testing.assert_array_equal(allreduce2, np_op.reduce(x))

  def test_2d_mesh(self):
    assert jax.device_count() == NDEVICES
    ni, nj = 4, NDEVICES // 4
    do_pscan, do_preduce = jit_pscan(
        mesh_shape=(ni, nj),
        axis_names=('i', 'j'),
        partition_spec=sharding.PartitionSpec('i', 'j'),
    )
    del do_preduce

    # one entry per device
    x = np.arange(ni * nj).reshape((ni, nj))

    scan_i = do_pscan(x, jnp.add, axis_name='i')
    scan_j = do_pscan(x, jnp.add, axis_name='j')

    np.testing.assert_array_equal(scan_i + x, jnp.cumsum(x, axis=0))
    np.testing.assert_array_equal(scan_j + x, jnp.cumsum(x, axis=1))

    # (2, 2) shape on each device
    x = np.arange(4 * ni * nj).reshape((2 * ni, 2 * nj))

    scan_i = do_pscan(x, jnp.add, axis_name='i')
    scan_j = do_pscan(x, jnp.add, axis_name='j')

    for i in range(2):
      for j in range(2):
        y = x[i::2, j::2]
        si = scan_i[i::2, j::2]
        sj = scan_j[i::2, j::2]
        np.testing.assert_array_equal(si + y, jnp.cumsum(y, axis=0))
        np.testing.assert_array_equal(sj + y, jnp.cumsum(y, axis=1))


if __name__ == '__main__':
  jax.config.update('jax_enable_x64', True)
  absltest.main()
