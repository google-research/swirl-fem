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

import collections
from functools import partial  # pylint: disable=g-importing-member
import os

from absl.testing import absltest
import jax
import jax.extend
import jax.numpy as jnp
import numpy as np
from swirl_fem.core import gather_scatter


def set_host_platform_device_count(num_devices: int):
  """Sets XLA flags for `num_devices` and returns a closure for undoing it."""
  # Forked from jax.test_util
  prev_xla_flags = os.getenv('XLA_FLAGS')
  flags_str = prev_xla_flags or ''
  # Don't override user-specified device count, or other XLA flags.
  if 'xla_force_host_platform_device_count' not in flags_str:
    os.environ['XLA_FLAGS'] = (
        flags_str + f' --xla_force_host_platform_device_count={num_devices}'
    )
  # Clear any cached backends so new CPU backend will pick up the env var.
  jax.extend.backend.get_backend.cache_clear()
  def undo():
    if prev_xla_flags is None:
      del os.environ['XLA_FLAGS']
    else:
      os.environ['XLA_FLAGS'] = prev_xla_flags
    jax.extend.backend.get_backend.cache_clear()
  return undo


class GatherScatterTest(absltest.TestCase):

  def test_exchange_noop(self):
    # a 1d mesh with 3 nodes without any nodes to exchange
    num_nodes = 3
    node_indices = np.arange(num_nodes, dtype=np.int32)
    gather_indices, unique_indices = (
        gather_scatter.get_exchange_indices(node_indices))

    # There's nothing to exchange.
    self.assertEqual(gather_indices.dtype, np.int32)
    self.assertEqual(gather_indices.shape, (0,))
    self.assertEqual(unique_indices.shape, (0,))
    self.assertSequenceAlmostEqual(
        jnp.arange(3.),
        gather_scatter.exchange(
            u=jnp.arange(3.),
            gather_indices=gather_indices, unique_indices=unique_indices,
        ),
    )

  def test_exchange_periodic(self):
    # a 1d mesh with 3 nodes with node 0 and 2 connected via a periodic link
    num_nodes = 3
    node_indices = np.arange(num_nodes, dtype=np.int32)
    unique_node_indices = gather_scatter.get_unique_node_indices(
        node_indices, periodic_links=np.array([[[0], [2]]]))
    gather_indices, unique_indices = (
        gather_scatter.get_exchange_indices(unique_node_indices))

    # We have two participating nodes.
    self.assertEqual(gather_indices.dtype, np.int32)
    self.assertEqual(gather_indices.shape, (2,))
    self.assertEqual(unique_indices.shape, (2,))
    # two boundary nodes get summed up
    self.assertSequenceAlmostEqual(
        jnp.array([4., 2., 4.]),
        gather_scatter.exchange(
            u=jnp.array([1., 2., 3.]),
            gather_indices=gather_indices, unique_indices=unique_indices,
        ),
    )

  def test_doubly_periodic(self):
    # create a 2d mesh with 4 elements and 9 nodes arranged as follows
    #   0 -- 1 -- 2
    #   |    |    |
    #   3 -- 4 -- 5
    #   |    |    |
    #   6 -- 7 -- 8
    num_nodes = 9
    node_indices = np.arange(num_nodes, dtype=np.int32)
    # There are 4 links for each of the facets along each axis.
    periodic_links = np.array([
        [[0, 1], [6, 7]],
        [[1, 2], [7, 8]],
        [[0, 3], [2, 5]],
        [[3, 6], [5, 8]],
    ], dtype=np.int32)

    unique_node_indices = gather_scatter.get_unique_node_indices(
        node_indices, periodic_links)
    gather_indices, unique_indices = (
        gather_scatter.get_exchange_indices(unique_node_indices))

    # local indices have all distinct elements. All but one nodes participate
    # in the periodic exchange. So it should have `num_nodes - 1` indices.
    self.assertCountEqual(
        collections.Counter(gather_indices).values(), [1] * (num_nodes - 1))
    # Make sure that the unique_indices has 1 value repeated 4 times and
    # 2 values repeated 2 times (and no other values).
    self.assertCountEqual(
        collections.Counter(unique_indices).values(), [4, 2, 2])

    # Four corners get the same value. The middle edge nodes sum up two nodes,
    # and the middle node remains unchanged.
    self.assertSequenceAlmostEqual(
        jnp.array([16., 8., 16., 8., 4., 8., 16., 8., 16.]),
        gather_scatter.exchange(
            u=jnp.arange(9.0),
            gather_indices=gather_indices, unique_indices=unique_indices,
        ),
    )


class GatherScatterPartitionedTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.cleanup = set_host_platform_device_count(4)

  def tearDown(self):
    self.cleanup()
    super().tearDown()

  def test_gather_indices(self):
    # Create a 1D mesh with 8 elements (9 nodes) partitioned into 4 cores.
    node_indices = np.array([[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 8]],
                            dtype=np.int32)

    gather_indices, unique_indices = (
        gather_scatter.get_exchange_indices(node_indices))

    # local indices need to exchange 3 node indices: (2, 4, 6)
    # which correspond to local indices 2 or 0 in different partitions.
    expected_gather_indices = np.array(
        [[2, -1, -1], [0, 2, -1], [-1, 0, 2], [-1, -1, 0]], dtype=np.int32)
    np.testing.assert_array_equal(gather_indices, expected_gather_indices)
    self.assertIsNone(unique_indices)

  def test_exchange(self):
    # Create a 1D mesh with 8 elements (9 nodes) partitioned into 4 cores.
    node_indices = np.array([[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 8]],
                            dtype=np.int32)
    gather_indices, unique_indices = (
        gather_scatter.get_exchange_indices(node_indices))

    # Create nodal values which need to be exchanged. The values are partitioned
    # into four cores in the same way as the mesh.
    u = jnp.arange(12.).reshape((4, 3)).astype(jnp.float32)
    # Actually place it on the 4 devices.
    u = jax.pmap(lambda x: x)(u)

    exchange_fn = partial(gather_scatter.exchange,
                          unique_indices=unique_indices, axis_name='i')

    u_exchanged = jax.pmap(exchange_fn, axis_name='i')(u, gather_indices)
    expected_u_exchanged = np.array(
        [[0, 1, 5], [5, 4, 11], [11, 7, 17], [17, 10, 11]], dtype=np.float32)
    np.testing.assert_array_equal(u_exchanged, expected_u_exchanged)

  def test_gather_indices_periodic(self):
    # Create a 1D mesh with 8 elements (9 nodes) partitioned into 4 cores.
    node_indices = np.array([[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 8]],
                            dtype=np.int32)
    # Dedup the node indices by taking into account periodicity between nodes
    # 0 and 8.
    node_indices = gather_scatter.get_unique_node_indices(
        node_indices, periodic_links=np.array([[[0], [8]]]))

    gather_indices, unique_indices = (
        gather_scatter.get_exchange_indices(node_indices))

    # local indices need to exchange 4 node indices: (0, 2, 4, 6)
    # which correspond to local indices 2 or 0 in different partitions.
    expected_gather_indices = np.array(
        [[0, 2, -1, -1], [-1, 0, 2, -1], [-1, -1, 0, 2], [2, -1, -1, 0]],
        dtype=np.int32)
    np.testing.assert_array_equal(gather_indices, expected_gather_indices)
    self.assertIsNone(unique_indices)

  def test_exchange_periodic(self):
    # Create a 1D mesh with 8 elements (9 nodes) partitioned into 4 cores.
    node_indices = np.array([[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 8]],
                            dtype=np.int32)
    # Dedup the node indices by taking into account periodicity between nodes
    # 0 and 8.
    node_indices = gather_scatter.get_unique_node_indices(
        node_indices, periodic_links=np.array([[[0], [8]]]))

    gather_indices, unique_indices = (
        gather_scatter.get_exchange_indices(node_indices))

    # Create nodal values which need to be exchanged. The values are partitioned
    # into four cores in the same way as the mesh.
    u = jnp.arange(12.).reshape((4, 3)).astype(jnp.float32)
    # Actually place it on the 4 devices.
    u = jax.pmap(lambda x: x)(u)

    exchange_fn = partial(gather_scatter.exchange,
                          unique_indices=unique_indices, axis_name='i')

    u_exchanged = jax.pmap(exchange_fn, axis_name='i')(u, gather_indices)
    expected_u_exchanged = np.array(
        [[11, 1, 5], [5, 4, 11], [11, 7, 17], [17, 10, 11]], dtype=np.float32)
    np.testing.assert_array_equal(u_exchanged, expected_u_exchanged)

  def test_exchange_doubly_periodic(self):
    # Create a 2D mesh with 4 elements partitioned into 4 cores, with one
    # element per core.
    #   0 -- 1 -- 2
    #   |    |    |
    #   3 -- 4 -- 5
    #   |    |    |
    #   6 -- 7 -- 8
    node_indices = np.array(
        [[0, 1, 3, 4], [1, 2, 4, 5], [3, 4, 6, 7], [4, 5, 7, 8]],
        dtype=np.int32)
    # There are 4 links for each of the facets along each axis.
    periodic_links = np.array([
        [[0, 1], [6, 7]],
        [[1, 2], [7, 8]],
        [[0, 3], [2, 5]],
        [[3, 6], [5, 8]],
    ], dtype=np.int32)

    # Dedup the node indices by taking into account periodicity
    node_indices = gather_scatter.get_unique_node_indices(
        node_indices, periodic_links=periodic_links)
    gather_indices, unique_indices = (
        gather_scatter.get_exchange_indices(node_indices))

    # Create nodal values which need to be exchanged. The values are partitioned
    # into four cores in the same way as the mesh.
    u = jnp.ones((4, 4)).astype(jnp.float32)
    # Actually place it on the 4 devices.
    u = jax.pmap(lambda x: x)(u)

    exchange_fn = partial(gather_scatter.exchange,
                          unique_indices=unique_indices, axis_name='i')

    u_exchanged = jax.pmap(exchange_fn, axis_name='i')(u, gather_indices)
    # due to symmetry, each global node receives a '1' value from its four local
    # node values
    expected_u_exchanged = 4 * u
    np.testing.assert_array_equal(u_exchanged, expected_u_exchanged)

  def test_within_partition_periodicity_not_implemented(self):
    # Create a 2D mesh with 4 elements partitioned into 4 cores, with one
    # element per core.
    #   0 -- 1 -- 2 -- 3 -- 4
    #   |    |    |    |    |
    #   5 -- 6 -- 7 -- 8 -- 9
    node_indices = np.array(
        [[0, 1, 5, 6], [1, 2, 6, 7], [2, 3, 7, 8], [3, 4, 8, 9]],
        dtype=np.int32)
    # Add 4 links to represent periodicity in the y-direction.
    periodic_links = np.array([
        [[0, 1], [5, 6]],
        [[1, 2], [6, 7]],
        [[2, 3], [7, 8]],
        [[3, 4], [8, 9]],
    ], dtype=np.int32)
    # Dedup the node indices by taking into account periodicity
    node_indices = gather_scatter.get_unique_node_indices(
        node_indices, periodic_links=periodic_links)

    # However, in the partitioned case, two periodic nodes cooccuring in the
    # same partition is not supported.
    with self.assertRaisesRegex(NotImplementedError, 'more than once'):
      gather_scatter.get_exchange_indices(node_indices)


if __name__ == '__main__':
  jax.config.update('jax_enable_x64', True)
  absltest.main()
