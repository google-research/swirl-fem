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

import os

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from swirl_fem.core import gather_scatter
from swirl_fem.core.interpolation import Nodes1D
from swirl_fem.core.interpolation import NodeType
from swirl_fem.core.mesh import Mesh
from swirl_fem.core.premesh import Premesh


# TODO(anudhyan): Move this function to test_util.py.
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
  jax.lib.xla_bridge.get_backend.cache_clear()
  def undo():
    if prev_xla_flags is None:
      del os.environ['XLA_FLAGS']
    else:
      os.environ['XLA_FLAGS'] = prev_xla_flags
    jax.lib.xla_bridge.get_backend.cache_clear()
  return undo


class PremeshTest(absltest.TestCase):

  def test_1d(self):
    num_elements = 8
    num_nodes = num_elements + 1
    node_coords = np.linspace(0, 1, num_nodes).reshape((num_nodes, 1))
    elements = np.array([[i, i + 1] for i in range(num_elements)])

    premesh = Premesh.create(node_coords=node_coords, elements=elements)
    self.assertEqual(premesh.ndim, 1)
    self.assertEqual(premesh.num_nodes, num_nodes)
    self.assertEqual(premesh.num_elements, num_elements)
    self.assertEqual(premesh.num_nodes_per_element, 2)
    self.assertFalse(premesh.is_partitioned())
    self.assertEqual(
        premesh.gridpoints_1d,
        Nodes1D.create(num_points=2, node_type=NodeType.NEWTON_COTES))

    mesh = premesh.finalize()
    self.assertIsInstance(mesh, Mesh)
    self.assertEqual(mesh.ndim, 1)
    self.assertEqual(mesh.num_nodes, num_nodes)
    self.assertEqual(mesh.num_elements, num_elements)
    self.assertEqual(mesh.num_nodes_per_element, 2)
    self.assertEqual(
        mesh.gridpoints_1d,
        Nodes1D.create(num_points=2, node_type=NodeType.NEWTON_COTES))

    # mesh.exchange does nothing
    self.assertSequenceAlmostEqual(jnp.arange(num_nodes),
                                   mesh.exchange(jnp.arange(num_nodes)))

  def test_1d_periodic(self):
    num_elements = 8
    num_nodes = num_elements + 1
    node_coords = np.linspace(0, 1, num_nodes).reshape((num_nodes, 1))
    elements = np.array([[i, i + 1] for i in range(num_elements)])

    # Add a periodic link between the two endpoints.
    premesh = Premesh.create(
        node_coords=node_coords, elements=elements,
        periodic_links=np.array([[[0], [num_nodes - 1]]], dtype=np.int32))

    mesh = premesh.finalize()
    self.assertIsInstance(mesh, Mesh)

    # mesh.exchange sums up the two endpoints
    self.assertSequenceAlmostEqual(np.array([10, 2, 3, 4, 5, 6, 7, 8, 10]),
                                   mesh.exchange(1 + jnp.arange(num_nodes)))

  def test_2d(self):
    # 6 nodes and two elements arranged as follows
    #     1--3--5
    #     |  |  |
    #     0--2--4
    num_elements = 2
    num_nodes = 6
    node_coords = np.array(
        [[0, 0], [0, 1], [1, 0], [1, 1], [0, 2], [2, 1]], dtype=np.float32
    )
    elements = np.array([[0, 1, 2, 3], [2, 3, 4, 5]], dtype=np.int32)

    premesh = Premesh.create(node_coords=node_coords, elements=elements)

    self.assertEqual(premesh.ndim, 2)
    self.assertEqual(premesh.num_nodes, num_nodes)
    self.assertEqual(premesh.num_elements, num_elements)
    self.assertEqual(premesh.num_nodes_per_element, 4)
    self.assertFalse(premesh.is_partitioned())

    mesh = premesh.finalize()
    self.assertIsInstance(mesh, Mesh)
    self.assertEqual(mesh.ndim, 2)
    self.assertEqual(mesh.num_nodes, num_nodes)
    self.assertEqual(mesh.num_elements, num_elements)
    self.assertEqual(mesh.num_nodes_per_element, 4)

    # mesh.exchange does nothing
    self.assertSequenceAlmostEqual(np.arange(num_nodes),
                                   mesh.exchange(jnp.arange(num_nodes)))

  def test_2d_periodic(self):
    # 6 nodes with a periodic link between left and right edges
    #     1--3--5
    #     |  |  |
    #     0--2--4
    num_nodes = 6
    node_coords = np.array(
        [[0, 0], [0, 1], [1, 0], [1, 1], [0, 2], [2, 1]], dtype=np.float32
    )
    elements = np.array([[0, 1, 2, 3], [2, 3, 4, 5]], dtype=np.int32)
    premesh = Premesh.create(
        node_coords=node_coords, elements=elements,
        periodic_links=np.array([[[0, 1], [4, 5]]], dtype=np.int32))

    mesh = premesh.finalize()
    self.assertIsInstance(mesh, Mesh)

    # mesh.exchange sums up nodes {0, 4} and {1, 5}
    self.assertSequenceAlmostEqual(
        np.array([1 + 5, 2 + 6, 3, 4, 5 + 1, 6 + 2], dtype=np.int32),
        mesh.exchange(1 + jnp.arange(num_nodes)))

  def test_2d_doubly_periodic(self):
    # 6 nodes with left-right and top-bottom periodic links
    #     1--3--5
    #     |  |  |
    #     0--2--4
    num_nodes = 6
    node_coords = np.array(
        [[0, 0], [0, 1], [1, 0], [1, 1], [0, 2], [2, 1]], dtype=np.float32)
    elements = np.array([[0, 1, 2, 3], [2, 3, 4, 5]], dtype=np.int32)
    periodic_links = np.array(
        [[[0, 1], [4, 5]], [[0, 2], [1, 3]], [[2, 4], [3, 5]]], dtype=np.int32)
    premesh = Premesh.create(
        node_coords=node_coords, elements=elements,
        periodic_links=periodic_links)

    mesh = premesh.finalize()
    self.assertIsInstance(mesh, Mesh)

    # mesh.exchange sums up nodes {0, 4} and {1, 5}. The corner nodes have 4
    # summands and the edge nodes have 2.
    self.assertSequenceAlmostEqual(
        np.array([4, 4, 2, 2, 4, 4], dtype=np.int32),
        mesh.exchange(jnp.ones(num_nodes)))

  def test_physical_masks(self):
    # 6 nodes and 2 elements physical groups
    #     1--3--5
    #     |  |  |
    #     0--2--4
    node_coords = np.array(
        [[0, 0], [0, 1], [1, 0], [1, 1], [0, 2], [2, 1]], dtype=np.float32)
    elements = np.array([[0, 1, 2, 3], [2, 3, 4, 5]], dtype=np.int32)

    physical_groups = {'left': np.array([[0, 1]]),
                       'top': np.array([[1, 3], [3, 5]])}
    premesh = Premesh.create(node_coords=node_coords, elements=elements,
                             physical_groups=physical_groups)

    mesh = premesh.finalize()
    self.assertIsInstance(mesh, Mesh)
    self.assertIsInstance(mesh.physical_masks, dict)
    self.assertIn('left', mesh.physical_masks)
    self.assertSequenceAlmostEqual(mesh.physical_masks['left'],
                                   np.array([1, 1, 0, 0, 0, 0], dtype=bool))
    self.assertIn('top', mesh.physical_masks)
    self.assertSequenceAlmostEqual(mesh.physical_masks['top'],
                                   np.array([0, 1, 0, 1, 0, 1], dtype=bool))


class PremeshPartitionedTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.cleanup = set_host_platform_device_count(4)

  def tearDown(self):
    self.cleanup()
    super().tearDown()

  def test_1d_basic(self):
    num_elements = 8
    num_nodes = num_elements + 1
    node_coords = np.linspace(0, 1, num_nodes).reshape((num_nodes, 1))
    elements = np.array([[i, i + 1] for i in range(num_elements)])
    # place two elements in each partition
    partitions = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)

    premesh = Premesh.create(node_coords=node_coords, elements=elements,
                             partitions=partitions)
    self.assertEqual(premesh.ndim, 1)
    self.assertEqual(premesh.num_nodes, num_nodes)
    self.assertEqual(premesh.num_elements, num_elements)
    self.assertEqual(premesh.num_nodes_per_element, 2)
    self.assertTrue(premesh.is_partitioned())
    self.assertEqual(
        premesh.gridpoints_1d,
        Nodes1D.create(num_points=2, node_type=NodeType.NEWTON_COTES))

    mesh = premesh.finalize(axis_name='i')
    self.assertIsInstance(mesh, Mesh)
    self.assertEqual(mesh.ndim, 1)
    # check the nodes and elements per partition
    self.assertEqual(mesh.num_nodes, 3)
    self.assertEqual(mesh.num_elements, 2)
    self.assertEqual(mesh.num_nodes_per_element, 2)
    self.assertEqual(
        mesh.gridpoints_1d,
        Nodes1D.create(num_points=2, node_type=NodeType.NEWTON_COTES))

  def test_1d_exchange(self):
    num_elements = 8
    num_nodes = num_elements + 1
    node_coords = np.linspace(0, 1, num_nodes).reshape((num_nodes, 1))
    elements = np.array([[i, i + 1] for i in range(num_elements)])
    # place two elements in each partition
    partitions = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)

    premesh = Premesh.create(node_coords=node_coords, elements=elements,
                             partitions=partitions)
    mesh = premesh.finalize(axis_name='i')

    # mesh.exchange sums up nodal values along shared nodes
    u = jnp.arange(num_nodes, dtype=jnp.float32)[mesh.node_indices]
    u = jax.pmap(lambda x: x)(u)
    u_exchanged = jax.pmap(
        lambda mesh, x: mesh.exchange(x), axis_name='i')(mesh, u)

    expected_u_exchanged = np.array(
        [[0, 1, 4], [4, 3, 8], [8, 5, 12], [12, 7, 8]], dtype=np.float32)
    np.testing.assert_array_almost_equal(expected_u_exchanged, u_exchanged)

  def test_1d_exchange_with_padding(self):
    num_elements = 6
    num_nodes = num_elements + 1
    node_coords = np.linspace(0, 1, num_nodes).reshape((num_nodes, 1))
    elements = np.array([[i, i + 1] for i in range(num_elements)])
    # two elements in the first two partitions and one element each in the rest.
    partitions = np.array([0, 0, 1, 1, 2, 3], dtype=np.int32)

    premesh = Premesh.create(node_coords=node_coords, elements=elements,
                             partitions=partitions)
    mesh = premesh.finalize(axis_name='i')

    # mesh.exchange sums up nodal values along shared nodes
    u = jnp.arange(num_nodes, dtype=jnp.float32)
    u_p = gather_scatter.gather(u, mesh.node_indices, fill_value=0.)
    u_p = jax.pmap(lambda x: x)(u_p)

    u_exchanged = jax.pmap(
        lambda mesh, x: mesh.exchange(x), axis_name='i')(mesh, u_p)
    expected_u_p = np.array(
        [[0, 1, 2], [2, 3, 4], [4, 5, 0], [5, 6, 0]], dtype=np.float32)
    np.testing.assert_array_almost_equal(expected_u_p, u_p)

    expected_u_exchanged = np.array(
        [[0, 1, 4], [4, 3, 8], [8, 10, 0], [10, 6, 0]], dtype=np.float32)
    np.testing.assert_array_almost_equal(expected_u_exchanged, u_exchanged)

  def test_1d_gather_with_padding(self):
    num_elements = 6
    num_nodes = num_elements + 1
    node_coords = np.linspace(0, 1, num_nodes).reshape((num_nodes, 1))
    elements = np.array([[i, i + 1] for i in range(num_elements)])
    # two elements in the first two partitions and one element each in the rest.
    partitions = np.array([0, 0, 1, 1, 2, 3], dtype=np.int32)

    premesh = Premesh.create(node_coords=node_coords, elements=elements,
                             partitions=partitions)
    mesh = premesh.finalize(axis_name='i')

    # mesh.exchange sums up nodal values along shared nodes
    u = jnp.arange(num_nodes, dtype=jnp.float32)
    u_p = gather_scatter.gather(u, mesh.node_indices, fill_value=0.)
    u_p = jax.pmap(lambda x: x)(u_p)
    u_local = jax.pmap(lambda mesh, x: mesh.gather(x))(mesh, u_p)
    expected_u_local = np.array(
        [
            [[0, 1], [1, 2]],
            [[2, 3], [3, 4]],
            [[4, 5], [0, 0]],
            [[5, 6], [0, 0]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_almost_equal(u_local, expected_u_local)

  def test_1d_exchange_nothing_to_exchange(self):
    num_elements = 4
    num_nodes = 8
    node_coords = np.linspace(0, 1, num_nodes).reshape((num_nodes, 1))
    elements = np.array([[2 * i, 2 * i + 1] for i in range(num_elements)])
    # each partition has exactly one element
    partitions = np.array([0, 1, 2, 3], dtype=np.int32)

    premesh = Premesh.create(node_coords=node_coords, elements=elements,
                             partitions=partitions)
    mesh = premesh.finalize(axis_name='i')

    # mesh.exchange sums up nodal values along shared nodes
    u = jnp.arange(num_nodes, dtype=jnp.float32)
    u_p = gather_scatter.gather(u, mesh.node_indices, fill_value=0.)
    u_p = jax.pmap(lambda x: x)(u_p)

    u_exchanged = jax.pmap(
        lambda mesh, x: mesh.exchange(x), axis_name='i')(mesh, u_p)
    expected_u_p = np.array(
        [[0, 1], [2, 3], [4, 5], [6, 7]], dtype=np.float32)
    np.testing.assert_array_almost_equal(expected_u_p, u_p)

    expected_u_exchanged = expected_u_p
    np.testing.assert_array_almost_equal(expected_u_exchanged, u_exchanged)

  def test_1d_periodic(self):
    num_elements = 8
    num_nodes = num_elements + 1
    node_coords = np.linspace(0, 1, num_nodes).reshape((num_nodes, 1))
    elements = np.array([[i, i + 1] for i in range(num_elements)])
    # place two elements in each partition
    partitions = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)

    premesh = Premesh.create(
        node_coords=node_coords,
        elements=elements,
        partitions=partitions,
        periodic_links=np.array([[[0], [num_nodes - 1]]], dtype=np.int32),
    )

    mesh = premesh.finalize(axis_name='i')

    # mesh.exchange sums up nodal values along shared nodes
    u = (1 + jnp.arange(num_nodes, dtype=jnp.float32))[mesh.node_indices]
    u = jax.pmap(lambda x: x)(u)
    u_exchanged = jax.pmap(
        lambda mesh, x: mesh.exchange(x), axis_name='i')(mesh, u)

    # The periodicity affects the first and last nodes
    # original values in `u` are
    #   [[1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 1]]
    # the exchange operation sums up along the partition boundaries as well
    # as the periodic link
    expected_u_exchanged = np.array(
        [[2, 2, 6], [6, 4, 10], [10, 6, 14], [14, 8, 2]], dtype=np.float32)
    np.testing.assert_array_almost_equal(expected_u_exchanged, u_exchanged)


if __name__ == '__main__':
  jax.config.update('jax_enable_x64', True)
  absltest.main()
