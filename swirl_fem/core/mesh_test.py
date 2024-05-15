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

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from swirl_fem.core.interpolation import Nodes1D
from swirl_fem.core.interpolation import NodeType
from swirl_fem.core.mesh import Mesh


class MeshTest(absltest.TestCase):

  def test_mesh_1d(self):
    num_elements = 8
    num_nodes = num_elements + 1
    node_coords = np.linspace(0, 1, num_nodes).reshape((num_nodes, 1))
    elements = np.array([[i, i + 1] for i in range(num_elements)])
    mesh = Mesh.create(node_coords=node_coords, elements=elements)

    self.assertEqual(1, mesh.order)
    self.assertEqual(1, mesh.ndim)
    self.assertEqual(num_nodes, mesh.num_nodes)
    self.assertEqual(num_elements, mesh.num_elements)
    self.assertEqual(2, mesh.num_nodes_per_element)

    # Verify that gathering nodal values works.
    nodal_values = jnp.flip(jnp.arange(num_nodes, dtype=jnp.float32), axis=0)
    expected = jnp.array(
        [[nodal_values[i], nodal_values[i + 1]] for i in range(num_elements)],
        dtype=jnp.float32)
    np.testing.assert_allclose(expected, mesh.gather(nodal_values))

    # Verify element coordinates.
    np.testing.assert_allclose(node_coords[elements], mesh.element_coords())

  def test_mesh_2d(self):
    # A 2D mesh with 2 elements and 6 nodes.
    num_elements = 2
    num_nodes = 6
    node_coords = np.array(
        [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]], dtype=np.float32)
    elements = np.array([[0, 1, 2, 3], [2, 3, 4, 5]], dtype=np.int32)
    mesh = Mesh.create(node_coords=node_coords, elements=elements)

    self.assertEqual(1, mesh.order)
    self.assertEqual(2, mesh.ndim)
    self.assertEqual(num_nodes, mesh.num_nodes)
    self.assertEqual(num_elements, mesh.num_elements)
    self.assertEqual(4, mesh.num_nodes_per_element)

    # Verify that gathering nodal values works.
    nodal_values = jnp.arange(num_nodes, dtype=jnp.float32)
    expected = jnp.array(elements, jnp.float32)
    np.testing.assert_allclose(expected, mesh.gather(nodal_values))

    # Verify element coordinates.
    np.testing.assert_allclose(node_coords[elements], mesh.element_coords())

  def test_invalid_input(self):
    # Create a mesh of order 1; but specify gridpoints with 3 nodes per element.
    num_elements = 8
    num_nodes = num_elements + 1
    node_coords = np.linspace(0, 1, num_nodes).reshape((num_nodes, 1))
    elements = np.array([[i, i + 1] for i in range(num_elements)])

    # Use the wrong number of nodes in `gridpoints`.
    gridpoints = Nodes1D.create(num_points=3, node_type=NodeType.NEWTON_COTES)
    with self.assertRaisesRegex(ValueError, 'number of nodes'):
      Mesh.create(node_coords=node_coords, elements=elements,
                  gridpoints_1d=gridpoints)

if __name__ == '__main__':
  jax.config.update('jax_enable_x64', True)
  absltest.main()
