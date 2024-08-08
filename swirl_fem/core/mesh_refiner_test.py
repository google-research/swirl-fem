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

import itertools

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from swirl_fem.common import facet_util
from swirl_fem.common.facet_util import FacetDimType
from swirl_fem.core.interpolation import Nodes1D
from swirl_fem.core.interpolation import NodeType
from swirl_fem.core.mesh_refiner import refine_premesh
from swirl_fem.core.premesh import Premesh
from swirl_fem.common.premesh_commons import unit_cube_mesh


class RefineMeshTest(parameterized.TestCase):

  @parameterized.parameters(1, 2, 3)
  def test_single_element(self, ndim):
    node_coords = np.array(
        list(itertools.product([0, 1], repeat=ndim)), dtype=np.float32)
    # Add a single cubical element containing all the nodes
    elements = np.array([list(range(len(node_coords)))], dtype=np.int32)
    premesh = Premesh.create(node_coords=node_coords, elements=elements)

    # Refine to a second-order mesh
    gridpoints = Nodes1D.create(3, NodeType.NEWTON_COTES)
    refined_premesh = refine_premesh(premesh, gridpoints_1d=gridpoints)
    self.assertEqual(refined_premesh.gridpoints_1d, gridpoints)
    self.assertEqual(refined_premesh.num_nodes, 3 ** ndim)
    self.assertEqual(refined_premesh.num_elements, 1)
    self.assertEqual(refined_premesh.num_nodes_per_element, 3 ** ndim)

  @parameterized.parameters((2, 2), (2, 3), (3, 3), (2, 5), (4, 5))
  def test_1d_gll(self, num_elements, order):
    # Create the first order 1D mesh to be refined
    num_nodes = num_elements + 1
    node_coords = np.arange(num_nodes, dtype=np.float32).reshape([num_nodes, 1])
    elements = np.array(
        [[i, i + 1] for i in range(num_elements)], dtype=np.int32)
    premesh = Premesh.create(node_coords=node_coords, elements=elements)
    gridpoints = Nodes1D.create(num_points=order + 1,
                                node_type=NodeType.GAUSS_LOBATTO_LEGENDRE)
    refined_premesh = refine_premesh(premesh, gridpoints_1d=gridpoints)
    self.assertEqual(refined_premesh.gridpoints_1d, gridpoints)
    self.assertEqual(refined_premesh.num_nodes, num_elements * order + 1)
    # number of elements remains the same
    self.assertEqual(refined_premesh.num_elements, num_elements)

  def test_two_elements_newton_cotes(self):
    # create a 2d mesh with 2 elements and 6 nodes arranged as follows
    #   1 -- 3 -- 5
    #   |    |    |
    #   0 -- 2 -- 4
    node_coords = np.array(list(itertools.product([0, 1, 2], [0, 1])),
                           dtype=np.float32)
    elements = np.array([[0, 1, 2, 3], [3, 5, 2, 4]], dtype=np.int32)
    premesh = Premesh.create(
        node_coords=node_coords,
        elements=elements,
    )
    refined_premesh = refine_premesh(
        premesh=premesh,
        gridpoints_1d=Nodes1D.create(
            num_points=3, node_type=NodeType.NEWTON_COTES))

    # the resulting refined mesh contains 7 edge nodes and 2 interior nodes for
    # a total of 15 nodes (including the original 6 nodes).
    self.assertEqual(refined_premesh.num_nodes, 15)
    # the number of elements remain the same
    self.assertEqual(refined_premesh.num_elements, 2)
    self.assertEqual(refined_premesh.num_nodes_per_element, 9)

  def test_two_elements_newton_cotes_common_facet(self):
    # create a 2d mesh with 2 elements and 6 nodes arranged as follows
    #   1 -- 3 -- 5
    #   |    |    |
    #   0 -- 2 -- 4
    node_coords = np.array(list(itertools.product([0, 1, 2], [0, 1])),
                           dtype=np.float32)
    # Put the second element in an order which makes the common facet appear
    # in 'reversed' order when seen from the canonical ordering. This tests out
    # the deduplication logic in the mesh refiner.
    elements = np.array([[0, 1, 2, 3], [3, 5, 2, 4]], dtype=np.int32)
    premesh = Premesh.create(
        node_coords=node_coords,
        elements=elements,
    )
    refined_premesh = refine_premesh(
        premesh=premesh,
        gridpoints_1d=Nodes1D.create(
            num_points=3, node_type=NodeType.NEWTON_COTES))

    # verify the common facet has the same nodes
    facet_type = (FacetDimType.LAST, FacetDimType.INNER)
    facet_slice = facet_util.slice_from_facet_type(
        facet_type, interior_nodes_only=False
    )
    right_facet = (
        refined_premesh.elements[0].reshape((3, 3))[facet_slice].flatten()
    )

    facet_type = (FacetDimType.INNER, FacetDimType.FIRST)
    facet_slice = facet_util.slice_from_facet_type(
        facet_type, interior_nodes_only=False
    )
    left_facet = (
        refined_premesh.elements[1].reshape((3, 3))[facet_slice].flatten()
    )

    # Make sure the facets are equal as sets.
    self.assertLen(right_facet, 3)
    self.assertEqual(set(right_facet), set(left_facet))

    # Check node coordinates of the right facet.
    expected_coords = np.array(
        list(itertools.product([1.0], [0.0, 0.5, 1.0])),
        dtype=np.float32,
    )
    np.testing.assert_array_almost_equal(
        refined_premesh.node_coords[right_facet], expected_coords)

  def test_two_elements_gll(self):
    # create a 2d mesh with 2 elements and 6 nodes arranged as follows
    #   1 -- 3 -- 5
    #   |    |    |
    #   0 -- 2 -- 4
    node_coords = np.array(list(itertools.product([0, 1, 2], [0, 1])),
                           dtype=np.float32)
    # Put the second element in an order which makes the common facet appear
    # in 'reversed' order when seen from the canonical ordering. This tests out
    # the deduplication logic in the mesh refiner.
    elements = np.array([[0, 1, 2, 3], [3, 5, 2, 4]], dtype=np.int32)
    premesh = Premesh.create(node_coords=node_coords, elements=elements)
    gridpoints = Nodes1D.create(num_points=5,
                                node_type=NodeType.GAUSS_LOBATTO_LEGENDRE)
    refined_premesh = refine_premesh(premesh=premesh, gridpoints_1d=gridpoints)
    self.assertEqual(refined_premesh.gridpoints_1d, gridpoints)
    self.assertEqual(refined_premesh.num_nodes, 45)
    self.assertEqual(refined_premesh.num_elements, 2)
    self.assertEqual(refined_premesh.num_nodes_per_element, 25)

  def test_two_elements_3d_order_2(self):
    # Create a mesh with two elements and 12 nodes
    #     3----7----11
    #    /|   /|   /|
    #   1----5----9 |
    #   | |  | |  | |
    #   | 2--|-6--|-10
    #   |/   |/   |/
    #   0----4----8
    node_coords = np.array(list(itertools.product([0, 1, 2], [0, 1], [0, 1])),
                           dtype=np.float32)
    # Put the second element in an order which makes the common 2d facet appear
    # in a non-canonical order when seen from the canonical ordering of the 3d
    # element. This tests out the deduplication logic in the mesh refiner.
    elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7], [4, 8, 5, 9, 6, 10, 7, 11]],
                        dtype=np.int32)
    premesh = Premesh.create(node_coords=node_coords, elements=elements)
    gridpoints = Nodes1D.create(num_points=3, node_type=NodeType.NEWTON_COTES)
    refined_premesh = refine_premesh(premesh=premesh, gridpoints_1d=gridpoints)
    self.assertEqual(refined_premesh.gridpoints_1d, gridpoints)
    self.assertEqual(refined_premesh.num_nodes, 9 * 5)
    self.assertEqual(refined_premesh.num_elements, 2)
    self.assertEqual(refined_premesh.num_nodes_per_element, 27)

  def test_two_elements_3d_order_2_common_facet(self):
    # Create a mesh with two elements and 12 nodes
    #     3----7----11
    #    /|   /|   /|
    #   1----5----9 |
    #   | |  | |  | |
    #   | 2--|-6--|-10
    #   |/   |/   |/
    #   0----4----8
    node_coords = np.array(list(itertools.product([0, 1, 2], [0, 1], [0, 1])),
                           dtype=np.float32)
    # Put the second element in an order which makes the common 2d facet appear
    # in a non-canonical order when seen from the canonical ordering of the 3d
    # element. This tests out the deduplication logic in the mesh refiner.
    elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7], [5, 7, 4, 6, 9, 11, 8, 10]],
                        dtype=np.int32)
    premesh = Premesh.create(node_coords=node_coords, elements=elements)
    gridpoints = Nodes1D.create(num_points=3, node_type=NodeType.NEWTON_COTES)
    refined_premesh = refine_premesh(premesh, gridpoints_1d=gridpoints)

    # Check the common facet. The right face of the first element and the left
    # face of the second element should contain the same node indices.
    facet_type = (FacetDimType.LAST, FacetDimType.INNER, FacetDimType.INNER)
    facet_slice = facet_util.slice_from_facet_type(
        facet_type, interior_nodes_only=False
    )
    right_facet = (
        refined_premesh.elements[0].reshape((3, 3, 3))[facet_slice].flatten()
    )

    facet_type = (FacetDimType.FIRST, FacetDimType.INNER, FacetDimType.INNER)
    facet_slice = facet_util.slice_from_facet_type(
        facet_type, interior_nodes_only=False
    )
    left_facet = (
        refined_premesh.elements[1].reshape((3, 3, 3))[facet_slice].flatten()
    )

    # Make sure the facets are equal as sets.
    self.assertLen(right_facet, 9)
    self.assertEqual(set(right_facet), set(left_facet))

    # Check node coordinates of the right facet.
    expected_coords = np.array(
        list(itertools.product([1.0], [0.0, 0.5, 1.0], [0.0, 0.5, 1.0])),
        dtype=np.float32,
    )
    np.testing.assert_array_almost_equal(
        refined_premesh.node_coords[right_facet], expected_coords)

  def test_two_elements_3d_order_4(self):
    # Create a mesh with two elements and 12 nodes
    #     3----7----11
    #    /|   /|   /|
    #   1----5----9 |
    #   | |  | |  | |
    #   | 2--|-6--|-10
    #   |/   |/   |/
    #   0----4----8
    node_coords = np.array(list(itertools.product([0, 1, 2], [0, 1], [0, 1])),
                           dtype=np.float32)
    # Put the second element in an order which makes the common 2d facet appear
    # in a non-canonical order when seen from the canonical ordering of the 3d
    # element. This tests out the deduplication logic in the mesh refiner.
    elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7], [4, 8, 5, 9, 6, 10, 7, 11]],
                        dtype=np.int32)
    premesh = Premesh.create(node_coords=node_coords, elements=elements)
    gridpoints = Nodes1D.create(num_points=5,
                                node_type=NodeType.GAUSS_LOBATTO_LEGENDRE)
    refined_premesh = refine_premesh(premesh, gridpoints_1d=gridpoints)
    self.assertEqual(refined_premesh.gridpoints_1d, gridpoints)
    self.assertEqual(refined_premesh.num_nodes, 25 * (5 + 4))
    self.assertEqual(refined_premesh.num_elements, 2)
    self.assertEqual(refined_premesh.num_nodes_per_element, 125)

  def test_refine_physical_groups(self):
    # create a 2d mesh with 2 elements and 6 nodes arranged as follows
    #   1 -- 3 -- 5
    #   |    |    |
    #   0 -- 2 -- 4
    node_coords = np.array(list(itertools.product([0, 1, 2], [0, 1])),
                           dtype=np.float32)
    elements = np.array([[0, 1, 2, 3], [3, 5, 2, 4]], dtype=np.int32)
    physical_groups = {'boundary': np.array([[0, 1], [4, 5]])}

    premesh = Premesh.create(
        node_coords=node_coords,
        elements=elements,
        physical_groups=physical_groups,
    )
    gridpoints = Nodes1D.create(num_points=3,
                                node_type=NodeType.GAUSS_LOBATTO_LEGENDRE)
    refined_premesh = refine_premesh(premesh, gridpoints_1d=gridpoints)
    self.assertEqual(refined_premesh.physical_groups.keys(),
                     premesh.physical_groups.keys())
    self.assertEqual(refined_premesh.physical_groups['boundary'].shape,
                     (2, gridpoints.num_points))

  def test_refine_periodic_links(self):
    # create a 2d mesh with 2 elements and 6 nodes arranged as follows
    #   1 -- 3 -- 5
    #   |    |    |
    #   0 -- 2 -- 4
    node_coords = np.array(list(itertools.product([0, 1, 2], [0, 1])),
                           dtype=np.float32)
    elements = np.array([[0, 1, 2, 3], [3, 5, 2, 4]], dtype=np.int32)
    # the boundary facets [0, 1] and [4, 5] have a periodic link.
    periodic_links = np.array([[[0, 1], [4, 5]]])

    premesh = Premesh.create(
        node_coords=node_coords,
        elements=elements,
        periodic_links=periodic_links,
    )
    gridpoints = Nodes1D.create(num_points=3,
                                node_type=NodeType.GAUSS_LOBATTO_LEGENDRE)
    refined_premesh = refine_premesh(premesh, gridpoints_1d=gridpoints)
    self.assertEqual(refined_premesh.periodic_links.shape,
                     (1, 2, gridpoints.num_points))

  def test_refine_discontinuous(self):
    # create a 2d mesh with 2 elements and 6 nodes arranged as follows
    #   1 -- 3 -- 5
    #   |    |    |
    #   0 -- 2 -- 4
    node_coords = np.array(list(itertools.product([0, 1, 2], [0, 1])),
                           dtype=np.float32)
    elements = np.array([[0, 1, 2, 3], [3, 5, 2, 4]], dtype=np.int32)
    premesh = Premesh.create(
        node_coords=node_coords,
        elements=elements,
    )
    gridpoints = Nodes1D.create(num_points=3,
                                node_type=NodeType.GAUSS_LEGENDRE)
    refined_premesh = refine_premesh(premesh, gridpoints_1d=gridpoints)
    # discontinuous gridpoints makes elements' nodes disjoint from each other
    self.assertEqual(refined_premesh.num_nodes,
                     len(elements) * gridpoints.num_points ** 2)


if __name__ == '__main__':
  absltest.main()
