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

import collections
import itertools

from absl.testing import absltest
import numpy as np
from swirl_fem.common import facet_util
from swirl_fem.common.facet_util import FacetDimType


class FacetUtilTest(absltest.TestCase):

  def test_slice_from_facet_type_1d(self):
    # Create a 1D element with 3 nodes
    #   0 -- 1 -- 2
    element = np.arange(3, dtype=np.int32)

    facet_slice = facet_util.slice_from_facet_type(
        (FacetDimType.FIRST,), interior_nodes_only=False)
    np.testing.assert_array_equal(element[facet_slice], [0])

    facet_slice = facet_util.slice_from_facet_type(
        (FacetDimType.INNER,), interior_nodes_only=False)
    np.testing.assert_array_equal(element[facet_slice], [0, 1, 2])

    facet_slice = facet_util.slice_from_facet_type(
        (FacetDimType.INNER,), interior_nodes_only=True)
    np.testing.assert_array_equal(element[facet_slice], [1])

    facet_slice = facet_util.slice_from_facet_type(
        (FacetDimType.LAST,), interior_nodes_only=False)
    np.testing.assert_array_equal(element[facet_slice], [2])

  def test_slice_from_facet_type_2d(self):
    # Create a 2D element with 9 nodes
    #   2 -- 5 -- 8
    #   |    |    |
    #   1 -- 4 -- 7
    #   |    |    |
    #   0 -- 3 -- 6
    element = np.arange(9, dtype=np.int32).reshape((3, 3))
    # Fetch the nodes from the facet comprising the first axis and the 'first'
    # node in the second axis.
    facet = (FacetDimType.INNER, FacetDimType.FIRST)
    facet_slice = facet_util.slice_from_facet_type(
        facet, interior_nodes_only=False)
    np.testing.assert_array_equal(element[facet_slice], [0, 3, 6])

  def test_slice_from_facet_type_interior(self):
    # Create a 2D element with 9 nodes
    #   2 -- 5 -- 8
    #   |    |    |
    #   1 -- 4 -- 7
    #   |    |    |
    #   0 -- 3 -- 6
    element = np.arange(9, dtype=np.int32).reshape((3, 3))
    # Fetch the interior nodes from the facet comprising the first axis and the
    # 'first' node in the second axis.
    facet = (FacetDimType.INNER, FacetDimType.FIRST)
    facet_slice = facet_util.slice_from_facet_type(
        facet, interior_nodes_only=True)
    np.testing.assert_array_equal(element[facet_slice], [3])

  def test_slice_from_facet_type_disjoint_union(self):
    # Create a 2D element with 9 nodes
    #   2 -- 5 -- 8
    #   |    |    |
    #   1 -- 4 -- 7
    #   |    |    |
    #   0 -- 3 -- 6
    element = np.arange(9, dtype=np.int32).reshape((3, 3))
    # A disjoint union of the all the facets reconstructs the element.
    # Keep a count of nodes which appear in each of the 3^2 facets.
    node_count = collections.Counter()
    for facet_type in itertools.product(
        [FacetDimType.FIRST, FacetDimType.INNER, FacetDimType.LAST], repeat=2):
      facet_slice = facet_util.slice_from_facet_type(
          facet_type, interior_nodes_only=True)
      facet = element[facet_slice]
      for node in facet.reshape((-1)).tolist():
        node_count[node] += 1

    self.assertLen(node_count, 9)
    for i in range(9):
      self.assertEqual(node_count[i], 1)

  def test_slice_from_facet_type_3d(self):
    # Create a 3D element with 8 nodes
    #     3----7
    #    /|   /|
    #   1----5 |
    #   | |  | |   ax2
    #   | 2--|-6   | / ax1
    #   |/   |/    |/
    #   0----4     ---> ax0
    element = np.arange(8, dtype=np.int32).reshape((2, 2, 2))

    # get the top-left edge along axis 1 (into the monitor), located at the
    # 'first' side along axis 0 (left) and 'last' side along axis 2 (top).
    facet_slice = facet_util.slice_from_facet_type(
        (FacetDimType.FIRST, FacetDimType.INNER, FacetDimType.LAST),
        interior_nodes_only=False,
    )
    np.testing.assert_array_equal(element[facet_slice], [1, 3])

  def test_get_facet_types_len(self):
    for ndim in [1, 2, 3]:
      self.assertLen(facet_util.get_facet_types(ndim), 3**ndim)

  def test_get_facet_types_1d(self):
    facet_types = set(facet_util.get_facet_types(ndim=1))
    self.assertLen(facet_types, 3)
    self.assertContainsSubset(
        [(FacetDimType.FIRST,), (FacetDimType.INNER,), (FacetDimType.LAST,)],
        facet_types,
    )

  def test_get_facet_types_facet_ndim(self):
    facet_types = set(facet_util.get_facet_types(ndim=2, facet_ndim=1))
    self.assertLen(facet_types, 4)
    self.assertContainsSubset(
        [
            (FacetDimType.FIRST, FacetDimType.INNER),
            (FacetDimType.INNER, FacetDimType.FIRST),
            (FacetDimType.LAST, FacetDimType.INNER),
            (FacetDimType.INNER, FacetDimType.LAST),
        ],
        facet_types,
    )

  def test_orderings_map_2d_num_points_2(self):
    # For number of target points 2, we simply get an identity mapping
    orderings = facet_util.get_orderings_mapping(ndim=2, num_points_1d=2)
    # there should be 8 orderings since 2^2 * 2! = 8
    self.assertLen(orderings, 8)
    for key in orderings.keys():
      self.assertEqual(tuple(orderings[key]), key)

  def test_orderings_map_1d_num_points_3(self):
    orderings = facet_util.get_orderings_mapping(ndim=1, num_points_1d=3)
    # there should be 2 orderings since 2^1 * 1! = 2
    self.assertLen(orderings, 2)

    # The first order and 2nd order elements in canonical ordering:
    #   0 -- 1      0 -- 1 -- 2
    # Verify both orderings
    self.assertEqual(tuple(orderings[(0, 1)]), (0, 1, 2))
    self.assertEqual(tuple(orderings[(1, 0)]), (2, 1, 0))

  def test_orderings_map_2d_num_points_3(self):
    orderings = facet_util.get_orderings_mapping(ndim=2, num_points_1d=3)
    # there should be 8 orderings since 2^2 * 2! = 8
    self.assertLen(orderings, 8)

    # The first order and 2nd order elements in canonical ordering:
    #             2 -- 5 -- 8
    #  1 -- 3     |    |    |
    #  |    |     1 -- 4 -- 7
    #  0 -- 2     |    |    |
    #             0 -- 3 -- 6

    # Verify a few orderings
    self.assertEqual(tuple(orderings[(0, 1, 2, 3)]), tuple(range(9)))
    self.assertEqual(tuple(orderings[(1, 3, 0, 2)]),
                     (2, 5, 8, 1, 4, 7, 0, 3, 6))
    self.assertEqual(tuple(orderings[(2, 0, 3, 1)]),
                     (6, 3, 0, 7, 4, 1, 8, 5, 2))


if __name__ == "__main__":
  absltest.main()
