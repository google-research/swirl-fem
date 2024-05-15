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

"""Utility functions for manipulating n-dimensional elements and their facets.

We consider quadrilateral or hexahedral n-dimensional elements (n-dimensional
cubes in general). An n-dimensional element has 2ⁿ vertices and (k - 1)ⁿ
interior nodes, where k is the element order.

We partition facets of an n-dimensional cube according to the facet dimensions.
An n-dimensional cube contains 3ⁿ facets. Specifically, we can partition all
nodes into the interior nodes of the facets. There are:
  (k + 1)ⁿ = ((k - 1) + 2)ⁿ nodes
           = (k - 1)ⁿ interior nodes
             + (2n) (k - 1)ⁿ⁻¹ interior nodes on 2n (n-1)-dimensional "faces"
             + ...
             + 2ⁿ vertices

Each of these 3ⁿ facets can be canonically identified by an n-tuple. The ith
element of the tuple specifies whether we take the 'first' vertex, the 'last'
vertex or the `k - 1` interior nodes along the ith axis. Indeed if using the
encoding t ∊ {-1, 0, 1}ⁿ, with "-1" for first, "0" for inner", and "1" for
last, then t is geometrically the center of the facet of the reference element
[-1, 1]ⁿ.
"""

from collections.abc import Iterable
import enum
import itertools
import more_itertools
import numpy as np


@enum.unique
class FacetDimType(enum.Enum):
  """Category of nodes included in a facet along a dimension."""
  FIRST = 'first'
  LAST = 'last'
  INNER = 'inner'


def slice_from_facet_type(facet_type: tuple[FacetDimType],
                          interior_nodes_only: bool):
  """Returns slice extracting the given facet from a tensor-product element.

  Given an index array `e` of shape `[k + 1, k + 1, ... (n times), k + 1]`
  containing the node indices of an element of order `k`, this function returns
  a slice `s` such that `e[s]` denotes the nodes forming a particular facet of
  the element. The facet is denoted by the signature `facet_type`.

  If `interior_nodes_only` is true, then for `INNER` facet dims, we return the
  slice corresponding only to the interior nodes in the facet, otherwise all
  the nodes of the facet are included.

  Args:
    facet_type: The signature of the facet.
    interior_nodes_only: Whether to restrict to interior nodes only.
  """
  mapping = {
      FacetDimType.FIRST: 0,
      FacetDimType.LAST: -1,
      FacetDimType.INNER: slice(1, -1) if interior_nodes_only else slice(None)
  }
  return tuple(map(lambda t: mapping[t], facet_type))


def get_facet_types(
    ndim: int, facet_ndim: int | None = None
) -> Iterable[tuple[FacetDimType]]:
  """Returns iterable representing facets of a `ndim`-dimensional element.

  Args:
    ndim: The dimension of the element.
    facet_ndim: The dimension of the facets of the element. If not None, returns
      only the facets of this dimension, otherwise, returns all facets.
  """
  facets = list(itertools.product(list(FacetDimType), repeat=ndim))
  if facet_ndim is None:
    return facets

  return [f for f in facets if f.count(FacetDimType.INNER) == facet_ndim]


def get_orderings_mapping(ndim: int, num_points_1d: int):
  """Returns a mapping from facet signatures to permutation describing it.

  A d-dimensional element can be represented as different tensor-product
  orderings of nodes within a mesh, when it appears as a facet of a larger
  element. For instance, consider the following elements:
               2 -- 5 -- 8
    1 -- 3     |    |    |
    |    |     1 -- 4 -- 7
    0 -- 2     |    |    |
               0 -- 3 -- 6

  The first order element can be represent by either of the node sequences
  `[0, 1, 2, 3]` or `[2, 3, 0, 1]`, or `[1, 3, 0, 2]` (among others).
  Corresponding orders of the 2nd order element are:
  `[0, 1, 2, 3, 4, 5, 6, 7, 8]`, `[6, 7, 8, 3, 4, 5, 0, 1, 2]` and
  `[2, 5, 8, 1, 4, 7, 0, 3, 6].

  In general, there could be 2^d * d! different orderings since the axes
  can appear in any order, and each axis can be independently reversed.

  This function returns a mapping from orderings of the primary nodes (of the
  element in the first order mesh) to the corresponding ordering of the higher
  order mesh. For instance in the above example `[2, 3, 0, 1]` would map to
  `[6, 7, 8, 3, 4, 5, 0, 1, 2]`.

  Args:
    ndim: The dimension of the element.
    num_points_1d: The number of points along each axes of the tensor-product
      elemenet (this is the order of the element plus 1).

  Returns:
    A mapping from permutations of {0, ..., 2^ndim - 1} representing possible
    orderings of a first order `ndim`-dimensional element, to the corresponding
    orderings of a `num_points_1d - 1` order element of the same dimension.
  """
  source_elem = np.arange(2 ** ndim, dtype=np.int32).reshape([2] * ndim)
  target_elem = np.arange(num_points_1d ** ndim, dtype=np.int32).reshape(
      [num_points_1d] * ndim)
  # Iterate through all possible orientations of `ndim`-dimensional facets and
  # map the source facet to the corresponding permutation of the target facet.
  orderings = {}
  for perm in itertools.permutations(range(ndim)):
    for axes in more_itertools.powerset(range(ndim)):
      ordered_source = np.flip(source_elem.transpose(perm), axes)
      ordered_target = np.flip(target_elem.transpose(perm), axes)
      orderings[tuple(ordered_source.flatten().tolist())] = (
          ordered_target.flatten())
  return orderings
