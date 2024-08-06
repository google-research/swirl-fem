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

"""A collection of premeshes."""

import collections
from collections.abc import Sequence
import itertools

import more_itertools
import numpy as np
from swirl_fem.common import facet_util
from swirl_fem.common.facet_util import FacetDimType
from swirl_fem.core.premesh import Premesh


def _unit_cube_facet_elements(
    num_elements_per_dim: int, facet: tuple[FacetDimType, ...]
) -> list[list[int]]:
  """Get elements of a unit cube mesh corresponding to the given facet.

  Args:
    num_elements_per_dim: Number of elements along each dimension.
    facet: Tuple of `FacetDimType` specifying the facet.

  Returns:
    List of elements on the facet. Each element is a list of node indices.
  """

  ndim = len(facet)
  # Create elements by iterating through all pairs of '1D' elements, which are
  # just pairwise node indices `{(i, i + 1) | 0 <= i < num_nodes_per_dim}`.
  elements_1d = list(
      more_itertools.pairwise(np.arange(num_elements_per_dim + 1)))
  dim_to_elem = {
      FacetDimType.FIRST: [(0,)],
      FacetDimType.INNER: elements_1d,
      FacetDimType.LAST: [(num_elements_per_dim,)],
  }

  # Convert node indices from N-tuples to the canonical global index.
  def _get_node_idx(multi_idx):
    return np.ravel_multi_index(multi_idx, [num_elements_per_dim + 1] * ndim)

  elements = []
  for element_tuple in itertools.product(*map(dim_to_elem.get, facet)):
    # From each element N-tuple, create the N-D element containing 2^N nodes.
    # E.g. in 2D, the element tuple ((0, 1), (5, 6)) becomes a single element
    # containing 4 nodes: [(0, 5), (0, 6), (1, 5), (1, 6)].
    element_nd = itertools.product(*element_tuple)
    element = map(_get_node_idx, element_nd)
    elements.append(list(element))
  return elements


def unit_cube_mesh(
    num_elements_per_dim: int,
    ndim: int = 2,
    a: float = 0.0,
    b: float = 1.0,
    periodic_dims: Sequence[int] = (),
    partitions: np.ndarray | None = None,
):
  """Returns a uniform mesh over the unit n-dimensional cube [a, b]^ndim.

  Args:
    num_elements_per_dim: Number of elements along each dimension.
    ndim: Dimension of the mesh.
    a: Coordinate of the first node along each dimension.
    b: Coordinate of the last node along each dimension.
    periodic_dims: Dimensions along which the mesh is periodic.
    partitions: Partitioning of the mesh, described as an `ndim`-dimensional
      array containing partition ids between 0 and `num_partitions - 1`, where
      num_partitions is `partitions.size`. E.g. `[[0, 1], [2, 3]]` partitions
      a 2D unit square into 4 equal, contiguous partitions.

  Returns:
    A `Premesh` object containing the unit cube mesh.
  """

  # Create node coordinates as the uniform cartesian grid over the unit cube.
  num_nodes_1d = num_elements_per_dim + 1
  num_nodes = num_nodes_1d**ndim
  node_coords_1d = np.linspace(a, b, num=num_nodes_1d)
  node_coords = np.meshgrid(*([node_coords_1d] * ndim), indexing='ij')
  node_coords = np.stack(node_coords, axis=-1).reshape(num_nodes, ndim)

  elements = _unit_cube_facet_elements(
      num_elements_per_dim, facet=(FacetDimType.INNER,) * ndim
  )

  # Go over all boundary facets of the unit cube. Each axis has two
  # boundary facets (normal to the axis).
  axis_to_facets = collections.defaultdict(list)
  for facet in facet_util.get_facet_types(ndim, facet_ndim=ndim - 1):
    facet_elements = _unit_cube_facet_elements(num_elements_per_dim, facet)
    facet_axis = (facet.index(FacetDimType.FIRST) if FacetDimType.FIRST in facet
                  else facet.index(FacetDimType.LAST))
    axis_to_facets[facet_axis].append(facet_elements)

  # Add each facet to periodic links if the axis is periodic, otherwise add it
  # to the boundary physical group.
  boundary_facets = []
  periodic_links = []
  for axis in range(ndim):
    if axis in periodic_dims:
      periodic_links.extend(np.array(axis_to_facets[axis]).transpose((1, 0, 2)))
    else:
      for facet in axis_to_facets[axis]:
        boundary_facets.extend(facet)

  elements = np.array(elements, dtype=np.int32)
  periodic_links = np.array(periodic_links, dtype=np.int32)
  boundary_facets = np.array(boundary_facets, dtype=np.int32)
  physical_groups = {}
  if boundary_facets.size:
    physical_groups['boundary'] = boundary_facets

  if partitions is not None:
    for axis in range(ndim):
      assert num_elements_per_dim % partitions.shape[axis] == 0, (
          partitions.shape)
      partitions = np.repeat(
          partitions,
          repeats=num_elements_per_dim // partitions.shape[axis],
          axis=axis)
    partitions = partitions.reshape(len(elements))

  return Premesh.create(
      node_coords=node_coords,
      elements=elements,
      periodic_links=periodic_links if periodic_links.size else None,
      physical_groups=physical_groups,
      partitions=partitions)
