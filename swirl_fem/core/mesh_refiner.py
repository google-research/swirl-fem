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

"""Performs p-refinement on first order meshes to form higher order meshes.

Defines the function `refine_premesh` which takes a `Premesh` (where each
element is a deformed n-dimensional cube) and performs p-refinement. The
resulting refined `Premesh` has a tensor-product structure and each element in
the mesh has the same order of refinement. The type of the resulting higher
order nodal distribution is described by a supplied `Nodes1D` object.
"""

from collections.abc import MutableMapping

import numpy as np

from swirl_fem.common import facet_util
from swirl_fem.common.facet_util import FacetDimType
from swirl_fem.core.interpolation import BarycentricInterpolator
from swirl_fem.core.interpolation import Nodes1D
from swirl_fem.core.premesh import Premesh


def refine_premesh(premesh: Premesh, gridpoints_1d: Nodes1D) -> Premesh:
  """Returns a p-refined premesh with the specified `gridpoints_1d` per element.

  The refined elements have a tensor-product ordering of nodes with the nodes in
  each dimension distributed according to the specified `gridpoints_1d`.

  Args:
    premesh: The premesh to be refined. Must have order 1.
    gridpoints_1d: The distribution of nodes within each element along each
      dimension.

  Returns:
    The refined premesh.

  Raises:
    ValueError: If the premesh has order different from 1.
    NotImplementedError: If the premesh has physical groups, or if the
      gridpoints are discontinuous.
  """
  if premesh.order != 1:
    raise ValueError(f'Expecting mesh of order 1. Got {premesh.order}.')

  return _MeshRefiner(premesh=premesh, gridpoints_1d=gridpoints_1d).refine()


class _MeshRefiner:
  """A single-use class which refines a premesh.

  Recall that a `Premesh` represents a mesh using numpy arrays, while
  `premesh.finalize()` creates a `Mesh`, which can be placed on a Jax device
  (possibly distributed). A `MeshRefiner` takes in a first order `Premesh` and
  gridpoints_1d (a `Nodes1D` object), and provides a single function `refine()`
  which creates a refined `Premesh` with the same number of elements and
  nodal distribution specified in `gridpoints_1d`.

  Attributes:
    premesh: The first order `Premesh` object to refine.
    gridpoints_1d: The nodal type of the resulting refined mesh. Typically this
      specifies a higher order Newton-Cotes, GLL or GL nodes.
    interpolator: An interpolator object used for interpolating the node
      coordinates from the first order mesh to the refined higher order mesh.
  """
  premesh: Premesh
  gridpoints_1d: Nodes1D
  interpolator: BarycentricInterpolator

  def __init__(
      self,
      premesh: Premesh,
      gridpoints_1d: Nodes1D,
  ):
    self.premesh = premesh
    self.gridpoints_1d = gridpoints_1d
    self.interpolator = BarycentricInterpolator(
        ndim=premesh.ndim,
        gridpoints_1d=premesh.gridpoints_1d,
        evalpoints_1d=gridpoints_1d)
    # Intialize the mesh node coords to the primary nodes. Secondary nodes
    # would be added on a call to refine(...).
    if self.gridpoints_1d.is_continuous():
      self._node_coords = list(premesh.node_coords)
    else:
      self._node_coords = []

    # A mapping from used to deduplicate nodes. Maps the source element facets
    # (represented as a sorted sequence of the facet node ids) to a pair:
    # the first entry of the pair is the facets node ids again in the ordering
    # that it first appeared in the mesh, and the second entry is the
    # refined facet nodes in the same (corresponding) ordering.
    self._inner_facets_mapping: MutableMapping[
        tuple[int, ...], tuple[tuple[int, ...], list[int]]] = {}

    # Maps each dimension to the facet orderings mapping of facets of that
    # dimension. Each ordering is a mapping from a permutation of the facet
    # nodes of order 1, to the corresponding permutation of the facet nodes
    # of the refinement order.
    self._dim_to_orderings_mapping: MutableMapping[
        int, MutableMapping[tuple[int, ...], list[int]]] = {}
    for k in range(1, premesh.ndim):
      self._dim_to_orderings_mapping[k] = facet_util.get_orderings_mapping(
          ndim=k, num_points_1d=self.gridpoints_1d.num_points - 2)

  def _num_target_points(self):
    return self.gridpoints_1d.num_points

  def _add_nodes(self, node_coords: list[np.ndarray]):
    """Adds new nodes and returns the indices of the newly added nodes."""
    node_indices = list(
        range(len(self._node_coords),
              len(self._node_coords) + len(node_coords)))
    self._node_coords.extend(node_coords)
    return node_indices

  def _facet_shape(self, facet_type: tuple[FacetDimType]):
    return tuple([self._num_target_points() - 2] *
                 facet_type.count(FacetDimType.INNER))

  def _add_facets(self, facet_type: tuple[FacetDimType],
                  facets: np.ndarray,
                  target_elements_nd: np.ndarray):
    """Adds the specified facet to the target element."""
    slice_nd = facet_util.slice_from_facet_type(facet_type,
                                                interior_nodes_only=True)
    facets_nd = facets.reshape([len(target_elements_nd)] +
                               list(self._facet_shape(facet_type)))
    target_elements_nd[(slice(None), *slice_nd)] = facets_nd
    return target_elements_nd

  def _refine_facets(
      self,
      facets: np.ndarray,
      ndim: int,
      target_node_coords: np.ndarray | None = None,
  ) -> np.ndarray:
    """Refines the given `ndim`-dimensional facets."""
    # There are 3^ndim subfacets of each facet. In the ith iteration, we refine
    # the subfacets which are of the ith type in a single batch. If target node
    # coords is not None, then the new nodes are also added to the global list
    # of nodes.
    num_facets = len(facets)
    facets_nd = facets.reshape([num_facets] + [2] * ndim)
    target_facets_nd = np.full(
        fill_value=-1,
        shape=[num_facets] + [self._num_target_points()] * ndim)

    target_node_coords_nd = None
    if target_node_coords is not None:
      target_node_coords_nd = target_node_coords.reshape((
          [num_facets] + [self._num_target_points()] * ndim + [ndim]))

    # Iterate through all 3^n facets of the elements.
    for facet_type in facet_util.get_facet_types(ndim):
      source_slice_nd = facet_util.slice_from_facet_type(
          facet_type, interior_nodes_only=False)
      target_slice_nd = facet_util.slice_from_facet_type(
          facet_type, interior_nodes_only=True)
      curr_facets = facets_nd[(slice(None), *source_slice_nd)]

      # If we're at a 0D facet (a primary node), just add the node unchanged.
      facet_dim = facet_type.count(FacetDimType.INNER)
      if facet_dim == 0:
        target_facets_nd = self._add_facets(
            facet_type, curr_facets, target_facets_nd)
        continue

      # Slice out the coordinates of the facet nodes from the precomputed
      # array of node coordinates.
      if target_node_coords_nd is not None:
        facet_node_coords = (
            target_node_coords_nd[(slice(None), *target_slice_nd, slice(None))])
      else:
        facet_node_coords = None

      # For full ndims there's no need for deduplication.
      if facet_dim == self.premesh.ndim and facet_node_coords is not None:
        target_nodes = self._add_nodes(
            list(facet_node_coords.reshape((-1, ndim))))
        target_facets_nd = self._add_facets(
            facet_type, np.array(target_nodes, dtype=np.int32),
            target_facets_nd)
        continue

      # For 1 < facet_dim < mesh.ndim, we need to check for duplicates.
      target_facets = []
      facet_perms = []
      for idx, facet in enumerate(curr_facets):
        facet = facet.reshape(-1).tolist()
        node_key = tuple(sorted(facet))

        facet_value = self._inner_facets_mapping.get(node_key, None)
        if facet_value is None:  # New facet.
          target_nodes = self._add_nodes(
              list(facet_node_coords[idx].reshape((-1, ndim))))
          target_facets.append(target_nodes)
          facet_perms.append(list(range(len(target_nodes))))
          self._inner_facets_mapping[node_key] = (
              (tuple(facet), target_nodes))
        else:  # Existing facet.
          facet_orientation, target_nodes = facet_value
          orientation_key = tuple(
              facet.index(k) for k in facet_orientation)
          facet_perms.append(
              self._dim_to_orderings_mapping[facet_dim][orientation_key])
          target_facets.append(target_nodes)

      target_facets = np.take_along_axis(
          np.array(target_facets, dtype=np.int32),
          np.array(facet_perms, dtype=np.int32),
          axis=-1,
      )
      target_facets_nd = self._add_facets(
          facet_type, target_facets, target_facets_nd)

    return target_facets_nd.reshape(
        (num_facets, self._num_target_points() ** ndim))

  def _refine_elements(self):
    """Refines elements and populate mapping from facets to secondary nodes."""
    target_node_coords = np.einsum(
        'mn,end->emd',
        self.interpolator.interpolation_matrix(),
        self.premesh.node_coords[self.premesh.elements],
    )

    # If discontinous, just add all the target nodes as is and we're done!
    ndim = self.premesh.ndim
    if not self.gridpoints_1d.is_continuous():
      target_elems = self._add_nodes(target_node_coords.reshape((-1, ndim)))
      return np.array(target_elems, dtype=np.int32).reshape(
          (self.premesh.num_elements, self._num_target_points() ** ndim))

    # Otherwise, refine elements with deduplication for the element boundaries.
    return self._refine_facets(
        facets=self.premesh.elements,
        ndim=ndim,
        target_node_coords=target_node_coords,
    )

  def refine(self) -> Premesh:
    """Return the refined mesh."""
    elements = self._refine_elements()
    node_coords = np.stack(self._node_coords)

    # By this time, all facets would have been added to `inner_facets_mapping`.
    # We use those to refine physical groups and periodic links.
    physical_groups = {}

    refine_facets_fn = lambda f: self._refine_facets(
        f, ndim=self.premesh.ndim - 1)
    if self.premesh.physical_groups and self.gridpoints_1d.is_continuous():
      for physical_name, facets in self.premesh.physical_groups.items():
        if not facets.size:
          raise ValueError(f'Got an empty physical group "{physical_name}".')
        physical_groups[physical_name] = refine_facets_fn(facets)

    # Refine facets which form periodic links.
    periodic_links = None
    if (
        self.premesh.periodic_links is not None
        and self.gridpoints_1d.is_continuous()
    ):
      periodic_links = np.stack([
          refine_facets_fn(self.premesh.periodic_links[:, 0, :]),
          refine_facets_fn(self.premesh.periodic_links[:, 1, :]),
      ], axis=-2)

    return Premesh.create(
        node_coords=node_coords,
        elements=np.array(elements, dtype=np.int32),
        gridpoints_1d=self.gridpoints_1d,
        physical_groups=physical_groups,
        periodic_links=periodic_links,
        partitions=self.premesh.partitions)
