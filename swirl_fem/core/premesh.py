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

"""A premesh data structure for staging meshes before refinement."""

from collections.abc import Mapping
from functools import partial  # pylint: disable=g-importing-member

import flax
import jax
from jax.tree_util import tree_map
import numpy as np

from swirl_fem.core import gather_scatter
from swirl_fem.core.interpolation import Nodes1D
from swirl_fem.core.interpolation import NodeType
from swirl_fem.core.mesh import Mesh


def _mask(facets, node_indices: np.ndarray) -> np.ndarray:
  """Returns a boolean mask of which `node_indices` are contained in facets."""
  indices = np.array(list(set(facets.flatten().tolist())), dtype=np.int32)
  return np.isin(node_indices, indices)


@flax.struct.dataclass
class Premesh:
  """An intermediate mesh format used to refine meshes.

  A `Premesh` is used to stage a `Mesh`. It may be refined (uniformly) to create
  another `Premesh`, or may be used to add or remove periodicity, or add or
  remove 'physical groups' used for boundary conditions. Finally,
  `Premesh.finalize()` converts it into a (possibly partitioned) `Mesh`.

  Attributes:
    order: The order of the mesh.
    gridpoints_1d: Defines the reference node type on each element.
    node_coords: N-dimensional coordinates of the nodes. This should have shape
      `(number of nodes, ndim)` where the mesh is `ndim` dimensional.
    elements: Integer-valued array containing the indices of nodes in each
      element; shaped `(number of elements, number of nodes per element)`.
      The entry `element[i]` contains the indices of the ith element.
    physical_groups: A mapping from labels to physical entities ('boundary',
      'inlet', 'wall' etc.) denoted by the set of facets comprising
      it. Each facet is defined by the indices of nodes contained in it.
    periodic_links: An array of shape `(num_facets, 2, num_nodes_per_facets)`
      denoting pairs of facets which are to be identified as periodic
      boundaries.
    partitions: If the mesh is partitioned, then this integer array, with shape
      `(number of elements,)` specifies the partition index for each
      element. The partition index must be between 0 and (number of partitions
      - 1), inclusive.
  """
  order: int
  gridpoints_1d: Nodes1D
  node_coords: np.ndarray
  elements: np.ndarray
  physical_groups: Mapping[str, np.ndarray]
  periodic_links: np.ndarray | None = None
  partitions: np.ndarray | None = None

  @classmethod
  def create(
      cls,
      node_coords: np.ndarray,
      elements: np.ndarray,
      order: int | None = None,
      gridpoints_1d: Nodes1D | None = None,
      physical_groups: Mapping[str, np.ndarray] | None = None,
      periodic_links: np.ndarray | None = None,
      partitions: np.ndarray | None = None,
  ) -> 'Premesh':
    """Creates a `Premesh` object."""
    ndim = node_coords.shape[-1]
    num_nodes_per_element = elements.shape[-1]

    # Default to uniformly distributed (aka Newton-Cotes) nodes if
    # `gridpoints_1d` is not specified.
    if gridpoints_1d is None:
      num_points = int(round(np.exp(np.log(num_nodes_per_element) / ndim)))
      gridpoints_1d = Nodes1D.create(num_points=num_points,
                                     node_type=NodeType.NEWTON_COTES)

    if num_nodes_per_element != gridpoints_1d.num_points**ndim:
      raise ValueError(
          'Expected the number of nodes in each element to be equal '
          f'to the number of gridpoints in {ndim} dimensions. But got '
          f'{num_nodes_per_element} != {gridpoints_1d.num_points} ** {ndim}.')

    if physical_groups is None:
      physical_groups = {}

    if order is None:
      order = gridpoints_1d.num_points - 1

    return cls(
        order=order,
        gridpoints_1d=gridpoints_1d,
        node_coords=node_coords,
        elements=elements,
        physical_groups=physical_groups,
        periodic_links=periodic_links,
        partitions=partitions,
    )

  @property
  def ndim(self) -> int:
    """Returns the dimension of the mesh."""
    return self.node_coords.shape[-1]

  @property
  def num_nodes(self) -> int:
    """Returns the number of nodes in the mesh."""
    return self.node_coords.shape[-2]

  @property
  def num_elements(self) -> int:
    """Returns the number of elements in the mesh."""
    return len(self.elements)

  @property
  def num_nodes_per_element(self) -> int:
    """Returns the number of nodes per element."""
    return self.elements.shape[-1]

  def is_partitioned(self) -> bool:
    """Returns whether the premesh is partitioned."""
    return self.partitions is not None

  def finalize(self, axis_name: str | None = None) -> Mesh:
    """Finalize the premesh into a mesh.

    Args:
      axis_name: In the partitioned case, the Jax axis name along which the
        mesh is partitioned.

    Returns:
      The mesh instance after finalization. If partitioned it's placed on the
      available Jax devices.
    """
    if not self.is_partitioned():
      node_indices = gather_scatter.get_unique_node_indices(
          node_indices=np.arange(self.num_nodes, dtype=np.int32),
          periodic_links=self.periodic_links)
      physical_masks = tree_map(
          partial(_mask, node_indices=node_indices), self.physical_groups)
      exchange_gather_indices, exchange_unique_indices = (
          gather_scatter.get_exchange_indices(node_indices))
      return Mesh.create(
          node_coords=self.node_coords,
          elements=self.elements,
          node_indices=node_indices,
          gridpoints_1d=self.gridpoints_1d,
          physical_masks=physical_masks,
          exchange_gather_indices=exchange_gather_indices,
          exchange_unique_indices=exchange_unique_indices,
      )

    assert self.partitions is not None
    if not axis_name:
      raise ValueError('If partitioned, we need a non-trivial axis_name')

    # Separate out the list of elements (num_elements, N) into the partitioned
    # list of elements of shape (num_partitions, num_elements_per_partition, N).
    element_indices = gather_scatter.group_by_partitions(self.partitions)
    elements = jax.vmap(partial(gather_scatter.gather, indices=element_indices),
                        in_axes=-1, out_axes=-1)(self.elements)

    # Obtain the node indices array containing the global node indices in each
    # partition. Also reindex the elements from global node indices to
    # partition-local node indices.
    #
    # For instance for the following element configuration:
    #   partition #0: [[0, 1], [1, 2]]
    #   partition #1: [[2, 3], [3, 4]]
    #
    # this function creates the node_indices array: [[0, 1, 2], [2, 3, 4]]
    # which is then used to obtain the partition-local reindexing of the
    # elements: [[[0, 1], [1, 2]], [[0, 1], [1, 2]]].
    node_indices, local_elements = (
        gather_scatter.get_local_elements(elements))

    # Use periodic links to dedup the node indices
    node_indices = gather_scatter.get_unique_node_indices(
        node_indices, periodic_links=self.periodic_links)

    # Compute auxiliary exchange indices for the Q mapping (as described in
    # gather_scatter.py)
    exchange_gather_indices, exchange_unique_indices = (
        gather_scatter.get_exchange_indices(node_indices))

    # Create boolean masks over the node indices for the specified physical
    # groups.
    physical_masks = tree_map(
        partial(_mask, node_indices=node_indices), self.physical_groups)

    # bind the static arrays which are not pmapped when placing on devices
    create_fn = partial(
        Mesh.create,
        gridpoints_1d=self.gridpoints_1d,
        exchange_unique_indices=exchange_unique_indices,
        axis_name=axis_name)

    # create and place the resulting `Mesh` on the available devices
    return jax.pmap(create_fn)(
        node_coords=self.node_coords[node_indices],
        elements=local_elements,
        node_indices=node_indices,
        physical_masks=physical_masks,
        exchange_gather_indices=exchange_gather_indices,
    )
