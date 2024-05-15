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

"""A mesh data structure for finite element simulations."""

from collections.abc import Mapping

import flax.struct
import jax
from jax import vmap
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np
from swirl_fem.core import gather_scatter
from swirl_fem.core.interpolation import Nodes1D
from swirl_fem.core.interpolation import NodeType


@flax.struct.dataclass
class Mesh:
  """Represents an N-dimensional mesh.

  This class assumes that each of the elements are of the same 'type' (e.g.,
  triangular, quadrilateral, tetrahedral or hexahedral etc.). Moreover each
  element must have the same 'order' of nodes associated with it. These
  assumptions guarantee that each element has the same number of nodes.

  The specific ordering of the primary and secondary nodes within an element is
  up to the downstream solver using this class. Currently, the supported
  elements in 2D and 3D are tensor-product quadrilateral (in 2D) and hexahedral
  (in 3D). These must have lexicographic (tensor-product) ordering of the nodes
  within each element.

  Note that for a given order and dimension, the number of nodes per element is
  then fixed to be `(order + 1) ** ndim`.

  Attributes:
    node_coords: Coordinates of the nodes in N-dimensional Euclidean space.
      This should have shape `(number of nodes, ndim)` where the mesh is `ndim`
      dimensional.
    elements: Integer-valued array containing the indices of nodes in each
      element; shaped `(number of elements, number of nodes per element)`.
      The entry `element[i]` contains the indices of the  ith element.
    node_indices: An array with the same shape as `node_coords.shape[:-1]` with
      the unique indices of the nodes. In the unpartitioned case this is just
      `arange(num_nodes)`. In the partitioned case, this mapping is used for
      identifying shared nodes. See also `gather_scatter.exchange`.
    order: The order of intra-element nodes associated with the mesh elements.
    gridpoints_1d: A `Nodes1D` object with `order + 1` nodes representing
      the nodal gridpoints on each element in each dimension. The values of the
      `gridpoints_1d` nodes are the coordinates on the 1D reference element.
      The corresponding `ndim`-dimensional gridpoints are derived from the
      tensor-product of the 1D nodes.
    physical_masks: A dict mapping string labels to physical entities
      ('boundary', 'inlet', 'wall' etc.) denoted by a boolean array with the
      same shape as `node_indices`.
    exchange_gather_indices: Indices used as gather indices for exchanging
      shared nodes. See also `gather_scatter.exchange`.
    exchange_unique_indices: Indices used as unique indices for exchanging
      shared nodes. See also `gather_scatter.exchange`.
    axis_name: If the mesh is partitioned, the Jax axis name along which it is
      partitioned.
  """
  node_coords: jax.Array
  elements: jax.Array
  node_indices: jax.Array
  order: int = flax.struct.field(pytree_node=False)
  gridpoints_1d: Nodes1D = flax.struct.field(pytree_node=False)

  physical_masks: Mapping[str, jax.Array] = flax.struct.field(
      default_factory=dict
  )
  exchange_gather_indices: jax.Array | None = None
  exchange_unique_indices: np.ndarray | None = flax.struct.field(
      pytree_node=False, default=None
  )
  axis_name: str | None = flax.struct.field(pytree_node=False, default=None)

  @classmethod
  def create(
      cls,
      node_coords: ArrayLike,
      elements: ArrayLike,
      node_indices: ArrayLike | None = None,
      gridpoints_1d: Nodes1D | None = None,
      physical_masks: Mapping[str, ArrayLike] | None = None,
      exchange_gather_indices: ArrayLike | None = None,
      exchange_unique_indices: ArrayLike | None = None,
      axis_name: str | None = None,
  ) -> 'Mesh':
    """Creates a `Mesh` object."""
    ndim = node_coords.shape[-1]  # pytype: disable=attribute-error  # numpy-scalars
    num_nodes_per_element = elements.shape[-1]  # pytype: disable=attribute-error  # numpy-scalars
    physical_masks = physical_masks or {}

    # Default to uniformly distributed (aka Newton-Cotes) nodes if
    # `gridpoints_1d` is not specified.
    if gridpoints_1d is None:
      num_points = int(round(np.exp(np.log(num_nodes_per_element) / ndim)))
      gridpoints_1d = Nodes1D.create(num_points=num_points,
                                     node_type=NodeType.NEWTON_COTES)

    if num_nodes_per_element != gridpoints_1d.num_points**ndim:
      raise ValueError(
          'Expected the number of nodes in each element of `mesh` to be equal '
          f'to the number of gridpoints in {ndim} dimensions. But got '
          f'{num_nodes_per_element} != {gridpoints_1d.num_points} ** {ndim}.')

    if node_indices is None:
      node_indices = jnp.arange(len(node_coords), dtype=jnp.int32)

    return cls(
        node_coords=node_coords,
        elements=elements,
        order=gridpoints_1d.num_points - 1,
        gridpoints_1d=gridpoints_1d,
        physical_masks=physical_masks,
        node_indices=node_indices,
        exchange_gather_indices=exchange_gather_indices,
        exchange_unique_indices=exchange_unique_indices,
        axis_name=axis_name,
    )

  @property
  def ndim(self) -> int:
    """Returns the dimension of the mesh."""
    return self.node_coords.shape[-1]

  @property
  def num_nodes(self) -> int:
    """Returns the number of nodes in the mesh (per partition)."""
    return self.node_coords.shape[-2]

  @property
  def num_elements(self) -> int:
    """Returns the number of elements in the mesh (per partition)."""
    return self.elements.shape[-2]

  @property
  def num_nodes_per_element(self) -> int:
    """Returns the number of nodes per element."""
    return self.elements.shape[-1]

  def gather(self, u: jax.Array) -> jax.Array:
    """Gathers nodal values specified on the nodes to each element."""
    if u.shape != (self.num_nodes,):
      raise ValueError(
          f'Expected `u` to have shape ({self.num_nodes},) but got: {u.shape}.')
    return gather_scatter.gather(u, indices=self.elements, fill_value=0.)

  # TODO(anudhyan): We should the flip the definitions of gather and scatter as
  # scatter should go from `u` (with unique dofs) to `u_local` (with duplicated)
  # nodes.
  def scatter(self, u_local: jax.Array) -> jax.Array:
    """Scatters elemental nodal values from each element to the nodes."""
    return gather_scatter.scatter(u_local, indices=self.elements,
                                  num_nodes=self.num_nodes)

  def element_coords(self) -> jax.Array:
    """Returns the coordinates of nodes within each element."""
    return vmap(self.gather, in_axes=-1, out_axes=-1)(self.node_coords)

  def exchange(self, u: jax.Array) -> jax.Array:
    """Perform an exchange operation on the given nodal values."""
    return gather_scatter.exchange(
        u, gather_indices=self.exchange_gather_indices,
        unique_indices=self.exchange_unique_indices,
        axis_name=self.axis_name)
