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

"""Utilities for gather/scatter between local and global indices.

We define operators for implementing the global-to-local map Q, expressed
as the matrix product uₚ = Qu. Q is a boolean matrix which takes global, unique
indices of dofs to the (possibly multiply-defined) local indices. We usually
apply the composed map QQᵀ to 'stay' in the local view.  See [1, 2] for more
details.

The global-to-local mapping needs to handle periodic links and domain
partitioning into multiple cores. We decompose the map Q = vstack(I, QₗQₚ),
where the identity matrix I captures global indices mapping to
singularly-defined local indices, and QₗQₚ has at least two non-zeros per row.

Qₚ ('p' for 'partition') maps global nodes to partition-local nodes -- it
handles both the shared dofs due to partitioning and inter-partition
periodicity. Qₗ ('l' for 'local') handles the intra-partition periodicity. We
explain the two maps in more detail below.

We use the input arrays `node_indices` and `periodic_links` to construct Qₚ
and Qₗ. The array `node_indices` has the same shape as uₚ and contains the
unique global index of nodes without accounting for periodicity links. The array
`periodic_links` is shaped `(num_facets, 2, num_nodes_per_facets)` and contains
parallel arrays of global node indices which share a periodic link. That is, for
every `i` and `j` the pairs of global node indices `periodic_links[i,0,j]` and
`periodic_links[i,1,j]` are mapped to the same dof.

### Intra-partition periodicity

Qₗ maps globally unique degrees of freedom to multiple nodes when these nodes
share a periodicity link and lie in the same partition. The boolean matrix Qₗ is
encoded by the array `unique_indices`. The `i`th row of Qₗ is the `k`th
standard basis vector eₖ (0-indexed) if `unique_indices[i] == k`.

For example, consider the following single-partition configuration of two
elements. Here, undeduped node_indices = `[0, 1, 2, 3, 4, 5]` and
`periodic_links = [[[0, 1], [4, 5]]]` indicating that the facets `[0, 1]` and
`[4, 5]` have a periodic link such that `0` and `1` are connected to `4` and `5`
respectively. The deduped (globally unique node indices) and the undeduped
node indices are as follows:

       node indices                      undeduped node indices
  (deduped using periodicity)    (not taking periodicity into account)

       1 -- 3 -- 1                           1 -- 3 -- 5
       |    |    |                           |    |    |
       0 -- 2 -- 0                           0 -- 2 -- 4

Here, we would have `unique_indices = [0, 1, 2, 3, 0, 1]` indicating the
mapping Qₗ = vstack(e₀; e₁; e₂; e₃; e₀; e₁). Since we have a single partition,
the map Qₚ is simply the identity map.

### Domain partitioning

In the case of domain-partitioning, we additionally use Qₚ to map from the
global dofs to the multiply-defined dofs due to shared nodes between partitions.
The boolean matrix Qₚ is encoded by the array `gather_indices`, which is
shaped `(number of global dofs)` in the unpartitioned case, and
`(number of partitions, number of global dofs)` in the partitioned case. The
`Rp + i`th row of Qₚ contains the `k`th standard basis vector eₖ (0-indexed)
if `gather_indices[p, k] == i` where R is the number of nodes per partition.
For the partitions which do not contain the `k`th global dof, we have the
sentinel value `gather_indices[p, k] == -1`.

Consider an example where three elements distributed in two different
partitions, where the first partition contains two elements.
Here, `node_indices = [[0, 1, 2, 3, 4, 5], [4, 5, 0, 1, -1, -1]]` indicating
that the `p`th partition's local node index `i` corresponds to the global node
index `node_indices[p, i]` (where sentinel `-1` values are used to fill up
missing nodes).

     global node indices       global node indices      local node indices
                                  (partitioned)            (partitioned)
      1 -- 3 -- 5 -- 1        1 -- 3 -- 5  5 -- 1       1 -- 3 -- 5  1 -- 3
      |    |    |    |        |    |    |  |    |       |    |    |  |    |
      0 -- 2 -- 4 -- 0        0 -- 2 -- 4  4 -- 0       0 -- 2 -- 4  0 -- 2

Since the global node indices {0, 1, 4, 5} are to be exchanged, the shared-only
index set contains 4 indices, renumbered to {0, 1, 2, 3}.

Here, we would have `gather_indices = [[0, 1, 4, 5], [2, 3, 0, 1]]` indicating
that Qₚ is the following mapping:
    partition 0: (0→0, 1→1, 2→4, 3→5);    partition 1: (0→2, 1→3, 2→0, 3→1)
from (shared-only) global indices to partition-local node indices. Since there
is no intra-partition periodicity, Qₗ is the identity map.

### References
  1. Fischer, P., Kerkemeier, S., Min, M., Lan, Y.H., Phillips, M., Rathnayake,
     T., Merzari, E., Tomboulides, A., Karakus, A., Chalmers, N. and Warburton,
     T., 2022. NekRS, a GPU-accelerated spectral element Navier–Stokes solver.
     Parallel Computing, 114, p.102982.
  2. Deville, M.O., Fischer, P.F. and Mund, E.H., 2002. High-order methods for
     incompressible fluid flow (Vol. 9). Cambridge university press.
"""
import collections

import jax
from jax import lax
import jax.numpy as jnp
import networkx as nx
import numpy as np


# A dummy index variable used to denote missing entries.
SENTINEL = -1


def gather(u, indices, fill_value=SENTINEL):
  """Gather u from indices while masking out sentinel values."""
  if u.ndim != 1:
    raise ValueError(f'Expecting a rank-1 array. Got {u.shape}')
  mask = indices != SENTINEL
  q_u = u[indices]
  return jnp.where(mask, q_u, jnp.full_like(q_u, fill_value=fill_value))


def scatter(u: jax.Array, indices: jax.Array, num_nodes: int) -> jax.Array:
  assert u.shape == indices.shape, f'Got: {u.shape} v/s {indices.shape}'
  mask = indices != SENTINEL
  return jnp.zeros(num_nodes, dtype=u.dtype).at[indices].add(mask * u)


def get_unique_node_indices(
    node_indices: np.ndarray, periodic_links: np.ndarray | None
) -> np.ndarray:
  """Returns deduped global node indices after accounting for periodic links.

  Args:
    node_indices: The unique node indices in each partition. If unpartitioned,
      then this is just arange(number of nodes). Has the same shape as the
      local nodal values uₚ, that is, shape `(num_nodes,)` or `(num_partitions,
      num_local_nodes)` in the unpartitioned and partitioned cases respectively.
    periodic_links: Specifies the periodic connections between nodes. See the
      specific format in the docstring at the top of this file. Shaped
      `(num_facets, 2, num_nodes_per_facets)`.
  """
  if periodic_links is None or len(periodic_links) == 0:  # pylint: disable=g-explicit-length-test
    return node_indices

  # Create a mapping from connected periodic nodes to a unique representative
  periodic_mapping = _get_periodic_mapping(periodic_links)

  # Map each index in node_indices to its unique representative
  def _replace(idx):
    return periodic_mapping[idx] if idx in periodic_mapping else idx

  return np.vectorize(_replace)(node_indices)


# TODO(anudhyan): Add a ExchangeIndices structure which encapsulates the
# `gather_indices` and `unique_indices`, and change the utility functions to
# accept and return instances of ExchangeIndices.
def get_exchange_indices(
    node_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
  """Returns auxiliary exchange arrays `gather_indices` and `unique_indices`.

  Args:
    node_indices: The unique node indices in each partition. Has the same shape
      as the local nodal values uₚ, that is, shape `(num_nodes,)` or
      `(num_partitions, num_local_nodes)` in the unpartitioned and partitioned
      cases respectively.

  Returns:
    A tuple `(gather_indices, unique_indices)` used for exchanging nodal values
    for periodicity or shared nodes between partitions.
  """
  if node_indices.ndim not in (1, 2):
    raise ValueError('node_indices must have ndim 1 or 2. Got '
                     f'{node_indices.ndim}')
  if node_indices.ndim == 2:
    return _get_exchange_indices_partitioned(node_indices)
  return _get_exchange_indices_unpartitioned(node_indices)


def exchange(
    u: jax.Array,
    gather_indices: jax.Array | None,
    unique_indices: np.ndarray | None = None,
    axis_name: str | None = None,
) -> jax.Array:
  """Exchanges node values for periodicity or shared nodes among partitions.

  Applies the map QQᵀ(u), where Q is the global-to-local map from unique dofs
  to the partition-local node indices. See the explanation at the top of this
  file.

  If `u` is partitioned, then this function must be `pmap`-ped across the nodal
  values and the axis_name must be non-empty. The 'replicated-view' of `u` has
  shape `(num_partitions, num_local_nodes)` but inside the `pmap` we see the
  shape `(num_local_nodes,)`. Similarly, `gather_indices` is also partitioned
  and pmapped across. Since `unique_indices` is same across all partitions,
  it's a static value (numpy array) and not partitioned or pmapped across.

  Args:
    u: Array to be exchanged. Should have shape `(num_nodes,)`.
    gather_indices: The gather indices corresponding to the exchange operation.
      Shaped `(num_gather_dofs,)` where `num_gather_dofs` is the number of
      participating dofs in Qₚ.
    unique_indices: The unique indices corresponding to the exchange
      operation. This array has shape `(num_gather_dofs)`.
    axis_name: The axis along which `u` is partitioned. If unpartitioned, this
      should be None.

  Returns:
    QQᵀ(u); the nodal values after the exchange.
  """
  if gather_indices is None or not gather_indices.size:
    return u

  # Grab the initial values at the indices which need to be exchanged.
  # TODO(anudhyan): Use an out-of-bounds index here (maybe UINT_MAX) instead of
  # the SENTINEL value -1, so the final `Qₚ` step becomes
  # `u[gather_indices].set(updates, mode='drop')``. At this point, we should be
  # able to have `unique_indices` be partitioned as set it to the globally
  # unique index of each gathered index.
  mask = gather_indices != SENTINEL
  # Indexing with the sentinel value gets masked out and set to zeros.
  initial_values = mask * u[gather_indices]

  # If unique indices are set, further consolidate into a smaller set of
  # indices.
  if unique_indices is not None:
    # The computation is static since `unique_indices` is a numpy array, which
    # means it's free for subsequent runs of jitted functions.
    # TODO(anudhyan): Add `num_unique` to the ExchangeInfo struct as metadata
    # to make this static computation more transparent.
    num_unique_nodes = 1 + unique_indices.max()
    updates = jnp.zeros(num_unique_nodes).at[unique_indices].add(initial_values)
  else:
    updates = initial_values

  # In the partitioned case, sum across all partitions.
  if axis_name is not None:
    updates = lax.psum(updates, axis_name=axis_name)

  # Expand back to the larger set of indices if unique indices were set.
  if unique_indices is not None:
    updates = updates[unique_indices]

  # Add the updates back to the exchange indices.
  # This is essentially `u.at[gather_indices].set(updates)`, but since
  # `gather_indices` could contain sentinel values, use `add` instead; as
  # otherwise the last index would get overwritten. Also, make sure the
  # corresponding values being added are zero; for instance, `gather_indices[i]`
  # has a sentinel if the current partition doesn't have the ith global dof.
  # The mask zeros out these positions as `updates[i]` is generally not zero.
  return u.at[gather_indices].add(mask * (updates - initial_values))


def _get_periodic_mapping(periodic_links: np.ndarray | None) -> dict[int, int]:
  """Create mapping from connected periodic nodes to a unique representative."""
  if periodic_links is None:
    return {}

  # periodic links is a list of pairs of facets which are periodic. To get the
  # connection between nodes, we transpose to a list of pairs of nodes.
  edges = np.transpose(periodic_links, axes=(0, 2, 1)).reshape((-1, 2)).tolist()
  periodicity_graph = nx.Graph(edges)

  # Map each node in a component to a unique representative - the minimum
  # node index in that component.
  periodic_mapping = {}
  for component in nx.connected_components(periodicity_graph):
    representative = min(component)
    for node in component:
      periodic_mapping[node] = representative
  return periodic_mapping


def _get_exchange_indices_unpartitioned(
    node_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
  """Returns auxiliary exchange arrays for the unpartitioned case."""
  # Create a frequency table for the set of global node indices
  assert node_indices.ndim == 1, node_indices.ndim
  counter = collections.Counter(node_indices.tolist())

  # Iterate through the unique nodes (counter.keys()) and add all the frequency
  # > 1 nodes to be exchanged
  indices_to_exchange = set()
  for node_idx, cnt in counter.items():
    if node_idx != SENTINEL and cnt > 1:
      indices_to_exchange.add(node_idx)

  # Create a mapping from unique indices to the exchange index; which is just
  # the corresponding rank (in a sorted array) of the unique indices
  exchange_mapping = dict(
      (idx, rank) for rank, idx in enumerate(sorted(indices_to_exchange)))

  # We set gather_indices as the positions of the participating indices, as Qₚ
  # is simply the identity map on those indices. The corresponding entry in
  # unique_indices to be the globally unique id of of the local index
  gather_indices = []
  unique_indices = []
  for pos, idx in enumerate(node_indices):
    if idx in indices_to_exchange:
      gather_indices.append(pos)
      unique_indices.append(exchange_mapping[idx])

  return (np.array(gather_indices, dtype=np.int32),
          np.array(unique_indices, dtype=np.int32))


def _get_exchange_indices_partitioned(
    node_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray | None]:
  """Returns auxiliary exchange arrays for the partitioned case."""
  assert node_indices.ndim == 2, node_indices.shape
  num_partitions = len(node_indices)

  # Create a frequency table for the set of global indices
  counter = collections.Counter(node_indices.flatten().tolist())

  # Iterate through the unique nodes (counter.keys()) and add all the frequency
  # > 1 nodes to be exchanged.
  indices_to_exchange = set()
  for node_idx, cnt in counter.items():
    if node_idx != SENTINEL and cnt > 1:
      indices_to_exchange.add(node_idx)

  # Create a mapping from unique indices to the exchange index; which is just
  # the corresponding rank (in a sorted array) of the unique indices
  exchange_mapping = dict(
      (idx, rank) for rank, idx in enumerate(sorted(indices_to_exchange)))

  gather_indices = np.full(shape=(len(node_indices), len(indices_to_exchange)),
                           fill_value=SENTINEL)
  for partition_idx in range(num_partitions):
    # Keep track of the node indices seen in this partition.
    seen = set()
    for pos, node_idx in enumerate(node_indices[partition_idx]):
      # skip indices added for padding or those that are not exchanged
      if node_idx == SENTINEL or node_idx not in indices_to_exchange:
        continue
      # If the same node index occurs more than once we have intra-partition
      # periodicity, which is currently not supported.
      if node_idx in seen:
        raise NotImplementedError(
            f'Found {node_idx=} occurring more than once in {partition_idx=}')
      gather_indices[partition_idx, exchange_mapping[node_idx]] = pos
      seen.add(node_idx)

  unique_indices = None
  return gather_indices, unique_indices


def _pad_evenly(indices: list[np.ndarray]) -> list[np.ndarray]:
  """Pads with sentinel values so that each list entry has the same length."""
  n = max(map(len, indices))
  return [
      np.hstack([i, np.full(n - len(i), fill_value=SENTINEL)]) for i in indices
  ]


def group_by_partitions(partitions: np.ndarray) -> np.ndarray:
  """Returns array with each partition's indices.

  Given a 1D array `partitions` where `partitions[i] == p` means that the `i`th
  index is placed on the `p`th partition, computes a 2D array `indices` of shape
  `(P, n)` where `P` is the total number of partitions and `n` is the maximum
  number of entries in `partitions` which any single partition. If any
  partition contains less than `n` indices, the remaining positions are filled
  with sentinel values.

  Args:
    partitions: A 1D array of partition indices, with each entry between `0` and
      `P - 1` where `P` is the total number of partitions.

  Returns:
    The 2D array `indices` where `indices[p]` contains all the `i`s such that
    `partitions[i] == p`.
  """
  assert partitions.ndim == 1, partitions.shape
  num_partitions = 1 + partitions.max()
  indices = [[] for _ in range(num_partitions)]
  for idx, p in enumerate(partitions):
    indices[p].append(idx)
  # in case of uneven number of indices in each partition, we pad to the maximum
  # number of indices among all the partitions.
  indices = _pad_evenly(list(map(np.asarray, indices)))
  # shape: (num_partitions, num_indices_per_partition)
  return np.array(indices, dtype=np.int32)


def get_local_elements(elements: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Reindex elements from global node indices to partition-local node indices.

  This function takes a partitioned list of indices which define 'elements',
  which index into a globally unique set of nodes. It then renumbers the nodes
  to partition-local indices, and returns both the renumbering mapping
  and the renumbered list of elements.

  The input array `elements` should have shape `(num_partitions,) + local_shape`
  and the returned array `local_elements` has the same shape as `elements`. The
  actual shape of the `elements` doesn't matter as long as it has a leading
  dimension of size `num_partitions` (which is simply inferred from the input).

  For example, suppose the elements are `[[2, 3], [3, 4]]` in the first
  partition and `[[4, 5], [5, 2]]` in the second partition. We create the
  mapping `2→0, 3→1, 4→2` for the first partition and `2→0, 4→1, 5→2` for the
  second partition. Using this mapping the elements are renumbered as
  `[[0, 1], [1, 2]]` and `[[1, 2], [2, 0]]` for the two partitions.

  Args:
    elements: An array containing elements indexing into the global set of
      nodes with leading dimension `num_partitions`.

  Returns:
    node_indices: An array representing the mapping from local to global node
      indices, of shape `(num_partitions, num_nodes_per_partition)`; where
      `num_nodes_per_partition` is the maximum number of nodes in any partition.
    local_elements: The renumbered elements with the same shape as `elements`.
  """
  # First, we find the set of node indices occurring in each partition.
  elements = list(elements)
  elements_flat = jax.tree.map(lambda x: x.flatten(), elements)
  dedup_fn = lambda indices: np.unique(indices[indices != SENTINEL])
  node_indices = _pad_evenly(jax.tree.map(dedup_fn, elements_flat))

  # Next, compute the local renumbering for each partition.
  def _local_mapping(indices):
    return {idx: pos for pos, idx in enumerate(indices) if idx != SENTINEL}

  local_mappings = jax.tree.map(_local_mapping, node_indices)

  # Finally, map the elements to use the local node index mappings.
  def _local_elements(indices, local_mapping):
    return np.vectorize(lambda i: local_mapping.get(i, SENTINEL))(indices)

  local_elements = jax.tree.map(_local_elements, elements, local_mappings)
  return np.stack(node_indices), np.stack(local_elements)
