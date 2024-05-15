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

"""Crystal router [1], a sparse dynamic all-to-all collective op.

[1] Fox GC, Johnson MA, Lyzenga GA, Otto SW, Salmon JK and Walker DW. Solving
    Problems on Concurrent Processors. Prentice-Hall, 1988.

    See also https://dl.acm.org/doi/10.1145/62297.62390.
"""
from functools import partial  # pylint: disable=g-importing-member
from typing import Any, Callable, Literal, overload, TypeVar

from flax import struct
import jax
from jax import lax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.typing import ArrayLike
import numpy as np
from swirl_fem.communication.jit_distributed import jit_distributed
from swirl_fem.communication.semi_traced_scalar import SemiTracedScalar


def crystal_router_setup(mesh: Mesh, axis_name):
  """Returns crystal router routine for given `mesh` and `axis_name`."""
  crystal_flat = _crystal_router_flat(mesh, axis_name)

  T = TypeVar('T')

  @overload
  def crystal_router(
      n: ArrayLike,
      data: T,
      target: jax.Array,
      return_source: Literal[True] = True,
  ) -> tuple[jax.Array, T, jax.Array]:
    ...

  @overload
  def crystal_router(
      n: ArrayLike, data: T, target: jax.Array, return_source: Literal[False]
  ) -> tuple[jax.Array, T]:
    ...

  def crystal_router(
      n: ArrayLike,
      data: T,
      target: jax.Array,
      return_source: bool = True,
  ) -> tuple[jax.Array, T, jax.Array] | tuple[jax.Array, T]:
    """Crystal router.

    Sparse all-to-all communication. Assumes `n`, `data`, and `target` are
    distributed on their first axis, with 1 entry per device (axis index).

      n_out, data_out, source = crystal_router(n, data, target)

    sends `data[p, j]` to the axis index `target[p, j]`, for 0 ≤ `j` < `n[p]`.
    That is, for 0 ≤ `j` < `n[p]`,
      data[p, j] == data_out[target[p, j], k]  for some k < n_out[target[p, j]].
    The ordering of data within each device `p` is unspecified.

    If `return_source` is true, the array `source` is returned such that, for
    0 ≤ `j` < `n_out[p]`,
      data_out[p, j] == data[source[p, j], k]  for some k < n[source[p, j]].

    In particular, a subsequent call
      n, data, target = crystal_router(n_out, data_out, source)
    should return the original `n`, `data`, and `target` arrays, up to
    re-ordering within each device `p`.

    `data` can be a pytree, provided all leaves have identical static and
    dynamic lengths. That is, they can have different shapes and types, but
    the size of the second axis must be the same. In this case the behavior is
    equivalent to mapping to each leaf, except that the communication is
    combined.

    Args:
      n: dynamic length of `data` and `target` arrays.
      data: array(s) of data to exchange.
      target: array specifying the axis_index to send each piece of data to.
        Must be the same static length as `data` on each device.
      return_source: Whether to return the `source` array.

    Returns:
      The new dynamic length `n` and exchanged data `data`. When `return_source`
      is true, also returns an array `source` that indicates where each piece of
      data came from.
    """
    data_flat, tree_def = jax.tree_util.tree_flatten(data)
    unflatten = lambda l: jax.tree_util.tree_unflatten(tree_def, l)
    out = crystal_flat(CrystalData([target] + data_flat, n), return_source)
    if return_source:
      return out.n, unflatten(out.data[1:-1]), out.data[-1]
    else:
      return out.n, unflatten(out.data[1:])

  return crystal_router


def _pad_to_size(data: list[jax.Array], size: int) -> list[jax.Array]:
  """Resize first dimension of `data` to `size`, padding if needed."""
  if not data:
    return data

  start_size = data[0].shape[0]
  if start_size == size:
    return data
  elif start_size > size:
    return [a[:size] for a in data]

  pad_size = size - start_size

  def pad_array(a):
    pad_shape = list(a.shape)
    pad_shape[0] = pad_size
    padding = jnp.zeros(pad_shape, dtype=a.dtype)
    return lax.concatenate((a, padding), dimension=0)

  return [pad_array(a) for a in data]


class CrystalData(struct.PyTreeNode):
  """List of arrays `data`, each with dynamic length `n`."""
  data: list[jax.Array]
  n: jax.Array

  def resize(self, size: int):
    """Returns a copy with new static length `size`."""
    return CrystalData(
        data=_pad_to_size(self.data, size=size),
        n=self.n,
    )

  def select(self, mask: ArrayLike, size: int):
    """[a[mask] for a in self.data], padded out to the static `size`."""
    n = jnp.sum(mask, axis=0, dtype=self.n.dtype)
    indices, = jnp.where(mask, size=size, fill_value=self.data[0].shape[0])
    data = [a.at[indices].get(mode='drop') for a in self.data]
    return CrystalData(data=data, n=n)

  def __add__(self, other):
    """Concatenate two CrystalDatas.

    Requires size1 ≥ self.n + size2, where
      size1 == a.shape[0]   for a in self.data,
      size2 == b.shape[0]   for b in other.data,
    because `dynamic_update_slice` adjusts `start_indices` so that *all* of `b`
    fits, including all the padding after its dynamic length `other.n`.
    (It would be nice if `dynamic_update_slice` had a mode that honored
    `start_indices` and dropped the entries that don't fit.)

    Args:
      other: `CrystalData` to concatenate.

    Returns:
      The concatenated `CrystalData`, with the same static size as `self`.
    """
    return CrystalData(
        data=[
            lax.dynamic_update_slice(a, b, start_indices=(self.n,))
            for (a, b) in zip(self.data, other.data)
        ],
        n=self.n + other.n,
    )


def _bit_ceil(n):
  """Rounds up to the nearest power of 2."""
  return 0 if n == 0 else 1 << np.frexp(n - 1)[1]


def _crystal_router_flat(
    mesh: Mesh, axis_name: Any
) -> Callable[[CrystalData, bool], CrystalData]:
  """Returns flat crystal router routine for `mesh` and `axis_name`."""
  jit_distr = partial(jit_distributed, mesh=mesh, axis_name=axis_name)
  loop = jit_distr(partial(_crystal_router_loop, axis_name=axis_name))

  @jit_distr
  def resize(crystal_data: CrystalData, *, buffer_size: int) -> CrystalData:
    return crystal_data.resize(buffer_size)

  @jit_distr
  def init_source(target: jax.Array) -> jax.Array:
    return target.at[:].set(lax.axis_index(axis_name))

  init_stage_index = jit_distr(lambda: 0)

  def crystal(crystal_data: CrystalData, append_source: bool) -> CrystalData:
    """The crystal router algorithm for a list of arrays.

    Invokes `crystal_router_loop`, resuming if necessary with increased
    `buffer_size`.

    Args:
      crystal_data: the data to exchange, with `crystal_data.data[0]` being the
        `target` array.
      append_source: whether to append a `source` array to `crystal_data.data`
        (initialized to `lax.axis_index`).

    Returns:
      The exchanged CrystalData.
    """
    buffer_size = _bit_ceil(crystal_data.data[0].shape[1])
    i = init_stage_index()
    crystal_data = resize(crystal_data, buffer_size=buffer_size)
    if append_source:
      crystal_data = CrystalData(
          data=crystal_data.data + [init_source(crystal_data.data[0])],
          n=crystal_data.n,
      )
    while True:
      crystal_data, max_n, i = loop(crystal_data, i, buffer_size=buffer_size)
      max_n = int(max_n[0])
      if max_n > buffer_size:
        buffer_size = _bit_ceil(max_n)
        crystal_data = resize(crystal_data, buffer_size=buffer_size)
      else:
        return CrystalData(data=[a[:, :max_n] for a in crystal_data.data],
                           n=crystal_data.n)

  return crystal


def _crystal_router_loop(
    crystal_data: CrystalData,
    start_stage: ArrayLike,
    *,
    axis_name: Any,
    buffer_size: int,
) -> tuple[CrystalData, ArrayLike, ArrayLike]:
  """Main crystal router implementation.

  Routes the data of each array in `crystal_data.data` (a list of arrays)
  to the axis index specified by the target array, which is taken to be the
  first array in the list. That is, for each array `arr` in `crystal_data.data`,
  `arr[j]` is transferred to axis index `target[j]` where `target` is
  `crystal_data.data[0]`.

  Intended to be called inside of a `shard_map` and `vmap` so that the mapped
  axis is not among the data dimensions.

  The crystal router algorithm is a divide-and-conquer algorithm implementing
  sparse all-to-all communication. At the start of each stage, the set of
  devices (axis indices) have been subdivided into groups, each consisting of
  a contiguous range of axis indices, such that communication is only required
  within each group and not between groups. During each stage, each group is
  subdivided into two subgroups of equal or nearly equal size. Each device of
  the first subgroup is assigned a partner from the second. Partners hand off
  data to each other: not just the data ultimately intended for the partner,
  but *all* data with any target in the partner's entire subgroup. This ensures
  that no further communication is required between subgroups, and we can
  recurse.

  For example, at the start of the stage we might have the group of devices

    4, 5, 6, 7

  which would be split into the two subgroups (4, 5) and (6, 7). Partners are
  assigned by reversing the range of indices within the group:

    4   5 │ 6   7
    ↓   ↓   ↓   ↓
    7   6 │ 5   4

  Device 4, for example, would send any data for devices 6 or 7 to device 7.
  Then, in the next round, the group (6, 7) would be subdivided into the
  singleton groups (6) and (7), and device 7 would send to device 6 any data
  intended for it (including, potentially, data it received from device 4 in the
  previous round.)

  To handle an odd-sized subgroup, we take the first subgroup to be the larger
  one. Instead of sending to itself, the last device of the first subgroup
  sends data to the first device of the second subgroup. For example,

    0   1   2 │ 3   4
    ↓   ↓ ↙     ↓   ↓
    4   3 │ 2   1   0

  This means that the middle device of the odd-sized group receives *no* data
  during the stage, while the subsequent device receives data from *two*
  devices. This particular way of assigning partners between subgroups is
  arbitrary but does ensure that every device receives data from two devices
  during at most one stage of the crystal router.

  Args:
    crystal_data: CrystalData to exchange, with `crystal_data.data[0]` being the
      `target` array.
    start_stage: Integer specifying the stage of the crystal router algorithm to
      start on. `0` for the first stage. Can be used to resume the algorithm
      partway through if `buffer_size` is found to be insufficient.
    axis_name: hashable Python object used to name a mapped axis.
    buffer_size: static length to use for arrays during exchange.

  Returns:
    A tuple (crystal_data_out, max_size, restart_stage).

    If the returned `max_size` ≤ `buffer_size`, then `buffer_size` was
    sufficient, the algorithm was successful, and `crystal_data_out` is the
    exchanged data.

    Otherwise, when `max_size` > `buffer_size`, then `buffer_size` was
    insufficient. The algorithm can be resumed by calling `crystal_router_loop`
    again, increasing `buffer_size` to at least `max_size`, and passing
    crystal_data=crystal_data_out, start_stage=restart_stage.
  """
  idx = SemiTracedScalar.axis_index(axis_name=axis_name)
  n = SemiTracedScalar.axis_size(axis_name=axis_name)
  base_lo = SemiTracedScalar.constant(0, axis_name=axis_name)

  def loop(crystal_data, max_size, i, n, base_lo):
    # group of devices is the interval [base_lo, base_lo + n)

    if np.max(n.global_) <= 1:  # all groups are singletons. done.
      return crystal_data, max_size, i

    # split group [base_low, base_low + n) into two subgroups
    n_lo = (n + 1) // 2
    base_hi = base_lo + n_lo
    is_lo = idx < base_hi
    # `target1` is the partner for the first round of communication.
    # reverse indices within group to assign partners between subgroups
    target1 = n - 1 - (idx - base_lo) + base_lo
    num_recv = SemiTracedScalar.constant(1, axis_name=axis_name)
    # In the odd case, the middle index is assigned to itself.
    # It will send its data to `base_hi` instead, but this must happen in a
    # second round of communication as `base_hi` is already exchanging data
    # with another device in the first round.
    # `target2` is the target for the second round of communication.
    is_self_target = target1 == idx
    is_recvn_two = (n % 2 == 1) & (idx == base_hi)
    num_recv = (num_recv - is_self_target) + is_recvn_two
    target2 = SemiTracedScalar.where(is_self_target, base_hi, idx)
    target2 = SemiTracedScalar.where(is_recvn_two, idx - 1, target2)

    next_data, max_size = lax.cond(
        i >= start_stage,
        lambda: _crystal_router_stage(  # pylint: disable=g-long-lambda
            crystal_data,
            cutoff=base_hi.local,
            num_recv=num_recv,
            axis_name=axis_name,
            perms=(target1.global_, target2.global_),
            buffer_size=buffer_size,
        ),
        lambda: (crystal_data.resize(buffer_size), max_size),
    )

    n = SemiTracedScalar.where(is_lo, n_lo, n - n_lo)
    base_lo = SemiTracedScalar.where(is_lo, base_lo, base_hi)

    # return early if `buffer_size` was too small
    return lax.cond(max_size > buffer_size,
                    lambda: (crystal_data.resize(buffer_size), max_size, i),
                    lambda: loop(next_data, max_size, i + 1, n, base_lo))

  max_size = jnp.zeros((), dtype=crystal_data.n.dtype)
  return loop(crystal_data, max_size, 0, n, base_lo)


def _crystal_router_stage(
    crystal_data: CrystalData,
    cutoff: ArrayLike,
    num_recv: SemiTracedScalar,
    axis_name: Any,
    perms: tuple[np.ndarray, np.ndarray],
    buffer_size: int,
) -> tuple[CrystalData, ArrayLike]:
  """Single stage of the crystal router.

  See the description of the algorithm in `crystal_router_loop` above.

  Args:
    crystal_data: data to exchange, with `crystal_data.data[0]` being the
      `target` array.
    cutoff: split point of the group. We are in the first subgroup if
      lax.axis_index(axis_name) < cutoff, and the second subgroup otherwise.
    num_recv: 0, 1, or 2, according to whether we are receiving data from
      0, 1, or 2 devices.
    axis_name: hashable Python object used to name a mapped axis.
    perms: pair of permutations for `pshuffle`. The second is used only if
      some device has num_recv == 2.
    buffer_size: static length of array to use. Needs to accomodate the
      (concatenation of) kept and received data.

  Returns:
    The exchanged crystal data `crystal_data_out` and its global maximum
    dynamic length `n_max == lax.pmax(crystal_data_out.n)`. If `n_max` is larger
    than `buffer_size`, then `buffer_size` was too small.
  """
  target = crystal_data.data[0]
  n = crystal_data.n
  data_size = target.shape[0]
  mask = jnp.arange(data_size, dtype=n.dtype) < n

  idx = lax.axis_index(axis_name=axis_name)
  lo_mask = mask & (target < cutoff)  # entries targeted for 1st subgroup
  # we keep the `lo_mask` entries if we are in the 1st subgroup (idx < cutoff)
  keep_mask = lo_mask ^ (mask & (idx >= cutoff))
  send_mask = mask ^ keep_mask

  # Note the size `buffer_size + data_size` is required here because
  # `lax.dynamic_update_slice`, invoked in CrystalData.__add__ below,
  # needs to fit the entire *static* length of the `send` arrays into the
  # `keep` arrays when the two are concatenated.
  # See comment in CrystalData.__add__, above.
  keep = crystal_data.select(keep_mask, size=buffer_size + data_size)
  send = crystal_data.select(send_mask, size=data_size)

  send = lax.pshuffle(send, axis_name=axis_name, perm=perms[0])
  keep = lax.cond(num_recv.local > 0, lambda: keep + send, lambda: keep)

  if np.max(num_recv.global_) > 1:
    send = lax.pshuffle(send, axis_name=axis_name, perm=perms[1])
    keep = lax.cond(num_recv.local == 2, lambda: keep + send, lambda: keep)

  return keep.resize(buffer_size), lax.pmax(keep.n, axis_name=axis_name)
