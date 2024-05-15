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

"""Parallel prefix scan (exclusive)."""

import numbers
from typing import Any

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

from swirl_fem.communication.semi_traced_scalar import SemiTracedScalar


def _dtype_range(dtype: jnp.dtype):
  """Returns tuple giving the minimum and maximum values of ``dtype``."""
  if dtype == jnp.dtype(bool):
    return (False, True)

  if issubclass(dtype.type, numbers.Integral):
    info = jnp.iinfo(dtype)
  else:
    info = jnp.finfo(dtype)

  return (info.min, info.max)


# Mapping to monoid unit (given dtype) of supported ops
unit_table = {
    jnp.add: lambda t: jnp.zeros((), dtype=t),
    jnp.multiply: lambda t: jnp.ones((), dtype=t),
    jnp.maximum: lambda t: _dtype_range(t)[0],
    jnp.minimum: lambda t: _dtype_range(t)[1],
    jnp.bitwise_and: lambda t: ~jnp.zeros((), dtype=t),
    jnp.bitwise_or: lambda t: jnp.zeros((), dtype=t),
    jnp.bitwise_xor: lambda t: jnp.zeros((), dtype=t),
}


def _fanin_fanout_flat(
    x: list[jax.Array],
    op: Any,
    axis_name: Any,
    prefix_scan: bool,
    reduction: bool,
) -> list[jax.Array] | tuple[list[jax.Array], list[jax.Array]]:
  """Implementation of fan-in/fan-out prefix scan and all-reduce.

  Computes the parallel prefix scan and/or all-reduce of the arrays ``x`` using
  the given ``op`` along the mapped axis ``axis_name``, using a binary fan-in/
  fan-out strategy.

  The devices are organized into a binary tree by recursively subdividing the
  axis in half, always taking the lower half to be larger when the size is odd.
  For example, when the axis size is 6, we get the following tree, labeled by
  the size of the group of devices (more precisely, axis indices) at each node:

         ╭───6───╮
         │       │
       ╭─3─╮   ╭─3─╮
       2   1   2   1
      1^1     1^1

  Within each group, we call the lowest index the "base". The communication tree
  arises by labeling each node in the tree with the "base" index of the group:

         ╭───0───╮
         │       │
       ╭─0─╮   ╭─3─╮
       0   2   3   5
      0^1     3^4

  Note the leaves are the range of axis indices, in order from left to right,
  while each parent inherits the index of the left child. Neighboring children
  in the tree exchange data in stages, first going up the tree (fan-in), then
  going down (fan-out).

      0↔1     3↔4
       0 ↔ 2   3 ↔ 5
         0   ↔   3
       0 → 2   3 → 5
      0→1     3→4

  Each device keeps track of two pieces of data, one that will become the
  prefix scan, and the other that will become the all-reduction. These are the
  variables `scan` and `red` in the code. `red` is initialized to the input `x`
  while `scan` is initialized to the unit of the monoid (e.g., 0 for summation).
  During the fan-in stage, the left child ("base low" index) merges the right
  child's ("base high" index) `red` into its own, while the right child assigns
  the left child's `red` as its own `scan`:

     scan_high ← red_low
     red_low ← op(red_low, red_high)

  At the end of each stage of the fan-in, the parent node's `red` is equal to
  the reduction of all of its leaves, and the right child's `scan` is equal to
  the reduction of all of the left child's leaves. In the last stage of fan-in,
  the sole right child also merges the left child's `red` into their own:

     red_high ← op(red_low, red_high)

  During fan-out, the left child does nothing: its `scan` and `red` are already
  the correct final prefix scan and all-reduction. The right child's `scan` is
  equal to the reduction of the left child's leaves, so it merges the left
  child's `scan` to obtain the final correct prefix scan, while simply taking
  the all-reduction from the left child:

     scan_high ← op(scan_low, scan_high)
     red_high ← red_low

  Below we show how the pair (`scan`, `red`) is updated during fan-in and
  fan-out for our running example for the case of summation (`op == jnp.add`).
  The input data here is a, b, c, d, e, f on devices 0 through 5, and
  s = a + b + c + d + e + f designates the sum (the all-reduce).

     0           1           2           3           4           5
     (0, a)      (0, b)      (0, c)      (0, d)      (0, e)      (0, f)

     0 ◀───────▶ 1           2           3 ◀───────▶ 4           5
     (0, a+b)    (a, b)      (0, c)      (0, d+e)    (d, e)      (0, f)

     0 ◀───────────────────▶ 2           3 ◀───────────────────▶ 5
     (0, a+b+c)  (a, b)      (a+b, c)    (0, d+e+f)  (d, e)      (d+e, f)

     0 ◀───────────────────────────────▶ 3           4           5
     (0, s)      (a, b)      (a+b, c)    (a+b+c, s)  (d, e)      (d+e, f)

     0 ────────────────────▶ 2           3 ────────────────────▶ 5
     (0, s)      (a, b)      (a+b, s)    (a+b+c, s)  (d, e)      (a+b+c+d+e, s)

     0 ────────▶ 1           2           3 ────────▶ 4           5
     (0, s)      (a, s)      (a+b, s)    (a+b+c, s)  (a+b+c+d, s)(a+b+c+d+e, s)

  Args:
    x: list of arrays with a mapped axis named ``axis_name``.
    op: the operation from `jax.numpy`, e.g., `jax.numpy.add`.
      May also be `multiply`, `maximum`, `minimum`, `bitwise_and`,
      `bitwise_xor`, or `bitwise_or`.
    axis_name: hashable Python object used to name a mapped axis.
    prefix_scan: whether to return the parallel exclusive prefix scan.
    reduction: whether to return the all-reduce.

  Returns:
    List(s) of array(s) with the same shape as those in ``x`` representing the
    result of an exclusive prefix scan and/or all-reduce using the operation
    ``op`` along the axis ``axis_name``.
  """

  assert prefix_scan or reduction

  idx = SemiTracedScalar.axis_index(axis_name=axis_name)
  n = SemiTracedScalar.axis_size(axis_name=axis_name)
  base_lo = SemiTracedScalar.constant(0, axis_name=axis_name)
  schedule = []

  # Prepare the fan-in / fan-out schedule
  while np.max(n.global_) > 1:
    # we are in the group of indices [base_lo, base_lo + n)
    n_lo = (n + 1) // 2
    base_hi = base_lo + n_lo
    is_lo = idx < base_hi
    is_base_lo = (n > 1) & (idx == base_lo)
    is_base_hi = (n > 1) & (idx == base_hi)
    target = SemiTracedScalar.where(is_base_lo, base_hi, idx)
    target = SemiTracedScalar.where(is_base_hi, base_lo, target)
    schedule.append({'target': target.global_,
                     'is_base_lo': is_base_lo.local,  # pytype: disable=attribute-error
                     'is_base_hi': is_base_hi.local})  # pytype: disable=attribute-error
    n = SemiTracedScalar.where(is_lo, n_lo, n - n_lo)
    base_lo = SemiTracedScalar.where(is_lo, base_lo, base_hi)

  list_op = lambda x, y: [op(a, b) for a, b in zip(x, y)]

  red = x
  if prefix_scan:
    scan = [jnp.full_like(a, fill_value=unit_table[op](a.dtype)) for a in x]

  y = None
  for s in reversed(schedule):  # Fan-in
    y = lax.pshuffle(red, axis_name=axis_name, perm=s['target'])
    red = lax.cond(s['is_base_lo'],
                   lambda: list_op(red, y),  # pylint: disable=cell-var-from-loop
                   lambda: red)
    if prefix_scan:
      scan = lax.cond(s['is_base_hi'],
                      lambda: y,
                      lambda: scan)

  if reduction and schedule:
    red = lax.cond(schedule[0]['is_base_hi'],
                   lambda: list_op(y, red),
                   lambda: red)

  if prefix_scan and reduction:
    data = (scan, red)
    merge = lambda x, y: (list_op(y[0], x[0]), y[1])
  elif prefix_scan:
    data = scan
    merge = lambda x, y: list_op(y, x)  # pylint: disable=arguments-out-of-order
  else:
    data = red
    merge = lambda x, y: y

  for s in schedule[1:]:  # Fan-out
    y = lax.pshuffle(data, axis_name=axis_name, perm=s['target'])
    data = lax.cond(s['is_base_hi'],
                    lambda: merge(data, y),  # pylint: disable=cell-var-from-loop
                    lambda: data)

  return data


def _fanin_fanout(x, op, axis_name, prefix_scan: bool, reduction: bool):
  """pytree version of `_fanin_fanout_flat`."""
  x_flat, tree_def = jax.tree_util.tree_flatten(x)
  unflatten = lambda l: jax.tree_util.tree_unflatten(tree_def, l)
  outs_flat = _fanin_fanout_flat(
      x_flat,
      op=op,
      axis_name=axis_name,
      prefix_scan=prefix_scan,
      reduction=reduction,
  )
  if prefix_scan and reduction:
    return unflatten(outs_flat[0]), unflatten(outs_flat[1])
  else:
    return unflatten(outs_flat)


def pscan(x, op, axis_name, reduction: bool = False):
  """Compute a prefix scan on ``x`` over the mapped axis ``axis_name``.

  If ``x`` is a pytree then the result is equivalent to mapping this function to
  each leaf in the tree.

  Args:
    x: array(s) with a mapped axis named ``axis_name``.
    op: the operation from `jax.numpy`, e.g., `jax.numpy.add`.
      May also be `multiply`, `maximum`, `minimum`, `bitwise_and`,
      `bitwise_xor`, or `bitwise_or`.
    axis_name: hashable Python object used to name a mapped axis.
    reduction: if True, additionally return the all-reduce on ``x`` with the
      same operation.

  Returns:
    Array(s) with the same shape as ``x`` representing the result of an
    exclusive prefix scan using the operation ``op`` along the axis
    ``axis_name``.

    If reduction is True, returns the all-reduce as well.
  """
  return _fanin_fanout(
      x, op=op, axis_name=axis_name, prefix_scan=True, reduction=reduction
  )


def preduce(x, op, axis_name):
  """Compute an all-reduce on ``x`` over the mapped axis ``axis_name``.

  If ``x`` is a pytree then the result is equivalent to mapping this function to
  each leaf in the tree.

  Args:
    x: array(s) with a mapped axis named ``axis_name``.
    op: the operation from `jax.numpy`, e.g., `jax.numpy.add`.
      May also be `multiply`, `maximum`, `minimum`, `bitwise_and`,
      `bitwise_xor`, or `bitwise_or`.
    axis_name: hashable Python object used to name a mapped axis.

  Returns:
    Array(s) with the same shape as ``x`` representing the result of an
    all-reduce using the operation ``op`` along the axis ``axis_name``.
  """
  if op == jnp.add:
    return lax.psum(x, axis_name=axis_name)
  elif op == jnp.maximum:
    return lax.pmax(x, axis_name=axis_name)
  elif op == jnp.minimum:
    return lax.pmin(x, axis_name=axis_name)
  else:
    return _fanin_fanout(
        x, op=op, axis_name=axis_name, prefix_scan=False, reduction=True
    )
