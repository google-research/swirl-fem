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

"""JIT compilation of a distributed computation."""
import functools
import inspect
from typing import Callable, Sequence

import jax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec  # pylint: disable=g-multiple-import


def jit_distributed(
    f: Callable,  # pylint: disable=g-bare-generic
    mesh: Mesh,
    axis_name,
    *,
    static_argnames: Sequence[str] | None = None,
) -> Callable:  # pylint: disable=g-bare-generic
  """JIT compilation of a distributed computation.

  Compiles `f` to run locally on each device in `mesh`, with inputs and
  outputs sharded along the given `axis_name`.

  The returned function assumes its inputs and outputs are sharded along their
  first dimension, assumed to have a size exactly matching the axis size. The
  inputs and outputs to `f` are missing this dimension. This mimics the
  semantics of `jax.pmap` with `in_axes=0` and `out_axes=0`.

  By default, any keyword-only arguments to `f` (those appearing after a `*`
  in its definition) are assumed to be static, though this can be overridden by
  supplying an explicit sequence of static argument names `static_argnames`.
  Static arguments are *not* sharded and are treated as constants during
  compilation.

  Args:
    f: Function to compile.
    mesh: jax.sharding.Mesh for the distributed computation.
    axis_name: hashable Python object used to name a mapped axis.
    static_argnames: names of static arguments, treated as constants during
      compilation. Defaults to all keyword-only arguments (those following a *).

  Returns:
    Compiled version of the distributed computation.
  """
  sig = inspect.signature(f)
  def bind(argnames, *args, **kwargs):
    """Subset `argnames` of BoundArguments for `f` from `*args, **kwargs`."""
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    args_to_remove = set(bound.arguments.keys()) - set(argnames)
    for x in args_to_remove:
      del bound.arguments[x]
    return bound

  if static_argnames is None:
    static_argnames = [
        x for x, p in sig.parameters.items() if p.kind == p.KEYWORD_ONLY
    ]

  nonsharded_args = set(static_argnames)
  sharded_args = set(sig.parameters.keys()) - nonsharded_args

  spec = PartitionSpec(axis_name)
  shmap = functools.partial(
      shard_map, mesh=mesh, in_specs=spec, out_specs=spec, check_rep=False
  )

  @functools.wraps(f)
  def pmapped(*args, **kwargs):
    nonsharded = bind(nonsharded_args, *args, **kwargs)
    sharded = bind(sharded_args, *args, **kwargs)
    if sharded.args or sharded.kwargs:
      g = functools.partial(f, *nonsharded.args, **nonsharded.kwargs)
      return shmap(jax.vmap(g, in_axes=0))(*sharded.args, **sharded.kwargs)
    else:
      # vmap requires at least one input. Use a (local) array of length 1,
      # which should be eliminated by DCE during compilation.
      def g(unused: jax.Array):
        del unused
        return f(*nonsharded.args, **nonsharded.kwargs)
      return shmap(lambda: jax.vmap(g, in_axes=0)(jnp.array([0], jnp.int32)))()

  return jax.jit(pmapped, static_argnames=static_argnames)
