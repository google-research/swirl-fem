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

"""The flax.nn lifted version of the SDE solver."""

import functools
from typing import Any, Callable, Sequence

from flax.core.lift import CollectionFilter
from flax.core.lift import pack
from flax.core.lift import PRNGSequenceFilter
from flax.linen.transforms import lift_transform
import jax
import jax.numpy as jnp
from swirl_fem.sde.sdeint import sdeint
from flax.typing import MutableVariableDict, RNGSequences


# The state at a single time point of the SDE can be any jax pytree.
State = Any


def core_sdeint(
    fn: Callable[..., Any],
    variables: CollectionFilter = True,
    rngs: PRNGSequenceFilter = True,
) -> Callable[..., Any]:
  """Lifted version of `sdeint`. See the documentation of `sdeint` for details.

  Args:
    fn: Scope function which should be integrated during application.
    variables: The variable collections that are lifted. By default all
      collections are lifted.
    rngs: The PRNG sequences that are lifted. By default all PRNG sequences
      are lifted.

  Returns:
    A wrapped version of `fn` transformed by sdeint.
  """
  def inner(scope_fn, repack_fn, variable_groups, rng_groups, *args):

    @functools.wraps(fn)
    def deriv_fn(x, t, dw,
                 variable_groups: MutableVariableDict,
                 rng_groups: RNGSequences, *args):
      # Recreate the flax scope objects from pure variables, rngs pytrees.
      scope = scope_fn(variable_groups, rng_groups)
      # Apply the model function.
      ys = fn(scope, x, t, dw, *args)
      return ys

    x, ts, dw, *additional_args = args
    scope = scope_fn(variable_groups, rng_groups)

    def is_scope_initializing(s):
      vs = s.variables()
      return 'params' not in vs or not vs['params']
    is_initializing = all(map(is_scope_initializing, scope))

    if is_initializing:
      drift, diffusion = fn(scope, x, ts[0], dw[0], *additional_args)
      out = jnp.stack([drift + diffusion] * len(ts[1:]))
      return out, repack_fn(scope)
    else:
      y_integrated = sdeint(
          deriv_fn, x, ts, dw, variable_groups, rng_groups, *additional_args)
      return y_integrated, repack_fn(scope)

  return pack(inner, (variables,), (variables,), (rngs,), name='sdeint')


# Make the functional-core transform above a full linen transform that can act
# as a class transform or a module method transform.
nn_sdeint = functools.partial(lift_transform, core_sdeint)
