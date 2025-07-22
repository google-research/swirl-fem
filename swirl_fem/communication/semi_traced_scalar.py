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

"""Semi-traced scalar."""

import dataclasses
import operator
from typing import Any, Callable

from jax import lax
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np


@dataclasses.dataclass(eq=False, frozen=True)
class SemiTracedScalar:
  """Simultaneous dynamic local and static global views of a scalar.

  An instance `x` of SemiTracedScalar behaves much like a scalar, in particular
  supporting arithmetic with other SemiTracedScalars or constants, but tracks
  both a "local" value of the scalar, specific to one index along a mapped
  axis, which may be a traced value, as well as a "global" view of the scalar
  as a concrete array of the values of the scalar across all indices of the
  mapped axis.

  The prototypical SemiTracedScalars are the axis index and axis size, which are
  given by

    idx, n = SemiTracedScalar.index_and_size(axis_name)

  These have local values

    idx.local == lax.axis_index(axis_name)  # abstract traced value
    n.local == lax.psum(1, axis_name)  # concrete integer

  while, e.g.,

    idx.global_ == np.arange(n.local).

  Arithmetic and comparison operators may be used to derive other
  SemiTracedScalars, for example `idx < n // 2`.
  """
  local: ArrayLike  # local value of scalar, potentially a tracer
  global_: np.ndarray  # global value of scalar across mapped axis, concrete

  # signal to pytype that additional methods (the overloaded arithmetic and
  # comparison operators) are added below the class definition
  _HAS_DYNAMIC_ATTRIBUTES = True

  @staticmethod
  def where(c, x, y):
    """x if c else y."""
    loc = jnp.where(c.local, x.local, y.local)
    gbl = np.where(c.global_, x.global_, y.global_)
    return SemiTracedScalar(local=loc, global_=gbl)

  @staticmethod
  def axis_index(axis_name):
    """`lax.axis_index(axis_name)` as a SemiTracedScalar."""
    i = lax.axis_index(axis_name=axis_name)
    n = int(lax.psum(1, axis_name=axis_name))
    return SemiTracedScalar(local=i, global_=np.arange(n))

  @staticmethod
  def axis_size(axis_name):
    """`lax.psum(1, axis_name)` as a SemiTracedScalar."""
    n = int(lax.psum(1, axis_name=axis_name))
    return SemiTracedScalar(local=n, global_=np.full(shape=(n,), fill_value=n))

  @staticmethod
  def constant(c, axis_name):
    """the constant `c` as a SemiTracedScalar."""
    n = int(lax.psum(1, axis_name=axis_name))
    return SemiTracedScalar(local=c, global_=np.full(shape=(n,), fill_value=c))


_SemiTracedScalarMethod = Callable[[SemiTracedScalar, Any], SemiTracedScalar]


def _lift_op(f: Callable[[Any, Any], Any]) -> _SemiTracedScalarMethod:
  """Lift the op `f` to SemiTracedScalars."""
  def impl(x: SemiTracedScalar, y: Any) -> SemiTracedScalar:
    if isinstance(y, SemiTracedScalar):
      return SemiTracedScalar(
          local=f(x.local, y.local), global_=f(x.global_, y.global_)
      )
    else:
      return SemiTracedScalar(local=f(x.local, y), global_=f(x.global_, y))

  return impl


def _lift_ops(
    f: Callable[[Any, Any], Any]
) -> tuple[_SemiTracedScalarMethod, _SemiTracedScalarMethod]:
  """Returns the op `f` and its reflection lifted to SemiTracedScalars."""
  return _lift_op(f), _lift_op(lambda x, y: f(y, x))


def _add_comparisons() -> None:
  """Adds the comparison operators to the SemiTracedScalar class."""
  for op_name in ['lt', 'le', 'eq', 'ne', 'gt', 'ge']:
    op = _lift_op(getattr(operator, f'__{op_name}__'))
    setattr(SemiTracedScalar, f'__{op_name}__', op)


def _add_ops() -> None:
  """Adds the arithmetic operators to the SemiTracedScalar class."""
  for op_name in ['add', 'sub', 'mul', 'truediv', 'floordiv', 'mod', 'pow',
                  'lshift', 'rshift', 'and', 'xor', 'or']:
    op, rop = _lift_ops(getattr(operator, f'__{op_name}__'))
    setattr(SemiTracedScalar, f'__{op_name}__', op)
    setattr(SemiTracedScalar, f'__r{op_name}__', rop)


_add_comparisons()
_add_ops()
