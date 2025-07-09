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

"""Implementation of conjugate gradients."""

from functools import partial
import operator

from jax import lax
import jax.numpy as jnp
from jax.tree_util import tree_leaves
from jax.tree_util import tree_map


def _vdot(a, b, dot_fn):
  return sum(tree_leaves(tree_map(dot_fn, a, b)))


def cg(
    A, b, x0=None, *, tol=1e-5, atol=0.0, maxiter=None, M=None, dot_fn=jnp.vdot
):
  """Conjugate gradient solver.

  This is based on jax.scipy.sparse.linalg.cg but with the difference that the
  residual is takes the preconditioner `M` into account. This version
  also supports a user-supplied dot product and more diagnostic information.

  Args:
    A: A linear operator.
    b: A pytree representing the right-hand side of the linear system.
    x0: An initial solution guess. Same structure as b.
    tol: The relative tolerance for the conjugate gradient solver.
    atol: The absolute tolerance for the conjugate gradient solver.
    maxiter: The maximum number of iterations.
    M: A preconditioner for the conjugate gradient solver.
    dot_fn: A function for computing the dot product.

  Returns:
    x: The solution of the linear system, or the last iterate after `max_iter`
      iterations was reached.
    info: A dictionary containing the residual and number of iterations.
  """
  if x0 is None:
    x0 = tree_map(jnp.zeros_like, b)

  if maxiter is None:
    size = sum(bi.size for bi in tree_leaves(b))
    maxiter = 10 * size  # copied from scipy

  if M is None:
    M = lambda x: x

  # tolerance handling uses the non-legacy behavior of scipy.sparse.linalg.cg
  bs = _vdot(b, b, dot_fn=dot_fn)
  atol2 = jnp.maximum(jnp.square(tol) * bs, jnp.square(atol))

  def cond_fun(value):
    _, _, gamma, _, k = value
    # the residual is changed from the factory implementation to take the
    # preconditioner into account.
    rs = gamma
    return (rs > atol2) & (k < maxiter)

  def body_fun(value):
    x, r, gamma, p, k = value
    Ap = A(p)
    alpha = gamma / _vdot(p, Ap, dot_fn=dot_fn).astype(dtype)

    x_ = tree_map(operator.add, x, tree_map(partial(operator.mul, alpha), p))
    r_ = tree_map(operator.sub, r, tree_map(partial(operator.mul, alpha), Ap))
    z_ = M(r_)
    gamma_ = _vdot(r_, z_, dot_fn=dot_fn).astype(dtype)
    beta_ = gamma_ / gamma
    p_ = tree_map(operator.add, z_, tree_map(partial(operator.mul, beta_), p))
    return x_, r_, gamma_, p_, k + 1

  r0 = tree_map(operator.sub, b, A(x0))
  p0 = z0 = M(r0)
  dtype = jnp.result_type(*tree_leaves(p0))
  gamma0 = _vdot(r0, z0, dot_fn=dot_fn).astype(dtype)
  initial_value = (x0, r0, gamma0, p0, 0)

  x_final, _, gamma, _, num_iters = lax.while_loop(cond_fun, body_fun,
                                                   initial_value)
  info = {'residual': gamma, 'num_iterations': num_iters}
  return x_final, info
