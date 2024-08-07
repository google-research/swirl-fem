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

"""An Euler-Heun SDE solver with an adjoint implementation."""

from collections.abc import Callable
from functools import partial
from typing import Any, Sequence

import jax
from jax import lax
import jax.extend as jex
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp


# The state at a single time point of the SDE can be any pytree.
State = Any


def brownian_path(
    rng: jax.Array, n: int, dtype=jnp.float32
) -> jax.Array:
  """Computes a standard Brownian sample path on `n` uniformly spaced points."""
  return (jnp.sqrt(1 / n) * jax.random.normal(rng, (n,))).astype(dtype)


def sdeint(
    f: Callable[[State, float], tuple[State, State]],
    y0: State,
    ts: Sequence[jax.Array],
    dw: jax.Array,
    *args
) -> Sequence[State]:
  """Euler-Heun SDE solver for Stratonovich SDEs.

  Solves a Stratonovich SDE on the time interval `[t0, t1]` using the Euler-Heun
  scheme [1]. The solver requires a standard brownian path sample `dw` with
  values given at uniformly-spaced time points on the interval `[0, 1]`.

  We solve a Stratonovich SDE of the following form:
    dYₜ = μ(Yₜ, t; θ) dt + σ(Yₜ, t; θ) ∘ dWₜ
  where μ and σ are user specified drift and diffusion functions.

  This function returns a tuple representing the final state and optionally, the
  path {dYₜ} corresponding to brownian path used. This solver is differentiable
  with respect to the arguments `y0` and `args` using an adjoint implementation
  based on [2].

  ### References.
    1. Thomas Schaffter. Numerical Integration of SDEs: A Short Tutorial. 2010.
    2. Li, X., Wong, T.L., Chen, R.T.Q. & Duvenaud, D.. Scalable Gradients for
       Stochastic Differential Equations. AISTATS 2020.

  Args:
    f: A function to evaluate the drift and diffusion terms of `y` at time `t`
      as `f(y, t, dw, *args)`, producing a tuple `(drift, diffusion)`, each of
      which must have the same pytree structure as `y0`. `drift` represents
      the term `μ(...)` and `diffusion` represents the term `σ(...) ∘ dWₜ`. Note
      that the function outputs the diffusion term after incorporating the
      brownian motion.
    y0: A pytree representing the initial value for the state.
    ts: A tuple containing a pair of floats `(t0, t1)`; the starting and ending
      times for the integration. We must have `0 <= t0 < t1 <= 1`. For numerical
      accuracy, `t0` and `t1` should be integer multiples of `1 / n` where
      `n = len(dw)`.
    dw: A brownian motion sample path.
    *args: Tuple of additional arguments for `f`.

  Returns:
    A tuple `(y1, path)`, where `y1` is the final state resulting from the
    integration; and `path` is the integration path.
  """
  t0, touts = ts[0], ts[1:]
  segmented_dw = dw.reshape((len(touts), -1, *dw.shape[1:]))

  def scan_fun(
      state: State, x: tuple[State, jax.Array]
  ) -> tuple[tuple[State, jax.Array], State]:
    dw, t_next = x
    y_curr, t_curr = state
    y_next = _sdeint_wrapper(f, False, y_curr, (t_curr, t_next), dw, *args)
    return (y_next, t_next), y_next

  _, ys = lax.scan(scan_fun, (y0, t0), (segmented_dw, touts))
  return ys


@partial(jax.jit, static_argnums=(0, 1))
def _sdeint_wrapper(f, reverse, y0, ts, dw, *args):
  f, consts = jax.custom_derivatives.closure_convert(f, y0, ts[0], dw[0], *args)
  y0, unravel = jax.flatten_util.ravel_pytree(y0)

  f = ravel_first_arg(f, unravel)
  ys = _sdeint(f, reverse, y0, ts, dw, *args, *consts)
  return unravel(ys)


@partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def _sdeint(f, reverse, y0, ts, dw, *args):
  """Implements SDE solver."""
  f_ = lambda x, t, dw: f(x, t, dw, *args)
  dt = (ts[1] - ts[0]) / len(dw)

  def step_fn(state, dw):
    t, x = state
    drift, diff = f_(x, t, dw)
    _, diffh = f_(x + diff, t + dt, dw)
    x = x + drift * dt + .5 * (diff + diffh)
    updated_state = lax.cond(
        jnp.logical_and(t >= ts[0], t < ts[1]), lambda: (t + dt, x), lambda:
        (t + dt, state[-1]))
    return updated_state, None

  t0 = ts[0]  # if not reverse else -ts[1]
  (_, x1), _ = lax.scan(step_fn, init=(t0, y0), xs=dw, reverse=reverse)
  return x1


def _sdeint_fwd(f, reverse, y0, ts, dw, *args):
  """Computes forward pass and residuals for sdeint."""
  y1 = _sdeint(f, reverse, y0, ts, dw, *args)
  return y1, (y1, ts, dw, args)


def _sdeint_rev(f, reverse, res, y1_bar):
  """Computes backward pass for `_sdeint`.

  We use the adjoint SDE algorithm from Li et al [1]. The arguments to this
  function are the non-differentiable arguments to `_sdeint`; followed by the
  residual arguments provided by the forward pass `_sdeint_fwd`; finally the
  cotangent corresponding to the output (`y1`) of the primal function. See
  the custom derivative tutorial in Jax documentation for more information about
  the jax custom derivative usage.
  https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html

  Args:
    f: Drift and diffusion parameterizing function.
    reverse: Boolean indicating whether to solve the time-reversed SDE for the
      backward pass.
    res: Residual argument values carried over from the forward pass.
    y1_bar: Cotangent for the primal function.

  Returns:
    Tuple of cotangent outputs for the inputs for the primal function.
  """
  y1, ts, dw, args = res

  def aug_f(aug_state, t, dw, *args):
    y, y_bar, *_ = aug_state
    drift_fn = lambda y, *args: f(y, -t, dw, *args)[0]
    diff_fn = lambda y, *args: f(y, -t, dw, *args)[1]

    drift_dot, vjpfun = jax.vjp(drift_fn, y, *args)
    drift_bar = (-drift_dot, *vjpfun(y_bar))

    diff_dot, vjpfun = jax.vjp(diff_fn, y, *args)
    diff_bar = (-diff_dot, *vjpfun(y_bar))
    return drift_bar, diff_bar

  # Run augmented system backwards.
  aug_init_state = (y1, y1_bar, jax.tree_util.tree_map(jnp.zeros_like, args))
  ts = (-ts[1], -ts[0])

  unused_y0, y_bar, args_bar = _sdeint_wrapper(
      aug_f, not reverse, aug_init_state, ts, dw, *args)

  # Note that the differential wrt `t` or `dw` is not implemented, so return
  # None for the corresponding cotangents.
  return (y_bar, None, None, *args_bar)


_sdeint.defvjp(_sdeint_fwd, _sdeint_rev)


def ravel_first_arg(f, unravel):
  return ravel_first_arg_(jex.linear_util.wrap_init(f), unravel).call_wrapped


@jex.linear_util.transformation
def ravel_first_arg_(unravel, y_flat, *args):
  y = unravel(y_flat)
  a, b = yield (y,) + args, {}
  a_flat, _ = ravel_pytree(a)
  b_flat, _ = ravel_pytree(b)
  yield a_flat, b_flat
