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

from functools import partial

from absl.testing import absltest
import jax
from jax import grad
from jax.experimental.ode import odeint
import jax.numpy as jnp
import numpy as np

from swirl_fem.sde.sdeint import brownian_path
from swirl_fem.sde.sdeint import sdeint


class SdeintTest(absltest.TestCase):

  def test_drift_simple(self):
    n = 512
    rng = jax.random.PRNGKey(42)
    dw = brownian_path(rng=rng, n=n)
    y0 = jnp.array(1., dtype=jnp.float64)
    ts = jnp.array([0., 1.], dtype=jnp.float64)

    def f(y, t):
      del t
      return y

    def fg(y, t, dw):
      del dw
      return f(y, t), 0.  # zero diffusion

    # An SDE with zero diffusion is just an ODE!
    y1 = sdeint(fg, y0, ts, dw)
    _, expected_y1 = odeint(f, y0, ts)
    np.testing.assert_allclose(expected_y1, y1, atol=1e-3, rtol=1e-3)

  def test_drift_multiple_timepoints(self):
    n = 512
    rng = jax.random.PRNGKey(42)
    dw = brownian_path(rng=rng, n=n)
    y0 = jnp.array(1., dtype=jnp.float64)
    ts = jnp.array([0, .5, 1], dtype=jnp.float64)

    def f(y, t):
      del t
      return y

    def fg(y, t, dw):
      del dw
      return f(y, t), 0.  # zero diffusion

    # An SDE with zero diffusion is just an ODE!
    y1 = sdeint(fg, y0, ts, dw)
    expected_y1 = odeint(f, y0, ts)
    np.testing.assert_allclose(expected_y1[1:], y1, atol=1e-3, rtol=1e-3)

  def test_diffusion_simple(self):
    n = 512
    rng = jax.random.PRNGKey(42)
    dw = brownian_path(rng=rng, n=n)
    y0 = jnp.array(1., dtype=jnp.float64)
    ts = jnp.array([0., 1.], dtype=jnp.float64)
    alpha = .2

    def f(y, t, dw):
      del t
      return 0., alpha * y * dw  # zero drift

    # Due to the linear diffusion term, we have an instance of geometric
    # brownian motion; with a closed form solution.
    y1 = sdeint(f, y0, ts, dw)
    w1 = dw.sum()
    np.testing.assert_allclose(
        y0 * jnp.exp(alpha * w1), y1, atol=1e-3, rtol=1e-3)

  def test_linear_drift_and_diffusion(self):
    n = 512
    rng = jax.random.PRNGKey(42)
    dw = brownian_path(rng=rng, n=n)
    y0 = jnp.array(1., dtype=jnp.float64)
    ts = jnp.array([0., 1.], dtype=jnp.float64)
    mu = 1.5
    sigma = .25

    def f(y, t, dw):
      del t
      return mu * y, sigma * y * dw

    # Instance of geometric brownian motion / Black-Scholes equation; with a
    # closed form solution.
    y1 = sdeint(f, y0, ts, dw)
    w1 = dw.sum()
    np.testing.assert_allclose(
        y0 * jnp.exp(mu + sigma * w1), y1, atol=1e-3, rtol=1e-2)

  def test_grad_wrt_state(self):
    n = 512
    rng = jax.random.PRNGKey(42)
    dw = brownian_path(rng=rng, n=n)
    y0 = jnp.array(2., dtype=jnp.float64)
    ts = jnp.array([0., 1.], dtype=jnp.float64)
    mu = 1.5
    sigma = .25

    def f(y, t, dw):
      del t
      return mu * y, sigma * y * dw

    # Instance of geometric brownian motion / Black-Scholes equation; with a
    # closed form solution.
    grad_wrt_y = grad(lambda z: sdeint(f, z, ts, dw)[0])(y0)
    w1 = dw.sum()

    np.testing.assert_allclose(
        jnp.exp(mu + sigma * w1), grad_wrt_y, atol=1e-3, rtol=1e-2)

  def test_grad_wrt_params(self):
    n = 1024
    rng = jax.random.PRNGKey(42)
    dw = brownian_path(rng=rng, n=n)
    y0 = jnp.array(1.5, dtype=jnp.float64)
    ts = jnp.array([0., 1.], dtype=jnp.float64)
    mu = 1.2
    sigma = .1

    def f(theta, y, t, dw):
      del t
      return theta[0] * y, theta[1] * y * dw

    # Instance of geometric brownian motion / Black-Scholes equation; with a
    # closed form solution.
    grad_wrt_theta = grad(
        lambda theta: sdeint(partial(f, theta), y0, ts, dw)[0])([mu, sigma])

    w1 = dw.sum()
    expected_grad = jnp.array(
        [y0 * jnp.exp(mu + sigma * w1), y0 * w1 * jnp.exp(mu + sigma * w1)])

    np.testing.assert_allclose(
        expected_grad, grad_wrt_theta, atol=1e-3, rtol=1e-2)

  def test_multivariate_grad_wrt_params(self):
    n = 1024
    rng = jax.random.PRNGKey(42)
    y0 = jnp.array([1., 1.5], dtype=jnp.float64)
    ts = jnp.array([0., 1.], dtype=jnp.float64)
    # theta has shape [2, 2, 2]. The first 2D matrix is used for the linear
    # drift and the second for the linear diffusion.
    # theta0 = jax.vmap(jnp.diag)(jnp.array([[1., 2.], [.3, .4]]))
    theta0 = jnp.array([[1., 2.], [.3, .4]], dtype=jnp.float64)
    # Sample a 2-dimensional brownian path.
    dw = jax.vmap(partial(brownian_path, n=n), out_axes=-1)(
        jax.random.split(rng, 2))

    def f(theta, y, t, dw):
      del t
      return theta[0] @ y, (theta[1] @ y) * dw

    # Instance of geometric brownian motion / Black-Scholes equation; with a
    # closed form solution.
    def loss(theta):
      theta = jax.vmap(jnp.diag)(theta)
      y1 = sdeint(partial(f, theta), y0, ts, dw)[0]
      return y1.sum()

    grad_wrt_theta = grad(loss)(theta0)

    w1 = dw.sum(axis=0)
    exp_term = jnp.exp(theta0[0] + theta0[1] * w1)
    expected_grad = jnp.stack([y0 * exp_term, y0 * w1 * exp_term])
    np.testing.assert_allclose(
        expected_grad, grad_wrt_theta, atol=1e-3, rtol=1e-2)


if __name__ == '__main__':
  jax.config.update('jax_enable_x64', True)
  absltest.main()