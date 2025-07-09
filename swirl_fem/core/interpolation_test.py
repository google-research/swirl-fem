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

import itertools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import scipy.special as sp

from swirl_fem.core.interpolation import BarycentricInterpolator
from swirl_fem.core.interpolation import Nodes1D
from swirl_fem.core.interpolation import NodeType
from swirl_fem.core.interpolation import Quadrature1D


def tensor_product_flat(xs, ndim):
  """Returns a lexicographically ordered `ndim`-fold tensor product of `xs`."""
  xs_outer = np.stack(np.meshgrid(*([xs] * ndim), indexing='ij'), axis=-1)
  xs_outer_flat = xs_outer.reshape((len(xs)**ndim, ndim))
  return xs_outer_flat


class InterpolationTest(parameterized.TestCase):

  def test_nodes_uniform(self):
    num_points = 3
    gridpoints = Nodes1D.create(num_points=num_points,
                                node_type=NodeType.NEWTON_COTES)
    np.testing.assert_equal(
        np.linspace(-1, 1, num=num_points), gridpoints.node_values)

  def test_nodes_gauss_legendre(self):
    gridpoints = Nodes1D.create(num_points=3, node_type=NodeType.GAUSS_LEGENDRE)
    np.testing.assert_equal(
        [-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)], gridpoints.node_values)

  @parameterized.parameters(1, 4)
  def test_nodes_gauss_lobatto_legendre(self, order):
    gridpoints = Nodes1D.create(num_points=order + 1,
                                node_type=NodeType.GAUSS_LOBATTO_LEGENDRE)
    expected = {
        1: [-1., 1.],
        4: [-1., -np.sqrt(21) / 7, 0, np.sqrt(21) / 7, 1.],
    }[order]
    np.testing.assert_equal(expected, gridpoints.node_values)

  def test_nodes_continuous(self):
    order = 3
    with self.subTest('newton_cotes'):
      gridpoints = Nodes1D.create(num_points=order + 1,
                                  node_type=NodeType.NEWTON_COTES)
      self.assertTrue(gridpoints.is_continuous())
    with self.subTest('gauss_legendre'):
      gridpoints = Nodes1D.create(num_points=order + 1,
                                  node_type=NodeType.GAUSS_LEGENDRE)
      self.assertFalse(gridpoints.is_continuous())
    with self.subTest('gauss_lobatto_legendre'):
      gridpoints = Nodes1D.create(num_points=order + 1,
                                  node_type=NodeType.GAUSS_LOBATTO_LEGENDRE)
      self.assertTrue(gridpoints.is_continuous())

  def test_quadrature_polynomial(self):
    quad = Quadrature1D.create(
        num_points=2, quadrature_type=NodeType.GAUSS_LEGENDRE)

    # integrate f(x) = 3x^2 - 4x^3 over [-1, 1].
    integral = np.einsum(
        'q,q->', quad.weights,
        np.vectorize(lambda x: 3 * x**2 - 4 * x**3)(quad.nodes.node_values))
    self.assertEqual(2., integral)

  def test_quadrature_trigonometric(self):
    quad = Quadrature1D.create(
        num_points=4, quadrature_type=NodeType.GAUSS_LEGENDRE)

    # integrate f(x) = cos(x) over [-1, 1].
    integral = np.einsum('q,q->', quad.weights,
                         np.vectorize(np.cos)(quad.nodes.node_values))
    self.assertAlmostEqual(np.sin(1.) - np.sin(-1.), integral, places=6)

  def test_quadrature_linear_gll(self):
    quad = Quadrature1D.create(
        num_points=2,
        quadrature_type=NodeType.GAUSS_LOBATTO_LEGENDRE)

    # integrate f(x) = 1 - x over [-1, 1].
    integral = np.einsum('q,q->', quad.weights,
                         np.vectorize(lambda x: 1 - x)(quad.nodes.node_values))
    self.assertAlmostEqual(2., integral, places=6)

  def test_quadrature_trigonometric_gll(self):
    quad = Quadrature1D.create(
        num_points=5,
        quadrature_type=NodeType.GAUSS_LOBATTO_LEGENDRE)

    # integrate f(x) = cos(x) over [-1, 1].
    integral = np.einsum('q,q->', quad.weights,
                         np.vectorize(np.cos)(quad.nodes.node_values))
    self.assertAlmostEqual(np.sin(1.) - np.sin(-1.), integral, places=6)

  @parameterized.parameters(1, 2, 3)
  def test_quadrature_weights_nd(self, ndim):
    quad = Quadrature1D.create(
        num_points=4, quadrature_type=NodeType.GAUSS_LEGENDRE)
    weights_nd = quad.weights_nd(ndim)

    # Each weight must be the tensor-product of the 1D weights, arranged in the
    # lexicographic order.
    shape = [quad.num_points] * ndim
    self.assertEqual(weights_nd.shape, (np.prod(shape),))
    for i, idx in enumerate(np.ndindex(*shape)):
      self.assertAlmostEqual(weights_nd[i], np.prod(quad.weights[[idx]]))

  @parameterized.parameters(1, 2, 5, 8)
  def test_quadrature_chebyshev(self, n):
    # integrate the square of the nth Chebyshev polynomial over [-1, 1].
    # it has degree 2n, so a Gauss-Legendre quadrature rule with n + 1 points
    # should be enough to integrate it exactly.
    quad = Quadrature1D.create(
        num_points=n + 1, quadrature_type=NodeType.GAUSS_LEGENDRE)
    values = [np.square(sp.eval_chebyt(n, x)) for x in quad.nodes.node_values]
    integral = np.einsum('q,q->', quad.weights, values)
    # Compare against the closed form solution.
    self.assertAlmostEqual((4 * n * n - 2) / (4 * n * n - 1), integral)

  @parameterized.parameters(
      itertools.product((3, 5, 7, 9),
                        (NodeType.NEWTON_COTES, NodeType.GAUSS_LEGENDRE,
                         NodeType.GAUSS_LOBATTO_LEGENDRE)))
  def test_barycentric_weights(self, n, node_type):
    # Test barycentric weights against the analytical formula involving the
    # lagrange polynomial.
    gridpoints = Nodes1D.create(num_points=n, node_type=node_type)
    # The evalpoints don't matter for the barycentric weights.
    interpolator = BarycentricInterpolator(
        ndim=1, gridpoints_1d=gridpoints, evalpoints_1d=gridpoints)

    def lagrange(x):
      return jnp.prod(jnp.stack([x - xi for xi in gridpoints.node_values]))

    # The barycentric weights are valid even if they're scaled by a constant; so
    # we normalize them before comparing.
    analytical_bary_weights = 1 / (
        jax.vmap(jax.grad(lagrange))(gridpoints.node_values))
    analytical_bary_weights /= jnp.max(jnp.abs(analytical_bary_weights))

    obtained_bary_weights = interpolator._barycentric_weights()
    obtained_bary_weights /= jnp.max(jnp.abs(obtained_bary_weights))
    np.testing.assert_allclose(analytical_bary_weights, obtained_bary_weights,
                               rtol=1e-3, atol=1e-3)

  def test_interpolate_1d(self):
    evalpoints = Nodes1D.create(num_points=2, node_type=NodeType.GAUSS_LEGENDRE)
    gridpoints = Nodes1D.create(num_points=3, node_type=NodeType.NEWTON_COTES)
    interpolator = BarycentricInterpolator(
        ndim=1, gridpoints_1d=gridpoints, evalpoints_1d=evalpoints)

    # use barycentric interpolator to interpolate (and integrate) a quadratic
    # function f(x) = 3x^2 + 2x over [-1, 1].
    # first, obtain the values at the nodal gridpoints.
    f = lambda x: 3 * x**2 + 2 * x
    nodal_values = np.vectorize(f)(gridpoints.node_values)

    # next, use the interpolation matrix to find the interpolated values at the
    # evaluation points.
    interpolated_values = np.einsum('qi,i->q',
                                    interpolator.interpolation_matrix(),
                                    nodal_values)
    # ensure that we recover the correct values.
    np.testing.assert_allclose(
        np.vectorize(f)(evalpoints.node_values), interpolated_values)

  def test_interpolate_2d(self):
    ndim = 2
    evalpoints = Nodes1D.create(num_points=3, node_type=NodeType.GAUSS_LEGENDRE)
    gridpoints = Nodes1D.create(num_points=3, node_type=NodeType.NEWTON_COTES)
    interpolator = BarycentricInterpolator(
        ndim=ndim, gridpoints_1d=gridpoints, evalpoints_1d=evalpoints)

    # use barycentric interpolator to interpolate (and integrate) a quadratic
    # function f(x, y) = x^2 + y over [-1, 1]^2.

    # first, obtain the values at the nodal gridpoints.
    f = lambda x: x[0]**2 + x[1]
    gridpoints_2d = tensor_product_flat(gridpoints.node_values, ndim)
    nodal_values = np.array([f(x) for x in gridpoints_2d])

    # next, use the interpolation matrix to find the interpolated values at the
    # evaluation points.
    interpolated_values = np.einsum('qi,i->q',
                                    interpolator.interpolation_matrix(),
                                    nodal_values)
    # ensure that we recover the correct values.
    evalpoints_2d = tensor_product_flat(evalpoints.node_values, ndim)
    np.testing.assert_allclose([f(x) for x in evalpoints_2d],
                               interpolated_values)

  def test_interpolate_1d_grad(self):
    evalpoints = Nodes1D.create(num_points=2, node_type=NodeType.GAUSS_LEGENDRE)
    gridpoints = Nodes1D.create(num_points=3, node_type=NodeType.NEWTON_COTES)
    interpolator = BarycentricInterpolator(
        ndim=1, gridpoints_1d=gridpoints, evalpoints_1d=evalpoints)

    # use barycentric interpolator to interpolate (and integrate) the derivative
    # of a quadratic function f(x) = 3x^2 + 2x over [-1, 1].

    # first, obtain the values at the nodal gridpoints.
    f = lambda x: 3 * x**2 + 2 * x
    nodal_values = np.vectorize(f)(gridpoints.node_values)

    # next, use the gradient interpolation matrix to find the interpolated
    # values of f'(x) at the evaluation points.
    interpolated_values = np.einsum('qid,i->qd',
                                    interpolator.interpolation_matrix_grad(),
                                    nodal_values)

    # ensure that we recover the correct values.
    df = lambda x: 6 * x + 2
    expected_values = np.vectorize(df)(evalpoints.node_values).reshape(
        (evalpoints.num_points, 1))
    np.testing.assert_allclose(expected_values, interpolated_values)

  def test_interpolate_2d_grad(self):
    ndim = 2
    evalpoints = Nodes1D.create(num_points=3, node_type=NodeType.GAUSS_LEGENDRE)
    gridpoints = Nodes1D.create(num_points=3, node_type=NodeType.NEWTON_COTES)
    interpolator = BarycentricInterpolator(
        ndim=ndim, gridpoints_1d=gridpoints, evalpoints_1d=evalpoints)

    # use barycentric interpolator to interpolate (and integrate) the gradient
    # of a quadratic function f(x, y) = x^2 + y over [-1, 1]^2.

    # first, obtain the values at the nodal gridpoints.
    f = lambda x: x[0]**2 + x[1]
    gridpoints_2d = tensor_product_flat(gridpoints.node_values, ndim)
    nodal_values = np.array([f(x) for x in gridpoints_2d])

    # next, use the gradient interpolation matrix to find the gradient
    # interpolated at the evaluation points.
    interpolated_values = np.einsum('qid,i->qd',
                                    interpolator.interpolation_matrix_grad(),
                                    nodal_values)

    # ensure that we recover the correct values of gradient(f).
    df = lambda x: (2 * x[0], 1)
    evalpoints_2d = tensor_product_flat(evalpoints.node_values, ndim)
    expected_values = np.array([df(x) for x in evalpoints_2d])
    np.testing.assert_allclose(expected_values, interpolated_values)

  @parameterized.parameters(1, 2, 3, 4)
  def test_bdf(self, order):
    # Recover the BDF formula by evaluating the derivative from Newton-Cotes
    # gridpoints (i.e. distributed uniformly in [-1, 1]) at the single point 1.
    gridpoints = Nodes1D.create(
        num_points=order + 1, node_type=NodeType.NEWTON_COTES)
    evalpoints = Nodes1D.create_single_point(node_value=1.)
    interpolator = BarycentricInterpolator(
        ndim=1, gridpoints_1d=gridpoints, evalpoints_1d=evalpoints)
    h = 2 / order
    coeffs = interpolator.interpolation_matrix_grad().reshape((-1)) * h
    expected_coeffs = {
        1: [-1, 1],
        2: [1 / 2, -4 / 2, 3 / 2],
        3: [-2 / 6, 9 / 6, -18 / 6, 11 / 6],
        4: [3 / 12, -16 / 12, 36 / 12, -48 / 12, 25 / 12],
    }
    np.testing.assert_allclose(coeffs, expected_coeffs[order])


if __name__ == '__main__':
  jax.config.update('jax_enable_x64', True)
  absltest.main()
