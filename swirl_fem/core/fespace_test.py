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

import itertools

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import grad
from jax import vmap
import jax.numpy as jnp
import numpy as np
import scipy.integrate

from swirl_fem.core.fespace import FiniteElementSpace
from swirl_fem.core.fespace import grad
from swirl_fem.core.fespace import div
from swirl_fem.core.interpolation import NodeType
from swirl_fem.core.interpolation import Quadrature1D
from swirl_fem.core.mesh import Mesh


def _make_unit_interval_mesh(num_elements, order):
  # Evaluate a function f(x) on a 1D mesh.
  # Create a 1D mesh on the unit interval [0, 1].
  num_nodes = 1 + order * num_elements
  node_coords = np.linspace(0, 1, num_nodes).reshape((num_nodes, 1))
  elements = np.array([
      list(range(order * i, order * (i + 1) + 1)) for i in range(num_elements)
  ])
  return Mesh.create(node_coords=node_coords, elements=elements)


def _quadrature_coords(fespace: FiniteElementSpace) -> jnp.ndarray:
  """Computes the coordinates of quadrature points on each element."""
  # Create a (num_elements, num_nodes_per_element, ndim)-shaped array containing
  # the nodal coordinates.
  elem_coords = fespace.mesh.element_coords()
  # Interpolate nodal coordinates to obtain the quadrature coordinates.
  return jnp.einsum('qn,mnd->mqd',
                    fespace.interpolator.interpolation_matrix(), elem_coords)


class FespaceTest(parameterized.TestCase):

  def test_integrate_linear_function_1d(self):
    mesh = _make_unit_interval_mesh(num_elements=8, order=1)

    # Set up FiniteElementSpace.
    quadrature = Quadrature1D.create(
        num_points=4, quadrature_type=NodeType.GAUSS_LEGENDRE)
    fespace = FiniteElementSpace.create(mesh, quadrature)

    # Evaluate a function on the node coordinates and interpolate it to the
    # quadrature points on each element. We use a linear (affine) function
    # since our mesh has order 1.
    f = lambda x: 1 + 5 * x[0]
    # Evaluate `f` on each node; then interpolate to quadrature points.
    integral = fespace.integrate(f)
    self.assertAlmostEqual(integral, 1 + 5 * .5)

  @parameterized.parameters(itertools.product(range(1, 3 + 1), range(1, 5)))
  def test_integrate_single_element(self, ndim, order):
    # Make a mesh containing a single element.
    num_nodes = (order + 1) ** ndim
    node_coords_per_dim = np.linspace(0, 1, num=order + 1)
    node_coords = np.meshgrid(*([node_coords_per_dim] * ndim), indexing='ij')
    node_coords = np.stack(node_coords, axis=-1).reshape(num_nodes, ndim)
    elements = np.arange(num_nodes, dtype=np.int32).reshape((1, num_nodes))
    mesh = Mesh.create(node_coords=node_coords, elements=elements)

    # Set up FiniteElementSpace.
    quadrature = Quadrature1D.create(
        num_points=order + 1, quadrature_type=NodeType.GAUSS_LEGENDRE)
    fespace = FiniteElementSpace.create(mesh, quadrature)

    # integrate a degree `order` function so that we get the exact answer
    f = lambda x: sum(x[i] ** order for i in range(ndim))
    # Integrating x^order yields 1 / (1 + order), and we have ndim identical
    # terms (by symmetry)
    self.assertAlmostEqual(fespace.integrate(f), ndim / (1 + order))

    # integrate the same function using nodal values
    u_local = jax.vmap(jax.vmap(f))(mesh.element_coords())
    nodal_f = fespace.scalar_function(u_local)
    self.assertAlmostEqual(fespace.integrate(nodal_f), ndim / (1 + order))

  @parameterized.parameters(itertools.product(range(1, 4), range(1, 5)))
  def test_integrate_grad_single_element(self, ndim, order):
    # Make a mesh containing a single element.
    num_nodes = (order + 1) ** ndim
    node_coords_per_dim = np.linspace(0, 1, num=order + 1)
    node_coords = np.meshgrid(*([node_coords_per_dim] * ndim), indexing='ij')
    node_coords = np.stack(node_coords, axis=-1).reshape(num_nodes, ndim)
    elements = np.arange(num_nodes, dtype=np.int32).reshape((1, num_nodes))
    mesh = Mesh.create(node_coords=node_coords, elements=elements)

    # Set up FiniteElementSpace.
    quadrature = Quadrature1D.create(
        num_points=order + 1, quadrature_type=NodeType.GAUSS_LEGENDRE)
    fespace = FiniteElementSpace.create(mesh, quadrature)

    # Evaluate a function on the node coordinates and interpolate it to the
    # quadrature points on each element. We use a degree `order` function.
    f = lambda x: sum(x[i] ** order for i in range(ndim))
    g = lambda x: grad(f)(x)[0]
    # grad(f) would be a `ndim`-shaped vector so check each of the dimensions
    # separately
    self.assertAlmostEqual(fespace.integrate(g), 1)

    # integrate the gradient of the function using nodal values
    u_local = jax.vmap(jax.vmap(f))(mesh.element_coords())
    nodal_f = fespace.scalar_function(u_local)
    nodal_g = lambda x: grad(nodal_f)(x)[0]
    self.assertAlmostEqual(fespace.integrate(nodal_g), 1.)

  def test_integrate_grad_generic_quad(self):
    # Make a mesh containing a single element; a quadrilateral where the X and Y
    # coordinates are not symmetric. This helps verify the jacobian and inverse
    # jacobian computations.
    node_coords = np.array([[0, 0], [0, 1], [1, 0], [1, 2]], dtype=np.float32)
    elements = np.arange(4, dtype=np.int32).reshape((1, 4))
    mesh = Mesh.create(node_coords=node_coords, elements=elements)

    # Set up FiniteElementSpace.
    quadrature = Quadrature1D.create(
        num_points=2, quadrature_type=NodeType.GAUSS_LEGENDRE)
    fespace = FiniteElementSpace.create(mesh, quadrature)

    # Evaluate a function on the node coordinates and interpolate it to the
    # quadrature points on each element.
    f = lambda x: 2 * x[0] - x[1] + 1
    u_local = jax.vmap(jax.vmap(f))(mesh.element_coords())
    nodal_f = fespace.scalar_function(u_local)

    # Integrate the 0th and 1st coordinate of the gradient separately. Since
    # the gradient is (2, -1) at every point the integral should be 2 and -1
    # times the area respectively.
    area = (1 + 2) / 2
    self.assertAlmostEqual(
        fespace.integrate(lambda x: grad(nodal_f)(x)[0]), 2 * area)
    self.assertAlmostEqual(
        fespace.integrate(lambda x: grad(nodal_f)(x)[1]), -area)

  def test_integrate_divergence_generic_quad(self):
    # Make a mesh containing a single element; a quadrilateral where the X and Y
    # coordinates are not symmetric. This helps verify the jacobian and inverse
    # jacobian computations.
    node_coords = np.array([[0, 0], [0, 1], [1, 0], [1, 2]], dtype=np.float32)
    elements = np.arange(4, dtype=np.int32).reshape((1, 4))
    mesh = Mesh.create(node_coords=node_coords, elements=elements)

    # Set up FiniteElementSpace.
    quadrature = Quadrature1D.create(
        num_points=2, quadrature_type=NodeType.GAUSS_LEGENDRE)
    fespace = FiniteElementSpace.create(mesh, quadrature)

    # Evaluate a function on the node coordinates and interpolate it to the
    # quadrature points on each element.
    f = lambda x: jnp.array([2 * x[0] - x[1], 3 * x[1]])
    u_local = jax.vmap(jax.vmap(f))(mesh.element_coords())
    nodal_f = fespace.vector_function(u_local)

    # Integrate the 0th and 1st coordinate of the gradient separately. Since
    # the gradient is (2, -1) at every point the integral should be 2 and -1
    # times the area respectively.
    area = (1 + 2) / 2
    # verify divergence
    self.assertAlmostEqual(fespace.integrate(div(nodal_f)), (2 + 3) * area)


  @parameterized.parameters(range(1, 4))
  def test_integrate_nodal_function_unit_interval(self, order):
    # Create a mesh of the given order on the unit interval.
    mesh = _make_unit_interval_mesh(num_elements=8, order=order)

    # Set up FiniteElementSpace.
    quadrature = Quadrature1D.create(
        num_points=order + 1, quadrature_type=NodeType.GAUSS_LEGENDRE)
    fespace = FiniteElementSpace.create(mesh, quadrature)

    # Integrating 1 should give the area of the mesh.
    f = lambda x: 1.
    u_local = jax.vmap(jax.vmap(f))(mesh.element_coords())
    nodal_f = fespace.scalar_function(u_local)
    self.assertAlmostEqual(fespace.integrate(nodal_f), 1.)

    # Using higher order meshes, we should be able to integrate
    # polynomials of degree up to `order`.
    f = lambda x: x[0] ** order
    u_local = jax.vmap(jax.vmap(f))(mesh.element_coords())
    nodal_f = fespace.scalar_function(u_local)
    self.assertAlmostEqual(fespace.integrate(nodal_f), 1 / (order + 1))

  @parameterized.parameters(range(1, 3))
  def test_covector_single_element_2d(self, order):
    # Create a mesh of the given order on the unit square.
    ndim = 2
    num_nodes = (order + 1) ** ndim
    node_coords_per_dim = np.linspace(0, 1, num=order + 1)
    node_coords = np.meshgrid(*([node_coords_per_dim] * ndim), indexing='ij')
    node_coords = np.stack(node_coords, axis=-1).reshape(num_nodes, ndim)
    elements = np.arange(num_nodes, dtype=np.int32).reshape((1, num_nodes))
    mesh = Mesh.create(node_coords=node_coords, elements=elements)

    # Set up FiniteElementSpace.
    quadrature = Quadrature1D.create(
        num_points=order + 1, quadrature_type=NodeType.GAUSS_LEGENDRE)
    fespace = FiniteElementSpace.create(mesh, quadrature)

    # Define a form which is simply the product of the two Q-functions
    def form(u, v):
      return lambda x: u(x) * v(x)

    # Integrating f * g should be the same as the dot product of nodal
    # evaluation of g with the covector of f.
    f = lambda x: (x[0] + 2 * x[1]) ** (order // 2)
    g = lambda x: (3 * x[0] - x[1]) ** ((order + 1) // 2)
    f_local = vmap(vmap(f))(mesh.element_coords())
    g_local = vmap(vmap(g))(mesh.element_coords())

    u = fespace.scalar_function(f_local)
    v = fespace.scalar_function(None)
    cov = fespace.local_covector(form, (u, v))

    # Compute expected integral by integrating f * g on the unit square.
    fg = lambda y, x: f([x, y]) * g([x, y])
    expected_integral, _ = scipy.integrate.dblquad(
        fg, 0, 1, lambda x: 0, lambda x: 1)
    obtained_integral = jnp.vdot(mesh.scatter(cov), mesh.scatter(g_local))
    self.assertAlmostEqual(expected_integral, obtained_integral)


if __name__ == '__main__':
  jax.config.update('jax_enable_x64', True)
  absltest.main()
