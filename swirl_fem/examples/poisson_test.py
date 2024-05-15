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

"""Tests for poisson solver."""

from functools import partial  # pylint: disable=g-importing-member
import itertools

from absl.testing import absltest
import jax
from jax import vmap
import jax.numpy as jnp
import more_itertools
import numpy as np

from swirl_fem.core.premesh import Premesh
from swirl_fem.examples.poisson import BCType
from swirl_fem.examples.poisson import solve_poisson


def _square_mesh(num_elements_per_dim) -> Premesh:
  """Returns a uniform mesh over [-1, 1]^2."""
  ndim = 2
  # Create node coordinates as the uniform cartesian grid over the unit square.
  num_nodes_per_dim = num_elements_per_dim + 1
  num_nodes = num_nodes_per_dim**ndim
  node_coords_per_dim = np.linspace(-1, 1, num=num_nodes_per_dim)
  node_coords = np.meshgrid(*([node_coords_per_dim] * ndim), indexing='ij')
  node_coords = np.stack(node_coords, axis=-1).reshape(num_nodes, ndim)

  # Create elements by iterating through all pairs of '1D' elements, which are
  # just pairwise node indices `{(i, i + 1) | 0 <= i < num_nodes_per_dim}`.
  elements = []
  elements_per_dim = more_itertools.pairwise(np.arange(num_nodes_per_dim))
  for element_tuple in itertools.product(elements_per_dim, repeat=ndim):
    # From each element N-tuple, create the N-D element containing 2^N nodes.
    # E.g. in 2D, the element tuple ((0, 1), (5, 6)) becomes a single element
    # containing 4 nodes: [(0, 5), (0, 6), (1, 5), (1, 6)].
    element_nd = itertools.product(*element_tuple)
    # Convert node indices from N-tuples to the canonical global index.
    element = map(
        partial(np.ravel_multi_index, dims=[num_nodes_per_dim] * ndim),
        element_nd)
    elements.append(list(element))

  # Boundary nodes contain either the first or the last node along either axis.
  boundary_nodes_1d = {0, num_nodes_per_dim - 1}
  boundary_nodes = []
  for node in itertools.product(np.arange(num_nodes_per_dim), repeat=ndim):
    if any(i in boundary_nodes_1d for i in node):
      boundary_nodes.append(
          np.ravel_multi_index(node, dims=[num_nodes_per_dim] * ndim))
  return Premesh.create(
      node_coords=node_coords,
      elements=np.array(elements),
      physical_groups={'boundary': np.array(boundary_nodes, dtype=np.int32)})


def _unit_circle_mesh(num_elements_per_dim):
  """Returns a mesh on the unit circle."""
  square_mesh = _square_mesh(num_elements_per_dim)

  def coons_patch(x):
    # Generate a coons patch on the unit circle.
    # https://en.wikipedia.org/wiki/Coons_patch
    rsqrt2 = 1 / np.sqrt(2)
    return jnp.array([
        x[0] * (jnp.cos(np.pi * x[1] / 4) - rsqrt2) + jnp.sin(np.pi * x[0] / 4),
        x[1] * (jnp.cos(np.pi * x[0] / 4) - rsqrt2) + jnp.sin(np.pi * x[1] / 4),
    ])

  # Map the node coordinates of the square mesh on [-1, 1] x [-1, 1] to the
  # unit circle. The boundary nodes remain the same.
  # Note: A high `num_elements_per_dim` may create degenerate elements.
  node_coords = vmap(coons_patch)(square_mesh.node_coords)
  return Premesh.create(
      node_coords=node_coords,
      elements=square_mesh.elements,
      physical_groups=square_mesh.physical_groups)


class PoissonTest(absltest.TestCase):

  def test_1d_unit_forcing(self):
    num_elements = 32
    num_nodes = num_elements + 1
    node_coords = np.linspace(0, 1, num_nodes).reshape((num_nodes, 1))
    elements = np.array([[i, i + 1] for i in range(num_elements)])
    physical_groups = {
        'boundary': np.array([[0, num_nodes - 1]], dtype=np.int32)}
    mesh = Premesh.create(
        node_coords=node_coords,
        elements=elements,
        physical_groups=physical_groups).finalize()

    forcing = jnp.ones(num_nodes, dtype=jnp.float32)
    boundary_conditions = {'boundary': (BCType.DIRICHLET, 0.)}
    u = solve_poisson(mesh, forcing, boundary_conditions=boundary_conditions)
    expected = vmap(lambda x: .5 * (x[0] - jnp.square(x[0])))(mesh.node_coords)
    np.testing.assert_allclose(u, expected)

  def test_1d_homogeneous_neumann(self):
    num_elements = 128
    num_nodes = num_elements + 1
    node_coords = np.linspace(0, 1, num_nodes).reshape((num_nodes, 1))
    elements = np.array([[i, i + 1] for i in range(num_elements)])
    mesh = Premesh.create(node_coords=node_coords, elements=elements).finalize()

    # Having an 'empty' set of boundary conditions is equivalent to homogeneous
    # Neumann boundary conditions.
    forcing_fn = lambda x: -.5 * jnp.square(np.pi) * jnp.cos(np.pi * x[0])
    forcing = vmap(forcing_fn)(mesh.node_coords)
    u = solve_poisson(mesh, forcing, boundary_conditions={}, rtol=1e-7)

    # Out of possible solutions differing by an additive constant, CG finds the
    # one with mean zero.
    expected_fn = lambda x: jnp.square(jnp.sin(.5 * np.pi * x[0])) - .5
    expected = vmap(expected_fn)(mesh.node_coords)
    np.testing.assert_allclose(u, expected, rtol=1e-4, atol=1e-5)

  def test_1d_linear_forcing(self):
    num_elements = 32
    num_nodes = num_elements + 1
    node_coords = np.linspace(0, 1, num_nodes).reshape((num_nodes, 1))
    elements = np.array([[i, i + 1] for i in range(num_elements)])
    physical_groups = {
        'boundary': np.array([[0, num_nodes - 1]], dtype=np.int32)}
    mesh = Premesh.create(
        node_coords=node_coords,
        elements=elements,
        physical_groups=physical_groups).finalize()

    forcing = vmap(lambda x: 6 * x[0])(mesh.node_coords)
    boundary_conditions = {'boundary': (BCType.DIRICHLET, 0.)}
    u = solve_poisson(mesh, forcing, boundary_conditions=boundary_conditions)
    expected = vmap(lambda x: x[0] - jnp.power(x[0], 3))(mesh.node_coords)
    np.testing.assert_allclose(u, expected)

  def test_square(self):
    # Test case from Example 1.1.1 of:
    #   Silvester, D., Elman, H., & Wathen, A. (2005). Finite Elements and Fast
    #   Iterative Solvers: With Applications in Incompressible Fluid Dynamics.
    #   Oxford University Press.
    mesh = _square_mesh(num_elements_per_dim=32).finalize()
    forcing = jnp.ones(mesh.num_nodes)
    boundary_conditions = {'boundary': (BCType.DIRICHLET, 0.)}
    u = solve_poisson(mesh, forcing, boundary_conditions=boundary_conditions)

    # Compute the analytical solution.
    def expected_u(x, num_terms=5):
      series = jnp.zeros_like(x[0])
      # sum over odd numbers.
      for k in range(1, 2 * num_terms, 2):
        const_term = 1 / (jnp.power(k, 3) * jnp.sinh(k * np.pi))
        sin_term = jnp.sin(k * np.pi * (1 + x[0]) / 2)
        sinh_term = (
            jnp.sinh(k * np.pi * (1 - x[1]) / 2) + jnp.sinh(k * np.pi *
                                                            (1 + x[1]) / 2))
        series += const_term * sin_term * sinh_term
      return (1 - jnp.square(x[0])) / 2 - (16 / jnp.power(np.pi, 3)) * series

    expected = vmap(expected_u)(mesh.node_coords)
    np.testing.assert_allclose(u, expected, rtol=1e-6, atol=1e-3)

  def test_unit_circle(self):
    mesh = _unit_circle_mesh(num_elements_per_dim=32).finalize()
    forcing = jnp.ones(mesh.num_nodes)
    boundary_conditions = {'boundary': (BCType.DIRICHLET, 0.)}
    u = solve_poisson(mesh, forcing, boundary_conditions=boundary_conditions)
    expected = vmap(lambda x: .25 * (1 - jnp.square(jnp.linalg.norm(x))))(
        mesh.node_coords)
    np.testing.assert_allclose(u, expected, rtol=1e-6, atol=1e-4)


if __name__ == '__main__':
  jax.config.update('jax_enable_x64', True)
  absltest.main()
