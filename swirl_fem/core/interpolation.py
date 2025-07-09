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

"""A module for interpolation and quadrature on tensor-product elements."""

import enum
import functools

import jax
import flax
from jax.typing import ArrayLike
import jax.numpy as jnp
import numpy as np
import scipy.special


@enum.unique
class NodeType(enum.Enum):
  """Type of distributions of collocation or quadrature nodes on [-1, 1]."""
  NEWTON_COTES = 'newton_cotes'
  GAUSS_LEGENDRE = 'gauss_legendre'
  GAUSS_LOBATTO_LEGENDRE = 'gauss_lobatto_legendre'
  SINGLE = 'single_point'


@flax.struct.dataclass
class Nodes1D:
  """Represents a sequence of 1D nodes on the reference element [-1, 1]."""

  num_points: int
  node_type: NodeType
  node_values: np.ndarray

  @classmethod
  def create_single_point(cls, node_value: np.ndarray) -> 'Nodes1D':
    """Creates a `Nodes1D` containing a single point."""
    return cls(num_points=1, node_type=NodeType.SINGLE,
               node_values=np.array([node_value], dtype=np.float64))

  @classmethod
  def create(
      cls, num_points: int, node_type: NodeType) -> 'Nodes1D':
    """Creates a `Nodes1D` object."""
    if node_type == NodeType.NEWTON_COTES:
      node_values = np.linspace(-1, 1, num=num_points, dtype=np.float64)
    elif node_type == NodeType.GAUSS_LEGENDRE:
      node_values, _ = np.polynomial.legendre.leggauss(deg=num_points)
    elif node_type == NodeType.GAUSS_LOBATTO_LEGENDRE:
      # The inner nodes for Gauss-Lobatto-Legendre quadrature coincide with the
      # roots of the derivative of the Legendre polynomial; which also form
      # the nodes of the Gauss-Jacobi quadrature so we can use scipy's
      # roots_jacobi. See https://mathworld.wolfram.com/LobattoQuadrature.html.
      if num_points == 2:
        inner_nodes = np.array([], dtype=np.float64)
      else:
        inner_nodes, _ = scipy.special.roots_jacobi(
            num_points - 2, alpha=1, beta=1)
      node_values = np.concatenate([[-1.], inner_nodes, [1.]])
    else:
      raise ValueError(f'Node type not recognized: {node_type}')

    return cls(
        num_points=num_points,
        node_type=node_type,
        node_values=node_values)

  def is_continuous(self):
    """Whether continuity is preserved at element boundaries."""
    # uses exact equality as the node values are set statically
    return self.node_values[0] == -1.0 and self.node_values[-1] == 1.0

  def __eq__(self, other):
    """Operator for equality of Node1d objects."""
    if self.node_type != other.node_type:
      return False
    if self.node_type == NodeType.SINGLE:
      return np.allclose(self.node_values, other.node_values, rtol=0,
                         atol=np.finfo(self.node_values.dtype).eps)
    return (self.node_type == other.node_type and
            self.num_points == other.num_points)


@flax.struct.dataclass
class Quadrature1D:
  """Represents a 1D quadrature rule on the reference element [-1, 1]."""

  num_points: int
  quadrature_type: NodeType
  nodes: Nodes1D
  weights: np.ndarray

  @classmethod
  def create_from_nodes_1d(cls, nodes: Nodes1D):
    """Initializes a Quadrature1D object from a Nodes1D object."""
    if nodes.node_type == NodeType.GAUSS_LEGENDRE:
      _, weights = np.polynomial.legendre.leggauss(deg=nodes.num_points)
    elif nodes.node_type == NodeType.GAUSS_LOBATTO_LEGENDRE:
      # See https://mathworld.wolfram.com/LobattoQuadrature.html.
      weights = (2 / (nodes.num_points * (nodes.num_points - 1))) / np.square(
          scipy.special.eval_legendre(nodes.num_points - 1, nodes.node_values))
    elif nodes.node_type == NodeType.NEWTON_COTES:
      weights = (1 / (nodes.num_points - 1)) * np.array(
          [1.] + (nodes.num_points - 2) * [2.] + [1.])
    else:
      raise ValueError(f'Quadrature type not recognized: {nodes.node_type}')
    return cls(num_points=nodes.num_points, quadrature_type=nodes.node_type,
               nodes=nodes, weights=weights)

  @classmethod
  def create(cls, num_points: int, quadrature_type: NodeType):
    """Initializes a Quadrature1D object.

    Args:
      num_points: The number of quadrature points in 1D. Depending on the type,
        the quadrature rule is exact for polynomials below a certain degree
        bound. (e.g. 2 * num_points - 1 for Gauss-Legendre quadrature).
      quadrature_type: The type of quadrature points. Typically, one can use
        Gauss-Legendre points for Q1/Q2 elements and GLL points for higher order
        spectral elements.

    Returns:
      A `Nodes1D` object.
    """
    return cls.create_from_nodes_1d(
        Nodes1D.create(num_points=num_points, node_type=quadrature_type))

  def weights_nd(self, ndim):
    """Computes `ndim`-dimensional (tensor-product) quadrature weights."""
    return functools.reduce(np.outer, [self.weights] * ndim).reshape(-1)


class BarycentricInterpolator:
  """Interpolates functions using barycentric interpolation.

  This class provides methods `interpolation_matrix` and
  `interpolation_matrix_grad`, which return statically computed matrices used
  for barycentric interpolation on the reference element `[-1, 1] ^ ndim`. These
  matrices operate on a vector with the integrand's values on the grid points
  and return the interpolated values on the given evaluation points.

  Attributes:
    ndim:  The dimension of the interpolation domain.
    gridpoints_1d: The nodal grid points and evaluation points represented in
      1D. The actual `ndim`-dimensional nodes are obtained as the `ndim`-fold
      tensor product of the 1D points.
    evalpoints_1d: The nodal grid points and evaluation points represented in
      1D. Similar to the gridpoints, the actual evaluation points are an
      `ndim`-fold tensor product.

  #### References
    1. Berrut, J., & Trefethen, L.N. (2004). Barycentric Lagrange Interpolation.
       SIAM Rev., 46, 501-517.
    2. Higham, N.J. (2004). The numerical stability of barycentric Lagrange
       interpolation. Ima Journal of Numerical Analysis, 24, 547-556.
    3. Wang, H., Huybrechs, D., & Vandewalle, S. (2014). Explicit barycentric
       weights for polynomial interpolation in the roots or extrema of classical
       orthogonal polynomials. Mathematics of Computation, 83(290), 2893-2914.
       https://arxiv.org/abs/1202.0154
  """
  ndim: int
  gridpoints_1d: Nodes1D
  evalpoints_1d: Nodes1D

  def __init__(self, ndim: int, gridpoints_1d: Nodes1D, evalpoints_1d: Nodes1D):
    self.ndim = ndim
    self.gridpoints_1d = gridpoints_1d
    self.evalpoints_1d = evalpoints_1d

  def _barycentric_weights(self):
    """Returns the barycentric weights."""
    if self.gridpoints_1d.node_type == NodeType.NEWTON_COTES:
      # For uniformly spaced grid points on [-1, 1], we use the closed-form
      # equation (5.1) from Berrut and Trefethen [1].
      order = self.gridpoints_1d.num_points - 1
      return np.array([
          np.power(-1, i) * scipy.special.binom(order, i)
          for i in range(self.gridpoints_1d.num_points)
      ])
    elif self.gridpoints_1d.node_type == NodeType.GAUSS_LEGENDRE:
      # We use equation 1.4 from [3].
      quad = Quadrature1D.create(
          num_points=self.gridpoints_1d.num_points,
          quadrature_type=self.gridpoints_1d.node_type)
      return np.array([
          np.power(-1, i) * np.sqrt((1 - np.square(x)) * w)
          for i, (x, w) in enumerate(zip(quad.nodes.node_values, quad.weights))
      ])
    elif self.gridpoints_1d.node_type == NodeType.GAUSS_LOBATTO_LEGENDRE:
      # We use equation 1.6 from [3].
      quad = Quadrature1D.create(
          num_points=self.gridpoints_1d.num_points,
          quadrature_type=self.gridpoints_1d.node_type)
      return np.array([
          np.power(-1, i) * np.sqrt(w) for i, w in enumerate(quad.weights)
      ])
    raise ValueError('Gridpoint type not supported: '
                     f'{self.gridpoints_1d.node_type}')

  def _interpolation_matrix_1d(self):
    """Computes the matrix {lagrange_j(quadrature_i)} for 1D quadrature."""
    weights = self._barycentric_weights()
    nodes = self.gridpoints_1d.node_values

    # Interpolate at a point `x` in [-1, 1] using the Barycentric formula.
    # Equation (4.2) in Berrut and Trefethen [1].
    def evaluate(x, i):
      # The exact floating point comparison is intentional. At x ~ nodes[i],
      # both the numerator and the denomator of the barycentric formula are
      # innacurate, but surprisingly "cancel out" in the correct way in IEEE
      # arithmetic. See section 7 of Berrut and Trefethen [1].
      if x == nodes[i]:
        return 1.
      terms = np.array([w / (x - xj) for w, xj in zip(weights, nodes)])
      return terms[i] / sum(terms)

    row = lambda x: np.array([evaluate(x, i) for i in range(len(nodes))])
    return np.vstack([row(x) for x in self.evalpoints_1d.node_values])

  def _differentiation_matrix_1d(self):
    """Computes the matrix {lagrange_j'(gridpoint_i)} for 1D quadrature."""
    weights = self._barycentric_weights()
    nodes = self.gridpoints_1d.node_values
    diff_matrix = np.zeros((len(nodes), len(nodes)))

    # Compute the values of the derivatives of the interpolating lagrange
    # polynomials at the gridpoints. We use formulas (9.4) and (9.5)
    # from Berrut and Trefethen [1].
    for i, j in np.ndindex(*diff_matrix.shape):
      if i != j:
        diff_matrix[i, j] = (weights[j] / weights[i]) / (nodes[i] - nodes[j])
    for i in range(len(nodes)):
      diff_matrix[i, i] = -diff_matrix[i, ...].sum()
    return diff_matrix

  def interpolation_matrix(self):
    """Returns the interpolation matrix."""
    interp_matrix_1d = self._interpolation_matrix_1d()
    # The interpolation matrix on the `ndim`-dimensional tensor-product
    # collocation nodes mapping to tensor-product quadrature nodes, is just the
    # kronecker product of the 1D interpolation matrices.
    return functools.reduce(np.kron, [interp_matrix_1d] * self.ndim)

  def interpolate(self, x: jax.Array) -> jax.Array:
    """Interpolate values specified on gridpoints to the evaluation points."""
    assert x.shape == (self.gridpoints_1d.num_points ** self.ndim,), x.shape  # pytype: disable=attribute-error  # numpy-scalars
    if self.gridpoints_1d == self.evalpoints_1d:
      return x

    # TODO(anudhyan): Take advantage of the tensor-product structure to obtain
    # O(N^(d + 1)) instead of O(N^(2d)) in matmul flops.
    return jnp.einsum('ij,j->i', self.interpolation_matrix(), x,
                      precision=jax.lax.Precision.HIGHEST)

  def interpolation_matrix_grad(self):
    """Returns the interpolation matrix for gradient evaluations."""
    interp_matrix_1d = self._interpolation_matrix_1d()

    # "Interpolate" the functions {lagrange_j} using the 1D interpolation
    # matrix. This yields the interpolation matrix for the derivative of the
    # given function.
    diff_matrix_1d = self._differentiation_matrix_1d()
    interp_matrix_1d_grad = interp_matrix_1d @ diff_matrix_1d

    # Compute the gradient matrix. The ith partial derivative is given by
    # M ⊗ M ⊗...⊗ D ⊗ ... ⊗ M, where the 'D' is in the ith position. E.g.
    # in 3D, the second partial derivative operator D_y is given by M ⊗ D ⊗ M.
    interp_matrix_grad = np.empty((self.ndim, self.ndim), dtype=object)
    for i, j in np.ndindex(*interp_matrix_grad.shape):
      if i == j:
        interp_matrix_grad[i][j] = interp_matrix_1d_grad
      else:
        interp_matrix_grad[i][j] = interp_matrix_1d

    return np.stack(
        [functools.reduce(np.kron, row) for row in interp_matrix_grad], axis=-1)

  def interpolate_grad(self, x):
    """Interpolate gradient values from gridpoints to the evaluation points."""
    assert x.shape == (self.gridpoints_1d.num_points ** self.ndim,), x.shape
    return jnp.einsum('qnd,n->qd', self.interpolation_matrix_grad(), x,
                      precision=jax.lax.Precision.HIGHEST)
