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

"""A finite element Poisson solver."""

import enum
from typing import Any, Callable, Mapping, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

from swirl_fem.core.fespace import FiniteElementSpace
from swirl_fem.core.fespace import grad
from swirl_fem.core.interpolation import NodeType
from swirl_fem.core.interpolation import Quadrature1D
from swirl_fem.core.mesh import Mesh

# TODO(anudhyan): Switch types to jax.Array and jax.typing.ArrayLike.
Scalar = Any
Array = Any
# The value of a boundary condition may be specified by a scalar, or an array
# denoting a scalar value at every boundary node, or a callable which accepts
# coordinates and outputs a scalar.
BCValue = Union[Scalar, Array, Callable]

# Disable pylint's variable name conventions for consistency with FEM programs.
# pylint: disable=invalid-name


@enum.unique
class BCType(enum.Enum):
  """Types of boundary conditions."""
  DIRICHLET = 'dirichlet'
  NEUMANN = 'neumann'


def solve_poisson(
    mesh: Mesh,
    forcing: Array,
    boundary_conditions: Mapping[str, Tuple[BCType, BCValue]],
    rtol: float = 1e-5,
    atol: float = 0.,
) -> jnp.ndarray:
  """Solves Poisson's equation on `mesh` for the given forcing term.

  We solve the following equation on the domain Ω defined by the mesh; with
  homogeneous Dirichlet boundary condition on the boundary ∂Ω.

      -∇²u = f; subject to u = 0 on ∂Ω   (1)

  Following the presentation in [1], for a discretized finite element space
  `Vₕ`, we can obtain the solution to Poisson's equation by finding `uₕ ∈ Vₕ`
  such that,

      a(uₕ, vₕ) = l(f, vₕ)  ∀vₕ ∈ Vₕ    (2)

  where the bilinear forms `a` and `l` are defined by:

      a(u, v) = ∫∇u ∇v dΩ;  l(u, v) = ∫uv dΩ    (3)

  The linear operators `K` and `M` defined by `K(u) := partial(a, u)` and
  `M(u) = partial(l, u)` are called the stiffness and mass operators,
  respectively. The linear system (2) is symmetric and positive definite, and
  may be solved by using conjugate gradients.

  The boundary conditions are specified as a mapping from the mesh physical
  groups to the boundary conditions on the physical group. Each boundary
  condition is specified as a tuple of `(BCType, BCValue)`. The `BCType`
  denotes the type of boundary condition (e.g., Dirichlet, Neumann) and the
  `BCValue` specifies the value of the boundary condition (i.e., `u` if
  Dirichlet, `du/dn` if Neumann) at each boundary node. The `BCValue` may be a
  scalar (for constant value), or an array giving the value at each boundary
  node, or a callable accepting the coordinates and outputting the value.

  TODO(anudhyan): Add support for non-homogeneous Dirichlet and Neumann
  boundary conditions.

  ### References
  1. Elman, H.C., Silvester, D.J., & Wathen, A.J. (2005). Finite Elements and
     Fast Iterative Solvers: with Applications in Incompressible Fluid Dynamics.

  Args:
    mesh: A `Mesh` object defining the domain.
    forcing: A scalar forcing value for the r.h.s. of the Poisson equation
      defined on the nodes of `mesh`. Should have shape `(mesh.num_nodes,)`.
    boundary_conditions: The mapping from the mesh physical groups to the
      boundary conditions on the physical group. See the expected semantics in
      the function docstring.
    rtol: The relative tolerance used for convergence of the residual in CG;
      convergence is achieved if: `norm(residual) <= max(rtol*norm(b), atol)`.
    atol: The absolute tolerance used for convergence of the residual in CG;
      convergence is achieved if: `norm(residual) <= max(rtol*norm(b), atol)`.

  Returns:
    `uₕ` the solution of the Poisson equation.

  """
  # We use enough quadrature points to represent products of two degree `order`
  # polynomials times a degree `ndim` jacobian determinant term.
  quadrature = Quadrature1D.create(
      num_points=mesh.order + (mesh.ndim + 1) // 2,
      quadrature_type=NodeType.GAUSS_LEGENDRE)
  # Define a finite element space on the mesh with the given quadrature nodes
  # for each element.
  fespace = FiniteElementSpace.create(mesh, quadrature)

  def with_bc(w):
    # Create an array which has 1s on the interior nodes and 0s on the boundary.
    # This is used for enforcing Dirichlet zero boundary conditions by setting
    # the corresponding 'rows' of the stiffness matrix and the RHS to zero.
    interior_mask = np.ones((mesh.num_nodes,))
    for physical_group, (bctype, bcvalue) in boundary_conditions.items():
      if not (np.isscalar(bcvalue) and bcvalue == 0):
        raise NotImplementedError('Only scalar-valued, homogeneous boundary '
                                  f'conditions are supported; got: {bcvalue}')
      if bctype == BCType.DIRICHLET:
        interior_mask *= (1 - mesh.physical_masks[physical_group])
    return w * interior_mask

  # Define the mass and stiffness linear forms.
  def l(u, v):
    return lambda x: u(x) * v(x)

  def a(u, v):
    return lambda x: jnp.vdot(grad(u)(x), grad(v)(x))

  # Given a candidate solution `u`, the stiffness operator returns the covector
  # `partial(a, u)`. With corresponding rows for the boundary nodes zeroed out.
  def A(u):
    u = fespace.scalar_function(mesh.gather(u))
    v = fespace.scalar_function(None)
    Au_local = fespace.local_covector(a, (u, v))
    Au = mesh.scatter(Au_local)
    return with_bc(Au)

  # Mass operator.
  def B(u):
    u = fespace.scalar_function(mesh.gather(u))
    v = fespace.scalar_function(None)
    Bu_local = fespace.local_covector(l, (u, v))
    Bu = mesh.scatter(Bu_local)
    return with_bc(Bu)

  # Obtain the RHS of the linear system. This is the covector `partial(l, f)`.
  b = B(forcing)

  # Solve the linear system `K(u) = M(f)` for `u` where `f` is the forcing term.
  # TODO(anudhyan): Jax CG doesn't currently detect nonconvergence. Add a check
  # manually to verify that the residual converged.
  u, _ = jax.scipy.sparse.linalg.cg(A, b, tol=rtol, atol=atol)

  return u
