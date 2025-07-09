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

"""A spectral element fractional step Navier-Stokes solver.

### References
  1. Deville, M.O., Fischer, P.F. and Mund, E.H., 2002. High-order methods for
     incompressible fluid flow (Vol. 9). Cambridge university press.
"""

from collections.abc import Sequence
import enum
from functools import partial
from typing import Any

import flax.struct
import jax
from jax import lax
from jax import vmap
import jax.numpy as jnp
import numpy as np
from swirl_fem.core.fespace import div
from swirl_fem.core.fespace import FiniteElementSpace
from swirl_fem.core.fespace import grad
from swirl_fem.core.interpolation import BarycentricInterpolator
from swirl_fem.core.interpolation import Nodes1D
from swirl_fem.core.interpolation import NodeType
from swirl_fem.core.interpolation import Quadrature1D
from swirl_fem.core.mesh import Mesh
from swirl_fem.core.mesh_refiner import refine_premesh
from swirl_fem.core.premesh import Premesh
from swirl_fem.linalg.cg import cg


# pylint: disable=invalid-name


def extk_coeffs(k: int) -> np.ndarray:
  """Linear extrapolation coefficients of order k."""
  gridpoints = Nodes1D.create(num_points=k + 1, node_type=NodeType.NEWTON_COTES)
  h = 2 / k
  evalpoints = Nodes1D.create_single_point(
      node_value=np.array(1 + h, dtype=np.float64))
  interpolator = BarycentricInterpolator(
      ndim=1, gridpoints_1d=gridpoints, evalpoints_1d=evalpoints)
  coeffs = interpolator.interpolation_matrix().reshape((-1))
  return coeffs


def bdfk_coeffs(k: int) -> np.ndarray:
  """Backward differentiation formula of order k."""
  gridpoints = Nodes1D.create(num_points=k + 1, node_type=NodeType.NEWTON_COTES)
  evalpoints = Nodes1D.create_single_point(
      node_value=np.array(1., dtype=np.float64))
  interpolator = BarycentricInterpolator(
      ndim=1, gridpoints_1d=gridpoints, evalpoints_1d=evalpoints)
  h = 2 / k
  coeffs = interpolator.interpolation_matrix_grad().reshape((-1)) * h
  return coeffs


def _pressure_project_out_nullspace(sem, p):
  """Remove the nullspace (all 1s vector) from p."""
  w = sem.pressure.exchange(p)
  q = jnp.ones_like(p)
  return w - (
      (jnp.vdot(q, sem.pressure.B(w)) / jnp.vdot(q, sem.pressure.B(q)))) * q


@enum.unique
class BCType(enum.Enum):
  """Types of boundary conditions."""
  DIRICHLET = 'dirichlet'
  NEUMANN = 'neumann'


def dirichlet_bc(mesh, boundary_conditions):
  """Create an interior mask based on the boundary conditions."""
  interior_mask = np.ones((mesh.num_nodes,))
  for physical_group, (bctype, unused_bcvalue) in boundary_conditions.items():
    if bctype == BCType.DIRICHLET:
      interior_mask *= 1 - mesh.physical_masks[physical_group]
  return interior_mask


@flax.struct.dataclass
class StokesPressure:
  """Pressure space for the Stokes problem."""
  pspace: FiniteElementSpace

  @classmethod
  def create(
      cls, premesh: Premesh, quadrature: Quadrature1D, order: int
  ) -> 'StokesPressure':
    """Create a `StokesPressure` instance on Gauss-Legendre nodes.

    Args:
      premesh: A `Premesh` for the domain.
      quadrature: A `Quadrature1D` for integration.
      order: The order of the pressure approximation.

    Returns:
      A `StokesPressure` instance.
    """
    # Refine the pressure mesh.
    gridpoints_1d = Nodes1D.create(
        num_points=order - 1, node_type=NodeType.GAUSS_LEGENDRE)
    pmesh = refine_premesh(premesh, gridpoints_1d=gridpoints_1d).finalize()
    pspace = FiniteElementSpace.create(mesh=pmesh, quadrature=quadrature)
    return cls(pspace=pspace)

  def gather(self, p: jax.Array) -> jax.Array:
    return self.pspace.mesh.gather(p)

  def scatter(self, p: jax.Array) -> jax.Array:
    return self.pspace.mesh.scatter(p)

  def B(self, p: jax.Array) -> jax.Array:
    """Apply the pressure mass matrix."""
    def l(u, v):
      return lambda x: u(x) * v(x)

    u = self.pspace.scalar_function(self.gather(p))
    v = self.pspace.scalar_function(None)
    return self.scatter(self.pspace.local_covector(l, (u, v)))

  def exchange(self, p: jax.Array) -> jax.Array:
    """Apply QQᵀ."""
    return self.pspace.mesh.exchange(p)


@flax.struct.dataclass
class StokesVelocity:
  """Velocity space for the Stokes system."""
  vspace: FiniteElementSpace
  overint_space: FiniteElementSpace
  interior_mask: jax.Array
  diag_qqt: jax.Array
  num_convection_overint_nodes: int = flax.struct.field(
      pytree_node=False, default=2)

  @classmethod
  def create(
      cls,
      premesh: Premesh,
      order: int,
      boundary_conditions,
      num_convection_overint_nodes: int = 2,
  ) -> 'StokesVelocity':
    """Create a `StokesVelocity` instance on Gauss-Lobatto-Legendre nodes.

    Args:
      premesh: A `Premesh` describing the domain.
      order: The order of the velocity approximation.
      boundary_conditions: A dictionary of boundary conditions. The keys are
        physical group names and the values are tuples of (`BCType`, value).
      num_convection_overint_nodes: The number of overintegration nodes to use
        for the convection term.

    Returns:
      A `StokesVelocity` instance.
    """
    # Refine the velocity mesh.
    gridpoints_1d = Nodes1D.create(
        num_points=order + 1, node_type=NodeType.GAUSS_LOBATTO_LEGENDRE)
    vmesh = refine_premesh(premesh, gridpoints_1d=gridpoints_1d).finalize()
    vspace = FiniteElementSpace.create(
        mesh=vmesh,
        quadrature=Quadrature1D.create_from_nodes_1d(gridpoints_1d))
    interior_mask = dirichlet_bc(vmesh, boundary_conditions)[:, np.newaxis]

    overint_gridpoints_1d = Nodes1D.create(
        num_points=gridpoints_1d.num_points + num_convection_overint_nodes,
        node_type=NodeType.GAUSS_LOBATTO_LEGENDRE)
    overint_space = FiniteElementSpace.create(
        mesh=vmesh,
        quadrature=Quadrature1D.create_from_nodes_1d(overint_gridpoints_1d))
    diag_qqt = vmesh.scatter(
        u_local=jnp.ones(vmesh.elements.shape, dtype=vmesh.node_coords.dtype))

    return cls(vspace=vspace, overint_space=overint_space,
               diag_qqt=diag_qqt,
               interior_mask=interior_mask)

  @property
  def local_shape(self) -> tuple[int, ...]:
    return (self.vspace.mesh.num_elements,
            self.vspace.mesh.num_nodes_per_element, self.vspace.mesh.ndim)

  def C(self, u: jax.Array) -> jax.Array:
    """Apply the convection operator with overintegration."""
    Cu_local = self.C_local(self.gather(u))
    return self.interior_mask * self.scatter(Cu_local)

  @property
  def mesh(self) -> Mesh:
    return self.vspace.mesh

  def gather(self, u: jax.Array) -> jax.Array:
    return vmap(self.vspace.mesh.gather, in_axes=-1, out_axes=-1)(u)

  def scatter(self, u: jax.Array) -> jax.Array:
    return vmap(self.vspace.mesh.scatter, in_axes=-1, out_axes=-1)(u)

  def exchange(self, u: jax.Array) -> jax.Array:
    """Apply QQᵀ."""
    return vmap(self.vspace.mesh.exchange, in_axes=-1, out_axes=-1)(u)

  def A_local(self, u_local: jax.Array) -> jax.Array:
    """Apply the velocity stiffness operator locally."""
    def a(u, v):
      return lambda x: jnp.einsum('ij,ij->', grad(u)(x), grad(v)(x))

    u = self.vspace.vector_function(u_local)
    v = self.vspace.vector_function(None)
    return self.vspace.local_covector(a, (u, v))

  def B_local(self, u_local: jax.Array) -> jax.Array:
    """Apply the velocity mass operator locally."""
    def l(u, v):
      return lambda x: jnp.vdot(u(x), v(x))

    u = self.vspace.vector_function(u_local)
    v = self.vspace.vector_function(None)
    return self.vspace.local_covector(l, (u, v))

  def C_local(self, u_local: jax.Array) -> jax.Array:
    """Apply the local convection operator."""
    def c(u, w, v):
      return lambda x: jnp.einsum('i,ij,j->', u(x), grad(w)(x), v(x))

    u = self.overint_space.vector_function(u_local)
    v = self.overint_space.vector_function(None)
    return self.overint_space.local_covector(c, (u, u, v))


@flax.struct.dataclass
class StokesSEM:
  """Linear operators for a Stokes solver using spectral elements."""

  velocity: StokesVelocity
  pressure: StokesPressure
  # diagonal mass matrix for the velocity space
  velocity_mass_diag: jax.Array

  @classmethod
  def create(cls,
             premesh: Premesh,
             boundary_conditions,
             order: int,
             num_convection_overint_nodes: int = 2) -> 'StokesSEM':
    """Create a `StokesSEM` instance for the given `order`.

    Args:
      premesh: A `Premesh` instance. The mesh must have order 1.
      boundary_conditions: A dictionary of boundary conditions. The keys are
        physical group names and the values are tuples of (`BCType`, value).
      order: The order of the velocity and pressure approximations. Note that
        the velocity space will be refined to have order `order + 1`.
      num_convection_overint_nodes: The number of overintegration nodes to use
        for the convection term.

    Returns:
      A `StokesSEM` instance.
    """
    if premesh.order != 1:
      raise ValueError(f'Expected mesh order 1; got {premesh.order}.')
    quadrature = Quadrature1D.create(
        num_points=order + 1, quadrature_type=NodeType.GAUSS_LOBATTO_LEGENDRE)

    pressure = StokesPressure.create(premesh, quadrature, order)
    velocity = StokesVelocity.create(
        premesh, order, boundary_conditions, num_convection_overint_nodes)

    velocity_mass_diag = velocity.scatter(
        velocity.B_local(jnp.ones(velocity.local_shape)))

    return cls(
        velocity=velocity,
        pressure=pressure,
        velocity_mass_diag=velocity_mass_diag,
    )

  def B(self, u: jax.Array) -> jax.Array:
    """Apply the mass operator to a velocity field."""
    return self.velocity.interior_mask * self.velocity_mass_diag * u

  def Bi(self, u: jax.Array) -> jax.Array:
    """Apply the inverse mass operator to a velocity field."""
    diag_qqti = 1 / self.velocity.exchange(self.velocity_mass_diag)
    return diag_qqti * self.velocity.exchange(u)

  def A(self, u: jax.Array) -> jax.Array:
    """Apply the stiffness operator to a velocity field."""
    return self.velocity.interior_mask * self.velocity.scatter(
        self.velocity.A_local(self.velocity.gather(u)))

  def C(self, u: jax.Array) -> jax.Array:
    """Apply the convection operator to a velocity field."""
    return self.velocity.C(u)

  def D_local(self, u_local: jax.Array) -> jax.Array:
    """Apply the local operator D."""
    def b(v, q):
      return lambda x: div(v)(x) * q(x)

    v = self.velocity.vspace.vector_function(u_local)
    p = self.pressure.pspace.scalar_function(None)
    return self.pressure.pspace.local_covector(b, (v, p))

  def Dt_local(self, p_local: jax.Array) -> jax.Array:
    """Apply the local operator Dᵀ."""
    def b(v, q):
      return lambda x: div(v)(x) * q(x)

    v = self.velocity.vspace.vector_function(None)
    p = self.pressure.pspace.scalar_function(p_local)
    return self.velocity.vspace.local_covector(b, (v, p))

  def D(self, u: jax.Array) -> jax.Array:
    """Velocity divergence matrix."""
    return self.pressure.scatter(self.D_local(self.velocity.gather(u)))

  def Dt(self, p: jax.Array) -> jax.Array:
    """Apply the pressure gradient oprator."""
    return self.velocity.interior_mask * self.velocity.scatter(
        self.Dt_local(self.pressure.gather(p)))

  def Q(self, u: jax.Array, dt: float, time_order: int):
    """Apply the operator Q = (Δt / βₖ) B⁻¹."""
    beta_k = bdfk_coeffs(time_order)[-1]
    return (dt / beta_k) * self.Bi(u)

  def E(self, p: jax.Array, dt: float, time_order: int):
    """Apply the operator E = DQDᵀ."""
    Q_ = partial(self.Q, dt=dt, time_order=time_order)
    return self.D(Q_(self.Dt(p)))

  def stokes_one_step(
      self, us: Sequence[jax.Array], ps: Sequence[jax.Array], f: jax.Array,
      mu: float, dt: float, time_order: int, alpha: float = 0.05,
      u_boundary: jax.Array | None = None,
      pressure_preconditioner=None,
      project_out_nullspace=True,
      tol: float = 1e-8, atol: float = 0,
  ) -> tuple[jax.Array, jax.Array, Any]:
    """Evolve the Stokes system forward by one time step.

    This function uses a fractional step method to solve the Stokes system:

    ∂u/∂t + (u ⋅ ∇) u = - ∇p + ν ∇²u + f
    ∇ ⋅ u = 0

    where u is the velocity, p is the pressure, ν is the kinematic viscosity,
    and f is the body force.

    The fractional step method used here is based on the following steps:

    1. Compute the tentative velocity u* by solving the advection-diffusion
       equation with the pressure term omitted. Here, we use a BDF
       timestepping scheme of order `time_order` to approximate the time
       derivative:

       (βₖ / Δt) B(u* - Σⱼ₌₁ᵏ⁻¹ βⱼ uⁿ⁻ʲ) =
           -B((1 / Δt) Σⱼ₌₁ᵏ⁻¹ βⱼ uⁿ⁻ʲ) +  f + D(p_ext)
       where:
         * B is the mass matrix.
         * D is the divergence operator.
         * p_ext is a linear extrapolation of the pressure.

    2. Solve for the pressure correction dp:
        DQDᵀ(dp) = -Du*
       where Q = (Δt / βₖ)B⁻¹.

    3. Correct the velocity:
        uⁿ⁺¹ = u* + QDᵀ(dp).

    4. Correct the pressure:
        pⁿ⁺¹ = p_ext + dp.

    This function returns the velocity and pressure at the next time step,
    as well as a dictionary of auxiliary information. For more details, see [1].

    Args:
      us: A sequence of `time_order` previous velocity fields, ordered from
        oldest to newest.
      ps: A sequence of `time_order` previous pressure fields, ordered from
        oldest to newest.
      f: The body force.
      mu: The kinematic viscosity.
      dt: The time step.
      time_order: The order of the BDF timestepping scheme.
      alpha: The filter coefficient for filter-based stabilization.
      u_boundary: The boundary conditions for the velocity.
      pressure_preconditioner: A preconditioner to use when solving the
        pressure Poisson equation.
      project_out_nullspace: Whether to project out the nullspace of the
        pressure operator.
      tol: The tolerance for the conjugate gradient solver.
      atol: The absolute tolerance for the conjugate gradient solver.

    Returns:
      A tuple containing:
        * The velocity field at the next time step.
        * The pressure field at the next time step.
        * A dictionary of auxiliary information.
    """
    if pressure_preconditioner is None and project_out_nullspace:
      pressure_preconditioner = partial(_pressure_project_out_nullspace, self)

    # Extrapolate pressure linearly.
    ext_coeffs = extk_coeffs(k=1)
    p_ext = sum(
        ext_coeffs[-i] * ps[-i] for i in range(1, len(ext_coeffs) + 1))
    f = f + self.Dt(p_ext)

    # Solve for H(u*) = b.
    beta_hist = bdfk_coeffs(time_order)[:-1]
    beta_k = bdfk_coeffs(time_order)[-1]
    H_ = lambda u: (beta_k / dt) * self.B(u) + mu * self.A(u)
    f = f - self.B((1 / dt) * sum(coef * u for coef, u in zip(beta_hist, us)))
    if u_boundary is not None:
      f = f - H_(u_boundary)

    u_star, info = lax.custom_linear_solve(
        H_, f, solve=partial(cg, M=self.velocity.exchange, tol=tol, atol=atol),
        symmetric=True, has_aux=True)
    if u_boundary is not None:
      u_star = u_star + u_boundary

    aux = dict()
    aux['u_star_info'] = info

    # Filter-based stabilization (α = 0.05)
    u_star = self.filter(u_star, alpha=alpha)

    # Obtain dp by solving DQDᵀ(dp) = -Du*.
    dp, info = lax.custom_linear_solve(
        partial(self.E, dt=dt, time_order=time_order), -self.D(u_star),
        solve=partial(cg, M=pressure_preconditioner, tol=tol, atol=atol),
        symmetric=True, has_aux=True)
    aux['dp_info'] = info

    Q_ = partial(self.Q, dt=dt, time_order=time_order)
    u = u_star + Q_(self.Dt(dp), dt=dt, time_order=time_order)
    p = p_ext + dp
    return u, p, aux

  def filter(self, u: jax.Array, alpha=0.05):
    """Apply filter-based stabilization to a velocity field."""
    low_gridpoints = Nodes1D.create(
        num_points=self.velocity.mesh.gridpoints_1d.num_points - 1,
        node_type=self.velocity.mesh.gridpoints_1d.node_type)
    low_interpolator = BarycentricInterpolator(
        ndim=self.velocity.mesh.ndim,
        gridpoints_1d=self.velocity.mesh.gridpoints_1d,
        evalpoints_1d=low_gridpoints)
    high_interpolator = BarycentricInterpolator(
        ndim=self.velocity.mesh.ndim,
        gridpoints_1d=low_gridpoints,
        evalpoints_1d=self.velocity.mesh.gridpoints_1d)

    def _filter_one_elem(x):
      return high_interpolator.interpolate(low_interpolator.interpolate(x))

    u_local = self.velocity.gather(u)
    filtered_u_local = vmap(vmap(_filter_one_elem), in_axes=-1, out_axes=-1)(
        u_local)
    filtered_u = (1 / self.velocity.diag_qqt[:, jnp.newaxis]) * (
        self.velocity.scatter(filtered_u_local))
    return (1 - alpha) * u + alpha * filtered_u

  def vorticity(self, u: jax.Array) -> jax.Array:
    """Compute the vorticity of a velocity field."""
    u = self.velocity.vspace.vector_function(self.velocity.gather(u))

    def _vorticity(x: jax.Array):
      grad_ux = grad(u)(x)
      return grad_ux[1, 0] - grad_ux[0, 1]

    vort_local = self.velocity.vspace._evaluate(_vorticity)  # pylint: disable=protected-access
    vmesh = self.velocity.vspace.mesh
    return (1. / self.velocity.diag_qqt) * vmesh.scatter(vort_local)
