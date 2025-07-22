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

from absl.testing import absltest
import jax
from jax import vmap
import jax.numpy as jnp
import numpy as np
import scipy.optimize

from swirl_fem.common import premesh_commons
from swirl_fem.linalg.cg import cg
from swirl_fem.navier_stokes.navier_stokes import BCType
from swirl_fem.navier_stokes.navier_stokes import bdfk_coeffs
from swirl_fem.navier_stokes.navier_stokes import extk_coeffs
from swirl_fem.navier_stokes.navier_stokes import StokesSEM


# pylint: disable=invalid-name


def _vdot(xs, ys):
  """inner product of arrays."""
  assert len(xs) == len(ys), f'{len(xs)} != {len(ys)}'
  return sum(x * y for x, y in zip(xs, ys))


def _make_stokes_vortices_premesh():
  """Uniform premesh on [-1, 1] x [-np.pi, np.pi] which is periodic in y."""
  premesh = premesh_commons.unit_cube_mesh(9, ndim=2, periodic_dims=(1,))
  translate_fn = lambda x: jnp.array([2 * x[0] - 1, 2 * np.pi * x[1] - np.pi])
  updated_node_coords = jax.vmap(translate_fn)(premesh.node_coords)
  return premesh.replace(node_coords=updated_node_coords)


def reference_soln_params(k=1., viscosity=1.):
  """Reference solution parameters for the stokes vortices test."""
  def _implicit(x):
    return k * np.tanh(k) + x * np.tan(x)

  mu = scipy.optimize.newton(_implicit, np.pi)
  sigma = -viscosity * (np.square(k) + np.square(mu))
  return mu, sigma


def reference_soln(vnode_coords, pnode_coords, t, k=1., viscosity=1.):
  """Returns the reference solution for the stokes vortices test at time `t`."""
  mu, sigma = reference_soln_params(k=k, viscosity=viscosity)
  f = lambda x: np.cos(mu) * jnp.cosh(k * x) - np.cosh(k) * jnp.cos(mu * x)
  g = lambda x: (1j / k) * (
      k * np.cos(mu) * jnp.sinh(k * x) + mu * np.cosh(k) * jnp.sin(mu * x))
  h = lambda x: -(sigma / k) * np.cos(mu) * jnp.sinh(k * x)

  leading_term_ = lambda x: jnp.exp(sigma * t) * jnp.exp(1j * k * x[1])
  u_ = lambda x: jnp.real(leading_term_(x) * jnp.array([f(x[0]), g(x[0])]))
  p_ = lambda x: jnp.real(leading_term_(x) * h(x[0]))
  return vmap(u_)(vnode_coords), vmap(p_)(pnode_coords)


class NavierStokesTest(absltest.TestCase):

  def test_premesh(self):
    premesh = _make_stokes_vortices_premesh()
    mesh = premesh.finalize()
    self.assertEqual(mesh.num_elements, 81)
    self.assertEqual(mesh.num_nodes, 100)

  def test_stokes_analytical_momentum(self):
    # Verify the equation B ∂ₜ u + Au − Dᵀp = 0.
    premesh = _make_stokes_vortices_premesh()
    boundary_conditions = {'boundary': (BCType.DIRICHLET, 0.0)}
    sem = StokesSEM.create(
        premesh, boundary_conditions=boundary_conditions, order=7)

    u, p = reference_soln(sem.velocity.vspace.mesh.node_coords,
                          sem.pressure.pspace.mesh.node_coords, t=0.0)
    _, sigma = reference_soln_params()

    Bdu_dt = sem.B(sigma * u)
    Au = sem.A(u)
    Dtp = sem.Dt(p)
    error = sem.velocity.exchange(Bdu_dt + Au - Dtp)
    self.assertLess(np.abs(error).max(), 1e-7)

  def test_stokes_analytical_divergence(self):
    # Verify div(u) = 0.
    premesh = _make_stokes_vortices_premesh()
    boundary_conditions = {'boundary': (BCType.DIRICHLET, 0.0)}
    sem = StokesSEM.create(
        premesh, boundary_conditions=boundary_conditions, order=7)

    u, _ = reference_soln(sem.velocity.vspace.mesh.node_coords,
                          sem.pressure.pspace.mesh.node_coords, t=0.0)

    error = sem.D(u)
    self.assertLess(np.abs(error).max(), 1e-10)

  def test_stokes_analytical_bdf(self):
    # Verify the equation B ∂ₜ u + Au − Dᵀp = 0 but use BDFk for the ∂ₜ u term.
    premesh = _make_stokes_vortices_premesh()
    boundary_conditions = {'boundary': (BCType.DIRICHLET, 0.0)}
    sem = StokesSEM.create(
        premesh, boundary_conditions=boundary_conditions, order=7)

    # Get the first `k + 1` states to apply the BDFk scheme.
    k = 3
    t0 = 0.
    dt = 0.001
    get_state = lambda t: reference_soln(
        sem.velocity.vspace.mesh.node_coords,
        sem.pressure.pspace.mesh.node_coords, t=t)
    us, ps = list(zip(*[get_state(t0 + i * dt) for i in range(k + 1)]))

    du_dt = (1 / dt) * _vdot(bdfk_coeffs(k), us)
    Bdu_dt = sem.B(du_dt)
    Au = sem.A(us[-1])
    Dtp = sem.Dt(ps[-1])

    error = sem.velocity.exchange(Bdu_dt + Au - Dtp)
    self.assertLess(np.abs(error).max(), 1e-7)

  def test_stokes_fractional_step_basic(self):
    # Verify the equation Huⁿ − Dᵀ(pⁿ − pⁿ⁻¹) = Bfⁿ + Dᵀp̂ⁿ
    # where fⁿ includes the terms from the BDFk scheme for ∂ₜ u.
    premesh = _make_stokes_vortices_premesh()
    boundary_conditions = {'boundary': (BCType.DIRICHLET, 0.0)}
    sem = StokesSEM.create(
        premesh, boundary_conditions=boundary_conditions, order=7)

    # Get the first `k + 1` states to apply the BDFk scheme. The first `k`
    # represent the history of states and `k + 1`st state is the new state.
    k = 3
    t0 = 0.
    dt = 0.001
    # Solution at time t.
    get_state = lambda t: reference_soln(
        sem.velocity.vspace.mesh.node_coords,
        sem.pressure.pspace.mesh.node_coords, t=t)
    us, ps = list(zip(*[get_state(t0 + i * dt) for i in range(k + 1)]))

    # Separate the history and the last term.
    us, u = us[:-1], us[-1]
    ps, p = ps[:-1], ps[-1]

    # Extrapolate pressure linearly.
    ext_coeffs = extk_coeffs(k=1)
    p_ext = sum(
        ext_coeffs[-i] * ps[-i] for i in range(1, len(ext_coeffs) + 1))

    # Augment the forcing term with BDFk expansion of the velocity history.
    # Using BDFk we have:
    #   B(βₖ uⁿ + βₖ₋₁uⁿ^ ⁻¹ + ⋯ + β₀uⁿ⁻ᵏ) = Δt (−Auⁿ + Dᵀp̂ⁿ + Bfⁿ).
    # Define, f̂ⁿ = fⁿ − (1 / Δt) (βₖ₋₁uⁿ^ ⁻¹ + ⋯ + β₀uⁿ⁻ᵏ).
    f = 0
    beta_hist = bdfk_coeffs(k)[:-1]
    beta_k = bdfk_coeffs(k)[-1]
    f = f - (1 / dt) * _vdot(beta_hist, us)
    b = sem.B(f) + sem.Dt(p_ext)

    # H = (βₖ / Δt) B + A.
    Hu = (beta_k / dt) * sem.B(u) + sem.A(u)
    dp = p - p_ext

    error = sem.velocity.exchange(Hu - sem.Dt(dp) - b)
    self.assertLess(np.abs(error).max(), 1e-7)

  def test_stokes_fractional_step_approx(self):
    # Verify the equation Huⁿ − HQDᵀ(pⁿ − pⁿ⁻¹) = Bfⁿ + Dᵀp̂ⁿ + O(Δt)²
    # where fⁿ includes the terms from the BDFk scheme for ∂ₜ u.
    premesh = _make_stokes_vortices_premesh()
    boundary_conditions = {'boundary': (BCType.DIRICHLET, 0.0)}
    sem = StokesSEM.create(
        premesh, boundary_conditions=boundary_conditions, order=7)

    # Get the first `k + 1` states to apply the BDFk scheme. The first `k`
    # represent the history of states and `k + 1`st state is the new state.
    k = 3
    t0 = 0.
    dt = 0.001
    # Solution at time t.
    get_state = lambda t: reference_soln(
        sem.velocity.vspace.mesh.node_coords,
        sem.pressure.pspace.mesh.node_coords, t=t)
    us, ps = list(zip(*[get_state(t0 + i * dt) for i in range(k + 1)]))

    # Separate the history and the last term.
    us, u = us[:-1], us[-1]
    ps, p = ps[:-1], ps[-1]

    # Extrapolate pressure linearly.
    ext_coeffs = extk_coeffs(k=1)
    p_ext = sum(
        ext_coeffs[-i] * ps[-i] for i in range(1, len(ext_coeffs) + 1))

    # Augment the forcing term with BDFk expansion of the velocity history.
    # Using BDFk we have:
    #   B(βₖ uⁿ + βₖ₋₁uⁿ^ ⁻¹ + ⋯ + β₀uⁿ⁻ᵏ) = Δt (−Auⁿ + Dᵀp̂ⁿ + Bfⁿ).
    # Define, f̂ⁿ = fⁿ − (1 / Δt) (βₖ₋₁uⁿ^ ⁻¹ + ⋯ + β₀uⁿ⁻ᵏ).
    f = 0
    beta_hist = bdfk_coeffs(k)[:-1]
    beta_k = bdfk_coeffs(k)[-1]
    f = f - (1 / dt) * _vdot(beta_hist, us)
    b = sem.B(f) + sem.Dt(p_ext)

    # H = (βₖ / Δt) B + A; Q = (Δt / βₖ) B⁻¹.
    H_ = lambda u: (beta_k / dt) * sem.B(u) + sem.A(u)
    Q_ = lambda u: (dt / beta_k) * sem.Bi(u)
    dp = p - p_ext

    error = sem.velocity.exchange(H_(u) - H_(Q_(sem.Dt(dp))) - b)
    self.assertLess(np.abs(error).max(), 10 * np.square(dt))

  def test_stokes_fractional_step_lu_factorization(self):
    # Obtain u* from the LU factorization of ((H -HQDt) (-D 0))
    # as u* = u - QDᵀdp; then verify Hu* = b.
    premesh = _make_stokes_vortices_premesh()
    boundary_conditions = {'boundary': (BCType.DIRICHLET, 0.0)}
    sem = StokesSEM.create(
        premesh, boundary_conditions=boundary_conditions, order=7)

    # Get the first `k + 1` states to apply the BDFk scheme. The first `k`
    # represent the history of states and `k + 1`st state is the new state.
    k = 3
    t0 = 0.
    dt = 0.001
    # Solution at time t.
    get_state = lambda t: reference_soln(
        sem.velocity.vspace.mesh.node_coords,
        sem.pressure.pspace.mesh.node_coords, t=t)
    us, ps = list(zip(*[get_state(t0 + i * dt) for i in range(k + 1)]))

    # Separate the history and the last term.
    us, u = us[:-1], us[-1]
    ps, p = ps[:-1], ps[-1]

    # Extrapolate pressure linearly.
    ext_coeffs = extk_coeffs(k=1)
    p_ext = sum(
        ext_coeffs[-i] * ps[-i] for i in range(1, len(ext_coeffs) + 1))

    # Augment the forcing term with BDFk expansion of the velocity history.
    # Using BDFk we have:
    #   B(βₖ uⁿ + βₖ₋₁uⁿ^ ⁻¹ + ⋯ + β₀uⁿ⁻ᵏ) = Δt (−Auⁿ + Dᵀp̂ⁿ + Bfⁿ).
    # Define, f̂ⁿ = fⁿ − (1 / Δt) (βₖ₋₁uⁿ^ ⁻¹ + ⋯ + β₀uⁿ⁻ᵏ).
    f = 0
    beta_hist = bdfk_coeffs(k)[:-1]
    beta_k = bdfk_coeffs(k)[-1]
    f = f - (1 / dt) * _vdot(beta_hist, us)
    b = sem.B(f) + sem.Dt(p_ext)

    # H = (βₖ / Δt) B + A; Q = (Δt / βₖ) B⁻¹.
    H_ = lambda u: (beta_k / dt) * sem.B(u) + sem.A(u)
    Q_ = lambda u: (dt / beta_k) * sem.Bi(u)
    dp = p - p_ext

    # Obtain u* as u - QDᵀdp; then verify Hu* = b.
    u_star = u - Q_(sem.Dt(dp))
    error = sem.velocity.exchange(H_(u_star) - b)
    self.assertLess(np.abs(error).max(), 10 * np.square(dt))

  def test_stokes_fractional_step_cg_solve(self):
    # Solve for u* from Hu* = b. Then verify that u* = u - QDᵀdp.
    premesh = _make_stokes_vortices_premesh()
    boundary_conditions = {'boundary': (BCType.DIRICHLET, 0.0)}
    sem = StokesSEM.create(
        premesh, boundary_conditions=boundary_conditions, order=7)

    # Get the first `k + 1` states to apply the BDFk scheme. The first `k`
    # represent the history of states and `k + 1`st state is the new state.
    k = 3
    t0 = 0.
    dt = 0.001
    # Solution at time t.
    get_state = lambda t: reference_soln(
        sem.velocity.vspace.mesh.node_coords,
        sem.pressure.pspace.mesh.node_coords, t=t)
    us, ps = list(zip(*[get_state(t0 + i * dt) for i in range(k + 1)]))

    # Separate the history and the last term.
    us, u = us[:-1], us[-1]
    ps, p = ps[:-1], ps[-1]

    # Extrapolate pressure linearly.
    ext_coeffs = extk_coeffs(k=1)
    p_ext = sum(
        ext_coeffs[-i] * ps[-i] for i in range(1, len(ext_coeffs) + 1))

    # Augment the forcing term with BDFk expansion of the velocity history.
    # Using BDFk we have:
    #   B(βₖ uⁿ + βₖ₋₁uⁿ^ ⁻¹ + ⋯ + β₀uⁿ⁻ᵏ) = Δt (−Auⁿ + Dᵀp̂ⁿ + Bfⁿ).
    # Define, f̂ⁿ = fⁿ − (1 / Δt) (βₖ₋₁uⁿ^ ⁻¹ + ⋯ + β₀uⁿ⁻ᵏ).
    f = 0
    beta_hist = bdfk_coeffs(k)[:-1]
    beta_k = bdfk_coeffs(k)[-1]
    f = f - (1 / dt) * _vdot(beta_hist, us)
    b = sem.B(f) + sem.Dt(p_ext)

    # H = (βₖ / Δt) B + A; Q = (Δt / βₖ) B⁻¹.
    H_ = lambda u: (beta_k / dt) * sem.B(u) + sem.A(u)
    Q_ = lambda u: (dt / beta_k) * sem.Bi(u)
    dp = p - p_ext

    # Obtain u* by solving Hu* = b.
    u_star, info = cg(H_, b, M=sem.velocity.exchange, tol=1e-15)
    del info

    residual = sem.velocity.exchange(H_(u_star) - b)
    self.assertLess(np.abs(residual).max(), 1e-12)

    # Verify u* = u - Q_(sem.Dt(dp))
    error = u_star - u + Q_(sem.Dt(dp))
    self.assertLess(np.abs(error).max(), 5 * np.square(dt))

  def test_stokes_solve_one_step(self):
    # Solve for u* from Hu* = b. Then verify that u* = u - QDᵀdp.
    premesh = _make_stokes_vortices_premesh()
    boundary_conditions = {'boundary': (BCType.DIRICHLET, 0.0)}
    sem = StokesSEM.create(
        premesh, boundary_conditions=boundary_conditions, order=7)

    # Get the first `k + 1` states to apply the BDFk scheme. The first `k`
    # represent the history of states and `k + 1`st state is the new state.
    k = 3
    t0 = 0.
    dt = 1e-3
    # Solution at time t.
    get_state = lambda t: reference_soln(
        sem.velocity.vspace.mesh.node_coords,
        sem.pressure.pspace.mesh.node_coords, t=t)
    us, ps = list(zip(*[get_state(t0 + i * dt) for i in range(k + 1)]))

    # Separate the history and the last term.
    us, u_expected = us[:-1], us[-1]
    ps, p_expected = ps[:-1], ps[-1]

    u_obtained, p_obtained, aux = sem.stokes_one_step(
        us, ps, f=0, mu=1, dt=dt, time_order=k, alpha=0.05,
        project_out_nullspace=True, tol=1e-12, atol=1e-12)

    error = u_obtained - u_expected
    self.assertLess(np.abs(error).max(), 5 * np.square(dt))

    error = p_obtained - p_expected
    self.assertLess(np.abs(error).max(), 50 * np.square(dt))

    self.assertLess(aux['u_star_info']['residual'], 1e-7)
    self.assertLess(aux['dp_info']['residual'], 1e-7)


if __name__ == "__main__":
  jax.config.update('jax_enable_x64', True)
  absltest.main()
