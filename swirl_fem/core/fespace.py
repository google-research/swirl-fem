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

"""Finite element space defined on a mesh."""

import dataclasses
from typing_extensions import Protocol

import jax
from jax import lax
from jax import vmap
import jax.numpy as jnp
from jax.custom_batching import custom_vmap

from swirl_fem.core.interpolation import BarycentricInterpolator
from swirl_fem.core.interpolation import Quadrature1D
from swirl_fem.core.mesh import Mesh


##############################
# Definitions of Q-functions #
##############################


class QFunction(Protocol):
  """Represents a q-function.

  A q-function with respect to a `Mesh` M, with ambient dimension `n` in
  Euclidean space, is a function from M ⊂ ℝⁿ to ℝᵏ. We can define functions
  using jax primitives on the coordinate `x` directly, or define nodal functions
  associated with a `FiniteElementSpace`, where the value at a point `x` in the
  mesh is interpolated from the nodal values on the 'nodes' in the mesh element
  containing `x`.

  The concept of q-functions is inspired from a similar usage in libCEED [1]:

  References:
   1. Barra, V., Brown, J., Thompson, J., & Dudouit, Y. (2020, July).
   High-performance operator evaluations with ease of use: libCEED’s Python
   interface. In Proceedings of the 19th Python in Science Conference
   (pp. 85-90).
  """

  def __call__(self, x: jax.Array) -> jax.Array:
    ...


class Form(Protocol):
  """A higher order function mapping q-functions to a functional.

  A `Form` is a map from multiple q-functions to a single scalar-valued
  q-function. We implicitly assume that target q-function is going to be
  integrated over the mesh to obtain the scalar integral. Thus, the `Form`
  implicitly represents a mapping from q-functions to a scalar. Typically, the
  q-functions are 'nodal' q-functions, which are vectors belonging to a
  `FiniteElementSpace`. So the `Form` may be thought of as a mapping from
  vectors to a scalar.
  """

  def __call__(self, *args: QFunction) -> QFunction:
    ...


@dataclasses.dataclass
class NodalQFunction(QFunction):
  """Represents a nodal function in a FiniteElementSpace.

  An abstract class which allows the action of evaluating the q-function over
  each quadrature point in each mesh element to be overridden in a custom
  fashion, allowing greater flexibility and efficient execution. For instance,
  a q-function can utilize precomputed per-quadrature point jacobians or other
  'global' information when evaluating the function over a mesh. To obtain more
  efficient versions of the evaluation, a q-function can invoke einsums over all
  elements and quadrature points instead of weighted sums at each quadrature
  point.

  Attributes:
    fespace: The underlying `FiniteElementSpace` that the nodal function is
      defined over.
    value_shape: The shape of each value of the q-function. The shape of the
      evaluation over every element's quadrature nodes would be `(num elements,
      num quadrature points per element) + value_shape`
      Default: ()
    u_local: The local (per-element) nodal values, shaped `(num elements,
      num nodes per element) + value_shape`. For instance, obtained by
      `u_local = fespace.mesh.gather(u)` from the nodal values `u` shaped
      as `(num_nodes,) + value_shape`.
  """
  fespace: 'FiniteElementSpace'
  value_shape: tuple[int, ...] = dataclasses.field(init=False)
  u_local: jax.Array | None = None

  def __init__(self,
               fespace: 'FiniteElementSpace',
               value_shape: tuple[int, ...],
               u_local: jax.Array | None):
    self.fespace = fespace
    self.value_shape = value_shape
    self.u_local = u_local
    expected_shape = (
        fespace.num_elements, fespace.mesh.num_nodes_per_element,
    ) + self.value_shape
    if u_local is not None and u_local.shape != expected_shape:
      raise ValueError('shape:', u_local.shape)

  def _evaluate(self) -> jax.Array:
    """Evaluates the function on each elements' quadrature points."""
    raise NotImplementedError

  def __call__(self, x) -> jax.Array:
    ndim = self.fespace.mesh.ndim

    # A function at a point x inside the domain. The implementation is never
    # reached except during tracing.
    @custom_vmap
    def _f(x):
      assert x.shape == (ndim,)
      return jnp.ones((), dtype=x.dtype)

    # The batching rule for _f specifies the function _f_batched which returns
    # evaluations on each quadrature point in a single, unspecified, element.
    @_f.def_vmap
    def _f_rule(axis_size, in_batched, xs):
      assert axis_size == (
          self.fespace.num_quadrature_points_per_element), axis_size
      return _f_batched(xs), in_batched[0]

    # Next, we define the evaluation of the function at each quadrature point
    # inside a single element. Similarly, the function implementation is never
    # reached except during tracing time.
    @custom_vmap
    def _f_batched(xs):
      assert xs.shape == (
          (self.fespace.num_quadrature_points_per_element, ndim,)), xs.shape
      return jnp.ones(len(xs), dtype=xs.dtype)

    # The batching rule for `_f_batched` specifies a function which evaluates on
    # each quadrature point in each element in the underlying mesh.
    @_f_batched.def_vmap
    def _f_batched_rule(axis_size, in_batched, xs):
      assert axis_size == self.fespace.num_elements, axis_size
      return _f_batched_batched(xs), in_batched[0]

    # Finally, we reach the twice vmapped function (once over each quadrature
    # nodes, and a second time over each element). We can now actually evaluate
    # the function using the nodal values on each element's quadrature points.
    def _f_batched_batched(xs):
      assert xs.shape == (self.fespace.num_elements,
                          self.fespace.num_quadrature_points_per_element,
                          ndim), xs.shape
      # The evaluation doesn't actually use the coordinates. Just the nodal
      # values should be enough.
      del xs
      return self._evaluate()

    return _f(x)


@dataclasses.dataclass
class ScalarNodalQFunction(NodalQFunction):
  """A scalar-valued function on a mesh interpolated from nodal values."""

  def __init__(self, fespace: 'FiniteElementSpace',
               u_local: jax.Array | None = None):
    super().__init__(fespace, value_shape=(), u_local=u_local)

  def _evaluate(self):
    return vmap(self.fespace.interpolator.interpolate)(self.u_local)


@dataclasses.dataclass
class ScalarNodalQFunctionGrad(NodalQFunction):
  """The gradient of a `ScalarNodalQFunction`."""

  def __init__(self, fespace: 'FiniteElementSpace',
               u_local: jax.Array | None = None):
    super().__init__(fespace, value_shape=(), u_local=u_local)

  def _evaluate(self):
    elem_grads = vmap(self.fespace.interpolator.interpolate_grad)(
        self.u_local)
    return jnp.einsum('mqi,mqji->mqj', elem_grads,
                      self.fespace.invjacs,
                      precision=lax.Precision.HIGHEST)


@dataclasses.dataclass
class VectorNodalQFunction(NodalQFunction):
  """A vector-valued function on a mesh interpolated from nodal values."""

  def __init__(self, fespace: 'FiniteElementSpace',
               u_local: jax.Array | None = None):
    super().__init__(fespace, value_shape=(fespace.mesh.ndim,),
                     u_local=u_local)

  def _evaluate(self):
    return vmap(vmap(self.fespace.interpolator.interpolate),
                in_axes=-1, out_axes=-1)(self.u_local)


@dataclasses.dataclass
class VectorNodalQFunctionGrad(NodalQFunction):
  """The gradient of a `VectorNodalQFunction`."""

  def __init__(self, fespace: 'FiniteElementSpace',
               u_local: jax.Array | None = None):
    super().__init__(fespace, value_shape=(fespace.mesh.ndim,),
                     u_local=u_local)

  def _evaluate(self):
    elem_grads = vmap(vmap(self.fespace.interpolator.interpolate_grad),
                      in_axes=-1, out_axes=-1)(self.u_local)
    return jnp.einsum('mqik,mqji->mqjk', elem_grads,
                      self.fespace.invjacs, precision=lax.Precision.HIGHEST)


############################
# Operators on Q-functions #
############################


def grad(f: QFunction) -> QFunction:
  """Gradient of a QFunction."""
  if isinstance(f, ScalarNodalQFunction):
    return ScalarNodalQFunctionGrad(fespace=f.fespace, u_local=f.u_local)
  elif isinstance(f, VectorNodalQFunction):
    return VectorNodalQFunctionGrad(fespace=f.fespace, u_local=f.u_local)
  # We expect that it's a jax function of the coordinate (x), in which case
  # fall back to the symbolic implementation of gradient.
  return jax.grad(f)


def div(f: QFunction) -> QFunction:
  """Divergence of a vector-valued QFunction."""
  def _divf(x: jax.Array) -> jax.Array:
    return jnp.trace(grad(f)(x))
  return _divf


######################
# FiniteElementSpace #
######################


@dataclasses.dataclass(frozen=True)
class FiniteElementSpace:
  """Represents a finite element space defined on a `Mesh`.

  This class defines a finite dimensional vector space V over reals whose basis
  elements are nodal basis functions defined on a mesh. The ith nodal basis
  function φᵢ attains a value of 1 the ith node of the mesh and 0 on every other
  node. Using this basis, each element `v ∈ V`, may be represented as a vector
  in ℝⁿ, where `n` is the number of nodes in the `Mesh`. See [2] or any other
  standard finite element textbooks for more details on the construction.

  This class primarily provides functionalities for (a) constructing functions
  in `V` from the primitives `evaluate` and `evaluate_grad`; (b) integrating
  functions `v ∈ V` over the mesh; and (c) obtaining representations of linear
  functionals `F ∊ V*` in the dual basis [1] to the one used for `V`.

  ### References
  1. Dual space. Wikipedia. https://en.wikipedia.org/wiki/Dual_space
  2. Elman, H.C., Silvester, D.J., & Wathen, A.J. (2005). Finite Elements and
     Fast Iterative Solvers: with Applications in Incompressible Fluid Dynamics.
  3. Riesz representation theorem. Wikipedia.
     https://en.wikipedia.org/wiki/Riesz_representation_theorem

  Attributes:
    mesh: The `Mesh` over which the continuous functions are defined.
    quadrature: A `Quadrature1D` instance which specifies the quadrature rule on
      the 1D reference elements. It is used to integrate over N-D elements using
      the tensor-product construction. Only the quadrature weights are used from
      this object; while the `interpolator` contains references to the
      quadrature nodes.
    interpolator: A `BarycentricInterpolator` used to interpolate nodal
      functions and their gradients on each element to the quadrature points.
      The `quadrature` nodes must agree with the interpolator's evaluation
      points: `interpolator.eval_points_1d`.
    invjacs: The inverse of the jacobian of the mapping from reference elements
      to the mesh element, precomputed for each element at its quadrature
      points. Shaped `(num elements, num quadrature points, ndim, ndim)`.
    jacdets: The determinant of the jacobian of the mapping from reference
      to the mesh element, precomputed for each element at its quadrature
      points. Shaped `(num elements, num quadrature points)`.
    quad_coords: Coordinates of the quadrature nodes on each element.
      Shaped `(num elements, num quadrature points, ndim)`.
  """
  mesh: Mesh
  quadrature: Quadrature1D
  interpolator: BarycentricInterpolator
  invjacs: jax.Array
  jacdets: jax.Array
  quad_coords: jax.Array

  @classmethod
  def create(cls, mesh: Mesh, quadrature: Quadrature1D) -> 'FiniteElementSpace':
    """Creates a `FiniteElementSpace` object.

    This function mainly precomputes jacobian inverses and determinants.

    Args:
      mesh: The underlying N-dimensional mesh on which the finite element space
        is defined.
      quadrature: A `Quadrature1D` object used for integration on each element.
        The quadrature nodes and weights are used for each dimension; the
        N-dimensional quadrature rule is derived from the tensor-product of
        the 1D nodes.

    Returns:
      The `FiniteElementSpace` object.
    """
    # Construct the `BarycentricInterpolator` from the given gridpoints and the
    # quadrature rule.
    interpolator = BarycentricInterpolator(
        ndim=mesh.ndim,
        gridpoints_1d=mesh.gridpoints_1d,
        evalpoints_1d=quadrature.nodes)
    # Gather the coordinates of the nodes within each element.
    elem_coords = mesh.element_coords()
    # vmap across elements and ndims.
    quad_coords = vmap(vmap(interpolator.interpolate, in_axes=-1, out_axes=-1))(
        elem_coords)

    # Compute jacobians by computing the N-dimensional derivative of the
    # coordinates. The result has shape
    # `(num elements, num quadrature points, ndim, ndim)`.
    jacs = jnp.einsum('mnj,qni->mqij', elem_coords,
                      interpolator.interpolation_matrix_grad(),
                      precision=lax.Precision.HIGHEST)

    # Next, for each element and each quadrature point within an element,
    # compute the `d x d` matrix inverse and its determinant. We vmap first over
    # the elements, and second over the quadrature nodes.
    invjacs = vmap(vmap(jnp.linalg.inv))(jacs)
    jacdets = vmap(vmap(jnp.linalg.det))(jacs)
    return cls(mesh=mesh, quadrature=quadrature, interpolator=interpolator,
               invjacs=invjacs, jacdets=jacdets, quad_coords=quad_coords)

  @property
  def num_elements(self) -> int:
    """Number of elements in the underlying mesh."""
    return self.mesh.num_elements

  @property
  def num_quadrature_points_per_element(self) -> int:
    """Number of quadrature points per element."""
    return int(self.quadrature.num_points ** self.mesh.ndim)

  def _evaluate(self, f: QFunction) -> jax.Array:
    """Evaluates a q-function on each element's quadrature nodes."""
    return vmap(vmap(f))(self.quad_coords)

  def scalar_function(self, u_local: jax.Array | None):
    """Returns a scalar-valued nodal function given the local nodal values."""
    expected_shape = (self.num_elements, self.mesh.num_nodes_per_element)
    if u_local is not None and u_local.shape != expected_shape:
      raise ValueError(
          f'Expecting shape {expected_shape} but got {u_local.shape=}')
    return ScalarNodalQFunction(fespace=self, u_local=u_local)

  def vector_function(self, u_local: jax.Array | None):
    """Returns a vector-valued nodal function given the local nodal values."""
    expected_shape = (self.num_elements, self.mesh.num_nodes_per_element,
                      self.mesh.ndim)
    if u_local is not None and u_local.shape != expected_shape:
      raise ValueError(
          f'Expecting shape {expected_shape} but got {u_local.shape=}')
    return VectorNodalQFunction(fespace=self, u_local=u_local)

  def integrate(self, f: QFunction) -> jax.Array:
    """Integrates a function evaluated on the mesh elements.

    We integrate using quadrature rule on the reference element; and multiply
    by the determinant of the Jacobian to obtain the corresponding integral on
    the mesh element. Finally, we sum up the integrals on each element to obtain
    the scalar integral.

    Args:
      f: The q-function to integrate

    Returns:
      The scalar integral value.
    """
    w = self._evaluate(f)
    expected_shape = (self.num_elements, self.num_quadrature_points_per_element)
    if w.shape != expected_shape:
      raise ValueError(
          'Expecting an array of shape (num elements, num quadrature points), '
          f'that is ({expected_shape}) but got: {w.shape}')
    quad_weights = self.quadrature.weights_nd(self.interpolator.ndim)
    return jnp.einsum('mq,mq,q->', w, self.jacdets, quad_weights,
                      precision=lax.Precision.HIGHEST)

  def local_covector(
      self, form: Form, funs: tuple[NodalQFunction, ...]
  ) -> jax.Array:
    """The local covector of a functional operating locally on each element.

    Returns the local covector representation of the linear functional described
    below, of shape (num elements, num nodes per element). The actual covector
    may be obtained by `self.mesh.scatter` on the local covector.

    The form is specified as a callable which accepts multiple q-functions, each
    representing a vector `f` in `V`, and outputs a single, scalar-valued
    q-function. The value of the form is obtained by subsequently integrating
    the resulting q-function using `integrate`.

    For example the bilinear form `L(u, v) = ∫uv dΩ` can be specified in this
    intermediate format as:

    ```python
      def l(u, v):
        return lambda x: u(x) * v(x)
    ```
    The actual value of the form `L(u, v)` is obtained at the vectors `(u0, v0)`
    as `fespace.integrate(l(u0, v0))`.

    The linear functional is formed by partial application of `funs` to the
    given multilinear form. In particular, a single element of `funs` must be
    a placeholder, represented by a `ScalarNodalQFunction` with nodal values
    as `None`. This placeholder is designated as the argument of the linear
    functional which is mapped to the scalar integral.

    For example, we can compute the local covector representation `b` of the
    linear functional `u ↦ L(f, u)`.

    ```python
      b_local = fespace.local_covector(l, f)
      b = fespace.mesh.scatter(b_local)
    ```

    Args:
      form: A multilinear form.
      funs: A sequence of q-functions on the underlying mesh, each of which
        represent vectors belonging to a finite element space.

    Returns:
      Local covector corresponding to the linear functional.
    """
    def _is_input(f):
      return isinstance(f, NodalQFunction) and f.u_local is None

    if sum(_is_input(f) for f in funs) != 1:
      raise ValueError('Exactly one `QFunction` must be a nodal function and '
                       'have `None` as nodal values')

    def _integrate_fn(v):
      replace_dual = (
          lambda f: dataclasses.replace(f, u_local=v) if _is_input(f) else f
      )
      funlist = list(map(replace_dual, funs))
      return self.integrate(form(*funlist))

    value_shape = [f.value_shape for f in funs if _is_input(f)][0]
    # compute the linear transpose of the function u ↦ integrate(form(u, ...))
    primal = jax.core.ShapedArray(
        shape=(
            (self.num_elements, self.mesh.num_nodes_per_element) + value_shape),
        dtype=self.jacdets.dtype)
    return jax.linear_transpose(_integrate_fn, primal)(1.)[0]
