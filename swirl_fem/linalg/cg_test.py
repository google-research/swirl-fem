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

import numpy as np
import jax.numpy as jnp
from swirl_fem.linalg.cg import cg
from jax import tree_util
import jax


class CgTest(absltest.TestCase):

  def test_cg_ndarray(self):
    A = lambda x: 2 * x
    b = jnp.arange(9.0).reshape((3, 3))
    expected = b / 2
    actual, _ = cg(A, b)
    np.testing.assert_allclose(expected, actual)

  def test_cg_pytree(self):
    A = lambda x: {'a': x['a'] + 0.5 * x['b'], 'b': 0.5 * x['a'] + x['b']}
    b = tree_util.tree_map(jnp.asarray, {'a': 1.0, 'b': -4.0})
    expected = {'a': 4.0, 'b': -6.0}
    actual, _ = cg(A, b)
    self.assertEqual(expected.keys(), actual.keys())
    self.assertAlmostEqual(expected['a'], actual['a'], places=6)
    self.assertAlmostEqual(expected['b'], actual['b'], places=6)

  def test_preconditioner(self):
    # The system: 2 * x[0] = 1; 0 * x[1] = 2 is not solvable, but if we use
    # M as lambda x: [x[0], 0], then the solver only cares about the first dim.
    A = lambda x: jnp.array([2 * x[0], 0 * x[1]])
    b = 1 + jnp.arange(2.0)
    M = lambda x: jnp.array([x[0], 0.])
    actual, _ = cg(A, b, M=M)
    expected = jnp.array([0.5, 0.])
    np.testing.assert_allclose(expected, actual)

if __name__ == '__main__':
  absltest.main()
