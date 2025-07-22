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

from collections.abc import Sequence

from absl.testing import absltest
import flax.linen as nn
import jax
import jax.numpy as jnp

from swirl_fem.sde.sdeint import brownian_path
from swirl_fem.sde.flax_nn_sde import nn_sdeint

class FlaxNNSdeTest(absltest.TestCase):

  def test_lifted_sdeint(self):

    class MLP(nn.Module):
      features: Sequence[int]

      @nn.compact
      def __call__(self, x, t, dw):
        x = jnp.concatenate([x, t[None]], axis=0)

        for feat in self.features[:-1]:
          x = nn.relu(nn.Dense(feat)(x))
          x = nn.Dense(self.features[-1])(x)

        drift = x
        diffusion = dw * jnp.sqrt(jnp.abs(x))
        return drift, diffusion

    class IntegratedMLP(nn.Module):
      features: Sequence[int]

      def setup(self):
        self.mlp = nn_sdeint(MLP)(self.features)

      def __call__(self, x, dw):
        ts = jnp.array([0., 1.])
        return self.mlp(x, ts, dw)

    key1, key2, key3 = jax.random.split(jax.random.PRNGKey(0), 3)
    x = jax.random.normal(key1, (3,))  # Dummy input
    dw = brownian_path(rng=key2, n=128)

    model = IntegratedMLP([128, 3])
    params = model.init(key3, x, dw)
    output = model.apply(params, x, dw)
    self.assertEqual(output.shape, (1, 3))


if __name__ == '__main__':
  absltest.main()
