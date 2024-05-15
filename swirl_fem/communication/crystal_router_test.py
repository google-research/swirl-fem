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

"""Tests for crystal_router."""

from absl.testing import absltest
import jax
from jax import sharding
from jax.experimental import mesh_utils
import numpy as np
from swirl_fem.communication.crystal_router import crystal_router_setup
from swirl_fem.communication.set_test_device_count import set_host_platform_device_count


NUM_DEVICES = 12


class CrystalRouterTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    set_host_platform_device_count(NUM_DEVICES)

  def test_crystal_router(self):
    assert jax.device_count() == NUM_DEVICES
    n = NUM_DEVICES
    m = 7  # static size of local array

    devices = mesh_utils.create_device_mesh((n,))
    mesh = sharding.Mesh(devices, axis_names='i')

    crystal = crystal_router_setup(mesh, 'i')

    rng = np.random.RandomState(seed=2)
    num = rng.randint(low=m // 2, high=m + 1, size=(n,), dtype=np.int32)
    target = rng.randint(low=0, high=n, size=(n, m), dtype=np.int32)
    data = rng.randint(100, size=(n, m), dtype=np.int32)

    mask = np.arange(m) < num[:, None]
    in_src = np.where(mask, np.arange(n)[:, None], -1)
    target = np.where(mask, target, -1)
    data = np.where(mask, data, -1)

    n_out, out, source = crystal(num, data, target)

    flat_target = target.flatten()[mask.flatten()]
    flat_source = in_src.flatten()[mask.flatten()]
    flat_data = data.flatten()[mask.flatten()]
    lexsorted = lambda *args: np.array([*args])[:, np.lexsort([*args])]
    for i in range(n):
      mask_i = flat_target == i
      expected = lexsorted(flat_data[mask_i], flat_source[mask_i])
      actual = lexsorted(out[i, :n_out[i]], source[i, :n_out[i]])
      np.testing.assert_array_equal(expected, actual)

    # output without source should be the same (up to re-ordering)
    n_out2, out2 = crystal(num, data, target, return_source=False)
    for i in range(n):
      np.testing.assert_array_equal(
          np.sort(out[i, : n_out[i]]), np.sort(out2[i, : n_out2[i]])
      )

    # Second invocation should restore data up to ordering.
    n_out, out, source = crystal(n_out, out, source)
    for i in range(n):
      expected = lexsorted(data[i, :num[i]], target[i, :num[i]])
      actual = lexsorted(out[i, :n_out[i]], source[i, :n_out[i]])
      np.testing.assert_array_equal(expected, actual)


if __name__ == '__main__':
  jax.config.update('jax_enable_x64', True)
  absltest.main()
