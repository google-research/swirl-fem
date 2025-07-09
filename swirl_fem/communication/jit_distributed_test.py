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

"""Tests for jit_distributed."""

from absl.testing import absltest
import jax
from jax import lax
from jax import sharding
from jax.experimental import mesh_utils
import numpy as np
from swirl_fem.communication.jit_distributed import jit_distributed
from swirl_fem.communication.set_test_device_count import set_host_platform_device_count


NUM_DEVICES = 4


def _mesh() -> sharding.Mesh:
  return sharding.Mesh(
      mesh_utils.create_device_mesh((NUM_DEVICES,)), axis_names='i'
  )


class JitDistributedTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    set_host_platform_device_count(NUM_DEVICES)

  def test_basic(self):
    f = jit_distributed(lambda x: x - lax.psum(x, 'i'), _mesh(), 'i')
    x = np.arange(NUM_DEVICES * 4, dtype=np.int32).reshape(NUM_DEVICES, 4)
    expected = x - np.sum(x, 0)
    np.testing.assert_array_equal(expected, f(x))

  def test_no_input(self):
    f = jit_distributed(lambda: lax.axis_index('i'), _mesh(), 'i')
    np.testing.assert_array_equal(np.arange(NUM_DEVICES), f())

  def test_static_arg(self):
    def local(x, *, s):
      return x - s * lax.psum(x, 'i')
    f = jit_distributed(local, _mesh(), 'i')
    x = np.arange(NUM_DEVICES * 4, dtype=np.int32).reshape(NUM_DEVICES, 4)
    expected = x - 2 * np.sum(x, 0)
    np.testing.assert_array_equal(expected, f(x, s=2))


if __name__ == '__main__':
  jax.config.update('jax_enable_x64', True)
  absltest.main()
