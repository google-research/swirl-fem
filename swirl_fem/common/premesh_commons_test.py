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

import itertools
import collections

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from swirl_fem.common.premesh_commons import unit_cube_mesh


class PremeshCommonsTest(parameterized.TestCase):

  @parameterized.parameters(itertools.product(range(1, 5 + 1), range(1, 3 + 1)))
  def test_num_nodes_and_elements(self, num_elements_per_dim, ndim):
    premesh = unit_cube_mesh(num_elements_per_dim, ndim=ndim)
    self.assertEqual(premesh.num_elements, num_elements_per_dim ** ndim)
    self.assertEqual(premesh.num_nodes, (num_elements_per_dim + 1) ** ndim)

  @parameterized.parameters(range(1, 5 + 1))
  def test_periodic(self, num_elements_per_dim):
    premesh = unit_cube_mesh(num_elements_per_dim, ndim=2,
                             periodic_dims=(0, 1))
    # there should be 2 * num_elements_per_dim periodic links
    self.assertEqual(premesh.periodic_links.shape,
                     (2 * num_elements_per_dim, 2, 2))

  def test_partitions(self):
    num_elements_per_dim = 12
    ndim = 2
    premesh = unit_cube_mesh(num_elements_per_dim,
                             ndim=ndim, partitions=np.array([[0, 1], [2, 3]]))
    # count each element of premesh.partitions and make sure there is equal
    # number of ids
    counter = collections.Counter(premesh.partitions.tolist())
    for count in counter.values():
      self.assertEqual(count, (num_elements_per_dim // 2) ** ndim)


if __name__ == "__main__":
  absltest.main()
