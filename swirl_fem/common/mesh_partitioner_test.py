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

import collections
import math

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import numpy as np

from swirl_fem.common import mesh_partitioner
from swirl_fem.common import mesh_reader
from swirl_fem.core.premesh import Premesh


def _unit_interval_mesh(num_elements: int) -> Premesh:
  """Creates a premesh on [0, 1] with the given number of elements."""
  num_nodes = 1 + num_elements
  node_coords = np.linspace(0, 1, num_nodes).reshape((num_nodes, 1))
  elements = np.array([
      list(range(i, (i + 1) + 1)) for i in range(num_elements)
  ])
  return Premesh.create(node_coords=node_coords, elements=elements)


class MeshPartitionerTest(parameterized.TestCase):

  @parameterized.parameters((2, 2), (8, 2), (16, 4), (15, 4), (35, 8))
  def test_partition_1d_mesh(self, num_elements, num_partitions):
    partitioned = mesh_partitioner.partition(
        _unit_interval_mesh(num_elements), num_partitions=num_partitions)
    self.assertEqual(partitioned.num_nodes, num_elements + 1)
    self.assertEqual(partitioned.num_elements, num_elements)
    self.assertLen(partitioned.partitions, num_elements)

    # Make sure partition ids are in [0, num_partitions) and that the partitions
    # are approximately equal-sized.
    partition_counts = collections.Counter(partitioned.partitions)
    for partition_id, count in partition_counts.items():
      self.assertBetween(partition_id, 0, num_partitions - 1)
      self.assertBetween(count,
                         int(math.floor(num_elements / num_partitions)),
                         int(math.ceil(num_elements / num_partitions)))

    # make sure each partition consists of a contiguous range of elements in 1d
    for p in range(num_partitions):
      elems = set(i for i, k in enumerate(partitioned.partitions) if p == k)
      self.assertSetEqual(elems, set(range(min(elems), 1 + max(elems))))

  @parameterized.parameters(2, 4, 8)
  def test_partition_unit_cube(self, num_partitions):
    # partition a 3d unit cube mesh
    filepath = epath.Path(__file__).parent.parent / 'testdata/cube.msh'
    partitioned = mesh_partitioner.partition(
        mesh_reader.read(filepath, ndim=3), num_partitions=num_partitions)
    self.assertEqual(partitioned.node_coords.shape, (125, 3))
    self.assertEqual(partitioned.elements.shape, (64, 8))

    # Make sure partition ids are in [0, num_partitions) and that the partitions
    # are approximately equal-sized.
    partition_counts = collections.Counter(partitioned.partitions)
    for partition_id, count in partition_counts.items():
      self.assertBetween(partition_id, 0, num_partitions - 1)
      self.assertBetween(
          count,
          int(math.floor(partitioned.num_elements / num_partitions)),
          int(math.ceil(partitioned.num_elements / num_partitions)),
      )

if __name__ == "__main__":
  absltest.main()
