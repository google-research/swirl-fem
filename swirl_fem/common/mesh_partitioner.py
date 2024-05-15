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

"""Partitions a mesh using metis for distributed computations."""

import networkx as nx
import pymetis
from swirl_fem.core.premesh import Premesh


def partition(premesh: Premesh, num_partitions: int) -> Premesh:
  """Returns a partitioned premesh with each element assigned to one partition.

  The `partitions` array associates an integer in [0, num_partitions) with
  each element. The underlying partitioning algorithm of pymetis approximately
  minimizes the edge cut of the graph defined by edges of shared nodes between
  elements, with each edge havign equal weight. Note that we do not consider
  periodic connections for defining edges.

  TODO: Consider exposing balance factor (`ufactor`) options from pymetis in
  case partition imbalance becomes an issue, particularly for large meshes.

  Args:
    premesh: The premesh to be partitioned.
    num_partitions: The desired number of partitions.

  Returns:
    A premesh with the associated partition ids of each element.
  """

  g = nx.Graph()
  for elem_idx, elem in enumerate(premesh.elements):
    g.add_edges_from((elem_idx, premesh.num_elements + i) for i in elem)

  adjlist = {}
  for source in range(premesh.num_elements):
    paths = nx.single_source_shortest_path_length(g, source, cutoff=2)
    adjlist[source] = [t for t, distance in paths.items() if distance == 2]
  # Partition the graph into `num_partitions` parts
  _, partitions = pymetis.part_graph(num_partitions, adjacency=adjlist)

  return premesh.replace(partitions=partitions)
