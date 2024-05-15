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

"""Routines for parsing a Gmsh mesh into a Swirl-FEM `Premesh`."""

from etils import epath
import meshio
import numpy as np
from swirl_fem.core.premesh import Premesh


# Mapping from dimension to permutation mapping the Gmsh node ordering to
# tensor product (lexicographic) node ordering. See below for the 2d and 3d
# node orderings which define the permutations.
#
#  Gmsh and tensor product node ordering for 2d (quadrilateral) elements:
#   3 -- 2     1 -- 3
#   |    |     |    |
#   0 -- 1     0 -- 2
#
#  Gmsh and tensor product node ordering for 3d (hexahedron) elements:
#     7----6       3----7
#    /|   /|      /|   /|
#   4----5 |     1----5 |
#   | |  | |     | |  | |
#   | 3--|-2     | 2--|-6
#   |/   |/      |/   |/
#   0----1       0----4
_NODE_ORDERING_PERMUTATIONS = {
    1: [0, 1],
    2: [0, 3, 1, 2],
    3: [0, 4, 3, 7, 1, 5, 2, 6],
}


def _get_periodic_links(mesh: meshio.Mesh, ndim: int) -> np.ndarray:
  """Obtains periodic links from the meshio Mesh.

  Args:
    mesh: The `meshio.Mesh` that we want to parse.
    ndim: The dimension of the mesh's ambient space.

  Returns:
    An array with a list of pairs of facets which share a periodic connection.
    See documentation of `Premesh` for the detailed specification.
  """
  # Create the mapping from source nodes to target nodes for every periodic
  # connection in the mesh among entities of dimension `ndim - 1`.
  src_tgt_mapping = dict()
  for entity_dim, _, _, node_pairs in mesh.gmsh_periodic:
    if entity_dim != ndim - 1:
      continue
    src_tgt_mapping.update(dict(node_pairs.tolist()))

  # For each `ndim - 1`-dimensional facet in the mesh, check if it is a periodic
  # facet. If so, save it as a periodic link.
  facet_elem_type = {1: 'line', 2: 'quad'}[ndim - 1]
  periodic_links = []
  for facet_elem in mesh.cells_dict[facet_elem_type]:
    if all(x in src_tgt_mapping for x in facet_elem):
      target_elem = list(map(lambda x: src_tgt_mapping[x], facet_elem))
      periodic_links.append(np.stack([facet_elem, target_elem]))

  return np.stack(periodic_links, dtype=np.int32)


def read(path: epath.PathLike, ndim: int) -> Premesh:
  """Reads a Gmsh mesh file from `path` and parses it into a `Premesh`.

  Note that restrictions aren't imposed on the mesh file format. While any mesh
  format supported by meshio should work with this function, only the Gmsh mesh
  format is tested. Due to limitations in meshio, only Gmsh meshes currently
  support periodicity.

  Args:
    path: The path at which the mesh file is located.
    ndim: The dimension of the mesh's ambient space.

  Returns:
    The `Premesh` object containing the parsed mesh.
  """
  if ndim not in [1, 2, 3]:
    raise ValueError(f'Invalid ndim: {ndim=}. Valid spatial dimensions are '
                     '1, 2 and 3.')
  mesh = meshio.read(path)
  node_coords = mesh.points[:, :ndim]

  elem_type = {1: 'line', 2: 'quad', 3: 'hexahedron'}[ndim]
  if elem_type not in mesh.cells_dict:
    raise ValueError(
        f'Reading mesh of {ndim=} but cells of type {elem_type=} not found '
        f'in {mesh.cells_dict.keys()=}')

  # reorder nodes in each element to follow the lexicographic ordering
  elements = mesh.cells_dict[elem_type][:, _NODE_ORDERING_PERMUTATIONS[ndim]]
  periodic_links = (
      _get_periodic_links(mesh, ndim=ndim) if mesh.gmsh_periodic else None)

  return Premesh.create(
      node_coords=node_coords,
      elements=elements,
      periodic_links=periodic_links,
  )