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
from etils import epath
import meshio
import numpy as np

from swirl_fem.common import mesh_reader


class MeshReaderTest(absltest.TestCase):

  def test_read_1d_mesh(self):
    filepath = epath.Path(__file__).parent.parent / 'testdata/line1d.msh'
    premesh = mesh_reader.read(filepath, ndim=1)

    self.assertEqual(premesh.node_coords.shape, (17, 1))
    self.assertEqual(premesh.elements.shape, (16, 2))
    # 1d meshes don't support periodicity in gmsh
    self.assertIsNone(premesh.periodic_links)

    np.testing.assert_array_almost_equal(
        np.sort(premesh.node_coords.flatten()), np.linspace(0, 1, num=17))

  def test_read_single_element_mesh_2d(self):
    # create a simple quadrilateral mesh
    node_coords = [[0, 0], [1, 0], [1, 1], [0, 1]]
    elements = [[0, 1, 2, 3]]
    mesh = meshio.Mesh(points=node_coords, cells=[('quad', elements)])

    # write the mesh to a Gmsh .msh file
    filepath = epath.Path(self.create_tempdir().full_path) / 'test.msh'
    meshio.write(filepath, mesh, file_format='gmsh')

    premesh = mesh_reader.read(filepath, ndim=2)
    self.assertEqual(premesh.node_coords.shape, (4, 2))
    np.testing.assert_array_equal(premesh.node_coords, np.array(node_coords))

    self.assertEqual(premesh.elements.shape, (1, 4))
    # order of elements changes to lexicographic ordering
    np.testing.assert_array_equal(premesh.elements, [[0, 3, 1, 2]])
    self.assertIsNone(premesh.periodic_links)

  def test_read_2d_periodic_mesh(self):
    # read a periodic 2D mesh
    filepath = epath.Path(__file__).parent.parent / 'testdata/kovasznay.msh'
    premesh = mesh_reader.read(filepath, ndim=2)

    self.assertEqual(premesh.node_coords.shape, (65, 2))
    self.assertEqual(premesh.elements.shape, (48, 4))
    self.assertEqual(premesh.periodic_links.shape, (4, 2, 2))

  def test_read_3d_mesh(self):
    # read a unit cube mesh
    filepath = epath.Path(__file__).parent.parent / 'testdata/cube.msh'
    premesh = mesh_reader.read(filepath, ndim=3)
    self.assertEqual(premesh.node_coords.shape, (125, 3))
    self.assertEqual(premesh.elements.shape, (64, 8))
    self.assertIsNone(premesh.periodic_links)

  def test_read_3d_mesh_periodic(self):
    # a triply periodic unit cube mesh
    filepath = epath.Path(__file__).parent.parent / 'testdata/periodic_cube.msh'
    premesh = mesh_reader.read(filepath, ndim=3)
    self.assertEqual(premesh.node_coords.shape, (125, 3))
    self.assertEqual(premesh.elements.shape, (64, 8))
    self.assertEqual(premesh.periodic_links.shape, (48, 2, 4))


if __name__ == "__main__":
  absltest.main()

