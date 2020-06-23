#include <fstream>
#include <iostream>


#include "mesh.h"

using std::cout;
using std::endl;

int main() {
  std::string pFile =
      "/Users/pangtao/PycharmProjects/contact_aware_control"
      "/contact_particle_filter/iiwa7_shifted_meshes/link_6.obj";
//  std::string pFile("cube.obj");
  TriangleMesh mesh(pFile);

  int trials = 1000;
  auto points = Eigen::Matrix3Xd(3, trials);

  for(size_t i = 0; i < trials; i++) {
    points.col(i) = mesh.SamplePointOnMesh();
  }

  std::ofstream file_points("points_sampled_on_mesh.csv");
  const Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols,
                                  ", ", "\n");
  file_points << points.format(CSVFormat);
  file_points.close();

  return 0;
}
