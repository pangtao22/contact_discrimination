#include "proximity_wrapper.h"

using Eigen::Vector3d;
using std::cout;
using std::endl;

int main() {
  std::string pFile =
      "/Users/pangtao/PycharmProjects/contact_aware_control"
      "/contact_particle_filter/iiwa7_shifted_meshes/link_6.obj";

  ProximityWrapper proximity_wrapper(pFile, 1e-3);
  std::vector<Vector3d> p_positions = {Vector3d(0.01, 0.01, 0.2),
                                       Vector3d(0, -0.2, -0.01),
                                       Vector3d(-0.2, -0.01, 0.01)};

  Vector3d p_mesh_L;
  Vector3d normal_L;
  size_t triangle_idx;
  double distance;
  for (const auto& p : p_positions) {
    proximity_wrapper.FindClosestPoint(p, &p_mesh_L, &normal_L, &triangle_idx,
                                       &distance);
    cout << "Collision result\n";
    cout << "p_mesh_L: " << p_mesh_L.transpose() << endl;
    cout << "normal_L: " << normal_L.transpose() << endl;
    cout << "tirnagle_idx: " << triangle_idx << endl;
    cout << "distance: " << distance << endl << endl;
  }
  return 0;
}
