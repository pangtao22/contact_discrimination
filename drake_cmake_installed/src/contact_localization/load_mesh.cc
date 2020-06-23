#include <chrono>
#include <memory>

#include <Eigen/Dense>
#include <fcl/narrowphase/collision.h>
#include <fcl/narrowphase/distance.h>

#include "mesh.h"

using namespace fcl;
typedef fcl::BVHModel<fcl::OBBRSSd> Model;

using Eigen::Vector3d;
using fcl::Triangle;
using std::cout;
using std::endl;

int main() {
  std::string pFile =
      "/Users/pangtao/PycharmProjects/contact_aware_control"
      "/contact_particle_filter/iiwa7_shifted_meshes/link_6.obj";
  //  std::string pFile = "./cube.obj";

  TriangleMesh mesh(pFile);

  cout << "Mesh info" << endl;
  cout << mesh.vertices_.size() << " " << mesh.triangles_.size() << endl;
  cout << mesh.num_primitives() << " " << mesh.num_vertices() << endl;
  cout << "End of mesh info" << endl;

  auto geom1 = std::make_shared<Model>();
  // add the mesh data into the BVHModel structure
  geom1->beginModel();
  geom1->addSubModel(mesh.vertices_, mesh.triangles_);
  geom1->endModel();
  geom1->computeLocalAABB();
  cout << "geom1 " << geom1->computeVolume() << endl;
  cout << geom1->aabb_local.height() << " " << geom1->aabb_local.width() << " "
       << geom1->aabb_local.depth() << endl;
  cout << geom1->aabb_local.center() << endl;
  cout << geom1->aabb_local.min_ << endl;
  cout << geom1->aabb_local.max_ << endl;

  auto geom2 = std::make_shared<Sphered>(1e-3);
  auto* obj1 = new CollisionObjectd(geom1);
  auto* obj2 = new CollisionObjectd(geom2);

  DistanceRequestd request;

  std::vector<Vector3d> p_positions = {Vector3d(0, 0, 0.2),
                                       Vector3d(0.2, -0.01, 0.01),
                                       Vector3d(-0.2, -0.01, 0.01)};

  for (const auto& p : p_positions) {
    obj2->setTranslation(p);
    auto start = std::chrono::high_resolution_clock::now();
    DistanceResultd result;

    distance(obj1, obj2, request, result);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = duration_cast<std::chrono::microseconds>(end - start);
    cout << endl << "Collision result" << endl;
    cout << "time " << duration.count() << endl;
    cout << result.min_distance << endl;
    cout << result.nearest_points[0] << endl;
    cout << result.nearest_points[1] << endl;
    cout << result.b1 << endl;
    auto p0 = mesh.get_primitive(result.b1);
    cout << p0[0] << " " << p0[1] << " " << p0[2] << endl;
    cout << result.b2 << endl;

    cout << "Normals:" << endl;
    cout << "Face: " << mesh.CalcFaceNormal(result.b1).transpose() << endl;
    cout
        << "Barycentric coordinates: "
        << mesh.CalcBarycentric(result.b1, result.nearest_points[0]).transpose()
        << endl;
    cout
        << "Barycentric normal: "
        << mesh.CalcPointNormal(result.b1, result.nearest_points[0]).transpose()
        << endl;
  }

  return 0;
}
