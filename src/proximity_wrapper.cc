#include "proximity_wrapper.h"

typedef fcl::BVHModel<fcl::OBBRSSd> Model;

ProximityWrapper::ProximityWrapper(const std::string& mesh_file,
                                    double epsilon)
    : epsilon_(epsilon) {
  mesh_ = std::make_unique<TriangleMesh>(mesh_file);
  auto geometry_mesh = std::make_shared<Model>();
  geometry_mesh->beginModel();
  geometry_mesh->addSubModel(mesh_->vertices_, mesh_->triangles_);
  geometry_mesh->endModel();

  auto geometry_point = std::make_shared<fcl::Sphered>(epsilon_);

  obj_mesh_ = std::make_unique<fcl::CollisionObjectd>(geometry_mesh);
  obj_point_ = std::make_unique<fcl::CollisionObjectd>(geometry_point);
}

void ProximityWrapper::FindClosestPoint(
    const Eigen::Ref<const Eigen::Vector3d>& p_L,
    drake::EigenPtr<Eigen::Vector3d> p_mesh_L,
    drake::EigenPtr<Eigen::Vector3d> normal_L, size_t* triangle_idx,
    double* distance) const {
  obj_point_->setTranslation(p_L);
  fcl::DistanceResultd result;
  fcl::distance(obj_mesh_.get(), obj_point_.get(), request_, result);

  *p_mesh_L = result.nearest_points[0];
  *normal_L = mesh_->CalcFaceNormal(result.b1);
  *triangle_idx = result.b1;
  *distance = result.min_distance + epsilon_;
}
