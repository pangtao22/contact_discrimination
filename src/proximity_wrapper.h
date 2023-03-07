#pragma once

#include <drake/common/eigen_types.h>
#include <fcl/narrowphase/collision.h>
#include <fcl/narrowphase/distance.h>

#include "mesh.h"

class ProximityWrapper {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ProximityWrapper)
  explicit ProximityWrapper(const std::string& mesh_file, double epsilon);
  /**
   * p_L: point coordinate in mesh frame (L).
   * p_mesh_L: point on mesh cloest to p_L, expressed in mesh frame L.
   * normal_L: normal of mesh at p_L, expressed in L.
   * triangle_idx: index of triangle into mesh_.triangle to which p_mesh_L
   *    belongs.
   * distance: distance between p_L and p_mesh_L.
   */
  void FindClosestPoint(const Eigen::Ref<const Eigen::Vector3d>& p_L,
                        drake::EigenPtr<Eigen::Vector3d> p_mesh_L,
                        drake::EigenPtr<Eigen::Vector3d> normal_L,
                        size_t* triangle_idx, double* distance) const;

  const TriangleMesh& get_mesh() { return *mesh_; }

 private:
  const double epsilon_{};
  std::unique_ptr<TriangleMesh> mesh_;
  std::unique_ptr<fcl::CollisionObjectd> obj_mesh_;
  std::unique_ptr<fcl::CollisionObjectd> obj_point_;
  fcl::DistanceRequestd request_;
};