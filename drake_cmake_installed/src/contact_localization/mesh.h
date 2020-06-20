#pragma once

#include <string>
#include <vector>

#include <Eigen/Dense>
#include <fcl/narrowphase/collision_object.h>

struct Mesh {
  explicit Mesh(const std::string& file_name);

  size_t num_primitives() const { return triangles_.size(); };
  size_t num_vertices() const { return vertices_.size(); };
  const fcl::Triangle& get_primitive(size_t i) const { return triangles_[i]; }
  Eigen::Vector3d CalcFaceNormal(size_t triangle_idx) const;
  Eigen::Vector3d CalcPointNormal(
      size_t triangle_idx, const Eigen::Ref<const Eigen::Vector3d>& p) const;
  Eigen::Vector3d CalcBarycentric(
      size_t triangle_idx, const Eigen::Ref<const Eigen::Vector3d>& p) const;

  std::vector<Eigen::Vector3d> vertices_;
  std::vector<Eigen::Vector3d> normals_;
  std::vector<fcl::Triangle> triangles_;
};
