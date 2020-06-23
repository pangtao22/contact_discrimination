#pragma once

#include <string>
#include <random>
#include <vector>

#include <Eigen/Dense>
#include <fcl/narrowphase/collision_object.h>

class TriangleMesh {
 public:
  explicit TriangleMesh(const std::string& file_name);

  size_t num_primitives() const { return triangles_.size(); };
  size_t num_vertices() const { return vertices_.size(); };
  const fcl::Triangle& get_primitive(size_t i) const { return triangles_[i]; }
  Eigen::Vector3d CalcFaceNormal(size_t triangle_idx) const;
  Eigen::Vector3d CalcPointNormal(
      size_t triangle_idx, const Eigen::Ref<const Eigen::Vector3d>& p) const;
  Eigen::Vector3d CalcBarycentric(
      size_t triangle_idx, const Eigen::Ref<const Eigen::Vector3d>& p) const;
  Eigen::Vector3d SamplePointOnMesh() const;
  void SamplePointOnMesh(Eigen::Vector3d* point_ptr, size_t* triangle_idx_ptr)
  const;
  double CalcFaceArea(size_t trianle_idx) const;

  std::vector<Eigen::Vector3d> vertices_;
  std::vector<Eigen::Vector3d> normals_;
  std::vector<fcl::Triangle> triangles_;
 private:
  std::vector<size_t> areas_cdf_;
  std::unique_ptr<std::mt19937> generator_;
  std::unique_ptr<std::mt19937_64> generator64_;
  std::unique_ptr<std::uniform_int_distribution<size_t>> distribution_idx_;
  std::unique_ptr<std::uniform_real_distribution<double>> distribution_uv_;
};
