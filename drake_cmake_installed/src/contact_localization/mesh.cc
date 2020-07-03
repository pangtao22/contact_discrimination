#include "mesh.h"

#include <numeric>

#include <assimp/Importer.hpp>   // C++ importer interface
#include <assimp/postprocess.h>  // Post processing flags
#include <assimp/scene.h>        // Output data structure

using Eigen::Vector3d;
using Eigen::VectorXd;
using std::cout;
using std::endl;

TriangleMesh::TriangleMesh(const std::string& file_name) {
  Assimp::Importer importer;
  const aiScene* scene = importer.ReadFile(
      file_name, aiProcess_CalcTangentSpace | aiProcess_Triangulate |
                     aiProcess_JoinIdenticalVertices | aiProcess_SortByPType);
  if(!scene) {
    throw std::runtime_error(file_name + " does not exist.");
  }
  assert(scene->mNumMeshes == 1);
  auto* mesh = scene->mMeshes[0];
  assert(mesh->mPrimitiveTypes == aiPrimitiveType_TRIANGLE);

  for (size_t i = 0; i < mesh->mNumVertices; i++) {
    auto* v = mesh->mVertices + i;
    auto* n = mesh->mNormals + i;
    vertices_.emplace_back(v->x, v->y, v->z);
    normals_.emplace_back(n->x, n->y, n->z);
  }

  for (size_t i = 0; i < mesh->mNumFaces; i++) {
    auto* t = mesh->mFaces + i;
    triangles_.emplace_back(t->mIndices[0], t->mIndices[1], t->mIndices[2]);
  }

  // Construct data structure for sampling.
  VectorXd areas(mesh->mNumFaces);
  double min_area = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < mesh->mNumFaces; i++) {
    areas[i] = CalcFaceArea(i);
    if (areas[i] > 0 && min_area > areas[i]) {
      min_area = areas[i];
    }
  }
  Eigen::Matrix<size_t, -1, 1> areas_int(mesh->mNumFaces);
  for (size_t i = 0; i < mesh->mNumFaces; i++) {
    areas_int[i] = areas[i] / min_area;
  }

  areas_cdf_.resize(areas_int.size());
  areas_cdf_[0] = areas_int[0];
  for (size_t i = 1; i < areas_cdf_.size(); i++) {
    areas_cdf_[i] = areas_cdf_[i - 1] + areas_int[i];
  }

  // Initialize random number generators.
  std::random_device rd;
  generator_ = std::make_unique<std::mt19937>(rd());
  generator64_ = std::make_unique<std::mt19937_64>(rd());
  distribution_idx_ = std::make_unique<std::uniform_int_distribution<size_t>>(
      0, areas_cdf_.back() - 1);
  distribution_uv_ = std::make_unique<std::uniform_real_distribution<>>(0, 1);

//  cout << min_area << endl;
//  cout << areas.minCoeff() << " " << areas.maxCoeff() << endl;
//  cout << areas_int.minCoeff() << " " << areas_int.maxCoeff() << endl;
//  cout << areas_cdf_[0] << " " << areas_cdf_.back() << endl;
  //  cout << distribution(generator) << endl;
}

double TriangleMesh::CalcFaceArea(size_t triangle_idx) const {
  const Vector3d& a = vertices_[triangles_[triangle_idx][0]];
  const Vector3d& b = vertices_[triangles_[triangle_idx][1]];
  const Vector3d& c = vertices_[triangles_[triangle_idx][2]];
  auto ab = b - a;
  auto ac = c - a;
  return ab.cross(ac).norm();
}

Eigen::Vector3d TriangleMesh::CalcFaceNormal(size_t triangle_idx) const {
  const Vector3d& a = vertices_[triangles_[triangle_idx][0]];
  const Vector3d& b = vertices_[triangles_[triangle_idx][1]];
  const Vector3d& c = vertices_[triangles_[triangle_idx][2]];
  auto ab = b - a;
  auto ac = c - a;
  return ab.cross(ac).normalized();
}

Eigen::Vector3d TriangleMesh::CalcBarycentric(
    size_t triangle_idx, const Eigen::Ref<const Eigen::Vector3d>& p) const {
  const Vector3d& a = vertices_[triangles_[triangle_idx][0]];
  const Vector3d& b = vertices_[triangles_[triangle_idx][1]];
  const Vector3d& c = vertices_[triangles_[triangle_idx][2]];
  auto ab = b - a;
  auto ac = c - a;
  auto ap = p - a;
  auto abc = ab.cross(ac).norm();
  auto abp = ab.cross(ap).norm();
  auto apc = ap.cross(ac).norm();
  auto u = abp / abc;
  auto v = apc / abc;
  assert(1 - u - v >= 0);
  return {1 - u - v, u, v};
}

Eigen::Vector3d TriangleMesh::CalcPointNormal(
    size_t triangle_idx, const Eigen::Ref<const Eigen::Vector3d>& p) const {
  const Vector3d& n0 = normals_[triangles_[triangle_idx][0]];
  const Vector3d& n1 = normals_[triangles_[triangle_idx][1]];
  const Vector3d& n2 = normals_[triangles_[triangle_idx][2]];
  auto barycentric = CalcBarycentric(triangle_idx, p);
  return n0 * barycentric[0] + n1 * barycentric[1] + n2 * barycentric[2];
}

Eigen::Vector3d TriangleMesh::SamplePointOnMesh() const {
  Vector3d point;
  size_t triangle_idx;
  SamplePointOnMesh(&point, &triangle_idx);
  return point;
}

void TriangleMesh::SamplePointOnMesh(Eigen::Vector3d* point_ptr,
                                     size_t* triangle_idx_ptr) const {
  size_t& triangle_idx = *triangle_idx_ptr;

  const size_t random_number = (*distribution_idx_)(*generator64_);
  triangle_idx =
      std::upper_bound(areas_cdf_.begin(), areas_cdf_.end(), random_number) -
      areas_cdf_.begin();
  //  cout << triangle_idx << endl;

  auto u = (*distribution_uv_)(*generator_);
  auto v = (*distribution_uv_)(*generator_);
  if (u + v > 1) {
    u = 1 - u;
    v = 1 - v;
  }

  *point_ptr = u * vertices_[triangles_[triangle_idx][0]] +
               v * vertices_[triangles_[triangle_idx][1]] +
               (1 - u - v) * vertices_[triangles_[triangle_idx][2]];
}
