#include "mesh.h"

#include <assimp/Importer.hpp>   // C++ importer interface
#include <assimp/postprocess.h>  // Post processing flags
#include <assimp/scene.h>        // Output data structure

using Eigen::Vector3d;

Mesh::Mesh(const std::string& file_name) {
  Assimp::Importer importer;

  const aiScene* scene = importer.ReadFile(
      file_name, aiProcess_CalcTangentSpace | aiProcess_Triangulate |
                     aiProcess_JoinIdenticalVertices | aiProcess_SortByPType);

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
}

Eigen::Vector3d Mesh::CalcFaceNormal(size_t triangle_idx) const {
  const Vector3d& a = vertices_[triangles_[triangle_idx][0]];
  const Vector3d& b = vertices_[triangles_[triangle_idx][1]];
  const Vector3d& c = vertices_[triangles_[triangle_idx][2]];
  auto ab = b - a;
  auto ac = c - a;
  return ab.cross(ac).normalized();
}

Eigen::Vector3d Mesh::CalcBarycentric(
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
  assert( 1 - u - v >= 0);
  return {1 - u - v, u, v};
}

Eigen::Vector3d Mesh::CalcPointNormal(
    size_t triangle_idx, const Eigen::Ref<const Eigen::Vector3d>& p) const {
  const Vector3d& n0 = normals_[triangles_[triangle_idx][0]];
  const Vector3d& n1 = normals_[triangles_[triangle_idx][1]];
  const Vector3d& n2 = normals_[triangles_[triangle_idx][2]];
  auto barycentric = CalcBarycentric(triangle_idx, p);
  return n0 * barycentric[0] + n1 * barycentric[1] + n2 * barycentric[2];
}
