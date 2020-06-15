#include "mesh.h"

#include <assimp/Importer.hpp>   // C++ importer interface
#include <assimp/postprocess.h>  // Post processing flags
#include <assimp/scene.h>        // Output data structure

Mesh::Mesh(const std::string file_name) {
  Assimp::Importer importer;

  const aiScene *scene = importer.ReadFile(
      file_name, aiProcess_CalcTangentSpace | aiProcess_Triangulate |
          aiProcess_JoinIdenticalVertices | aiProcess_SortByPType);

  assert(scene->mNumMeshes == 1);
  auto *mesh = scene->mMeshes[0];
  assert(mesh->mPrimitiveTypes == aiPrimitiveType_TRIANGLE);

  for (size_t i = 0; i < mesh->mNumVertices; i++) {
    auto *v = mesh->mVertices + i;
    auto *n = mesh->mNormals + i;
    vertices_.emplace_back(v->x, v->y, v->z);
    normals_.emplace_back(n->x, n->y, n->z);
  }

  for(size_t i = 0; i < mesh->mNumFaces; i++) {
    auto* t = mesh->mFaces + i;
    triangles_.emplace_back(t->mIndices[0], t->mIndices[1], t->mIndices[2]);
  }
}
