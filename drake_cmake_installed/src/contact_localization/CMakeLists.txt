
add_library(mesh mesh.h mesh.cc)
target_link_libraries(mesh fcl ${ASSIMP_LIBRARY})

add_executable(load_mesh load_mesh.cc)
target_link_libraries(load_mesh mesh fcl lcm::lcm drake::drake osqp::osqp)