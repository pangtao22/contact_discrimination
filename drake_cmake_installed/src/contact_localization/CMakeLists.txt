
add_library(mesh mesh.h mesh.cc)
target_link_libraries(mesh fcl ${ASSIMP_LIBRARY})

add_library(osqp_wrapper osqp_wrapper.h osqp_wrapper.cc)
target_link_libraries(osqp_wrapper osqp::osqp drake::drake)

add_library(gradient_calculator gradient_calculator.h gradient_calculator.cc)
target_link_libraries(gradient_calculator mesh osqp_wrapper drake::drake )

add_executable(load_mesh load_mesh.cc)
target_link_libraries(load_mesh mesh lcm::lcm drake::drake osqp::osqp)

add_executable(try_osqp try_osqp.cc)
target_link_libraries(try_osqp osqp_wrapper)

add_executable(try_gradient_calculator try_gradient_calculator.cc)
target_link_libraries(try_gradient_calculator gradient_calculator)