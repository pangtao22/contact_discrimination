
add_library(mesh mesh.h mesh.cc)
target_link_libraries(mesh fcl ${ASSIMP_LIBRARY})

add_library(osqp_wrapper osqp_wrapper.h osqp_wrapper.cc)
target_link_libraries(osqp_wrapper osqp::osqp drake::drake)

add_library(proximity_wrapper proximity_wrapper.h proximity_wrapper.cc)
target_link_libraries(proximity_wrapper mesh drake::drake)

add_library(gradient_calculator gradient_calculator.h gradient_calculator.cc)
target_link_libraries(gradient_calculator mesh osqp_wrapper drake::drake )

add_executable(load_mesh try_loading_mesh.cc)
target_link_libraries(load_mesh mesh lcm::lcm drake::drake osqp::osqp)

add_executable(try_osqp try_osqp.cc)
target_link_libraries(try_osqp osqp_wrapper)

add_executable(try_gradient_decent_on_3link_arm try_gradient_decent_on_3link_arm.cc)
target_link_libraries(try_gradient_decent_on_3link_arm gradient_calculator yaml-cpp)

add_executable(try_proximity_wrapper try_proximity_wrapper.cc)
target_link_libraries(try_proximity_wrapper proximity_wrapper)

add_executable(try_gradient_descent_on_iiwa_link6 try_gradient_descent_on_iiwa_link6.cc)
target_link_libraries(try_gradient_descent_on_iiwa_link6
        gradient_calculator proximity_wrapper yaml-cpp)

add_executable(try_sampling_mesh try_sampling_mesh.cc)
target_link_libraries(try_sampling_mesh mesh)