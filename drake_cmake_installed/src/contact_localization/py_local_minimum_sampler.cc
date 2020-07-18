#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "local_minimum_sampler.h"

namespace py = pybind11;

PYBIND11_MODULE(py_local_minimum_sampler, m) {
  using Class = LocalMinimumSampler;

  py::class_<Class>(m, "LocalMinimumSampler")
      .def(py::init<std::string>())
      .def("UpdateJacobians", &Class::UpdateJacobians)
      .def("SampleLocalMinimum",
           [](const Class* self, const Eigen::Ref<const Eigen::VectorXd>& q,
              const Eigen::Ref<const Eigen::VectorXd>& tau_ext,
              const size_t contact_link_idx,
              Eigen::Ref<Eigen::Vector3d>& p_LQ_L_final,
              Eigen::Ref<Eigen::Vector3d>& normal_L_final,
              Eigen::Ref<Eigen::Vector3d>& f_W_final) {
             double dlduv_norm_final;
             double l_star_final;
             return self->SampleLocalMinimum(q, tau_ext, contact_link_idx,
                                      &p_LQ_L_final,
                                      &normal_L_final, &f_W_final,
                                      &dlduv_norm_final, &l_star_final, true);
           })
      .def("RunGradientDescentFromPointOnMesh",
           [](const Class* self,
              const Eigen::Ref<const Eigen::VectorXd>& tau_ext,
              const size_t contact_link_idx,
              const Eigen::Ref<const Eigen::Vector3d>& p_LQ_L_initial,
              const Eigen::Ref<const Eigen::Vector3d>& normal_L_initial,
              Eigen::Ref<Eigen::Vector3d>& p_LQ_L_final,
              Eigen::Ref<Eigen::Vector3d>& normal_L_final,
              Eigen::Ref<Eigen::Vector3d>& f_W_final) {
             double dlduv_norm_final;
             double l_star_final;
             return self->RunGradientDescentFromPointOnMesh(
                 tau_ext, contact_link_idx, p_LQ_L_initial,
                 normal_L_initial, &p_LQ_L_final, &normal_L_final, &f_W_final,
                 &dlduv_norm_final, &l_star_final, true);
           })
      .def("get_points_log", &Class::get_points_log)
      .def("get_normals_log", &Class::get_normals_log)
      .def("get_dlduv_norm_log", &Class::get_dlduv_norm_log)
      .def("get_l_star_log", &Class::get_l_star_log);
}
