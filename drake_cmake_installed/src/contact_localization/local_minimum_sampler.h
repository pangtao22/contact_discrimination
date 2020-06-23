#pragma once

#include "gradient_calculator.h"
#include "proximity_wrapper.h"

class LocalMinimumSampler {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(LocalMinimumSampler)
  LocalMinimumSampler(const std::string& robot_sdf_path,
                      const std::string& model_name,
                      const std::vector<std::string>& link_names, int num_rays,
                      const std::string& link_mesh_path, double epsilon);

  void SampleLocalMinimum(const Eigen::Ref<const Eigen::VectorXd>& q,
                          const Eigen::Ref<const Eigen::VectorXd>& tau_ext,
                          size_t contact_link_idx, const size_t iteration_limit,
                          drake::EigenPtr<Eigen::Vector3d> p_LQ_L_final,
                          drake::EigenPtr<Eigen::Vector3d> normal_L_final,
                          drake::EigenPtr<Eigen::Vector3d> f_W_final,
                          double* dlduv_norm_final,
                          double* l_star_final,
                          std::vector<Eigen::Vector3d>* log_points_L,
                          std::vector<Eigen::Vector3d>* log_normals_L) const;

 private:
  const double epsilon_;
  std::unique_ptr<GradientCalculator> calculator_;
  std::unique_ptr<ProximityWrapper> p_query_;
};