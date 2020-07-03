#pragma once

#include "gradient_calculator.h"
#include "proximity_wrapper.h"

class LocalMinimumSampler {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(LocalMinimumSampler)
  LocalMinimumSampler(const std::string &robot_sdf_path,
                      const std::string &model_name,
                      const std::vector<std::string> &link_names,
                      const std::vector<std::string> &link_mesh_paths,
                      const std::vector<int> &active_link_indices,
                      int num_rays,
                      double epsilon);

  bool RunGradientDescentFromPointOnMesh(
      const Eigen::Ref<const Eigen::VectorXd>& q,
      const Eigen::Ref<const Eigen::VectorXd>& tau_ext,
      const size_t contact_link_idx,
      const size_t iteration_limit,
      const Eigen::Ref<const Eigen::VectorXd>& p_LQ_L_initial,
      const Eigen::Ref<const Eigen::VectorXd>& normal_L_initial,
      drake::EigenPtr<Eigen::Vector3d> p_LQ_L_final,
      drake::EigenPtr<Eigen::Vector3d> normal_L_final,
      drake::EigenPtr<Eigen::Vector3d> f_W_final, double* dlduv_norm_final,
      double* l_star_final, bool is_logging) const;

  bool SampleLocalMinimum(const Eigen::Ref<const Eigen::VectorXd>& q,
                          const Eigen::Ref<const Eigen::VectorXd>& tau_ext,
                          const size_t contact_link_idx,
                          const size_t iteration_limit,
                          drake::EigenPtr<Eigen::Vector3d> p_LQ_L_final,
                          drake::EigenPtr<Eigen::Vector3d> normal_L_final,
                          drake::EigenPtr<Eigen::Vector3d> f_W_final,
                          double* dlduv_norm_final, double* l_star_final,
                          bool is_logging) const;

  std::vector<Eigen::Vector3d> get_points_log() const { return log_points_L_; }
  std::vector<Eigen::Vector3d> get_normals_log() const {
    return log_normals_L_;
  }
  std::vector<double> get_dlduv_norm_log() const { return log_dlduv_norm_; }
  std::vector<double> get_l_star_log() const { return log_l_star_; }

 private:
  const double epsilon_;
  const double gradient_norm_convergence_threshold_;
  std::unique_ptr<GradientCalculator> calculator_;
  std::vector<std::unique_ptr<ProximityWrapper>> p_queries_;

  mutable std::vector<Eigen::Vector3d> log_points_L_;
  mutable std::vector<Eigen::Vector3d> log_normals_L_;
  mutable std::vector<double> log_dlduv_norm_;
  mutable std::vector<double> log_l_star_;
};