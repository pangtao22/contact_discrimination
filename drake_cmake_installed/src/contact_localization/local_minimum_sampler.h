#pragma once

#include "gradient_calculator.h"
#include "proximity_wrapper.h"

struct LocalMinimumSamplerConfig {
  GradientCalculatorConfig gradient_calculator_config;

  std::vector<std::string> link_mesh_paths;
  std::vector<size_t> active_link_indices;
  size_t num_links{0};

  // distance above the mesh, used for proximity queries.
  double epsilon{5e-4};

  // Gradient descent parameters.
  size_t line_search_steps_limit{10};
  double gradient_norm_convergence_threshold{1e-3};
  double alpha{0.4};
  double beta{0.5};
  double max_step_size{0.02};
};

LocalMinimumSamplerConfig LoadLocalMinimumSamplerConfigFromYaml(
    const std:: string& file_path);


class LocalMinimumSampler {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(LocalMinimumSampler)
  LocalMinimumSampler(const LocalMinimumSamplerConfig& config);

  bool RunGradientDescentFromPointOnMesh(
      const Eigen::Ref<const Eigen::VectorXd>& q,
      const Eigen::Ref<const Eigen::VectorXd>& tau_ext,
      const size_t contact_link_idx,
      const size_t iteration_limit,
      const Eigen::Ref<const Eigen::VectorXd>& p_LQ_L_initial,
      const Eigen::Ref<const Eigen::VectorXd>& normal_L_initial,
      drake::EigenPtr<Eigen::Vector3d> p_LQ_L_final,
      drake::EigenPtr<Eigen::Vector3d> normal_L_final,
      drake::EigenPtr<Eigen::Vector3d> f_L_final, double* dlduv_norm_final,
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
  const LocalMinimumSamplerConfig config_;

  std::unique_ptr<GradientCalculator> calculator_;
  std::vector<std::unique_ptr<ProximityWrapper>> p_queries_;

  mutable std::vector<Eigen::Vector3d> log_points_L_;
  mutable std::vector<Eigen::Vector3d> log_normals_L_;
  mutable std::vector<double> log_dlduv_norm_;
  mutable std::vector<double> log_l_star_;
};