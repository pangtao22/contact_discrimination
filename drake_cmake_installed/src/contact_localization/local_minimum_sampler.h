#pragma once
#include <tuple>
#include <iostream>

#include <drake/lcmt_contact_discrimination.hpp>
#include <drake/lcmt_iiwa_status.hpp>
#include <lcm/lcm-cpp.hpp>

#include "gradient_calculator.h"
#include "proximity_wrapper.h"

struct LocalMinimumSamplerConfig {
  GradientCalculatorConfig gradient_calculator_config;

  std::vector<std::string> link_mesh_paths;
  std::vector<size_t> active_link_indices;
  size_t num_links{0};
  size_t num_samples_per_link{0};
  // distance above the mesh, used for proximity queries.
  double epsilon{5e-4};

  double tau_ext_infinity_norm_threshold{0};
  // Rejection sampling
  double optimal_cost_threshold;

  // Gradient descent parameters.
  size_t iterations_limit{0};
  size_t line_search_steps_limit{10};
  double gradient_norm_convergence_threshold{1e-3};
  double alpha{0.4};
  double beta{0.5};
  double max_step_size{0.02};

  // Samples and normals
  bool load_samples_from_files;
  std::vector<std::string> points_L_paths;
  std::vector<std::string> outward_normals_L_paths;
};

LocalMinimumSamplerConfig LoadLocalMinimumSamplerConfigFromYaml(
    const std::string& file_path);

class LocalMinimumSampler {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(LocalMinimumSampler)
  explicit LocalMinimumSampler(const LocalMinimumSamplerConfig& config);
  explicit LocalMinimumSampler(const std::string& config_file_path);

  void UpdateJacobians(const Eigen::Ref<const Eigen::VectorXd>& q) const {
    calculator_->UpdateJacobians(q, config_.active_link_indices);
  }
  void ComputeOptimalCostForSamples(
      const Eigen::Ref<const Eigen::VectorXd>& tau_ext) const;
  void RunGradientDescentOnSmallCostSamples(
      const Eigen::Ref<const Eigen::VectorXd>& tau_ext) const;

  bool RunGradientDescentFromPointOnMesh(
      const Eigen::Ref<const Eigen::VectorXd>& tau_ext,
      const size_t contact_link_idx,
      const Eigen::Ref<const Eigen::Vector3d>& p_LQ_L_initial,
      const Eigen::Ref<const Eigen::Vector3d>& normal_L_initial,
      drake::EigenPtr<Eigen::Vector3d> p_LQ_L_final,
      drake::EigenPtr<Eigen::Vector3d> normal_L_final,
      drake::EigenPtr<Eigen::Vector3d> f_L_final, double* dlduv_norm_final,
      double* l_star_final, bool is_logging, bool project_to_mesh) const;

  bool SampleLocalMinimum(const Eigen::Ref<const Eigen::VectorXd>& q,
                          const Eigen::Ref<const Eigen::VectorXd>& tau_ext,
                          const size_t contact_link_idx,
                          drake::EigenPtr<Eigen::Vector3d> p_LQ_L_final,
                          drake::EigenPtr<Eigen::Vector3d> normal_L_final,
                          drake::EigenPtr<Eigen::Vector3d> f_W_final,
                          double* dlduv_norm_final, double* l_star_final,
                          bool is_logging, bool project_to_mesh) const;

  void InitializeContactDiscriminationMsg() const;
  void PublishSamples() const;
  void PublishGradientDescentMessages() const;
  void PublishNoContactMessages() const;
  int get_max_num_small_cost_samples() const;
  int get_max_num_converged_samples() const;
  inline void set_msg_time() const;
  void HandleIiwaStatusMessage(const lcm::ReceiveBuffer* rbuf,
                               const std::string& chan,
                               const drake::lcmt_iiwa_status* msg);
  void RunLcm();

  std::vector<Eigen::Vector3d> get_points_log() const { return log_points_L_; }
  std::vector<Eigen::Vector3d> get_normals_log() const {
    return log_normals_L_;
  }
  std::vector<double> get_dlduv_norm_log() const { return log_dlduv_norm_; }
  std::vector<double> get_l_star_log() const { return log_l_star_; }
  std::tuple<std::vector<Eigen::Matrix3Xd>, std::vector<Eigen::Matrix3Xd>>
  get_mesh_samples() const {
    return {samples_L_, normals_L_};
  }
  std::vector<Eigen::VectorXd> get_samples_optimal_cost() const {
    return samples_optimal_cost_;
  }
  std::vector<size_t> get_num_line_searches() const {
    return log_num_line_searches_;
  }

  void print_num_small_cost_samples() const {
    size_t num_small_cost_samples = 0;
    size_t num_samples = 0;

    for (const auto i : config_.active_link_indices) {
      num_samples += samples_L_[i].cols();
    }

    for (const auto i : config_.active_link_indices) {
      num_small_cost_samples += small_cost_indices_[i].size();
    }

    std::cout << "Num small cost samples / all samples: "
              << num_small_cost_samples << "/" << num_samples << std::endl;
  }

 private:
  void GenerateSamples();

  const LocalMinimumSamplerConfig config_;
  std::unique_ptr<GradientCalculator> calculator_;
  std::vector<std::unique_ptr<ProximityWrapper>> p_queries_;
  std::unique_ptr<lcm::LCM> lcm_;
  std::vector<Eigen::Matrix3Xd> samples_L_;
  std::vector<Eigen::Matrix3Xd> normals_L_;

  mutable drake::lcmt_contact_discrimination msg_;

  // Rejection sampling.
  mutable std::vector<Eigen::VectorXd> samples_optimal_cost_;
  mutable std::vector<std::vector<int>> small_cost_indices_;

  // Converged samples from rejection sampling.
  mutable std::vector<std::vector<int>> converged_samples_indices_;
  mutable std::vector<std::vector<Eigen::Vector3d>> converged_samples_L_;
  mutable std::vector<std::vector<Eigen::Vector3d>>
      converged_samples_normals_L_;

  // Logging.
  mutable std::vector<Eigen::Vector3d> log_points_L_;
  mutable std::vector<Eigen::Vector3d> log_normals_L_;
  mutable std::vector<double> log_dlduv_norm_;
  mutable std::vector<double> log_l_star_;
  mutable std::vector<size_t> log_num_line_searches_;
};