#include "local_minimum_sampler.h"

#include <yaml-cpp/yaml.h>

using Eigen::Vector3d;
using std::cout;
using std::endl;
using std::string;

LocalMinimumSamplerConfig LoadLocalMinimumSamplerConfigFromYaml(
    const std::string& file_path) {
  LocalMinimumSamplerConfig config;
  config.gradient_calculator_config =
      LoadGradientCalculatorConfigFromYaml(file_path);

  YAML::Node config_yaml = YAML::LoadFile(file_path);
  config.num_links = config_yaml["num_links"].as<size_t>();
  for (int i = 0; i < config.num_links; i++) {
    config.link_mesh_paths.emplace_back(
        config_yaml["link_mesh_paths_prefix"].as<string>() + std::to_string(i) +
        config_yaml["link_mesh_paths_suffix"].as<string>());
  }
  config.active_link_indices =
      config_yaml["active_link_indices"].as<std::vector<size_t>>();

  config.epsilon = config_yaml["epsilon"].as<double>();
  config.line_search_steps_limit =
      config_yaml["line_search_steps_limit"].as<size_t>();
  config.gradient_norm_convergence_threshold =
      config_yaml["gradient_norm_convergence_threshold"].as<double>();
  config.alpha = config_yaml["alpha"].as<double>();
  config.beta = config_yaml["beta"].as<double>();
  config.max_step_size = config_yaml["max_step_size"].as<double>();

  return config;
}

LocalMinimumSampler::LocalMinimumSampler(
    const LocalMinimumSamplerConfig& config)
    : config_(config) {
  calculator_ =
      std::make_unique<GradientCalculator>(config.gradient_calculator_config);
  for (int i = 0; i < config.num_links; i++) {
    p_queries_.push_back(nullptr);
  }
  for (const auto& i : config.active_link_indices) {
    p_queries_[i] = std::make_unique<ProximityWrapper>(
        config.link_mesh_paths[i], config.epsilon);
  }
}

bool LocalMinimumSampler::RunGradientDescentFromPointOnMesh(
    const Eigen::Ref<const Eigen::VectorXd>& q,
    const Eigen::Ref<const Eigen::VectorXd>& tau_ext,
    const size_t contact_link_idx, const size_t iteration_limit,
    const Eigen::Ref<const Eigen::VectorXd>& p_LQ_L_initial,
    const Eigen::Ref<const Eigen::VectorXd>& normal_L_initial,
    drake::EigenPtr<Vector3d> p_LQ_L_final,
    drake::EigenPtr<Vector3d> normal_L_final,
    drake::EigenPtr<Vector3d> f_L_final, double* dlduv_norm_final,
    double* l_star_final, bool is_logging) const {
  // Initialize quantities needed for gradient descent.
  Vector3d dldp;
  double l_star;
  Vector3d dlduv = Vector3d::Constant(std::numeric_limits<double>::infinity());
  Vector3d f_L;
  Vector3d p_LQ_L = p_LQ_L_initial;
  Vector3d normal_L = normal_L_initial;
  size_t iter_count{0};

  log_points_L_.clear();
  log_normals_L_.clear();
  log_dlduv_norm_.clear();
  log_l_star_.clear();

  bool is_stuck = false;

  while (iter_count < iteration_limit) {
    if (!calculator_->CalcDlDp(q, contact_link_idx, p_LQ_L, -normal_L, tau_ext,
                               &dldp, &f_L, &l_star)) {
      return false;
    }
    dlduv = dldp - normal_L * dldp.dot(normal_L);

    // Logging.
    if (is_logging) {
      log_points_L_.push_back(p_LQ_L);
      log_normals_L_.push_back(normal_L);
      log_dlduv_norm_.push_back(dlduv.norm());
      log_l_star_.push_back(l_star);
    }

    if (dlduv.norm() < config_.gradient_norm_convergence_threshold) {
      break;
    }

    // Line search
    double t = std::min(config_.max_step_size / dlduv.norm(), 1.);
    size_t line_search_steps = 0;
    double l_star_ls;
    while (true) {
      if (!calculator_->CalcContactQp(q, contact_link_idx, p_LQ_L - t * dlduv,
                                      -normal_L, tau_ext, &f_L, &l_star_ls)) {
        return false;
      }
      if (l_star_ls < l_star - config_.alpha * t * dlduv.squaredNorm()) {
        break;
      }
      t *= config_.beta;
      line_search_steps++;
      if (line_search_steps > config_.line_search_steps_limit) {
        is_stuck = true;
        break;
      }
    }
    if (is_stuck) {
      break;
    }

    p_LQ_L += -t * dlduv;

    // Project p_LQ_L back to mesh
    Vector3d p_LQ_L_mesh;
    size_t triangle_idx;
    double distance;
    p_LQ_L += normal_L * 2 * config_.epsilon;
    p_queries_[contact_link_idx]->FindClosestPoint(
        p_LQ_L, &p_LQ_L_mesh, &normal_L, &triangle_idx, &distance);
    p_LQ_L = p_LQ_L_mesh;

    iter_count++;
  }

  if (dlduv.norm() < config_.gradient_norm_convergence_threshold ||
      sqrt(2 * l_star) < tau_ext.norm() * 1e-2) {
    *p_LQ_L_final = p_LQ_L;
    *normal_L_final = normal_L;
    *f_L_final = f_L;
    *dlduv_norm_final = dlduv.norm();
    *l_star_final = l_star;
    return true;
  }
  return false;
}

bool LocalMinimumSampler::SampleLocalMinimum(
    const Eigen::Ref<const Eigen::VectorXd>& q,
    const Eigen::Ref<const Eigen::VectorXd>& tau_ext,
    const size_t contact_link_idx, const size_t iteration_limit,
    drake::EigenPtr<Vector3d> p_LQ_L_final,
    drake::EigenPtr<Vector3d> normal_L_final,
    drake::EigenPtr<Vector3d> f_W_final, double* dlduv_norm_final,
    double* l_star_final, bool is_logging) const {
  // Sample a point on mesh and get its normal.
  Vector3d p_LQ_L;
  Vector3d normal_L;
  size_t triangle_idx;
  p_queries_[contact_link_idx]->get_mesh().SamplePointOnMesh(&p_LQ_L,
                                                             &triangle_idx);
  normal_L =
      p_queries_[contact_link_idx]->get_mesh().CalcFaceNormal(triangle_idx);

  return RunGradientDescentFromPointOnMesh(
      q, tau_ext, contact_link_idx, iteration_limit, p_LQ_L, normal_L,
      p_LQ_L_final, normal_L_final, f_W_final, dlduv_norm_final, l_star_final,
      is_logging);
}
