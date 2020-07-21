#include "local_minimum_sampler.h"

#include <fstream>
#include <iostream>

#include <lcmtypes/bot_core/double_array_t.hpp>
#include <yaml-cpp/yaml.h>

using Eigen::Vector3d;
using std::cout;
using std::endl;
using std::string;

template <class M>
M LoadCsv(const std::string& path) {
  std::ifstream indata(path);
  std::string line;
  std::vector<double> values;
  uint rows = 0;
  while (std::getline(indata, line)) {
    std::stringstream lineStream(line);
    std::string cell;
    while (std::getline(lineStream, cell, ',')) {
      values.push_back(std::stod(cell));
    }
    ++rows;
  }

  return Eigen::Map<
      const Eigen::Matrix<typename M::Scalar, M::RowsAtCompileTime,
                          M::ColsAtCompileTime, Eigen::RowMajor>>(
      values.data(), rows, values.size() / rows);
}

LocalMinimumSamplerConfig LoadLocalMinimumSamplerConfigFromYaml(
    const std::string& file_path) {
  LocalMinimumSamplerConfig config;
  config.gradient_calculator_config =
      LoadGradientCalculatorConfigFromYaml(file_path);

  YAML::Node config_yaml = YAML::LoadFile(file_path);
  config.num_links = config.gradient_calculator_config.link_names.size();
  for (int i = 0; i < config.num_links; i++) {
    config.link_mesh_paths.emplace_back(
        config_yaml["link_mesh_paths_prefix"].as<string>() + std::to_string(i) +
        config_yaml["link_mesh_paths_suffix"].as<string>());
  }
  config.active_link_indices =
      config_yaml["active_link_indices"].as<std::vector<size_t>>();
  config.num_samples_per_link =
      config_yaml["num_samples_per_link"].as<size_t>();

  config.epsilon = config_yaml["epsilon"].as<double>();

  // Rejection sampling
  const double std = config_yaml["standard_deviation"].as<double>();
  config.optimal_cost_threshold =
      -std * std * std::log(config_yaml["likelihood_threhold"].as<double>());

  // Gradient descent parameters.
  config.iterations_limit = config_yaml["iterations_limit"].as<size_t>();
  config.line_search_steps_limit =
      config_yaml["line_search_steps_limit"].as<size_t>();
  config.gradient_norm_convergence_threshold =
      config_yaml["gradient_norm_convergence_threshold"].as<double>();
  config.alpha = config_yaml["alpha"].as<double>();
  config.beta = config_yaml["beta"].as<double>();
  config.max_step_size = config_yaml["max_step_size"].as<double>();

  // Samples and normals
  config.load_samples_from_files =
      config_yaml["load_samples_from_files"].as<bool>();
  config.points_L_paths =
      config_yaml["points_L_paths"].as<std::vector<string>>();
  config.outward_normals_L_paths =
      config_yaml["normals_L_paths"].as<std::vector<string>>();

  return config;
}

LocalMinimumSampler::LocalMinimumSampler(
    const LocalMinimumSamplerConfig& config)
    : config_(config) {
  lcm_ = std::make_unique<lcm::LCM>();
  DRAKE_THROW_UNLESS(lcm_->good());

  calculator_ =
      std::make_unique<GradientCalculator>(config.gradient_calculator_config);

  for (int i = 0; i < config.num_links; i++) {
    p_queries_.push_back(nullptr);
    samples_optimal_cost_.emplace_back(Eigen::VectorXd::Constant(1, NAN));
    small_cost_indices_.emplace_back(std::vector<size_t>());
  }
  converged_samples_L_.resize(config.num_links);
  converged_samples_normals_L_.resize(config.num_links);

  for (const auto& link_idx : config.active_link_indices) {
    // Initialize proximity query.
    p_queries_[link_idx] = std::make_unique<ProximityWrapper>(
        config.link_mesh_paths[link_idx], config.epsilon);

    // Initialize data for rejection sampling.
    samples_optimal_cost_[link_idx].resize(config_.num_samples_per_link);
  }

  // Generate samples on links.
  GenerateSamples();

  InitializeContactDiscriminationMsg();

  // publish samples
  for (const auto& link_idx : config.active_link_indices) {
    bot_core::double_array_t point_array;
    point_array.num_values = samples_L_[link_idx].size();
    Eigen::Map<Eigen::ArrayXd> data(samples_L_[link_idx].data(),
                                    samples_L_[link_idx].size());
    point_array.values.resize(point_array.num_values);
    for (int i = 0; i < point_array.num_values; i++) {
      point_array.values[i] = data[i];
    }
    lcm_->publish("SAMPLES_LINK_" + std::to_string(link_idx), &point_array);
  }
}

LocalMinimumSampler::LocalMinimumSampler(const string& config_file_path)
    : LocalMinimumSampler(
          LoadLocalMinimumSamplerConfigFromYaml(config_file_path)) {}

void LocalMinimumSampler::GenerateSamples() {
  for (int i = 0; i < config_.num_links; i++) {
    samples_L_.emplace_back(Eigen::Matrix3Xd::Constant(3, 1, NAN));
    normals_L_.emplace_back(Eigen::Matrix3Xd::Constant(3, 1, NAN));
  }

  if (config_.load_samples_from_files) {
    for (const auto& link_idx : config_.active_link_indices) {
      // Load samples and normals from files.
      samples_L_[link_idx] =
          LoadCsv<Eigen::MatrixXd>(config_.points_L_paths[link_idx])
              .transpose();
      normals_L_[link_idx] =
          LoadCsv<Eigen::MatrixXd>(config_.outward_normals_L_paths[link_idx])
              .transpose();

      DRAKE_THROW_UNLESS(samples_L_[link_idx].cols() ==
                         config_.num_samples_per_link);
      DRAKE_THROW_UNLESS(normals_L_[link_idx].cols() ==
                         config_.num_samples_per_link);
    }
  } else {
    for (const auto& link_idx : config_.active_link_indices) {
      // Sample points on mesh.
      Vector3d point_L;
      size_t triangle_idx;
      samples_L_[link_idx].resize(3, config_.num_samples_per_link);
      normals_L_[link_idx].resize(3, config_.num_samples_per_link);

      for (size_t i = 0; i < config_.num_samples_per_link; i++) {
        p_queries_[link_idx]->get_mesh().SamplePointOnMesh(&point_L,
                                                           &triangle_idx);
        samples_L_[link_idx].col(i) = point_L;
        normals_L_[link_idx].col(i) =
            p_queries_[link_idx]->get_mesh().CalcFaceNormal(triangle_idx);
      }
    }
  }
}

void LocalMinimumSampler::ComputeOptimalCostForSamples(
    const Eigen::Ref<const Eigen::VectorXd>& tau_ext) const {
  double l_star;
  Vector3d f_L;
  for (const auto& link_idx : config_.active_link_indices) {
    small_cost_indices_[link_idx].clear();

    for (size_t i = 0; i < config_.num_samples_per_link; i++) {
      calculator_->CalcContactQp(link_idx, samples_L_[link_idx].col(i),
                                 -normals_L_[link_idx].col(i), tau_ext, &f_L,
                                 &l_star);
      samples_optimal_cost_[link_idx][i] = l_star;

      if (l_star <= config_.optimal_cost_threshold) {
        small_cost_indices_[link_idx].push_back(i);
      }
    }
  }
  //  cout << "cost threshold: " << config_.optimal_cost_threshold << endl;
  //  for (const auto& link_idx : config_.active_link_indices) {
  //    cout << "num small cost samples for link " << link_idx << " "
  //         << small_cost_indices_[link_idx].size() << endl;
  //  }
}

void LocalMinimumSampler::RunGradientDescentOnSmallCostSamples(
    const Eigen::Ref<const Eigen::VectorXd>& tau_ext) const {
  Vector3d p_LQ_L_final;
  Vector3d normal_L_final;
  Vector3d f_L_final;
  double dlduv_norm_final;
  double l_star_final;

  for (const auto& link_idx : config_.active_link_indices) {
    converged_samples_L_[link_idx].clear();
    converged_samples_normals_L_[link_idx].clear();

    for (const auto i : small_cost_indices_[link_idx]) {
      bool is_success = RunGradientDescentFromPointOnMesh(
          tau_ext, link_idx, samples_L_[link_idx].col(i),
          normals_L_[link_idx].col(i), &p_LQ_L_final, &normal_L_final,
          &f_L_final, &dlduv_norm_final, &l_star_final, false);

      if (is_success && l_star_final < samples_optimal_cost_[link_idx][i]) {
        converged_samples_L_[link_idx].push_back(p_LQ_L_final);
        converged_samples_normals_L_[link_idx].push_back(normal_L_final);
      }
    }
  }
  //  for (int i = 0; i < config_.num_links; i++) {
  //    cout << "num converged samples for link " << i << " "
  //         << converged_samples_L_[i].size() << endl;
  //  }
}

void LocalMinimumSampler::InitializeContactDiscriminationMsg() const {
  msg_.num_links = config_.active_link_indices.size();
  msg_.num_per_link = config_.num_samples_per_link;

  msg_.optimal_cost.resize(msg_.num_links);
  msg_.num_small_cost_per_link.resize(msg_.num_links);
  msg_.small_cost_indices.resize(msg_.num_links);
  msg_.num_converged_per_link.resize(msg_.num_links);
  msg_.points_L.resize(msg_.num_links);

  for (const auto& link_idx : config_.active_link_indices) {
    const auto link_idx2 = link_idx - config_.active_link_indices[0];
    msg_.optimal_cost[link_idx2].resize(msg_.num_per_link);
  }
}

void LocalMinimumSampler::PublishGradientDescentResults() const {
  msg_.max_num_small_cost_per_link = get_max_num_small_cost_samples();
  msg_.max_num_converged_per_link = get_max_num_converged_samples();

  for (const auto& link_idx : config_.active_link_indices) {
    // Optimal cost for all samples.
    const auto link_idx2 = link_idx - config_.active_link_indices[0];

    for (int i = 0; i < msg_.optimal_cost[link_idx2].size(); i++) {
      msg_.optimal_cost[link_idx2][i] = samples_optimal_cost_[link_idx][i];
    }

    // Small cost samples.
    msg_.num_small_cost_per_link[link_idx2] =
        small_cost_indices_[link_idx].size();
    msg_.small_cost_indices[link_idx2].resize(msg_.max_num_small_cost_per_link);
    for (int i = 0; i < small_cost_indices_[link_idx].size(); i++) {
      msg_.small_cost_indices[link_idx2][i] = small_cost_indices_[link_idx][i];
    }

    // Converged samples.
    msg_.num_converged_per_link[link_idx2] =
        converged_samples_L_[link_idx].size();
    msg_.points_L[link_idx2].resize(msg_.max_num_converged_per_link);

    for (int i = 0; i < msg_.max_num_converged_per_link; i++) {
      msg_.points_L[link_idx2][i].resize(3);
    }

    for (int i = 0; i < converged_samples_L_[link_idx].size(); i++) {
      const Vector3d& p = converged_samples_L_[link_idx][i];
      msg_.points_L[link_idx2][i][0] = p[0];
      msg_.points_L[link_idx2][i][1] = p[1];
      msg_.points_L[link_idx2][i][2] = p[2];
    }
  }
  lcm_->publish("CONTACT_DISCRIMINATION", &msg_);
}

int LocalMinimumSampler::get_max_num_small_cost_samples() const {
  int n_max = 0;
  for (const auto& a : small_cost_indices_) {
    int n = a.size();
    //    cout << "haha " << n << endl;
    if (n_max < n) {
      n_max = n;
    }
  }
  return n_max;
}

int LocalMinimumSampler::get_max_num_converged_samples() const {
  int n_max = 0;
  //  cout << "huh? " << converged_samples_L_.size() << endl;
  for (const auto& a : converged_samples_L_) {
    int n = a.size();
    //    cout << "hehe " << n << endl;
    if (n_max < n) {
      n_max = n;
    }
  }
  return n_max;
}

bool LocalMinimumSampler::RunGradientDescentFromPointOnMesh(
    const Eigen::Ref<const Eigen::VectorXd>& tau_ext,
    const size_t contact_link_idx,
    const Eigen::Ref<const Eigen::Vector3d>& p_LQ_L_initial,
    const Eigen::Ref<const Eigen::Vector3d>& normal_L_initial,
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

  while (iter_count < config_.iterations_limit) {
    if (!calculator_->CalcDlDp(contact_link_idx, p_LQ_L, -normal_L, tau_ext,
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
      if (!calculator_->CalcContactQp(contact_link_idx, p_LQ_L - t * dlduv,
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
    const size_t contact_link_idx, drake::EigenPtr<Vector3d> p_LQ_L_final,
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
  UpdateJacobians(q);
  return RunGradientDescentFromPointOnMesh(
      tau_ext, contact_link_idx, p_LQ_L, normal_L, p_LQ_L_final, normal_L_final,
      f_W_final, dlduv_norm_final, l_star_final, is_logging);
}
