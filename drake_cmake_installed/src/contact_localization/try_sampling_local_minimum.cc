#include <fstream>
#include <iostream>

#include <drake/common/find_resource.h>
#include <yaml-cpp/yaml.h>

#include "local_minimum_sampler.h"

using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using std::cout;
using std::endl;

const char kIiwa7Config[] =
    "/Users/pangtao/PycharmProjects/contact_aware_control"
    "/contact_discrimination/config_iiwa7.yml";

const char kLink6MeshPath[] =
    "/Users/pangtao/PycharmProjects/contact_aware_control"
    "/contact_particle_filter/iiwa7_shifted_meshes/link_6.obj";

int main() {
  YAML::Node config = YAML::LoadFile(kIiwa7Config);

  LocalMinimumSampler lm_sampler(
      drake::FindResourceOrThrow(kIiwaSdf),
      config["model_instance_name"].as<std::string>(),
      config["link_names"].as<std::vector<std::string>>(),
      config["num_friction_cone_rays"].as<size_t>(), kLink6MeshPath, 5e-4);

  const size_t nq = 7;
  VectorXd q(nq);
  q << -0.47839296, -0.07140746, -1.62651793, 1.37309486, 0.22398543,
      0.67425391, 2.79916161;
  VectorXd tau_ext(nq);
  tau_ext << 2.19172843, -2.52495118, 2.1875039, -2.28800535, 0.19415752,
      0.14975149, 0.;
  const size_t contact_link_idx = 6;

  const size_t iteration_limit = 30;
  Vector3d p_LQ_L_final;
  Vector3d normal_L_final;
  Vector3d f_W_final;
  double dlduv_norm_final;
  double l_star_final;

  std::vector<double> l_star_final_log;
  std::vector<Vector3d> p_LQ_L_final_log;

  for (int i = 0; i < 20; i++) {
    lm_sampler.SampleLocalMinimum(q, tau_ext, contact_link_idx, iteration_limit,
                                  &p_LQ_L_final, &normal_L_final, &f_W_final,
                                  &dlduv_norm_final, &l_star_final, nullptr,
                                  nullptr);
    if (dlduv_norm_final < 1e-3) {
      cout << i << ": " << l_star_final << " " << p_LQ_L_final.transpose() <<
      endl;
      l_star_final_log.push_back(l_star_final);
      p_LQ_L_final_log.push_back(p_LQ_L_final);
    }
  }

  // IO
  MatrixXd local_minima(3, l_star_final_log.size());
  VectorXd optimal_values(l_star_final_log.size());
  for (size_t i = 0; i < l_star_final_log.size(); i++) {
    local_minima.col(i) = p_LQ_L_final_log[i];
    optimal_values[i] = l_star_final_log[i];
  }

  const std::string name = "link_6_local_minima";
  std::ofstream file_points(name + "_points.csv");
  std::ofstream file_values(name + "_values.csv");
  const Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols,
                                  ", ", "\n");
  file_points << local_minima.format(CSVFormat);
  file_values << optimal_values.format(CSVFormat);

  file_points.close();
  file_values.close();

  return 0;
}