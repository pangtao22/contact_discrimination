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


int main() {
  YAML::Node config = YAML::LoadFile(kIiwa7Config);

  std::vector<std::string> active_link_mesh_paths;
  for(int i = 0; i < 7; i++) {
    active_link_mesh_paths.emplace_back(
        "/Users/pangtao/PycharmProjects/contact_aware_control"
        "/contact_particle_filter/iiwa7_shifted_meshes/link_" +
        std::to_string(i) + ".obj");
  }

  LocalMinimumSampler lm_sampler(
      drake::FindResourceOrThrow(kIiwaSdf),
      config["model_instance_name"].as<std::string>(),
      config["link_names"].as<std::vector<std::string>>(),
      active_link_mesh_paths,
      {5, 6},
      config["num_friction_cone_rays"].as<size_t>(),
      5e-4);

  const size_t nq = 7;
  VectorXd q(nq);
  q << -0.72660612, -1.91476416, -0.88434094, 0.56057658, 0.09017497,
      0.77990924, -2.50260755;
  VectorXd tau_ext(nq);
  tau_ext << 4.58945169, 2.29842181, 0.66350414, -1.72077332, 0.23138661, 0.,
      0.;
  const size_t contact_link_idx = 5;

  const size_t iteration_limit = 50;
  Vector3d p_LQ_L_final;
  Vector3d normal_L_final;
  Vector3d f_W_final;
  double dlduv_norm_final;
  double l_star_final;

  //  Vector3d p_LQ_L_initial(-0.05276329, -0.04265036,  0.03206141);
  //  Vector3d normal_L_initial(-0.76607425, -0.64275209,  0.        );
  //
  //  lm_sampler.RunGradientDescentFromPointOnMesh(q, tau_ext, contact_link_idx,
  //      iteration_limit, p_LQ_L_initial, normal_L_initial,
  //                                  &p_LQ_L_final, &normal_L_final,
  //                                  &f_W_final, &dlduv_norm_final,
  //                                  &l_star_final, false);
  //  cout << l_star_final << " " << dlduv_norm_final << endl;
  //  cout << "f_W_final: " << f_W_final.transpose() << endl;
  //  cout << "p_LQ_L_final: " << p_LQ_L_final.transpose() << endl;
  //  cout << "normal_L_final: " << normal_L_final.transpose() << endl;
  //  cout << "dot: " << f_W_final.normalized().dot(-normal_L_final) << endl;

  std::vector<double> l_star_final_log;
  std::vector<Vector3d> p_LQ_L_final_log;

  for (int i = 0; i < 100; i++) {
    dlduv_norm_final = 1e6;
    bool is_successful = lm_sampler.SampleLocalMinimum(
        q, tau_ext, contact_link_idx, iteration_limit, &p_LQ_L_final,
        &normal_L_final, &f_W_final, &dlduv_norm_final, &l_star_final, false);
    if (is_successful) {
      cout << i << ": " << l_star_final << " " << dlduv_norm_final << " " <<
          p_LQ_L_final.transpose() << endl;
      l_star_final_log.push_back(l_star_final);
      p_LQ_L_final_log.push_back(p_LQ_L_final);
    } else {
      cout << i << ": "
           << "did not converge." << endl;
    }
  }

  // IO
  MatrixXd local_minima(3, l_star_final_log.size());
  VectorXd optimal_values(l_star_final_log.size());
  for (size_t i = 0; i < l_star_final_log.size(); i++) {
    local_minima.col(i) = p_LQ_L_final_log[i];
    optimal_values[i] = l_star_final_log[i];
  }

  const std::string name = "link_local_minima";
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