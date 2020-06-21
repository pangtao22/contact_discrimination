#include <chrono>
#include <iostream>

#include <yaml-cpp/yaml.h>

#include "gradient_calculator.h"

using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using std::cout;
using std::endl;

const char k3linkArmConfig[] =
    "/Users/pangtao/PycharmProjects/contact_aware_control"
    "/contact_discrimination/config_3link_arm.yml";

int main() {
  YAML::Node config = YAML::LoadFile(k3linkArmConfig);

  GradientCalculator calculator(kPlanarArmSdf,
      config["model_instance_name"].as<std::string>(),
      config["link_names"].as<std::vector<std::string>>(),
      config["num_friction_cone_rays"].as<size_t>());

  const size_t nq = 3;
  VectorXd q(nq);
  q << M_PI / 2 * 1.2, -M_PI / 2 * 0.7, -M_PI / 2;
  const size_t contact_link_idx = 3;
  VectorXd tau_ext(nq);
  tau_ext << -0.835809, 0.424264, 0.424264;

  Vector3d p_LQ_L(0, 0.2, -0.05);
  Vector3d normal_L(0, 0, -1); // outward normal
  Vector3d dldy;
  dldy.setZero();
  Vector3d dldp;
  double fy;

  size_t iter_count{0};
  while (true) {
    auto start = std::chrono::high_resolution_clock::now();
    calculator.CalcDlDp(q, contact_link_idx, p_LQ_L, -normal_L, tau_ext, &dldp,
        &fy);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = duration_cast<std::chrono::microseconds>(end - start);

    dldy[1] = dldp[1];
    cout << "Iteration: " << iter_count << endl;
    cout << "dldy: " << dldy.transpose() << endl;
    cout << "dldp: " << dldp.transpose() << endl;
    cout << "fy: " << fy << endl;
    cout << "Gradient time: " << duration.count() << endl;
    if (dldy.norm() < 1e-4) {
      break;
    }

    start = std::chrono::high_resolution_clock::now();
    // Line search
    double alpha = 0.4;
    double beta = 0.9;
    double t = 1;
    size_t line_search_steps = 0;
    while (calculator.CalcContactQp(q, contact_link_idx, p_LQ_L - t * dldy,
                                     -normal_L, tau_ext) >
           fy - alpha * t * dldy.squaredNorm()) {
      t *= beta;
      line_search_steps++;
    }
    p_LQ_L += -t * dldy;
    end = std::chrono::high_resolution_clock::now();
    duration = duration_cast<std::chrono::microseconds>(end - start);
    cout << "Line search steps: " << line_search_steps << endl;
    cout << "Line search time: " << duration.count() << endl << endl;

    iter_count++;
  }
  cout << "Final position: " << p_LQ_L.transpose() << endl;

  return 0;
}