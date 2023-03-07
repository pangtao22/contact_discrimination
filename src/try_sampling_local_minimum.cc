#include <fstream>
#include <iostream>

#include <drake/common/find_resource.h>

#include "local_minimum_sampler.h"

using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using std::cout;
using std::endl;

const char kIiwa7Config[] =
    "/Users/pangtao/drake-external-examples/drake_cmake_installed/"
    "/src/contact_localization/iiwa_config.yml";

int main() {
  LocalMinimumSampler lm_sampler(kIiwa7Config);

  const size_t nq = 7;
  VectorXd q(nq);
  q << -1.75956086,  0.74755266,  2.72834054,  2.05278647,  2.3270361 ,
      -1.27883698,  0.61326711;
  VectorXd tau_ext(nq);
  tau_ext << 9.66613398e-01, -2.82202806e+00, -1.60744425e-03,  6.62280818e-01,
      3.19086728e-02,  8.17417009e-02,  0;

  Vector3d p_LQ_L_final;
  Vector3d normal_L_final;
  Vector3d f_W_final;

  lm_sampler.UpdateJacobians(q);
  auto start = std::chrono::high_resolution_clock::now();
  lm_sampler.ComputeOptimalCostForSamples(tau_ext);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = duration_cast<std::chrono::microseconds>(end - start);
  cout << "QP time for all samples: " << duration.count() << endl;

  lm_sampler.print_num_small_cost_samples();

  start = std::chrono::high_resolution_clock::now();
  lm_sampler.RunGradientDescentOnSmallCostSamples(tau_ext);
  end = std::chrono::high_resolution_clock::now();
  duration = duration_cast<std::chrono::microseconds>(end - start);
  cout << "Gradient descent time for small-cost samples: "
       << duration.count() << endl;

  start = std::chrono::high_resolution_clock::now();
  lm_sampler.PublishGradientDescentMessages();
  end = std::chrono::high_resolution_clock::now();
  duration = duration_cast<std::chrono::microseconds>(end - start);
  cout << "publish time: " << duration.count() << endl;

  return 0;
}