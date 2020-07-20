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
  const size_t contact_link_idx = 5;

  lm_sampler.UpdateJacobians(q);
  Vector3d p_LQ_L_final;
  Vector3d normal_L_final;
  Vector3d f_W_final;
  double dlduv_norm_final;
  double l_star_final;

  lm_sampler.ComputeOptimalCostForSamples(tau_ext);

  return 0;
}