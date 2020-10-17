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
  lm_sampler.RunLcm();

  return 0;
}