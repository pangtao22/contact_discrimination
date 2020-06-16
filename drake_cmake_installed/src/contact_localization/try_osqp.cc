#include <iostream>

#include "osqp_wrapper.h"


int main(int argc, char **argv) {
  const size_t num_vars = 2;
  auto qp_solver = OsqpWrapper(num_vars);

  Eigen::MatrixXd P(num_vars, num_vars);
  P << 2.005, 0.005, 0.005, 0.005;
  Eigen::VectorXd q(num_vars);
  q << -2.83549819e+00, -2.07106781e-03;

  qp_solver.Solve(P, q);

  return 0;
}