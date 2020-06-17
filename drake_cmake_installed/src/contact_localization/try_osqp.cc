#include <iostream>

#include "osqp_wrapper.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;

int main(int argc, char **argv) {
  const size_t num_vars = 2;
  auto qp_solver = OsqpWrapper(num_vars);

  Eigen::MatrixXd P(num_vars, num_vars);
  P << 2.5, 0.5, 0.5, 0.5;
  Eigen::VectorXd b(num_vars);
  b << -3.53553391, -0.70710678;

  VectorXd x_star(num_vars), dldb(num_vars);
  MatrixXd dldQ(num_vars, num_vars);

  qp_solver.Solve(P, b, &x_star, &dldQ, &dldb);
  cout << "x_star\n" << x_star << endl;
  cout << "dldb\n" << dldb << endl;
  cout << "dldQ\n" << dldQ << endl;

  return 0;
}