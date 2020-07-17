#include <iostream>
#include <chrono>

#include <Eigen/Dense>
#include <drake/math/autodiff.h>
#include <drake/math/autodiff_gradient.h>

using Eigen::MatrixXd;
using Eigen::Matrix;
using AutoDiff3d = Eigen::AutoDiffScalar<Eigen::Vector3d>;
using std::cout;
using std::endl;


int main() {
  AutoDiff3d a;
  a.derivatives().setZero();
  cout << a.value() << endl;
  cout << a.derivatives().transpose() << endl;

  Matrix<AutoDiff3d, 3, 3> A;
  Matrix<AutoDiff3d, 6, 7> J;
  A.setIdentity();
  J.setIdentity();

  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++) {
      cout << i << " " << j << ": " << A(i, j).derivatives().transpose() <<
      endl;
    }
  }


  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++) {
      cout << i << " " << j << ": " << A(i, j).derivatives().transpose() <<
                                    endl;
    }
  }

  A(2, 2) = 0;
  cout << A(0, 0).derivatives().transpose() << endl;
  cout << A(2, 2) .derivatives().transpose() << endl;
  cout << A * J.topRows(3) + J.bottomRows(3) << endl;
  return 0;
}