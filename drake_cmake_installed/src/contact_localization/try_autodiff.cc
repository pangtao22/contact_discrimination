#include <fstream>
#include <iostream>
#include <vector>

#include <Eigen/Dense>
#include <drake/math/autodiff.h>
#include <drake/math/autodiff_gradient.h>

using Eigen::Matrix;
using Eigen::MatrixXd;
using AutoDiff3d = Eigen::AutoDiffScalar<Eigen::Vector3d>;
using std::cout;
using std::endl;

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

  return Eigen::Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime,
                                 M::ColsAtCompileTime, Eigen::RowMajor>>(
      values.data(), rows, values.size() / rows);
}

int main() {
  std::string file_path =
      "/Users/pangtao/PycharmProjects/contact_aware_control"
      "/contact_discrimination/link_6_points.txt";

  auto points_L = LoadCsv<MatrixXd>(file_path);
  cout << points_L << endl;

  return 0;
}