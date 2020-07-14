#pragma once

#include <vector>

#include <Eigen/Dense>
#include <osqp.h>
#include <drake/common/eigen_types.h>

constexpr size_t kNumRays = 4;

class OsqpWrapper {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(OsqpWrapper);
  explicit OsqpWrapper(size_t x_size);
  bool SolveGradient(const Eigen::Ref<Eigen::MatrixXd> &Q,
                     const Eigen::Ref<Eigen::VectorXd> &b,
                     drake::EigenPtr<Eigen::VectorXd> x_star,
                     double *l_star,
                     drake::EigenPtr<Eigen::MatrixXd> dlDQ,
                     drake::EigenPtr<Eigen::VectorXd> dldb) const;
  bool Solve(const Eigen::Ref<Eigen::MatrixXd> &Q,
             const Eigen::Ref<Eigen::VectorXd> &b,
             drake::EigenPtr<Eigen::VectorXd> x_star,
             double *l_star) const;
  ~OsqpWrapper();

 private:
  void UpdateQpParameters(const Eigen::Ref<Eigen::MatrixXd>& Q,
                          const Eigen::Ref<Eigen::VectorXd>& b) const;

  const size_t num_vars_;
  OSQPWorkspace* work_{nullptr};
  OSQPSettings* settings_{nullptr};
  OSQPData* data_{nullptr};

  const c_int P_nnz_{};
  c_int* P_i_{nullptr};
  c_int* P_p_{nullptr};

  c_float* A_x_{nullptr};
  const c_int A_nnz_{};
  c_int* A_i_{nullptr};
  c_int* A_p_{nullptr};

  c_float* l_{nullptr};
  c_float* u_{nullptr};

  mutable std::vector<c_float> P_x_{};
  mutable std::vector<c_float> q_{};
};
