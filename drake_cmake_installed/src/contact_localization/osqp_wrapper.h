#include <vector>

#include <Eigen/Dense>
#include <osqp.h>
#include <drake/common/eigen_types.h>

class OsqpWrapper {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(OsqpWrapper);
  explicit OsqpWrapper(size_t x_size);
  void Solve(const Eigen::Ref<Eigen::MatrixXd>& Q,
             const Eigen::Ref<Eigen::VectorXd>& b,
             drake::EigenPtr<Eigen::VectorXd> x_star,
             drake::EigenPtr<Eigen::MatrixXd> dlDQ,
             drake::EigenPtr<Eigen::VectorXd> dldb);
  ~OsqpWrapper();

 private:
  const size_t num_vars_;
  OSQPWorkspace* work_{nullptr};
  OSQPSettings* settings_{nullptr};
  OSQPData* data_{nullptr};

  c_float* P_x_{nullptr};
  const c_int P_nnz_{};
  c_int* P_i_{nullptr};
  c_int* P_p_{nullptr};

  c_float* A_x_{nullptr};
  const c_int A_nnz_{};
  c_int* A_i_{nullptr};
  c_int* A_p_{nullptr};

  c_float* q_{nullptr};
  c_float* l_{nullptr};
  c_float* u_{nullptr};

  std::vector<c_float> P_x_new_{};
  std::vector<c_float> q_new_{};
};