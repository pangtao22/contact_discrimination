#include <vector>

#include <Eigen/Dense>
#include <osqp.h>

class OsqpWrapper {
 public:
  explicit OsqpWrapper(size_t x_size);
  void Solve(const Eigen::Ref<Eigen::MatrixXd>& P,
             const Eigen::Ref<Eigen::VectorXd>& q);
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