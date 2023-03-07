#include "osqp_wrapper.h"

#include <iostream>
#include <numeric>

using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;

template <class T>
void PrintArray(T* ptr, size_t size, const std::string& name = "") {
  if (!name.empty()) {
    cout << name << ": ";
  }
  for (size_t i = 0; i < size; i++) {
    cout << *(ptr + i) << " ";
  }
  cout << endl;
}

OsqpWrapper::OsqpWrapper(size_t x_size)
    : num_vars_(x_size), P_nnz_((x_size + 1) * x_size / 2), A_nnz_(x_size) {
  if(num_vars_ != kNumRays) {
    throw std::runtime_error("QP problem size and kNumRays do not match.");
  }

  // Allocate memory.
  settings_ = static_cast<OSQPSettings*>(c_malloc(sizeof(OSQPSettings)));
  data_ = static_cast<OSQPData*>(c_malloc(sizeof(OSQPData)));

  P_i_ = static_cast<c_int*>(c_malloc(sizeof(c_float) * P_nnz_));
  P_p_ = static_cast<c_int*>(c_malloc(sizeof(c_float) * (num_vars_ + 1)));

  A_x_ = static_cast<c_float*>(c_malloc(sizeof(c_float) * A_nnz_));
  A_i_ = static_cast<c_int*>(c_malloc(sizeof(c_float) * A_nnz_));
  A_p_ = static_cast<c_int*>(c_malloc(sizeof(c_float) * (num_vars_ + 1)));

  l_ = static_cast<c_float*>(c_malloc(sizeof(c_float) * num_vars_));
  u_ = static_cast<c_float*>(c_malloc(sizeof(c_float) * num_vars_));

  P_x_.resize(P_nnz_);
  q_.resize(num_vars_);

  // Initialize P_x_ and q_ so that the problem is convex.
  MatrixXd Q(num_vars_, num_vars_);
  Q.setIdentity();
  VectorXd b(num_vars_);
  b.setZero();
  UpdateQpParameters(Q, b);

  // Populate matrices.
  size_t idx = 0;
  for (size_t j = 0; j < num_vars_; j++) {
    for (size_t i = 0; i <= j; i++) {
      P_i_[idx] = i;
      idx++;
    }
  }

  P_p_[0] = 0;
  for (size_t i = 1; i < num_vars_ + 1; i++) {
    P_p_[i] = P_p_[i - 1] + i;
  }

  for (size_t i = 0; i < num_vars_; i++) {
    A_x_[i] = 1;
    A_i_[i] = i;
    A_p_[i] = i;
    l_[i] = 0;
    u_[i] = std::numeric_limits<c_float>::infinity();
  }
  A_p_[num_vars_] = num_vars_;

  // Populate data.
  if (data_) {
    data_->n = num_vars_;
    data_->m = num_vars_;
    data_->P = csc_matrix(data_->n, data_->n, P_nnz_, P_x_.data(), P_i_, P_p_);
    data_->q = q_.data();
    data_->A = csc_matrix(data_->m, data_->n, A_nnz_, A_x_, A_i_, A_p_);
    data_->l = l_;
    data_->u = u_;
  }

  // Define solver settings as default
  if (settings_) {
    osqp_set_default_settings(settings_);
    settings_->alpha = 1.0;  // Change alpha parameter
    settings_->verbose = false;
  }

  // Setup workspace
  auto setup_status = osqp_setup(&work_, data_, settings_);
  if (setup_status != 0) {
    throw std::runtime_error("osqp workspace not setup properly.");
  }

  //
  G_.setIdentity();
  G_ *= -1;
}

void OsqpWrapper::UpdateQpParameters(const Eigen::Ref<Eigen::MatrixXd>& Q,
                                     const Eigen::Ref<Eigen::VectorXd>& b)
                                     const {
  size_t idx = 0;
  for (size_t j = 0; j < num_vars_; j++) {
    for (size_t i = 0; i <= j; i++)  {
      P_x_[idx] = Q(i, j);
      idx++;
    }
  }

  for (size_t i = 0; i < num_vars_; i++) {
    q_[i] = b[i];
  }
}

bool OsqpWrapper::Solve(const Eigen::Ref<Eigen::MatrixXd> &Q,
                        const Eigen::Ref<Eigen::VectorXd> &b,
                        drake::EigenPtr<Eigen::VectorXd> x_star,
                        double *l_star) const {
  // Update data.
  UpdateQpParameters(Q, b);
  osqp_update_P(work_, P_x_.data(), OSQP_NULL, P_nnz_);
  osqp_update_lin_cost(work_, q_.data());

  // Solve Problem
  osqp_solve(work_);

  // Extract solution.
  *x_star = Map<VectorXd>(work_->solution->x, work_->data->n);
  *l_star = work_->info->obj_val;
  return work_->info->status_val == 1;
}

bool OsqpWrapper::SolveGradient(const Eigen::Ref<Eigen::MatrixXd> &Q,
                                const Eigen::Ref<Eigen::VectorXd> &b,
                                drake::EigenPtr<Eigen::VectorXd> x_star,
                                double *l_star,
                                drake::EigenPtr<Eigen::MatrixXd> dlDQ,
                                drake::EigenPtr<Eigen::VectorXd> dldb) const {
  // Solve problem
  if(!Solve(Q, b, x_star, l_star)) {
    return false;
  }

  const size_t nx = work_->data->n;
  const size_t n_lambda = work_->data->m;
  if(nx != num_vars_ || n_lambda != num_vars_) {
    throw std::runtime_error("nx != num_vars_ || n_lambda != num_vars_");
  }

  // Gradient
  for (size_t i = 0; i < n_lambda; i++) {
    double yi = (work_->solution->y)[i];
    lambda_star_[i] = -(yi < 0 ? yi : 0);
  }

  a0_.setZero();
  if (lambda_star_.norm() > 1e-6) {
    // When the inequlaity constraints are tight.
    A_inverse_.setZero();
    h_.setZero();

    A_inverse_.topLeftCorner(nx, nx) = Q;
    A_inverse_.topRightCorner(nx, n_lambda) = G_.transpose();
    A_inverse_.bottomLeftCorner(n_lambda, nx) = lambda_star_.asDiagonal() * G_;
    A_inverse_.bottomRightCorner(n_lambda, n_lambda).diagonal() =
        G_ * (*x_star) - h_;

    A_ = A_inverse_.inverse();
    const MatrixXd& A11 = A_.topLeftCorner(nx, nx);
    a0_ = A11.transpose() * (b + Q * (*x_star));
    //    cout << "large multipliers\n" << endl;
  }

  a1_ = 0.5 * x_star->transpose() - a0_.transpose();
  *dlDQ = a1_.transpose() * x_star->transpose();
  *dldb = -a0_ + *x_star;

  //  cout << "A_inverse\n" << A_inverse << endl;
  //  cout << "A\n" << A << endl;
  //  cout << "A11\n" << A11 << endl;
  //  cout << "a1\n" << a1 << endl;
  return true;
}

OsqpWrapper::~OsqpWrapper() {
  osqp_cleanup(work_);
//  c_free(data_->P->x);
  c_free(data_->P->i);
  c_free(data_->P->p);
  c_free(data_->P);
  c_free(data_->A->x);
  c_free(data_->A->i);
  c_free(data_->A->p);
  c_free(data_->A);
  c_free(l_);
  c_free(u_);
  c_free(data_);
  c_free(settings_);
}
