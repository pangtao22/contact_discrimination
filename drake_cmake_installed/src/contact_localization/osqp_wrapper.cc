#include <iostream>
#include <numeric>

#include "osqp_wrapper.h"

using std::cout;
using std::endl;

template <class T>
void PrintArray(T* ptr, size_t size, const std::string& name = "") {
  if(!name.empty()) {
    cout << name << ": ";
  }
  for(size_t i = 0; i < size; i++) {
    cout << *(ptr + i) << " ";
  }
  cout << endl;
}

OsqpWrapper::OsqpWrapper(size_t x_size)
    : num_vars_(x_size), P_nnz_((x_size + 1) * x_size / 2), A_nnz_(x_size) {
  // Allocate memory.
  settings_ = static_cast<OSQPSettings*>(c_malloc(sizeof(OSQPSettings)));
  data_ = static_cast<OSQPData*>(c_malloc(sizeof(OSQPData)));

  P_x_ = static_cast<c_float*>(c_malloc(sizeof(c_float) * P_nnz_));
  P_i_ = static_cast<c_int*>(c_malloc(sizeof(c_float) * P_nnz_));
  P_p_ = static_cast<c_int*>(c_malloc(sizeof(c_float) * (num_vars_ + 1)));

  A_x_ = static_cast<c_float*>(c_malloc(sizeof(c_float) * A_nnz_));
  A_i_ = static_cast<c_int*>(c_malloc(sizeof(c_float) * A_nnz_));
  A_p_ =  static_cast<c_int*>(c_malloc(sizeof(c_float) * (num_vars_ + 1)));

  q_ = static_cast<c_float*>(c_malloc(sizeof(c_float) * num_vars_));
  l_ = static_cast<c_float*>(c_malloc(sizeof(c_float) * num_vars_));
  u_ = static_cast<c_float*>(c_malloc(sizeof(c_float) * num_vars_));

  P_x_new_.resize(P_nnz_);
  q_new_.resize(num_vars_);

  // Populate matrices.
  size_t idx = 0;
  for (size_t i = 0; i < num_vars_; i++) {
    for (size_t j = i; j < num_vars_; j++) {
      P_i_[idx] = i;
      idx++;
    }
  }

  P_p_[0] = 0;
  for(size_t i = 1; i < num_vars_ + 1; i++) {
    P_p_[i] = P_p_[i-1] + i;
  }

  for(size_t i = 0; i < num_vars_; i++) {
    A_x_[i] = 1;
    A_i_[i] = i;
    A_p_[i] = i;
    l_[i] = 0;
    u_[i] = std::numeric_limits<c_float>::infinity();
  }
  A_p_[num_vars_] = num_vars_;

//    PrintArray(P_i_, P_nnz_, "P_i");
//    PrintArray(P_p_, num_vars_ + 1, "P_p");
//    PrintArray(A_x_, num_vars_, "A_x");
//    PrintArray(A_i_, num_vars_, "A_i");
//    PrintArray(A_p_, num_vars_ + 1, "A_p_");
//    PrintArray(l_, num_vars_, "l");
//    PrintArray(u_, num_vars_, "u");

  // Populate data.
  if (data_) {
    data_->n = num_vars_;
    data_->m = num_vars_;
    data_->P = csc_matrix(data_->n, data_->n, P_nnz_, P_x_, P_i_, P_p_);
    data_->q = q_;
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
  assert(osqp_setup(&work_, data_, settings_) == 0);
}

void OsqpWrapper::Solve(const Eigen::Ref<Eigen::MatrixXd>& P,
           const Eigen::Ref<Eigen::VectorXd>& q) {
  size_t idx = 0;
  for (size_t i = 0; i < num_vars_; i++) {
    for (size_t j = i; j < num_vars_; j++) {
      P_x_new_[idx] = P(i, j);
      idx++;
    }
  }

  for(size_t i = 0; i < num_vars_; i++) {
    q_new_[i] = q[i];
  }

  PrintArray(P_x_new_.data(), P_x_new_.size(), "P_new");
  PrintArray(q_new_.data(), q_new_.size(), "q");

  osqp_update_P(work_, P_x_new_.data(), OSQP_NULL, P_nnz_);
  osqp_update_lin_cost(work_, q_new_.data());

  // Solve Problem
  osqp_solve(work_);
//    cout << "Status: " << work_->info->status << endl;
//    cout << "Status val: " << work_->info->status_val << endl;
//    cout << "Runtime: " << work_->info->run_time << endl;
//    cout << "Obj: " << work_->info->obj_val << endl;
//    PrintArray(work_->solution->x, num_vars_, "x");
//    PrintArray(work_->solution->y, num_vars_, "y");
}

OsqpWrapper::~OsqpWrapper() {
  osqp_cleanup(work_);
  c_free(data_->P->x);
  c_free(data_->P->i);
  c_free(data_->P->p);
  c_free(data_->P);
  c_free(data_->A->x);
  c_free(data_->A->i);
  c_free(data_->A->p);
  c_free(data_->A);
  c_free(q_);
  c_free(l_);
  c_free(u_);
  c_free(data_);
  c_free(settings_);
}

