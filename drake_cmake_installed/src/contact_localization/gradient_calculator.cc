#include "gradient_calculator.h"

#include <cmath>

#include <drake/math/autodiff.h>
#include <drake/math/autodiff_gradient.h>
#include <drake/multibody/parsing/parser.h>

using drake::AutoDiffXd;
using drake::MatrixX;
using drake::VectorX;
using drake::math::autoDiffToGradientMatrix;
using drake::math::DiscardGradient;
using drake::math::initializeAutoDiff;
using drake::multibody::MultibodyPlant;
using Eigen::ArrayXd;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;

using std::cout;
using std::endl;

Eigen::MatrixXd CalcFrictionConeRays(
    const Eigen::Ref<const Eigen::Vector3d>& normal, double mu, size_t nd) {
  DRAKE_THROW_UNLESS(nd == 2 || nd == 4);
  // tangent vectors

  Eigen::Matrix3Xd dC;
  dC.resize(3, nd);
  if (nd == 2) {
    dC.col(0) = Vector3d(1, 0, 0).cross(normal);
    dC.col(1) = -dC.col(0);
  } else {
    if (normal.head(2).norm() < 1e-6) {
      dC.col(0) << 0, normal[2], -normal[1];
    } else {
      dC.col(0) << normal[1], -normal[0], 0;
    }
    dC.col(0).normalize();
    dC.col(1) = normal.cross(dC.col(0));
    dC.col(2) = -dC.col(0);
    dC.col(3) = -dC.col(1);
  }

  Eigen::Matrix3Xd vC;
  vC.resize(3, nd);
  vC = (mu * dC).colwise() + normal;
  vC /= sqrt(1 + mu * mu);

  return vC;
}

GradientCalculator::GradientCalculator(
    const std::string& robot_sdf_path, const std::string& model_name,
    const std::vector<std::string>& link_names, size_t num_rays)
    : num_rays_{num_rays},
      qp_solver_(std::make_unique<OsqpWrapper>(num_rays_)),
      plant_(std::make_unique<MultibodyPlant<double>>(1e-3)) {
  drake::multibody::Parser parser(plant_.get());
  parser.AddModelFromFile(robot_sdf_path);
  plant_->WeldFrames(plant_->world_frame(),
                     plant_->GetFrameByName(link_names[0]));
  plant_->mutable_gravity_field().set_gravity_vector({0, 0, 0});
  plant_->Finalize();

  plant_ad_ = drake::systems::System<double>::ToAutoDiffXd(*plant_);
  robot_model_ = plant_->GetModelInstanceByName(model_name);
  for (const auto& name : link_names) {
    frame_indices_.emplace_back(plant_->GetFrameByName(name).index());
  }
  plant_context_ = plant_->CreateDefaultContext();
  plant_context_ad_ = plant_ad_->CreateDefaultContext();

  // Initialize autodiff variables.
  J_ad_.resize(6, plant_->num_actuated_dofs(robot_model_));
  J_.resize(J_ad_.rows(), J_ad_.cols());
  Jc_a3d_.resize(3, J_ad_.cols());
  Jv_a3d_.resize(num_rays_, J_ad_.cols());

  vC_W_.resize(3, num_rays_);
  vC_W_a3d_.resize(vC_W_.rows(), vC_W_.cols());
  for (size_t i = 0; i < vC_W_a3d_.rows(); i++) {
    for (size_t j = 0; j < vC_W_a3d_.cols(); j++) {
      vC_W_a3d_(i, j).derivatives().setZero();
    }
  }

  dQdy_.resize(num_rays_ * num_rays_, 3);
  dbdy_.resize(num_rays_, 3);
  dldQ_.resize(num_rays_, num_rays_);
  dldb_.resize(num_rays_);

  tau_ext_a3d_.resize(plant_->num_positions());
  for (size_t i = 0; i < tau_ext_a3d_.size(); i++) {
    tau_ext_a3d_(i).derivatives().setZero();
  }

  Q_.resize(num_rays_, num_rays_);
  b_.resize(num_rays_);
  Q_a3d_.resize(num_rays_, num_rays_);
  b_a3d_.resize(num_rays_);
}

void GradientCalculator::CalcFrictionConeRaysWorld(
    const Eigen::Ref<const Eigen::VectorXd>& q, size_t contact_link,
    const Eigen::Ref<const Eigen::Vector3d>& p_LoQ_L,
    const Eigen::Ref<const Eigen::Vector3d>& normal_L) const {
  const auto& contact_frame_idx = frame_indices_[contact_link];
  plant_->SetPositions(plant_context_.get(), q);
  X_WL_ = plant_->CalcRelativeTransform(*plant_context_, plant_->world_frame(),
                                        plant_->get_frame(contact_frame_idx));
  vC_W_ = X_WL_.rotation() * CalcFrictionConeRays(normal_L, 1, num_rays_);
  for (size_t i = 0; i < vC_W_a3d_.rows(); i++) {
    for (size_t j = 0; j < vC_W_a3d_.cols(); j++) {
      vC_W_a3d_(i, j).value() = vC_W_(i, j);
    }
  }
}

void GradientCalculator::UpdateA3DVariables(
    const Eigen::Ref<const Eigen::VectorXd>& tau_ext) const {
  for (size_t i = 0; i < Jc_a3d_.rows(); i++) {
    for (size_t j = 0; j < Jc_a3d_.cols(); j++) {
      Jc_a3d_(i, j).value() = J_ad_(i + 3, j).value();
      int d_size = J_ad_(i + 3, j).derivatives().size();
      assert(d_size == 0 || d_size == 3);
      if (d_size == 3) {
        Jc_a3d_(i, j).derivatives() = J_ad_(i + 3, j).derivatives();
      } else {
        Jc_a3d_(i, j).derivatives().setZero();
      }
    }
  }

  for (size_t i = 0; i < tau_ext.size(); i++) {
    tau_ext_a3d_[i].value() = tau_ext[i];
  }
}

void GradientCalculator::CalcDlDp(
    const Eigen::Ref<const Eigen::VectorXd>& q, size_t contact_link_idx,
    const Eigen::Ref<const Eigen::Vector3d>& p_LQ_L,
    const Eigen::Ref<const Eigen::Vector3d>& normal_L,
    const Eigen::Ref<const Eigen::VectorXd>& tau_ext,
    drake::EigenPtr<Eigen::Vector3d> dldy_ptr, double* f_star_ptr) const {
  CalcFrictionConeRaysWorld(q, contact_link_idx, p_LQ_L, normal_L);

  const auto& contact_frame_idx = frame_indices_[contact_link_idx];

  auto p_LQ_L_ad = initializeAutoDiff(p_LQ_L);
  plant_ad_->SetPositions(plant_context_ad_.get(), q);
  plant_ad_->CalcJacobianSpatialVelocity(
      *plant_context_ad_, drake::multibody::JacobianWrtVariable::kQDot,
      plant_ad_->get_frame(contact_frame_idx), p_LQ_L_ad,
      plant_ad_->world_frame(), plant_ad_->world_frame(), &J_ad_);
  UpdateA3DVariables(tau_ext);

  Jv_a3d_ = vC_W_a3d_.transpose() * Jc_a3d_;  // CalcJ
  Q_a3d_ = Jv_a3d_ * Jv_a3d_.transpose();     // CalcQ
  b_a3d_ = -Jv_a3d_ * tau_ext_a3d_;           // Calcb

  for (int j = 0; j < num_rays_; j++) {
    dbdy_.row(j) = b_a3d_(j).derivatives();
    b_(j) = b_a3d_(j).value();
    for (int i = 0; i < num_rays_; i++) {
      dQdy_.row(i + j * num_rays_) = Q_a3d_(i, j).derivatives();
      Q_(i, j) = Q_a3d_(i, j).value();
    }
  }

  VectorXd x_star(num_rays_);
  *f_star_ptr = qp_solver_->SolveGradient(Q_, b_, &x_star, &dldQ_, &dldb_) +
                0.5 * tau_ext.squaredNorm();

  Eigen::Map<ArrayXd> dldQ_v(dldQ_.data(), dldQ_.size());
  Vector3d dldy = Vector3d::Zero();
  for (size_t i = 0; i < 3; i++) {
    dldy[i] += (dldQ_v * dQdy_.col(i).array()).sum();
    dldy[i] += (dldb_.array() * dbdy_.col(i).array()).sum();
  }

  *dldy_ptr = dldy;
}

void GradientCalculator::CalcContactQp(
    const Eigen::Ref<const Eigen::VectorXd>& q, size_t contact_link_idx,
    const Eigen::Ref<const Eigen::Vector3d>& p_LQ_L,
    const Eigen::Ref<const Eigen::Vector3d>& normal_L,
    const Eigen::Ref<const Eigen::VectorXd>& tau_ext,
    drake::EigenPtr<Eigen::Vector3d> f_W, double* l_star) const {
  CalcFrictionConeRaysWorld(q, contact_link_idx, p_LQ_L, normal_L);

  const auto& contact_frame_idx = frame_indices_[contact_link_idx];

  plant_->SetPositions(plant_context_.get(), q);
  plant_->CalcJacobianSpatialVelocity(
      *plant_context_, drake::multibody::JacobianWrtVariable::kQDot,
      plant_->get_frame(contact_frame_idx), p_LQ_L, plant_->world_frame(),
      plant_->world_frame(), &J_);

  // CalcJ
  MatrixXd J = vC_W_.transpose() * J_.bottomRows(3);
  MatrixXd Q = J * J.transpose();  // CalcQ
  VectorXd b = -J * tau_ext;       // Calcb

  VectorXd x_star(num_rays_);
  *l_star = qp_solver_->Solve(Q, b, &x_star) + 0.5 * tau_ext.squaredNorm();
  *f_W = vC_W_ * x_star;
}
