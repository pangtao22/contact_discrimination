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
using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;

using std::cout;
using std::endl;

Eigen::MatrixXd CalcFrictionConeRays(
    const Eigen::Ref<const Eigen::Vector3d>& normal, double mu) {
  // tangent vectors

  Eigen::Matrix<double, 3, kNumRays> dC;
  if (kNumRays == 2) {
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

  Eigen::Matrix<double, 3, kNumRays> vC;
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
  DRAKE_THROW_UNLESS(num_rays == kNumRays);

  drake::multibody::Parser parser(plant_.get());
  parser.AddModelFromFile(robot_sdf_path);
  plant_->WeldFrames(plant_->world_frame(),
                     plant_->GetFrameByName(link_names[0]));
  plant_->mutable_gravity_field().set_gravity_vector({0, 0, 0});
  plant_->Finalize();
  DRAKE_THROW_UNLESS(plant_->num_positions() == kNumPositions);

  plant_ad_ = drake::systems::System<double>::ToAutoDiffXd(*plant_);
  robot_model_ = plant_->GetModelInstanceByName(model_name);
  for (const auto& name : link_names) {
    frame_indices_.emplace_back(plant_->GetFrameByName(name).index());
  }
  plant_context_ = plant_->CreateDefaultContext();
  plant_context_ad_ = plant_ad_->CreateDefaultContext();
}

void GradientCalculator::CalcFrictionConeRaysWorld(
    const Eigen::Ref<const Eigen::VectorXd>& q, size_t contact_link,
    const Eigen::Ref<const Eigen::Vector3d>& p_LoQ_L,
    const Eigen::Ref<const Eigen::Vector3d>& normal_L) const {
  const auto& contact_frame_idx = frame_indices_[contact_link];
  plant_->SetPositions(plant_context_.get(), q);
  X_WL_ = plant_->CalcRelativeTransform(*plant_context_, plant_->world_frame(),
                                        plant_->get_frame(contact_frame_idx));
  vC_W_ = X_WL_.rotation() * CalcFrictionConeRays(normal_L, 1);
}

bool GradientCalculator::CalcDlDp(
    const Eigen::Ref<const Eigen::VectorXd>& q, size_t contact_link_idx,
    const Eigen::Ref<const Eigen::Vector3d>& p_LQ_L,
    const Eigen::Ref<const Eigen::Vector3d>& normal_L,
    const Eigen::Ref<const Eigen::VectorXd>& tau_ext,
    drake::EigenPtr<Eigen::Vector3d> dldy_ptr,
    drake::EigenPtr<Eigen::Vector3d> fW_ptr, double* l_star_ptr) const {
  CalcFrictionConeRaysWorld(q, contact_link_idx, p_LQ_L, normal_L);

  const auto& contact_frame_idx = frame_indices_[contact_link_idx];
  //---------------------------------------------------------------------------
  plant_->CalcJacobianSpatialVelocity(
      *plant_context_, drake::multibody::JacobianWrtVariable::kQDot,
      plant_->get_frame(contact_frame_idx), Vector3d::Zero(),
      plant_->world_frame(), plant_->get_frame(contact_frame_idx), &J_);

  Matrix<AutoDiff3d, 3, 1> p_LQ_L_a3d;
  for (int i = 0; i < 3; i++) {
    p_LQ_L_a3d[i].value() = p_LQ_L[i];
    p_LQ_L_a3d[i].derivatives().setZero();
    p_LQ_L_a3d[i].derivatives()[i] = 1;
  }

  Matrix<AutoDiff3d, 3, 3> Sp;
  Sp.setZero();
  Sp(0, 1) = -p_LQ_L_a3d[2];
  Sp(1, 0) = p_LQ_L_a3d[2];
  Sp(0, 2) = p_LQ_L_a3d[1];
  Sp(2, 0) = -p_LQ_L_a3d[1];
  Sp(1, 2) = -p_LQ_L_a3d[0];
  Sp(2, 1) = p_LQ_L_a3d[0];

  //  cout << "Jc_old\n" << Jc_a3d_ << endl;
  //  cout << "Jc_new\n" << Jc_a3d << endl;
  //  cout << "derivatives\n" << endl;
  //  for(int i = 0; i < Jc_a3d.rows(); i++) {
  //    for(int j = 0; j < Jc_a3d.cols(); j++) {
  //      cout << i << " " << j << ": " << endl;
  //      cout << Jc_a3d_(i, j).derivatives().transpose() << endl;
  //      cout << Jc_a3d(i, j).derivatives().transpose() << endl;
  //    }
  //  }

  //---------------------------------------------------------------------------

  Jv_a3d_ =
      vC_W_.transpose() *
      (X_WL_.rotation() * (-Sp * J_.topRows(3) + J_.bottomRows(3)));  // CalcJ
  Q_a3d_ = Jv_a3d_ * Jv_a3d_.transpose();                             // CalcQ
  b_a3d_ = -Jv_a3d_ * tau_ext;                                        // Calcb

  for (int j = 0; j < num_rays_; j++) {
    b_(j) = b_a3d_(j).value();
    for (int i = 0; i < num_rays_; i++) {
      Q_(i, j) = Q_a3d_(i, j).value();
    }
  }

  const bool is_qp_solved =
      qp_solver_->SolveGradient(Q_, b_, &x_star_, l_star_ptr, &dldQ_, &dldb_);
  if (!is_qp_solved) {
    return false;
  }
  *l_star_ptr += 0.5 * tau_ext.squaredNorm();

  Vector3d dldy = Vector3d::Zero();
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < kNumRays; j++) {
      for (size_t k = 0; k < kNumRays; k++) {
        dldy[i] += dldQ_(j, k) * Q_a3d_(j, k).derivatives()[i];
      }
    }
    for (size_t j = 0; j < kNumRays; j++) {
      dldy[i] += dldb_[j] * b_a3d_[j].derivatives()[i];
    }
  }

  *fW_ptr = vC_W_ * x_star_;
  *dldy_ptr = dldy;

  return true;
}

bool GradientCalculator::CalcContactQp(
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

  MatrixXd J = vC_W_.transpose() * J_.bottomRows(3);  // CalcJ
  MatrixXd Q = J * J.transpose();                     // CalcQ
  VectorXd b = -J * tau_ext;                          // Calcb

  VectorXd x_star(num_rays_);
  if (qp_solver_->Solve(Q, b, &x_star, l_star)) {
    *f_W = vC_W_ * x_star;
    *l_star += +0.5 * tau_ext.squaredNorm();
    return true;
  }
  return false;
}
