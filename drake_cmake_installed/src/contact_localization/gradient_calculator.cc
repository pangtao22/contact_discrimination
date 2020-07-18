#include "gradient_calculator.h"

#include <cmath>

#include <drake/common/find_resource.h>
#include <drake/math/autodiff.h>
#include <drake/math/autodiff_gradient.h>
#include <drake/multibody/parsing/parser.h>
#include <yaml-cpp/yaml.h>

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
using std::string;

GradientCalculatorConfig LoadGradientCalculatorConfigFromYaml(
    const string& file_path) {
  YAML::Node config_yaml = YAML::LoadFile(file_path);

  GradientCalculatorConfig config;
  config.robot_sdf_path =
      drake::FindResourceOrThrow(config_yaml["iiwa_sdf_path"].as<string>());
  config.model_name = config_yaml["model_instance_name"].as<string>();
  config.link_names = config_yaml["link_names"].as<std::vector<string>>();

  config.num_rays = config_yaml["num_friction_cone_rays"].as<size_t>();
  config.mu = config_yaml["friction_coefficient"].as<double>();
  return config;
}

template <class T>
inline Matrix<T, 3, 3> skew_symmetric(
    const Eigen::Ref<const Matrix<T, 3, 1>>& p) {
  Matrix<T, 3, 3> Sp;
  Sp.setZero();
  Sp(0, 1) = -p[2];
  Sp(1, 0) = p[2];
  Sp(0, 2) = p[1];
  Sp(2, 0) = -p[1];
  Sp(1, 2) = -p[0];
  Sp(2, 1) = p[0];

  return Sp;
}

GradientCalculator::GradientCalculator(const GradientCalculatorConfig& config)
    : num_rays_(config.num_rays),
      mu_(config.mu),
      qp_solver_(std::make_unique<OsqpWrapper>(num_rays_)),
      plant_(std::make_unique<MultibodyPlant<double>>(1e-3)) {
  DRAKE_THROW_UNLESS(num_rays_ == kNumRays);

  drake::multibody::Parser parser(plant_.get());
  parser.AddModelFromFile(config.robot_sdf_path);
  plant_->WeldFrames(plant_->world_frame(),
                     plant_->GetFrameByName(config.link_names[0]));
  plant_->mutable_gravity_field().set_gravity_vector({0, 0, 0});
  plant_->Finalize();
  DRAKE_THROW_UNLESS(plant_->num_positions() == kNumPositions);

  robot_model_ = plant_->GetModelInstanceByName(config.model_name);
  for (const auto& name : config.link_names) {
    frame_indices_.emplace_back(plant_->GetFrameByName(name).index());
  }
  plant_context_ = plant_->CreateDefaultContext();
}

void GradientCalculator::CalcFrictionConeRays(
    const Eigen::Ref<const Eigen::Vector3d>& normal, double mu) const {
  // tangent vectors
  if (num_rays_ == 2) {
    vC_.col(0) = Vector3d(1, 0, 0).cross(normal);
    vC_.col(1) = -vC_.col(0);
  } else {
    if (normal.head(2).norm() < 1e-6) {
      vC_.col(0) << 0, normal[2], -normal[1];
    } else {
      vC_.col(0) << normal[1], -normal[0], 0;
    }
    vC_.col(0).normalize();
    vC_.col(1) = normal.cross(vC_.col(0));
    vC_.col(2) = -vC_.col(0);
    vC_.col(3) = -vC_.col(1);
  }

  for (int i = 0; i < vC_.cols(); i++) {
    vC_.col(i) *= mu;
    vC_.col(i) += normal;
  }

  vC_ /= sqrt(1 + mu * mu);
}

Matrix<double, 3, kNumRays> GradientCalculator::CalcFrictionConeRaysWorld(
    const Eigen::Ref<const Eigen::VectorXd>& q, size_t contact_link,
    const Eigen::Ref<const Eigen::Vector3d>& p_LoQ_L,
    const Eigen::Ref<const Eigen::Vector3d>& normal_L) const {
  plant_->SetPositions(plant_context_.get(), q);
  const auto& contact_frame_idx = frame_indices_[contact_link];
  auto X_WL =
      plant_->CalcRelativeTransform(*plant_context_, plant_->world_frame(),
                                    plant_->get_frame(contact_frame_idx));
  CalcFrictionConeRays(normal_L, 1);
  return X_WL.rotation() * vC_;
}

/*
 * dQ_ij / dJv_kl
 */
double GradientCalculator::dQdJv(int i, int j, int k, int l) const {
  if (k == i && k != j) {
    return J_(j, l);
  }
  if (k != i && k == j) {
    return J_(i, l);
  }
  if (k == i && k == j) {
    return 2 * J_(k, l);
  }
  return 0;
}

void GradientCalculator::UpdateKinematics(
    const Eigen::Ref<const Eigen::VectorXd>& q, size_t contact_link_idx,
    const Eigen::Ref<const Eigen::Vector3d>& normal_L) const {
  CalcFrictionConeRays(normal_L, mu_);
  plant_->SetPositions(plant_context_.get(), q);
  const auto& contact_frame_idx = frame_indices_[contact_link_idx];
  plant_->CalcJacobianSpatialVelocity(
      *plant_context_, drake::multibody::JacobianWrtVariable::kQDot,
      plant_->get_frame(contact_frame_idx), Vector3d::Zero(),
      plant_->world_frame(), plant_->get_frame(contact_frame_idx), &J_L_);
}

bool GradientCalculator::CalcDlDpAutoDiff(
    const Eigen::Ref<const Eigen::VectorXd>& q, size_t contact_link_idx,
    const Eigen::Ref<const Eigen::Vector3d>& p_LQ_L,
    const Eigen::Ref<const Eigen::Vector3d>& normal_L,
    const Eigen::Ref<const Eigen::VectorXd>& tau_ext,
    drake::EigenPtr<Eigen::Vector3d> dldy_ptr,
    drake::EigenPtr<Eigen::Vector3d> f_L_ptr, double* l_star_ptr) const {
  UpdateKinematics(q, contact_link_idx, normal_L);

  Matrix<AutoDiff3d, 3, 1> p_LQ_L_a3d;
  for (int i = 0; i < 3; i++) {
    p_LQ_L_a3d[i].value() = p_LQ_L[i];
    p_LQ_L_a3d[i].derivatives().setZero();
    p_LQ_L_a3d[i].derivatives()[i] = 1;
  }

  auto Sp_a3d = skew_symmetric<AutoDiff3d>(p_LQ_L_a3d);

  // CalcJ
  J_a3d_ = vC_.transpose() * (-Sp_a3d * J_L_.topRows(3) + J_L_.bottomRows(3));
  Q_a3d_ = J_a3d_ * J_a3d_.transpose();  // CalcQ
  b_a3d_ = -J_a3d_ * tau_ext;            // Calcb

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

  dldp_.setZero();
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < kNumRays; j++) {
      for (size_t k = 0; k < kNumRays; k++) {
        dldp_[i] += dldQ_(j, k) * Q_a3d_(j, k).derivatives()[i];
      }
    }
    for (size_t j = 0; j < kNumRays; j++) {
      dldp_[i] += dldb_[j] * b_a3d_[j].derivatives()[i];
    }
  }

  *f_L_ptr = vC_ * x_star_;
  *dldy_ptr = dldp_;
  return true;
}

bool GradientCalculator::CalcDlDp(
    const Eigen::Ref<const Eigen::VectorXd>& q, size_t contact_link_idx,
    const Eigen::Ref<const Eigen::Vector3d>& p_LQ_L,
    const Eigen::Ref<const Eigen::Vector3d>& normal_L,
    const Eigen::Ref<const Eigen::VectorXd>& tau_ext,
    drake::EigenPtr<Eigen::Vector3d> dldy_ptr,
    drake::EigenPtr<Eigen::Vector3d> f_L_ptr, double* l_star_ptr) const {
  UpdateKinematics(q, contact_link_idx, normal_L);
  J_ = vC_.transpose() *
       (-skew_symmetric<double>(p_LQ_L) * J_L_.topRows(3) + J_L_.bottomRows(3));
  Q_ = J_ * J_.transpose();  // CalcQ
  b_ = -J_ * tau_ext;        // Calcb

  const bool is_qp_solved =
      qp_solver_->SolveGradient(Q_, b_, &x_star_, l_star_ptr, &dldQ_, &dldb_);
  if (!is_qp_solved) {
    return false;
  }
  *l_star_ptr += 0.5 * tau_ext.squaredNorm();

  dldJ_.setZero();
  for (int k = 0; k < dldJ_.rows(); k++) {
    for (int l = 0; l < dldJ_.cols(); l++) {
      for (int i = 0; i < dldQ_.rows(); i++) {
        for (int j = 0; j < dldQ_.cols(); j++) {
          dldJ_(k, l) += dldQ_(i, j) * dQdJv(i, j, k, l);
        }
      }
    }
  }

  dldJ_ += -dldb_ * tau_ext.transpose();
  dldE_ = dldJ_ * J_L_.topRows(3).transpose();
  dldSp_ = -vC_ * dldE_;
  dldp_[0] = dldSp_(2, 1) - dldSp_(1, 2);
  dldp_[1] = dldSp_(0, 2) - dldSp_(2, 0);
  dldp_[2] = dldSp_(1, 0) - dldSp_(0, 1);

  *f_L_ptr = vC_ * x_star_;
  *dldy_ptr = dldp_;
  return true;
}

bool GradientCalculator::CalcContactQp(
    const Eigen::Ref<const Eigen::VectorXd>& q, size_t contact_link_idx,
    const Eigen::Ref<const Eigen::Vector3d>& p_LQ_L,
    const Eigen::Ref<const Eigen::Vector3d>& normal_L,
    const Eigen::Ref<const Eigen::VectorXd>& tau_ext,
    drake::EigenPtr<Eigen::Vector3d> f_L, double* l_star) const {
  UpdateKinematics(q, contact_link_idx, normal_L);
  J_ = vC_.transpose() *
       (-skew_symmetric<double>(p_LQ_L) * J_L_.topRows(3) + J_L_.bottomRows(3));
  Q_ = J_ * J_.transpose();  // CalcQ
  b_ = -J_ * tau_ext;        // Calcb

  if (qp_solver_->Solve(Q_, b_, &x_star_, l_star)) {
    *f_L = vC_ * x_star_;
    *l_star += 0.5 * tau_ext.squaredNorm();
    return true;
  }
  return false;
}
