#pragma once

#include <vector>

#include <drake/multibody/plant/multibody_plant.h>

#include "osqp_wrapper.h"

using AutoDiff3d = Eigen::AutoDiffScalar<Eigen::Vector3d>;

const char kIiwaSdf[] =
    "drake/manipulation/models/iiwa_description/iiwa7/"
    "iiwa7_no_collision.sdf";

const char kPlanarArmSdf[] =
    "/Users/pangtao/PycharmProjects/contact_aware_control/plan_runner/models/"
    "three_link_arm.sdf";

constexpr size_t kNumPositions = 7;
constexpr size_t kNumRays = 4;

Eigen::MatrixXd CalcFrictionConeRays(
    const Eigen::Ref<const Eigen::Vector3d>& normal, double mu, size_t nd);

class GradientCalculator {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(GradientCalculator);
  GradientCalculator(const std::string& robot_sdf_path,
                     const std::string& model_name,
                     const std::vector<std::string>& link_names,
                     size_t num_rays);
  void CalcFrictionConeRaysWorld(
      const Eigen::Ref<const Eigen::VectorXd>& q, size_t contact_link,
      const Eigen::Ref<const Eigen::Vector3d>& p_LoQ_L,
      const Eigen::Ref<const Eigen::Vector3d>& normal_L) const;
  bool CalcDlDp(const Eigen::Ref<const Eigen::VectorXd> &q,
                size_t contact_link_idx,
                const Eigen::Ref<const Eigen::Vector3d> &p_LQ_L,
                const Eigen::Ref<const Eigen::Vector3d> &normal_L,
                const Eigen::Ref<const Eigen::VectorXd> &tau_ext,
                drake::EigenPtr<Eigen::Vector3d> dldy_ptr,
                drake::EigenPtr<Eigen::Vector3d> fW_ptr,
                double *l_star_ptr) const;
  bool CalcContactQp(const Eigen::Ref<const Eigen::VectorXd>& q,
                     size_t contact_link_idx,
                     const Eigen::Ref<const Eigen::Vector3d>& p_LQ_L,
                     const Eigen::Ref<const Eigen::Vector3d>& normal_L,
                     const Eigen::Ref<const Eigen::VectorXd>& tau_ext,
                     drake::EigenPtr<Eigen::Vector3d> f_W,
                     double* l_star) const;


 private:
  const size_t num_rays_{};
  std::unique_ptr<drake::multibody::MultibodyPlant<double>> plant_;
  std::unique_ptr<drake::multibody::MultibodyPlant<drake::AutoDiffXd>>
      plant_ad_;
  std::unique_ptr<OsqpWrapper> qp_solver_;
  drake::multibody::ModelInstanceIndex robot_model_;
  std::vector<drake::multibody::FrameIndex> frame_indices_;

  mutable std::unique_ptr<drake::systems::Context<double>> plant_context_;
  mutable std::unique_ptr<drake::systems::Context<drake::AutoDiffXd>>
      plant_context_ad_;
  mutable drake::math::RigidTransformd X_WL_;
  mutable Eigen::Matrix<AutoDiff3d, kNumRays, kNumPositions> Jv_a3d_;
  mutable Eigen::Matrix<double, 6, kNumPositions> J_;
  mutable Eigen::Matrix<double, 3, kNumRays> vC_W_;
  mutable Eigen::Matrix<double, kNumRays, kNumRays> dldQ_;
  mutable Eigen::Matrix<double, kNumRays, 1> dldb_;
  mutable Eigen::Matrix<double, kNumRays, kNumRays> Q_;
  mutable Eigen::Matrix<AutoDiff3d, kNumRays, kNumRays> Q_a3d_;
  mutable Eigen::Matrix<double, kNumRays, 1> b_;
  mutable Eigen::Matrix<AutoDiff3d, kNumRays, 1> b_a3d_;
  mutable Eigen::Matrix<double, kNumRays, 1> x_star_;
};
