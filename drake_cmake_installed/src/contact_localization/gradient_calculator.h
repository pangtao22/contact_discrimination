#pragma once

#include <vector>

#include <drake/multibody/plant/multibody_plant.h>

#include "osqp_wrapper.h"

const char kIiwaSdf[] =
    "drake/manipulation/models/iiwa_description/iiwa7/"
    "iiwa7_no_collision.sdf";

const char kPlanarArmSdf[] =
    "/Users/pangtao/PycharmProjects/contact_aware_control/plan_runner/models/"
    "three_link_arm.sdf";

Eigen::MatrixXd CalcFrictionConeRays(
    const Eigen::Ref<const Eigen::Vector3d>& normal, double mu, size_t nd);

class GradientCalculator {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(GradientCalculator);
  explicit GradientCalculator(const std::string& robot_sdf_path);
  void CalcFrictionConeRaysWorld(const Eigen::Ref<const Eigen::VectorXd>& q,
      size_t contact_link, const Eigen::Ref<const Eigen::Vector3d>& p_LoQ_L)
      const;
  Eigen::Vector3d CalcInwardNormal(size_t contact_link_idx,
      const Eigen::Ref<const Eigen::Vector3d>& p_LQ_L) const;
  void CalcDlDy(const Eigen::Ref<const Eigen::VectorXd>& q,
                    size_t contact_link_idx,
                    const Eigen::Ref<const Eigen::Vector3d>& p_LQ_L,
                    const Eigen::Ref<const Eigen::VectorXd>& tau_ext,
                    drake::EigenPtr<Eigen::Vector3d> dldy_ptr,
                    double* f_star_ptr) const;
  double CalcContactQp(const Eigen::Ref<const Eigen::VectorXd>& q,
                       size_t contact_link_idx,
                       const Eigen::Ref<const Eigen::Vector3d>& p_LQ_L,
                       const Eigen::Ref<const Eigen::VectorXd>& tau_ext) const;

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
  mutable drake::MatrixX<drake::AutoDiffXd> Jc_ad_;
  mutable Eigen::MatrixXd Jc_;
  mutable drake::math::RigidTransformd X_WL_;
  mutable Eigen::MatrixXd vC_W_;
};
