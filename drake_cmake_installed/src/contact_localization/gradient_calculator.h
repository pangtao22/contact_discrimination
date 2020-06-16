#pragma once

#include <vector>

#include <drake/multibody/plant/multibody_plant.h>
#include <drake/solvers/osqp_solver.h>

class GradientCalculator {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(GradientCalculator);
  explicit GradientCalculator(const std::string& robot_sdf_path);
  void CalcKinematicsFromContext(size_t contact_link, Eigen::Vector3d p_LoQ_L);

 private:
  std::unique_ptr<drake::multibody::MultibodyPlant<double>> plant_;
  std::unique_ptr<drake::multibody::MultibodyPlant<drake::AutoDiffXd>>
      plant_ad_;
  drake::multibody::ModelInstanceIndex robot_model_;
  std::vector<drake::multibody::FrameIndex> frame_indices_;
  mutable std::unique_ptr<drake::systems::Context<double>> plant_context_;
  mutable Eigen::MatrixXd Jc_;
  mutable drake::math::RigidTransformd X_WL_;
  std::unique_ptr<drake::solvers::OsqpSolver> solver_;
};

const char kIiwaSdf[] =
    "drake/manipulation/models/iiwa_description/iiwa7/"
    "iiwa7_no_collision.sdf";

const char kPlanarArmSdf[] =
    "/Users/pangtao/PycharmProjects/contact_aware_control/plan_runner/models/"
    "three_link_arm.sdf";
