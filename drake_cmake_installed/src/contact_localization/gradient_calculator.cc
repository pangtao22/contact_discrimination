#include "gradient_calculator.h"

#include "drake/multibody/parsing/parser.h"
//#include <drake/manipulation/robot_plan_runner/robot_plans/plan_utilities.h>

using drake::multibody::MultibodyPlant;

GradientCalculator::GradientCalculator(const std::string& robot_sdf_path) {
  solver_ = std::make_unique<drake::solvers::OsqpSolver>> ();
  plant_ = std::make_unique<MultibodyPlant<double>>(1e-3);

  drake::multibody::Parser parser(plant_.get());
  parser.AddModelFromFile(robot_sdf_path);
  plant_->WeldFrames(plant_->world_frame(), plant_->GetFrameByName("link_0"));
  plant_->mutable_gravity_field().set_gravity_vector({0, 0, 0});
  plant_->Finalize();

  plant_ad_ = drake::systems::System<double>::ToAutoDiffXd(*plant_);
  robot_model_ = plant_->GetModelInstanceByName("three_link_arm");
  std::vector<std::string> link_names{"link_0", "link_1", "link_2", "link_ee"};
  for (const auto& name : link_names) {
    frame_indices_.emplace_back(plant_->GetFrameByName(name).index());
  }
  plant_context_ = plant_->CreateDefaultContext();

  Jc_.resize(6, plant_->num_actuated_dofs(robot_model_));
}

void GradientCalculator::CalcKinematicsFromContext(size_t contact_link,
                                                   Eigen::Vector3d p_LoQ_L) {
  const auto& contact_frame_idx = frame_indices_[contact_link];

  plant_->CalcJacobianSpatialVelocity(
      *plant_context_, drake::multibody::JacobianWrtVariable::kQDot,
      plant_->get_frame(contact_frame_idx), p_LoQ_L,
      plant_->world_frame(), plant_->world_frame(), &Jc_);

  X_WL_ = plant_->CalcRelativeTransform(*plant_context_, plant_->world_frame(),
                                    plant_->get_frame(contact_frame_idx));

}
