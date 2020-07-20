#pragma once

#include <vector>

#include <drake/multibody/plant/multibody_plant.h>

#include "osqp_wrapper.h"

using AutoDiff3d = Eigen::AutoDiffScalar<Eigen::Vector3d>;

const char kPlanarArmSdf[] =
    "/Users/pangtao/PycharmProjects/contact_aware_control/plan_runner/models/"
    "three_link_arm.sdf";

constexpr size_t kNumPositions = 7;

struct GradientCalculatorConfig {
  std::string robot_sdf_path;
  std::string model_name;
  std::vector<std::string> link_names;

  size_t num_rays{0};
  double mu{0};
};

GradientCalculatorConfig LoadGradientCalculatorConfigFromYaml(
    const std::string& file_path);


class GradientCalculator {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(GradientCalculator);
  explicit GradientCalculator(const GradientCalculatorConfig& config);
  void UpdateJacobians(
      const Eigen::Ref<const Eigen::VectorXd>& q,
      const std::vector<size_t>& active_link_indices) const;
  bool CalcDlDp(size_t contact_link_idx,
                const Eigen::Ref<const Eigen::Vector3d>& p_LQ_L,
                const Eigen::Ref<const Eigen::Vector3d>& normal_L,
                const Eigen::Ref<const Eigen::VectorXd>& tau_ext,
                drake::EigenPtr<Eigen::Vector3d> dldy_ptr,
                drake::EigenPtr<Eigen::Vector3d> f_L_ptr,
                double* l_star_ptr) const;
  bool CalcDlDpAutoDiff(size_t contact_link_idx,
                        const Eigen::Ref<const Eigen::Vector3d>& p_LQ_L,
                        const Eigen::Ref<const Eigen::Vector3d>& normal_L,
                        const Eigen::Ref<const Eigen::VectorXd>& tau_ext,
                        drake::EigenPtr<Eigen::Vector3d> dldy_ptr,
                        drake::EigenPtr<Eigen::Vector3d> f_L_ptr,
                        double* l_star_ptr) const;
  bool CalcContactQp(size_t contact_link_idx,
                     const Eigen::Ref<const Eigen::Vector3d>& p_LQ_L,
                     const Eigen::Ref<const Eigen::Vector3d>& normal_L,
                     const Eigen::Ref<const Eigen::VectorXd>& tau_ext,
                     drake::EigenPtr<Eigen::Vector3d> f_L,
                     double* l_star) const;

 private:
  void CalcFrictionConeRays(const Eigen::Ref<const Eigen::Vector3d>& normal,
                            double mu) const;
  double dQdJv(int i, int j, int k, int l) const;
  Eigen::Matrix<double, 3, kNumRays> CalcFrictionConeRaysWorld(
      const Eigen::Ref<const Eigen::VectorXd>& q, size_t contact_link,
      const Eigen::Ref<const Eigen::Vector3d>& p_LoQ_L,
      const Eigen::Ref<const Eigen::Vector3d>& normal_L) const;

  const size_t num_rays_{};
  const double mu_{};
  std::unique_ptr<drake::multibody::MultibodyPlant<double>> plant_;
  std::unique_ptr<OsqpWrapper> qp_solver_;
  drake::multibody::ModelInstanceIndex robot_model_;
  std::vector<drake::multibody::FrameIndex> frame_indices_;

  // UpdateKinematics
  mutable std::unique_ptr<drake::systems::Context<double>> plant_context_;
  mutable std::vector<Eigen::Matrix<double, 6, kNumPositions>> J_L_;
  mutable Eigen::Matrix<double, 3, kNumRays> vC_;

  // Sovling QP and its gradient
  mutable Eigen::Matrix<double, kNumRays, kNumRays> Q_;
  mutable Eigen::Matrix<double, kNumRays, 1> b_;
  mutable Eigen::Matrix<double, kNumRays, kNumRays> dldQ_;
  mutable Eigen::Matrix<double, kNumRays, 1> dldb_;
  mutable Eigen::Matrix<double, kNumRays, 1> x_star_;

  // CalcGradientAutoDiff
  mutable Eigen::Matrix<AutoDiff3d, kNumRays, kNumPositions> J_a3d_;
  mutable Eigen::Matrix<AutoDiff3d, kNumRays, kNumRays> Q_a3d_;
  mutable Eigen::Matrix<AutoDiff3d, kNumRays, 1> b_a3d_;

  // CalcGradient
  mutable Eigen::Matrix<double, kNumRays, kNumPositions> J_;
  mutable Eigen::Matrix<double, kNumRays, kNumPositions> dldJ_;
  mutable Eigen::Matrix<double, kNumRays, 3> dldE_;
  mutable Eigen::Matrix<double, 3, 1> dldp_;
  mutable Eigen::Matrix<double, 3, 3> dldSp_;
};
