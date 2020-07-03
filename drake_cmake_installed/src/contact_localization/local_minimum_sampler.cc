
#include "local_minimum_sampler.h"

using Eigen::Vector3d;
using std::cout;
using std::endl;

LocalMinimumSampler::LocalMinimumSampler(
    const std::string& robot_sdf_path, const std::string& model_name,
    const std::vector<std::string>& link_names,
    const std::vector<std::string>& link_mesh_paths,
    const std::vector<int>& active_link_indices, int num_rays, double epsilon)
    : epsilon_(epsilon), gradient_norm_convergence_threshold_(1e-3) {
  calculator_ = std::make_unique<GradientCalculator>(robot_sdf_path, model_name,
                                                     link_names, num_rays);
  const int num_links = *(std::max_element(active_link_indices.begin(),
                                           active_link_indices.end())) +
                        1;

  for (int i = 0; i < num_links; i++) {
    p_queries_.push_back(nullptr);
  }

  for (const auto& i : active_link_indices) {
    p_queries_[i] =
        std::make_unique<ProximityWrapper>(link_mesh_paths[i], epsilon_);
  }
}

bool LocalMinimumSampler::RunGradientDescentFromPointOnMesh(
    const Eigen::Ref<const Eigen::VectorXd>& q,
    const Eigen::Ref<const Eigen::VectorXd>& tau_ext,
    const size_t contact_link_idx, const size_t iteration_limit,
    const Eigen::Ref<const Eigen::VectorXd>& p_LQ_L_initial,
    const Eigen::Ref<const Eigen::VectorXd>& normal_L_initial,
    drake::EigenPtr<Vector3d> p_LQ_L_final,
    drake::EigenPtr<Vector3d> normal_L_final,
    drake::EigenPtr<Vector3d> f_W_final, double* dlduv_norm_final,
    double* l_star_final, bool is_logging) const {
  // Initialize quantities needed for gradient descent.
  Vector3d dldp;
  double l_star;
  Vector3d dlduv = Vector3d::Constant(std::numeric_limits<double>::infinity());
  Vector3d f_W;
  Vector3d p_LQ_L = p_LQ_L_initial;
  Vector3d normal_L = normal_L_initial;
  size_t iter_count{0};

  const size_t line_search_steps_limit = 50;

  log_points_L_.clear();
  log_normals_L_.clear();
  log_dlduv_norm_.clear();
  log_l_star_.clear();

  while (iter_count < iteration_limit) {
    if (!calculator_->CalcDlDp(q, contact_link_idx, p_LQ_L, -normal_L, tau_ext,
                               &dldp, &f_W, &l_star)) {
      return false;
    }
    dlduv = dldp - normal_L * dldp.dot(normal_L);

    // Logging.
    if (is_logging) {
      log_points_L_.push_back(p_LQ_L);
      log_normals_L_.push_back(normal_L);
      log_dlduv_norm_.push_back(dlduv.norm());
      log_l_star_.push_back(l_star);
    }

    if (dlduv.norm() < gradient_norm_convergence_threshold_) {
      break;
    }

    // Line search
    double alpha = 0.4;
    double beta = 0.5;
    double t = std::min(0.02 / dlduv.norm(), 1.);
    size_t line_search_steps = 0;
    double l_star_ls;

    while (true) {
      if (!calculator_->CalcContactQp(q, contact_link_idx, p_LQ_L - t * dlduv,
                                      -normal_L, tau_ext, &f_W, &l_star_ls)) {
        return false;
      }
      if (l_star_ls < l_star - alpha * t * dlduv.squaredNorm()) {
        break;
      }
      t *= beta;
      line_search_steps++;
      if (line_search_steps > line_search_steps_limit) {
        return false;
      }
    }
    p_LQ_L += -t * dlduv;

    // Project p_LQ_L back to mesh
    Vector3d p_LQ_L_mesh;
    size_t triangle_idx;
    double distance;
    p_LQ_L += normal_L * 2 * epsilon_;
    p_queries_[contact_link_idx]->FindClosestPoint(
        p_LQ_L, &p_LQ_L_mesh, &normal_L, &triangle_idx, &distance);
    p_LQ_L = p_LQ_L_mesh;

    iter_count++;
  }

  if (dlduv.norm() < gradient_norm_convergence_threshold_) {
    *p_LQ_L_final = p_LQ_L;
    *normal_L_final = normal_L;
    *f_W_final = f_W;
    *dlduv_norm_final = dlduv.norm();
    *l_star_final = l_star;
    return true;
  }
  return false;
}

bool LocalMinimumSampler::SampleLocalMinimum(
    const Eigen::Ref<const Eigen::VectorXd>& q,
    const Eigen::Ref<const Eigen::VectorXd>& tau_ext,
    const size_t contact_link_idx, const size_t iteration_limit,
    drake::EigenPtr<Vector3d> p_LQ_L_final,
    drake::EigenPtr<Vector3d> normal_L_final,
    drake::EigenPtr<Vector3d> f_W_final, double* dlduv_norm_final,
    double* l_star_final, bool is_logging) const {
  // Sample a point on mesh and get its normal.
  Vector3d p_LQ_L;
  Vector3d normal_L;
  size_t triangle_idx;
  p_queries_[contact_link_idx]->get_mesh().SamplePointOnMesh(&p_LQ_L,
                                                             &triangle_idx);
  normal_L =
      p_queries_[contact_link_idx]->get_mesh().CalcFaceNormal(triangle_idx);

  return RunGradientDescentFromPointOnMesh(
      q, tau_ext, contact_link_idx, iteration_limit, p_LQ_L, normal_L,
      p_LQ_L_final, normal_L_final, f_W_final, dlduv_norm_final, l_star_final,
      is_logging);
}
