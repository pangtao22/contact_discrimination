
#include "local_minimum_sampler.h"

using Eigen::Vector3d;
using std::cout;
using std::endl;

LocalMinimumSampler::LocalMinimumSampler(
    const std::string& robot_sdf_path, const std::string& model_name,
    const std::vector<std::string>& link_names, int num_rays,
    const std::string& link_mesh_path, double epsilon)
    : epsilon_(epsilon) {
  calculator_ = std::make_unique<GradientCalculator>(robot_sdf_path, model_name,
                                                     link_names, num_rays);
  p_query_ = std::make_unique<ProximityWrapper>(link_mesh_path, epsilon);
}

void LocalMinimumSampler::SampleLocalMinimum(
    const Eigen::Ref<const Eigen::VectorXd>& q,
    const Eigen::Ref<const Eigen::VectorXd>& tau_ext, size_t contact_link_idx,
    const size_t iteration_limit, drake::EigenPtr<Vector3d> p_LQ_L_final,
    drake::EigenPtr<Vector3d> normal_L_final,
    drake::EigenPtr<Vector3d> f_W_final, double* dlduv_norm_final,
    double* l_star_final,
    std::vector<Vector3d>* log_points_L,
    std::vector<Vector3d>* log_normals_L) const {
  // Sample a point on mesh and get its normal.
  Vector3d p_LQ_L;
  Vector3d normal_L;
  size_t triangle_idx;
  p_query_->get_mesh().SamplePointOnMesh(&p_LQ_L, &triangle_idx);
  normal_L = p_query_->get_mesh().CalcFaceNormal(triangle_idx);

  // Initialize quantities needed for gradient descent.
  Vector3d dldp;
  double l_star;
  Vector3d dlduv = Vector3d::Constant(std::numeric_limits<double>::infinity());
  Vector3d f_W;
  size_t iter_count{0};

  if (log_points_L && log_normals_L) {
    log_points_L->clear();
    log_normals_L->clear();
    log_points_L->push_back(p_LQ_L);
    log_normals_L->push_back(normal_L);
  }

  while (iter_count < iteration_limit) {
    calculator_->CalcDlDp(q, contact_link_idx, p_LQ_L, -normal_L, tau_ext,
                          &dldp, &l_star);
    dlduv = dldp - normal_L * dldp.dot(normal_L);

//    cout << iter_count << endl;
//    cout << "dlduv: " << dlduv.transpose() << endl;

    if (dlduv.norm() < 1e-3) {
      break;
    }

    // Line search
    double alpha = 0.4;
    double beta = 0.5;
    double t = std::min(0.02 / dlduv.norm(), 1.);
    size_t line_search_steps = 0;
    double l_star_ls;

    while (true) {
      calculator_->CalcContactQp(q, contact_link_idx, p_LQ_L - t * dlduv,
                                 -normal_L, tau_ext, &f_W, &l_star_ls);
      if (l_star_ls < l_star - alpha * t * dlduv.squaredNorm()) {
        break;
      }
      t *= beta;
      line_search_steps++;
    }
    p_LQ_L += -t * dlduv;

    // Project p_LQ_L back to mesh
    Vector3d p_LQ_L_mesh;
    size_t triangle_idx;
    double distance;
    p_LQ_L += normal_L * 2 * epsilon_;
    p_query_->FindClosestPoint(p_LQ_L, &p_LQ_L_mesh, &normal_L, &triangle_idx,
                               &distance);
    p_LQ_L = p_LQ_L_mesh;

    // Logging.
    if (log_points_L && log_normals_L) {
      log_points_L->push_back(p_LQ_L);
      log_normals_L->push_back(normal_L);
    }

    iter_count++;
  }

  *p_LQ_L_final = p_LQ_L;
  *normal_L_final = normal_L;
  *f_W_final = f_W;
  *dlduv_norm_final = dlduv.norm();
  *l_star_final = l_star;
//  cout << "dlduv: " << dlduv.transpose() << endl;

}