#include <fstream>
#include <iostream>

#include <drake/common/find_resource.h>
#include <yaml-cpp/yaml.h>

#include "gradient_calculator.h"
#include "proximity_wrapper.h"

using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;
using std::cout;
using std::endl;

const char kIiwa7Config[] =
    "/Users/pangtao/drake-external-examples/drake_cmake_installed/"
    "/src/contact_localization/iiwa_config.yml";

const char kLink6MeshPath[] =
    "/Users/pangtao/PycharmProjects/contact_aware_control"
    "/contact_particle_filter/iiwa7_shifted_meshes/link_6.obj";

int main() {
  GradientCalculator calculator(
    LoadGradientCalculatorConfigFromYaml(kIiwa7Config));

  const double epsilon = 5e-4;
  ProximityWrapper p_query(kLink6MeshPath, epsilon);

  const size_t nq = 7;
  VectorXd q(nq);
  q << -0.47839296, -0.07140746, -1.62651793, 1.37309486, 0.22398543,
      0.67425391, 2.79916161;
  const size_t contact_link_idx = 6;  // link 6
  VectorXd tau_ext(nq);
  tau_ext << 2.19172843, -2.52495118, 2.1875039, -2.28800535, 0.19415752,
      0.14975149, 0.;

  // point on z axis
  Vector3d p_LQ_L(0.0054828, -0.00315267, 0.0649383);
  Vector3d normal_L(0.0329616, 0.114744, 0.992848);

  // point on y axis
  //  Vector3d p_LQ_L(-3e-06, -0.08533, 0.011023);
  //  Vector3d normal_L(0.0336766, -0.986436, -0.160653);

  Vector3d dldp;
  double l_star;
  Vector3d dlduv;
  Vector3d f_L;
  size_t iter_count{0};

  std::vector<Vector3d> log_normals_L = {normal_L};
  std::vector<Vector3d> log_points_L = {p_LQ_L};

  calculator.UpdateJacobians(q, {5, 6});
  while (iter_count < 30) {
    auto start = std::chrono::high_resolution_clock::now();
    calculator.CalcDlDp(contact_link_idx,
                        p_LQ_L,
                        -normal_L,
                        tau_ext,
                        &dldp,
                        &f_L,
                        &l_star);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = duration_cast<std::chrono::microseconds>(end - start);

    dlduv = dldp - normal_L * dldp.dot(normal_L);
    cout << "Iteration: " << iter_count << endl;
    cout << "normal: " << normal_L.transpose() << endl;
    cout << "dldp: " << dldp.transpose() << endl;
    cout << "dlduv: " << dlduv.transpose() << endl;
    cout << "l_star: " << l_star << endl;
    cout << "Gradient time: " << duration.count() << endl;

    calculator.CalcDlDpAutoDiff(contact_link_idx,
                                p_LQ_L,
                                -normal_L,
                                tau_ext,
                                &dldp,
                                &f_L,
                                &l_star);
    cout << "dldp_autodiff: " << dldp.transpose() << endl;

    if (dlduv.norm() < 1e-3) {
      break;
    }

    start = std::chrono::high_resolution_clock::now();
    // Line search
    double alpha = 0.4;
    double beta = 0.2;
    double t = std::min(0.02 / dlduv.norm(), 1.);
    size_t line_search_steps = 0;
    double l_star_ls;  // line search

    while (true) {
      calculator.CalcContactQp(contact_link_idx, p_LQ_L - t * dlduv, -normal_L,
                               tau_ext, &f_L, &l_star_ls);
      if(l_star_ls < l_star - alpha * t * dlduv.squaredNorm()) {
        break;
      }
      t *= beta;
      line_search_steps++;

      if (line_search_steps > 10) {
        break;
      }
    }
    if (line_search_steps > 10) {
      break;
    }
    p_LQ_L += -t * dlduv;

    end = std::chrono::high_resolution_clock::now();
    duration = duration_cast<std::chrono::microseconds>(end - start);
    cout << "Line search steps: " << line_search_steps << endl;
    cout << "Step size t: " << t << endl;
    cout << "Line search time: " << duration.count() << endl << endl;

    // Project p_LQ_L back to mesh
    Vector3d p_LQ_L_mesh;
    size_t triangle_idx;
    double distance;
    p_LQ_L += normal_L * 2 * epsilon;

    start = std::chrono::high_resolution_clock::now();
    p_query.FindClosestPoint(p_LQ_L, &p_LQ_L_mesh, &normal_L, &triangle_idx,
                             &distance);
    end = std::chrono::high_resolution_clock::now();
    duration = duration_cast<std::chrono::microseconds>(end - start);

    cout << "proximity time: " << duration.count() << endl;
    cout << "p_LQ_L: " << p_LQ_L.transpose() << endl;
    cout << "p_LQ_L_mesh: " << p_LQ_L_mesh.transpose() << endl;
    cout << "new_normal_L: " << normal_L.transpose() << endl;
    cout << "tirnagle_idx: " << triangle_idx << endl;
    cout << "distance: " << distance << endl << endl;

    p_LQ_L = p_LQ_L_mesh;

    log_normals_L.push_back(normal_L);
    log_points_L.push_back(p_LQ_L);

    iter_count++;
  }
  cout << "\nFinal position: " << p_LQ_L.transpose() << endl;
  cout << "f_L: " << f_L.transpose() << endl;
  // Save logs to files
  const std::string name = "gradient_descent_on_link_6";
  std::ofstream file_points(name + "_points.csv");
  std::ofstream file_normals(name + "_normals.csv");
  const Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols,
                                  ", ", "\n");

  MatrixXd points(3, iter_count + 1);
  MatrixXd normals(3, iter_count + 1);
  for (size_t i = 0; i < iter_count + 1; i++) {
    points.col(i) = log_points_L[i];
    normals.col(i) = log_normals_L[i];
  }

  file_points << points.format(CSVFormat);
  file_normals << normals.format(CSVFormat);

  file_points.close();
  file_normals.close();

  return 0;
}
