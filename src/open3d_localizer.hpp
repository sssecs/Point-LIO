// -----------------------------------------------------------------------------
//  open3d_localizer.hpp
//
//  Point-cloud-to-map relocalization core built on Open3D registration. This
//  code is deliberately FREE of any ROS type: it only depends on Eigen, PCL (as
//  the interchange cloud type) and Open3D, so it can be reused in any LiDAR
//  pipeline without pulling in rclcpp.
//
//  Pipeline (ported from deepglint/FAST_LIO_LOCALIZATION_HUMANOID, open3d_loc):
//    1. FPFH features + RANSAC coarse registration,
//    2. coarse -> fine multiscale ICP refinement,
//    3. EvaluateRegistration() to get the overlap fitness.
//
//  Inside the node it is used to refine the RViz /initialpose guess: the
//  current scan (base_link frame) is aligned against the loaded map and the
//  result (a refined T_map_base) seeds the EKF instead of the raw guess.
//  "success" means the pipeline ran AND the result passed the validity
//  thresholds (res.fitness >= min_fitness, translation/rotation deltas within
//  limits), i.e. the initial guess was good enough to trust the match.
// -----------------------------------------------------------------------------
#pragma once

#include <Eigen/Dense>
#include <open3d/Open3D.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <vector>

namespace point_lio {

// Tunable Open3D registration parameters (mirror the yaml localization block).
struct Open3DParams {
  // ---- Coarse (FPFH + RANSAC) + fine (multiscale ICP) registration --------
  double voxel_size          = 0.2;  // voxel used to downsample each SCAN for FPFH & ICP [m]
  double map_voxel_size      = 0.2;  // voxel used to downsample the LOADED MAP target [m].
                                     // Independent of voxel_size: raise it to build the map
                                     // target faster / with less memory.
  std::vector<double> scale  = {1.0, 4.0, 6.0};  // multiscale ICP voxel ladder (× voxel_size, coarse first)
  int    icp_method          = 1;    // 0 = point-to-point, 1 = point-to-plane, 2 = GICP
  int    icp_iteration       = 30;   // iterations per ICP level
  bool   use_fpfh            = true; // run coarse FPFH+RANSAC before ICP
  int    ransac_max_iteration = 100000; // coarse RANSAC budget. With a good initial guess a few
                                        // hundred/thousand suffice; raise if guesses are bad.
  bool   mutual_filter       = true; // FPFH correspondence mutual filtering
  bool   statistical_filter  = true; // RemoveStatisticalOutliers on scan+map
  int    filter_neighbors    = 50;   //   statistical filter neighbors
  double filter_std_ratio    = 3.0;  //   statistical filter std ratio
  // Crop the map to a cube (±crop_radius m) around the initial guess before
  // building the registration target. Huge speedup on large maps: FPFH, RANSAC
  // and ICP only see the local region. 0 disables the crop.
  double crop_radius         = 60.0;

  // ---- Validity thresholds (decide whether the initial guess was good) ----
  double min_fitness            = 0.5;   // min overlap ratio of the final transform (0..1)
  double max_translation_delta  = 5.0;   // max correction the pipeline may apply [m]
  double max_rotation_delta     = 0.785; // max correction the pipeline may apply [rad]
};

// Result of one alignment. Used to decide whether the match is trustworthy.
struct AlignResult {
  bool success = false;           // passed all validity thresholds
  double fitness = 0.0;           // overlap ratio of the final transform (0..1, higher better)
  double rmse = 0.0;              // inlier RMSE of the final transform [m]
  double translation_delta = 0.0; // |correction| applied to the guess [m]
  double rotation_delta = 0.0;    // |correction| applied to the guess [rad]
  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();  // refined T_map_base
  // Diagnostics from the coarse FPFH stage.
  double fpfh_fitness = 0.0;
  double fpfh_rmse = 0.0;
  // Timing of the last alignment [ms] — useful when tuning speed.
  double total_ms  = 0.0;  // whole align() call
  double target_ms = 0.0;  // map crop + target preprocess (downsample + normals + FPFH)
  double coarse_ms = 0.0;  // scan preprocess + FPFH+RANSAC (0 if use_fpfh is false)
  double icp_ms    = 0.0;  // multiscale ICP

  // ---- Debug diagnostics (for diagnosing far-off matches) ----
  // Cumulative distance the scan center moved from the ICP start pose after
  // EACH multiscale ICP level [m] (level 0 = coarsest). Measured as the
  // Euclidean distance between positions (the same metric as
  // translation_delta). If level 0 already shows the full jump, the coarse step
  // is the culprit; if it creeps up across levels it is a slide.
  std::vector<double> icp_level_delta;
  std::vector<double> icp_level_fitness;  // ICP fitness after each level
  size_t scan_points   = 0;  // raw scan points fed to align()
  double scan_radius   = 0.0;  // max radius of raw scan points from the sensor origin [m]
  size_t target_points = 0;  // cropped + preprocessed target points used by ICP
};

class Open3DLocalizer {
 public:
  using Cloud = pcl::PointCloud<pcl::PointXYZ>;

  Open3DLocalizer() = default;
  explicit Open3DLocalizer(const Open3DParams& params) { setParams(params); }

  void setParams(const Open3DParams& params) { params_ = params; }

  // Build the registration target from a global map. The map is expected to be
  // expressed in the map frame already. It is converted to Open3D and
  // pre-downsampled to map_voxel_size once here; each align() then crops this
  // coarse cloud around the initial guess instead of re-processing the full map.
  void setMap(const Cloud::Ptr& map);

  // Register one lidar scan against the map.
  //   scan  : current scan, expressed in the base_link frame.
  //   guess : initial estimate of T_map_base (homogeneous 4x4). Typically the
  //           /initialpose guess the first time, then the previous result.
  // Returns an AlignResult whose .success indicates whether the match is
  // trustworthy (see struct docs for the threshold checks).
  AlignResult align(const Cloud::Ptr& scan, const Eigen::Matrix4d& guess);

  size_t mapPoints() const { return map_points_; }

 private:
  Open3DParams params_;
  // Full map in Open3D form. The registration target is cropped around the
  // initial guess and preprocessed inside align().
  std::shared_ptr<open3d::geometry::PointCloud> map_ori_;
  size_t map_points_ = 0;
};

}  // namespace point_lio
