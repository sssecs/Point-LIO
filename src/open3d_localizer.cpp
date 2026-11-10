// -----------------------------------------------------------------------------
//  open3d_localizer.cpp — Open3D FPFH+RANSAC / multiscale-ICP relocalization.
//
//  Ported from deepglint/FAST_LIO_LOCALIZATION_HUMANOID (open3d_loc /
//  open3d_registration.cpp):
//    1. FPFH features + RANSACBasedOnFeatureMatching  -> coarse matrix
//    2. RegistrationMultiScaleIcp (coarse -> fine)   -> fine matrix
//    3. EvaluateRegistration                         -> fitness (overlap ratio)
//
//  Compatible with the system Open3D 0.14 (RemoveNonFinitePoints is in-place;
//  RegistrationRANSACBasedOnFeatureMatching's `seed` argument is optional).
// -----------------------------------------------------------------------------
#include "open3d_localizer.hpp"

#include <chrono>

namespace point_lio {

namespace reg = open3d::pipelines::registration;

namespace {

using O3dCloud = open3d::geometry::PointCloud;
using O3dFeature = open3d::pipelines::registration::Feature;

// Milliseconds since the given steady-clock timestamp (for per-stage timing).
inline double elapsed_ms(const std::chrono::steady_clock::time_point& from) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - from)
      .count();
}

// PCL (PointXYZ) -> Open3D (Vector3d) copy.
std::shared_ptr<O3dCloud> ToOpen3d(const Open3DLocalizer::Cloud::Ptr& pcl_cloud) {
  auto cloud = std::make_shared<O3dCloud>();
  cloud->points_.reserve(pcl_cloud->size());
  for (const auto& p : pcl_cloud->points) {
    cloud->points_.emplace_back(p.x, p.y, p.z);
  }
  return cloud;
}

// -------- coarse registration: FPFH + RANSAC --------------------------------
reg::RegistrationResult RegistrationFpfh(
    const std::shared_ptr<O3dCloud>& source,
    const std::shared_ptr<O3dCloud>& target,
    const std::shared_ptr<O3dFeature>& source_fpfh,
    const std::shared_ptr<O3dFeature>& target_fpfh,
    double voxel_size, bool mutual_filter, int ransac_max_iteration) {
  const double distance_threshold = 1.5 * voxel_size;

  std::vector<std::reference_wrapper<const reg::CorrespondenceChecker>> checkers;
  auto checker_edge = reg::CorrespondenceCheckerBasedOnEdgeLength(0.9);
  auto checker_dist = reg::CorrespondenceCheckerBasedOnDistance(distance_threshold);
  checkers.push_back(checker_edge);
  checkers.push_back(checker_dist);

  // NOTE: the `seed` argument (last, optional) of
  // RegistrationRANSACBasedOnFeatureMatching is omitted — it defaults to
  // nullopt. Re-add it if you need reproducible RANSAC runs.
  return reg::RegistrationRANSACBasedOnFeatureMatching(
      *source, *target, *source_fpfh, *target_fpfh,
      mutual_filter, distance_threshold,
      reg::TransformationEstimationPointToPoint(false),
      4,  // ransac_n (min 3)
      checkers,
      reg::RANSACConvergenceCriteria(ransac_max_iteration, 0.999));
}

// -------- fine registration: single-scale ICP --------------------------------
// Pre-transforms a copy of `source` by `init_matrix`, then solves the residual
// with the identity start (same convention as the reference pcd_tools code).
reg::RegistrationResult RegistrationIcp(
    const std::shared_ptr<O3dCloud>& source,
    const std::shared_ptr<O3dCloud>& target,
    double icp_distance_threshold,
    const Eigen::Matrix4d& init_matrix,
    int icp_method, int icp_iteration) {
  auto source_t = std::make_shared<O3dCloud>(*source);
  source_t->Transform(init_matrix);

  const auto criteria = reg::ICPConvergenceCriteria(1e-6, 1e-6, icp_iteration);
  switch (icp_method) {
    case 0:
      return reg::RegistrationICP(*source_t, *target, icp_distance_threshold,
                                  Eigen::Matrix4d::Identity(),
                                  reg::TransformationEstimationPointToPoint(),
                                  criteria);
    case 2:
      return reg::RegistrationGeneralizedICP(*source_t, *target, icp_distance_threshold,
                                             Eigen::Matrix4d::Identity(),
                                             reg::TransformationEstimationForGeneralizedICP(),
                                             criteria);
    case 1:
    default:
      return reg::RegistrationICP(*source_t, *target, icp_distance_threshold,
                                  Eigen::Matrix4d::Identity(),
                                  reg::TransformationEstimationPointToPlane(),
                                  criteria);
  }
}

// -------- fine registration: coarse -> fine multiscale ICP ------------------
// Refines the pose from the coarsest voxel level down to the fine voxel,
// chaining each level's result into the next. Source and target are downsampled
// with their own voxel bases (`source_voxel`/`target_voxel`), so a coarse map
// target and a fine scan both behave sensibly.
// `scan_center` is the map-frame position of the scan origin where ICP starts
// (the guess translation, or the guess moved by the coarse stage). If
// `level_delta`/`level_fitness` are non-null they are filled with, per level,
// the cumulative distance the scan center moved from `scan_center` [m] and the
// ICP fitness. (Measured as position displacement, NOT the relative-pose
// translation — see the translation_delta note in align().)
Eigen::Matrix4d RegistrationMultiScaleIcp(
    const std::shared_ptr<O3dCloud>& source,
    const std::shared_ptr<O3dCloud>& target,
    double source_voxel, double target_voxel, int icp_method,
    const std::vector<double>& scale,
    const Eigen::Vector3d& scan_center,
    std::vector<double>* level_delta = nullptr,
    std::vector<double>* level_fitness = nullptr) {
  struct Pair {
    std::shared_ptr<O3dCloud> src, tgt;
    double threshold;
  };
  std::vector<Pair> pairs;
  pairs.reserve(scale.size());
  for (double s : scale) {
    Pair pair;
    const double sv = source_voxel * s;
    const double tv = target_voxel * s;
    pair.threshold = std::max(sv, tv) * 1.5;
    pair.src = source->VoxelDownSample(sv);
    pair.tgt = target->VoxelDownSample(tv);
    // Point-to-plane (and GICP) need target normals.
    pair.tgt->EstimateNormals(
        open3d::geometry::KDTreeSearchParamHybrid(tv * 2, 30));
    pairs.push_back(pair);
  }

  Eigen::Matrix4d matrix = Eigen::Matrix4d::Identity();
  for (auto it = pairs.rbegin(); it != pairs.rend(); ++it) {
    const auto r = RegistrationIcp(it->src, it->tgt, it->threshold, matrix,
                                   icp_method, 30);
    matrix = r.transformation_ * matrix;
    if (level_delta) {
      const Eigen::Vector3d center =
          matrix.block<3, 3>(0, 0) * scan_center + matrix.block<3, 1>(0, 3);
      level_delta->push_back((center - scan_center).norm());
    }
    if (level_fitness) level_fitness->push_back(r.fitness_);
  }
  return matrix;
}

// -------- preprocessing: clean -> downsample -> (normals + FPFH) ------------
// Normals + FPFH are only needed for the coarse FPFH+RANSAC stage; when it is
// disabled the features are skipped (much faster). An empty Feature is returned
// in that case. Point-to-plane ICP only needs target normals, which the
// multiscale-ICP ladder computes itself.
std::tuple<std::shared_ptr<O3dCloud>, std::shared_ptr<O3dFeature>> Preprocess(
    const std::shared_ptr<O3dCloud>& cloud,
    double voxel_size, bool statistical_filter,
    int filter_neighbors, double filter_std_ratio,
    bool compute_features) {
  auto out = cloud;
  out->RemoveNonFinitePoints(true, true);  // in-place
  if (statistical_filter) {
    out = std::get<0>(out->RemoveStatisticalOutliers(filter_neighbors,
                                                     filter_std_ratio, false));
  }
  out = out->VoxelDownSample(voxel_size);
  if (!compute_features) {
    return {out, std::make_shared<O3dFeature>()};
  }
  out->EstimateNormals(
      open3d::geometry::KDTreeSearchParamHybrid(voxel_size * 2, 30));
  out->OrientNormalsToAlignWithDirection();
  auto fpfh = reg::ComputeFPFHFeature(
      *out, open3d::geometry::KDTreeSearchParamHybrid(voxel_size * 5, 100));
  return {out, fpfh};
}

}  // namespace

void Open3DLocalizer::setMap(const Cloud::Ptr& map) {
  if (!map || map->empty()) {
    map_points_ = 0;
    map_ori_.reset();
    return;
  }
  map_points_ = map->size();

  // Convert to Open3D and pre-downsample the FULL map once here. Every align()
  // then crops this coarse cloud around the guess instead of hashing, copying
  // and outlier-filtering the full-resolution map on each call — that used to
  // dominate the target stage (thousands of ms on dense maps). The registration
  // target never needs finer resolution than map_voxel_size, so pre-downsampling
  // is lossless for align().
  map_ori_ = ToOpen3d(map);
  map_ori_->RemoveNonFinitePoints(true, true);
  map_ori_ = map_ori_->VoxelDownSample(params_.map_voxel_size);
}

AlignResult Open3DLocalizer::align(const Cloud::Ptr& scan,
                                   const Eigen::Matrix4d& guess) {
  AlignResult res;
  if (!scan || scan->empty() || !map_ori_) return res;

  const auto t_start = std::chrono::steady_clock::now();

  // ---- debug: scan size / extent (diagnosing far-off matches) --------------
  res.scan_points = scan->size();
  double r2max = 0.0;
  for (const auto& p : scan->points) {
    const double r2 = p.x * p.x + p.y * p.y + p.z * p.z;
    if (r2 > r2max) r2max = r2;
  }
  res.scan_radius = std::sqrt(r2max);

  // ---- target: crop the map around the guess, then preprocess ---------------
  std::shared_ptr<O3dCloud> target = map_ori_;
  if (params_.crop_radius > 0.0) {
    const Eigen::Vector3d center = guess.block<3, 1>(0, 3);
    const Eigen::Vector3d lo = center - Eigen::Vector3d::Constant(params_.crop_radius);
    const Eigen::Vector3d hi = center + Eigen::Vector3d::Constant(params_.crop_radius);
    const open3d::geometry::AxisAlignedBoundingBox box(lo, hi);
    target = map_ori_->Crop(box);  // Crop already returns a new, detached cloud
    if (!target || target->IsEmpty()) return res;
  }
  std::shared_ptr<O3dFeature> target_fpfh;
  std::tie(target, target_fpfh) = Preprocess(
      target, params_.map_voxel_size, params_.statistical_filter,
      params_.filter_neighbors, params_.filter_std_ratio, params_.use_fpfh);
  if (params_.use_fpfh && (!target_fpfh || target_fpfh->Num() == 0)) return res;
  res.target_points = target->points_.size();  // debug: cropped target size
  res.target_ms = elapsed_ms(t_start);

  // ---- source: PCL -> Open3D, clean, keep a copy, apply the guess ----------
  auto source = ToOpen3d(scan);
  source->RemoveNonFinitePoints(true, true);
  if (source->IsEmpty()) return res;
  auto source_ori = std::make_shared<O3dCloud>(*source);  // for final evaluation
  source->Transform(guess);

  // ---- preprocess source (downsample + normals + FPFH) ---------------------
  const auto t_coarse = std::chrono::steady_clock::now();
  std::shared_ptr<O3dFeature> source_fpfh;
  std::tie(source, source_fpfh) = Preprocess(
      source, params_.voxel_size, params_.statistical_filter,
      params_.filter_neighbors, params_.filter_std_ratio, params_.use_fpfh);
  if (params_.use_fpfh && (!source_fpfh || source_fpfh->Num() == 0)) return res;

  // Coarse matching between the two densities uses the coarser voxel as the
  // distance base, so the scan can find correspondences on a sparse map.
  const double coarse_voxel = std::max(params_.voxel_size, params_.map_voxel_size);

  // ---- coarse: FPFH + RANSAC ------------------------------------------------
  Eigen::Matrix4d matrix = Eigen::Matrix4d::Identity();
  if (params_.use_fpfh) {
    const auto coarse = RegistrationFpfh(source, target, source_fpfh,
                                         target_fpfh, coarse_voxel,
                                         params_.mutual_filter,
                                         params_.ransac_max_iteration);
    res.fpfh_fitness = coarse.fitness_;
    res.fpfh_rmse = coarse.inlier_rmse_;
    matrix = coarse.transformation_;
    source->Transform(coarse.transformation_);  // move source under the coarse pose
  }
  res.coarse_ms = elapsed_ms(t_coarse);

  // ---- fine: multiscale ICP --------------------------------------------------
  const auto t_icp = std::chrono::steady_clock::now();
  // Map-frame position of the scan origin where ICP starts (for the per-level
  // diagnostic deltas). With use_fpfh the coarse stage already moved the scan;
  // otherwise matrix is Identity and this is just the guess translation.
  const Eigen::Vector3d scan_center = (matrix * guess).block<3, 1>(0, 3);
  const auto icp = RegistrationMultiScaleIcp(source, target, params_.voxel_size,
                                             params_.map_voxel_size,
                                             params_.icp_method, params_.scale,
                                             scan_center,
                                             &res.icp_level_delta,
                                             &res.icp_level_fitness);
  matrix = icp * matrix;
  res.icp_ms = elapsed_ms(t_icp);

  // ---- final transform: scan(base_link) -> map -------------------------------
  res.transform = matrix * guess;

  // ---- fitness (overlap ratio) of the final transform ------------------------
  // Evaluate against the preprocessed target (already at map_voxel_size) so the
  // full map is not downsampled again. The correspondence distance follows the
  // coarser of the two densities.
  auto src_eval = source_ori->VoxelDownSample(params_.voxel_size);
  const double eval_voxel = std::max(params_.voxel_size, params_.map_voxel_size);
  const auto eval = reg::EvaluateRegistration(*src_eval, *target,
                                              eval_voxel * 1.5,
                                              res.transform);
  res.fitness = eval.fitness_;
  res.rmse = eval.inlier_rmse_;

  // ---- validity check on the initial guess ---------------------------------
  // Distance the pipeline moved the pose from the initial guess. A trustworthy
  // match moves little; a large jump usually means a bad initial guess or a
  // match into a local minimum.
  //
  // NOTE: translation must be measured as the Euclidean distance between the
  // found and guess POSITIONS. Using the translation of the relative pose
  // (res.transform * guess.inverse()) instead is wrong when the robot is far
  // from the map origin: that value equals (t_found - t_guess) + (I - R_rel)*t_guess,
  // whose (I - R_rel)*t_guess term grows linearly with |t_guess| (~0.3 rad of
  // relative rotation at |t_guess| = 60 m inflates it by ~18 m), causing false
  // rejections of perfectly good matches.
  const Eigen::Vector3d guess_pos = guess.block<3, 1>(0, 3);
  const Eigen::Vector3d found_pos = res.transform.block<3, 1>(0, 3);
  res.translation_delta = (found_pos - guess_pos).norm();
  // Rotation is a relative quantity, so the angle of the relative pose is fine.
  const Eigen::Matrix4d correction = res.transform * guess.inverse();
  res.rotation_delta = Eigen::AngleAxisd(correction.block<3, 3>(0, 0)).angle();

  res.success = res.fitness >= params_.min_fitness &&
                res.translation_delta <= params_.max_translation_delta &&
                res.rotation_delta <= params_.max_rotation_delta;
  res.total_ms = elapsed_ms(t_start);
  return res;
}

}  // namespace point_lio
