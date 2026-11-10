// Point-LIO ROS2 node implementation. See laserMapping.hpp.

#include "laserMapping.hpp"

#include <omp.h>

#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

namespace
{
// Debug timing accumulators (per scan), formerly file-scope globals.
double match_time = 0, solve_time = 0, propag_time = 0, update_time = 0;

rclcpp::Time stamp_from_sec(double sec)
{
  return rclcpp::Time(static_cast<int64_t>(sec * 1e9));
}

pcl::PointCloud<pcl::PointXYZ>::Ptr to_point_xyz(const PointCloudXYZI::Ptr & cloud)
{
  auto out = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  out->reserve(cloud->size());
  for (const auto & p : cloud->points)
  {
    out->push_back(pcl::PointXYZ(p.x, p.y, p.z));
  }
  return out;
}

// Build the initial guess T_map_base from an RViz "2D Pose Estimate" message.
// The click is 2D: z is forced to the configured init_z height.
Eigen::Matrix4d initial_pose_to_matrix(
  const geometry_msgs::msg::PoseWithCovarianceStamped & pose, double init_z)
{
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  const auto & p = pose.pose.pose.position;
  const auto & q = pose.pose.pose.orientation;
  Eigen::Quaterniond quat(q.w, q.x, q.y, q.z);
  T.block<3, 3>(0, 0) = quat.toRotationMatrix();
  T(0, 3) = p.x;
  T(1, 3) = p.y;
  T(2, 3) = init_z;
  return T;
}
}  // namespace

namespace point_lio
{

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

LaserMappingNode::LaserMappingNode()
: Node("point_lio_mapping")
{
  readParameters(this);
  load_params();

  Q_input_ = process_noise_cov_input();
  Q_output_ = process_noise_cov_output();

  RCLCPP_INFO(
    this->get_logger(), "lidar topic: %s, imu topic: %s, frames merged: %d, frames: %s <- %s -> %s",
    lid_topic.c_str(), imu_topic.c_str(), frame_merge_num, map_frame_.c_str(), base_frame_.c_str(),
    lidar_frame_.c_str());

  frame_merge_num_ = std::max(1, frame_merge_num);

  ivox_ = std::make_shared<IVoxType>(ivox_options_);

  path_.header.stamp = this->get_clock()->now();
  path_.header.frame_id = map_frame_;

  memset(point_selected_surf, true, sizeof(point_selected_surf));
  down_size_filter_surf_.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
  down_size_filter_map_.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);

  Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
  Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);

  if (extrinsic_est_en)
  {
    if (!use_imu_as_input)
    {
      kf_output.x_.offset_R_L_I = Lidar_R_wrt_IMU;
      kf_output.x_.offset_T_L_I = Lidar_T_wrt_IMU;
    }
    else
    {
      kf_input.x_.offset_R_L_I = Lidar_R_wrt_IMU;
      kf_input.x_.offset_T_L_I = Lidar_T_wrt_IMU;
    }
  }

  p_imu->imu_en = imu_en;

  kf_input.init_dyn_share_modified_2h(get_f_input, df_dx_input, h_model_input);
  kf_output.init_dyn_share_modified_3h(get_f_output, df_dx_output, h_model_output, h_model_IMU_output);
  Eigen::Matrix<double, 24, 24> P_init;
  reset_cov(P_init);
  kf_input.change_P(P_init);
  Eigen::Matrix<double, 30, 30> P_init_output;
  reset_cov_output(P_init_output);
  kf_output.change_P(P_init_output);

  // --- static lidar -> base TF ----------------------------------------------
  // Required to express the lidar scan and the IMU in the robot base frame.
  // Retry for a while: with use_sim_time the lookup timeout is measured on the
  // (possibly not yet published) sim clock, so a single lookup can fail before
  // the static_transform_publisher shows up.
  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
  bool tf_ok = false;
  for (int attempt = 0; attempt < 20 && !tf_ok; ++attempt)
  {
    try
    {
      tf_lidar_to_base_ = tf_buffer_->lookupTransform(
        base_frame_, lidar_frame_, tf2::TimePointZero, std::chrono::milliseconds(200));
      tf_ok = true;
    }
    catch (const tf2::TransformException & ex)
    {
      RCLCPP_WARN(
        this->get_logger(), "Waiting for static TF %s -> %s: %s", lidar_frame_.c_str(),
        base_frame_.c_str(), ex.what());
      std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }
  }
  if (!tf_ok)
  {
    RCLCPP_ERROR(
      this->get_logger(), "Could not get transform %s -> %s.", lidar_frame_.c_str(),
      base_frame_.c_str());
    throw std::runtime_error("lidar -> base static TF not available at startup");
  }
  RCLCPP_INFO(
    this->get_logger(), "Static TF %s -> %s available.", lidar_frame_.c_str(), base_frame_.c_str());

  // --- pre-built map ----------------------------------------------------------
  if (loc_params_.enable_localization)
  {
    load_map();
    // Localization mode waits for the operator's first /initialpose.
    localization_seeded_ = false;
  }
  else
  {
    init_map_ = false;
    localization_seeded_ = true;  // mapping mode self-starts from the first scan
  }
  initial_pose_set_.store(false);

  // --- ROS graph --------------------------------------------------------------
  // Deep queues: while the node is busy (e.g. inside the relocalization ICP)
  // the bag keeps publishing; the buffers must survive several seconds of
  // backlog (mid360 IMU ~200 Hz).
  sub_pcl_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    lid_topic, rclcpp::SensorDataQoS().keep_last(200),
    [this](const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) { this->lidar_callback(msg); });
  sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(
    imu_topic, rclcpp::SensorDataQoS().keep_last(8000),
    [this](const sensor_msgs::msg::Imu::ConstSharedPtr msg) { this->imu_callback(msg); });
  sub_init_pose_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
    loc_params_.initial_pose_topic, 10,
    [this](const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
      this->initial_pose_callback(msg);
    });

  pub_laser_cloud_full_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered", 20);
  pub_laser_cloud_full_body_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
    "/cloud_registered_body", 20);
  // transient_local so RViz (subscribing after the map was published once at
  // startup) still receives the loaded map.
  pub_laser_cloud_map_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
    "/Laser_map", rclcpp::QoS(1).transient_local());
  pub_odom_ = this->create_publisher<nav_msgs::msg::Odometry>("/Odometry", 2000);
  pub_path_ = this->create_publisher<nav_msgs::msg::Path>("/path", 20);
  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

  // Publish the (loaded) map once so RViz can display it before the first pose.
  publish_map();

  map_save_srv_ = this->create_service<std_srvs::srv::Trigger>(
    "map_save",
    std::bind(&LaserMappingNode::map_save_callback, this, std::placeholders::_1, std::placeholders::_2));

  // Main loop: poll the buffers and run one scan per synchronized package
  // (the ROS1 version used ros::Rate(500)).
  timer_ = rclcpp::create_timer(
    this, this->get_clock(), std::chrono::milliseconds(2),
    std::bind(&LaserMappingNode::timer_callback, this));

  const std::string pos_log_dir = std::string(ROOT_DIR) + "/Log/pos_log.txt";
  fp_ = fopen(pos_log_dir.c_str(), "w");
  open_file();

  RCLCPP_INFO(this->get_logger(), "Node init finished.");
}

LaserMappingNode::~LaserMappingNode()
{
  fout_out.close();
  fout_imu_pbp.close();
  if (fp_)
  {
    fclose(fp_);
  }
  if (pcl_wait_save_->size() > 0 && pcd_save_en)
  {
    const std::string file_name = "scans.pcd";
    const std::string all_points_dir = std::string(ROOT_DIR) + "PCD/" + file_name;
    pcl::PCDWriter pcd_writer;
    pcd_writer.writeBinary(all_points_dir, *pcl_wait_save_);
  }
}

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

void LaserMappingNode::load_params()
{
  this->declare_parameter<std::string>("frame.map_frame", "map");
  this->declare_parameter<std::string>("frame.base_frame", "base_link");
  this->declare_parameter<std::string>("frame.lidar_frame", "livox_frame");

  this->declare_parameter<bool>("localization.enable_localization", false);
  this->declare_parameter<bool>("localization.enable_map_incremental", true);
  this->declare_parameter<double>("localization.leaf_size", 0.1);
  this->declare_parameter<std::string>("localization.map_path", "");
  this->declare_parameter<double>("localization.init_z", 1.2);
  this->declare_parameter<double>("localization.tf_lookup_timeout", 5.0);
  this->declare_parameter<std::string>("localization.initial_pose_topic", "/initialpose");
  this->declare_parameter<bool>("localization.failure_detect_en", true);
  this->declare_parameter<double>("localization.max_pose_to_map_dist", 3.0);
  this->declare_parameter<double>("localization.max_pose_speed", 3.0);
  this->declare_parameter<bool>("localization.relocalization_verify_en", true);
  this->declare_parameter<double>("localization.relocalization_verify_window", 2.0);
  this->declare_parameter<double>("localization.relocalization_verify_max_drift", 0.1);
  this->declare_parameter<double>("localization.relocalization_verify_max_path", 0.2);
  // Open3D relocalization
  this->declare_parameter<bool>("localization.relocalization.enable", true);
  this->declare_parameter<double>("localization.relocalization.voxel_size", 0.2);
  this->declare_parameter<double>("localization.relocalization.map_voxel_size", 0.2);
  this->declare_parameter<std::vector<double>>("localization.relocalization.scale", std::vector<double>{1.0, 4.0, 6.0});
  this->declare_parameter<int>("localization.relocalization.icp_method", 1);
  this->declare_parameter<int>("localization.relocalization.icp_iteration", 30);
  this->declare_parameter<bool>("localization.relocalization.use_fpfh", true);
  this->declare_parameter<int>("localization.relocalization.ransac_max_iteration", 100000);
  this->declare_parameter<bool>("localization.relocalization.mutual_filter", true);
  this->declare_parameter<bool>("localization.relocalization.statistical_filter", true);
  this->declare_parameter<int>("localization.relocalization.filter_neighbors", 50);
  this->declare_parameter<double>("localization.relocalization.filter_std_ratio", 3.0);
  this->declare_parameter<double>("localization.relocalization.crop_radius", 60.0);
  this->declare_parameter<double>("localization.relocalization.min_fitness", 0.5);
  this->declare_parameter<double>("localization.relocalization.max_translation_delta", 5.0);
  this->declare_parameter<double>("localization.relocalization.max_rotation_delta", 0.785);

  this->get_parameter("frame.map_frame", map_frame_);
  this->get_parameter("frame.base_frame", base_frame_);
  this->get_parameter("frame.lidar_frame", lidar_frame_);

  this->get_parameter("localization.enable_localization", loc_params_.enable_localization);
  this->get_parameter("localization.enable_map_incremental", loc_params_.enable_map_incremental);
  this->get_parameter("localization.relocalization.enable", loc_params_.relocalization_en);
  this->get_parameter("localization.leaf_size", loc_params_.leaf_size);
  this->get_parameter("localization.map_path", loc_params_.map_path);
  this->get_parameter("localization.init_z", loc_params_.init_z);
  this->get_parameter("localization.tf_lookup_timeout", loc_params_.tf_lookup_timeout);
  this->get_parameter("localization.initial_pose_topic", loc_params_.initial_pose_topic);
  this->get_parameter("localization.failure_detect_en", loc_params_.failure_detect_en);
  this->get_parameter("localization.max_pose_to_map_dist", loc_params_.max_pose_to_map_dist);
  this->get_parameter("localization.max_pose_speed", loc_params_.max_pose_speed);
  this->get_parameter("localization.relocalization_verify_en", loc_params_.relocalization_verify_en);
  this->get_parameter("localization.relocalization_verify_window", loc_params_.relocalization_verify_window);
  this->get_parameter("localization.relocalization_verify_max_drift", loc_params_.relocalization_verify_max_drift);
  this->get_parameter("localization.relocalization_verify_max_path", loc_params_.relocalization_verify_max_path);
  loc_params_.open3d.voxel_size = this->get_parameter("localization.relocalization.voxel_size").as_double();
  loc_params_.open3d.map_voxel_size = this->get_parameter("localization.relocalization.map_voxel_size").as_double();
  loc_params_.open3d.scale = this->get_parameter("localization.relocalization.scale").as_double_array();
  loc_params_.open3d.icp_method = this->get_parameter("localization.relocalization.icp_method").as_int();
  loc_params_.open3d.icp_iteration = this->get_parameter("localization.relocalization.icp_iteration").as_int();
  loc_params_.open3d.use_fpfh = this->get_parameter("localization.relocalization.use_fpfh").as_bool();
  loc_params_.open3d.ransac_max_iteration = this->get_parameter("localization.relocalization.ransac_max_iteration").as_int();
  loc_params_.open3d.mutual_filter = this->get_parameter("localization.relocalization.mutual_filter").as_bool();
  loc_params_.open3d.statistical_filter = this->get_parameter("localization.relocalization.statistical_filter").as_bool();
  loc_params_.open3d.filter_neighbors = this->get_parameter("localization.relocalization.filter_neighbors").as_int();
  loc_params_.open3d.filter_std_ratio = this->get_parameter("localization.relocalization.filter_std_ratio").as_double();
  loc_params_.open3d.crop_radius = this->get_parameter("localization.relocalization.crop_radius").as_double();
  loc_params_.open3d.min_fitness = this->get_parameter("localization.relocalization.min_fitness").as_double();
  loc_params_.open3d.max_translation_delta = this->get_parameter("localization.relocalization.max_translation_delta").as_double();
  loc_params_.open3d.max_rotation_delta = this->get_parameter("localization.relocalization.max_rotation_delta").as_double();

  localizer_.setParams(loc_params_.open3d);
}

// ---------------------------------------------------------------------------
// Map loading / relocalization
// ---------------------------------------------------------------------------

void LaserMappingNode::load_map()
{
  RCLCPP_INFO(this->get_logger(), "Loading GlobalMap from %s ...", loc_params_.map_path.c_str());

  PointCloudXYZI::Ptr localization_map(new PointCloudXYZI());
  if (pcl::io::loadPCDFile(loc_params_.map_path, *localization_map) < 0)
  {
    RCLCPP_ERROR(this->get_logger(), "Failed to load map file: %s", loc_params_.map_path.c_str());
    throw std::runtime_error("localization map file not found: " + loc_params_.map_path);
  }
  map_raw_ = localization_map;
  RCLCPP_INFO(this->get_logger(), "GlobalMap point num: %zu", map_raw_->size());

  // Downsample the loaded map once; the iVox (and the scan matching) uses this
  // resolution. The full-resolution cloud feeds the Open3D relocalizer.
  PointCloudXYZI::Ptr map_ds(new PointCloudXYZI());
  pcl::VoxelGrid<PointType> voxel_grid_filter;
  voxel_grid_filter.setLeafSize(loc_params_.leaf_size, loc_params_.leaf_size, loc_params_.leaf_size);
  voxel_grid_filter.setInputCloud(localization_map);
  voxel_grid_filter.filter(*map_ds);
  RCLCPP_INFO(this->get_logger(), "GlobalMap ds point num: %zu", map_ds->size());

  ivox_->AddPoints(map_ds->points);
  init_map_ = true;

  map_kdtree_.reset(new pcl::KdTreeFLANN<PointType>());
  map_kdtree_->setInputCloud(map_ds);
  {
    // Self-check: the nearest neighbor of a map point must be ~0 away.
    PointType q = map_ds->points.front();
    std::vector<int> nn_idx(1);
    std::vector<float> nn_sqdist(1);
    if (map_kdtree_->nearestKSearch(q, 1, nn_idx, nn_sqdist) > 0)
    {
      RCLCPP_INFO(
        this->get_logger(), "Map kd-tree ready (self-check nearest dist: %.3f m).",
        std::sqrt(static_cast<double>(nn_sqdist[0])));
    }
    else
    {
      RCLCPP_ERROR(this->get_logger(), "Map kd-tree self-check FAILED.");
    }
  }

  if (loc_params_.relocalization_en)
  {
    auto map_xyz = std::make_shared<Open3DLocalizer::Cloud>();
    map_xyz->reserve(map_raw_->size());
    for (const auto & p : map_raw_->points)
    {
      map_xyz->push_back(pcl::PointXYZ(p.x, p.y, p.z));
    }
    localizer_.setMap(map_xyz);
    localizer_ready_ = true;
    RCLCPP_INFO(this->get_logger(), "Open3D relocalizer map ready: %zu points.", localizer_.mapPoints());
  }

  RCLCPP_INFO(this->get_logger(), "Loaded GlobalMap from %s!", loc_params_.map_path.c_str());
}

bool LaserMappingNode::relocalize_and_seed()
{
  // Initial guess: T_map_base built from the RViz /initialpose click
  // (a 2D click — z is forced to the configured init_z height).
  Eigen::Matrix4d T_map_base = initial_pose_to_matrix(init_pose_, loc_params_.init_z);

  // Refine the guess by matching the current scan against the loaded map.
  // Only a trustworthy match may seed the EKF: the raw RViz click is a 2D
  // approximation that can land in a local minimum and later fly away, so a
  // rejected match is treated as a failure (stay stopped, wait for a better
  // initial pose) instead of a fallback that seeds the raw guess.
  bool match_ok = false;
  if (localizer_ready_ && !Measures.lidar->empty())
  {
    const auto scan_xyz = to_point_xyz(Measures.lidar);
    const auto res = localizer_.align(scan_xyz, T_map_base);
    if (res.success)
    {
      RCLCPP_INFO(
        this->get_logger(),
        "Relocalization accepted: fitness %.3f, correction %.2f m / %.2f rad "
        "(%.0f ms, scan %zu pts).",
        res.fitness, res.translation_delta, res.rotation_delta, res.total_ms, res.scan_points);
      T_map_base = res.transform;
      match_ok = true;
    }
    else
    {
      RCLCPP_WARN(
        this->get_logger(),
        "Relocalization rejected (fitness %.3f, correction %.2f m / %.2f rad, "
        "scan %zu pts); not seeding the EKF.",
        res.fitness, res.translation_delta, res.rotation_delta, res.scan_points);
    }
  }
  else if (!localizer_ready_)
  {
    RCLCPP_WARN(
      this->get_logger(), "Relocalizer not ready (disabled); using the RViz initial pose.");
    match_ok = true;
  }

  if (!match_ok)
  {
    return false;
  }

  // Seed both filter states with the trusted pose. The robot is stationary
  // during relocalization, so velocity is zero and the gravity vector comes
  // from the configured map-frame gravity.
  const Eigen::Vector3d p_map = T_map_base.block<3, 1>(0, 3);
  const Eigen::Matrix3d R_map = T_map_base.block<3, 3>(0, 0);
  const V3D grav(VEC_FROM_ARRAY(gravity));

  state_in = kf_input.get_x();
  state_in.pos = p_map;
  state_in.rot = SO3(R_map);
  state_in.vel = Zero3d;
  state_in.bg = Zero3d;
  state_in.ba = Zero3d;
  state_in.gravity = grav;
  state_in.offset_R_L_I = SO3(Lidar_R_wrt_IMU);
  state_in.offset_T_L_I = Lidar_T_wrt_IMU;
  kf_input.change_x(state_in);

  state_out = kf_output.get_x();
  state_out.pos = p_map;
  state_out.rot = SO3(R_map);
  state_out.vel = Zero3d;
  state_out.omg = Zero3d;
  state_out.bg = Zero3d;
  state_out.ba = Zero3d;
  state_out.gravity = grav;
  state_out.acc = -R_map.transpose() * grav;  // at rest: rot*acc + gravity = 0
  state_out.offset_R_L_I = SO3(Lidar_R_wrt_IMU);
  state_out.offset_T_L_I = Lidar_T_wrt_IMU;
  kf_output.change_x(state_out);

  // Re-arm the covariance of both filters around the seed.
  Eigen::Matrix<double, 24, 24> P_init;
  reset_cov(P_init);
  kf_input.change_P(P_init);
  Eigen::Matrix<double, 30, 30> P_init_output;
  reset_cov_output(P_init_output);
  kf_output.change_P(P_init_output);

  // Skip the data-driven IMU gravity initialization: gravity and orientation
  // are already known in the map frame.
  p_imu->imu_need_init_ = false;
  p_imu->after_imu_init_ = true;

  localization_seeded_ = true;
  localization_failed_ = false;
  rearm_relocalization(p_map, Measures.lidar_beg_time);

  RCLCPP_INFO(
    this->get_logger(), "EKF seeded: pos=(%.3f, %.3f, %.3f).", p_map.x(), p_map.y(), p_map.z());
  return true;
}

void LaserMappingNode::rearm_relocalization(const Eigen::Vector3d & seed_pos, double lidar_beg_time)
{
  // Resume localization from this seed: clear any previous failure flag and
  // use the seed as the motion reference, so a teleport via /initialpose is
  // not mistaken for an implausible scan-to-scan jump by the health check.
  localization_failed_ = false;
  last_good_valid_ = true;
  last_good_pos_ = seed_pos;
  last_good_time_ = lidar_beg_time;
  pos_hist_.clear();
  pos_hist_.emplace_back(lidar_beg_time, seed_pos);
  // Arm the post-relocalization fly-away guard: the robot is expected to be
  // stationary right after a re-seed, so the pose must neither leave the
  // re-seeded position nor travel a long path during the verification window
  // (checked in localization_health_check()).
  relocalized_verify_ = loc_params_.relocalization_verify_en;
  relocalized_pos_ = last_good_pos_;
  relocalized_time_ = last_good_time_;
  relocalized_path_ = 0.0;
}

bool LaserMappingNode::localization_health_check(double lidar_beg_time)
{
  // Only active for static-map localization; in mapping mode the map is being
  // built so the support check does not apply.
  if (!loc_params_.enable_localization || !loc_params_.failure_detect_en)
  {
    return true;
  }

  const Eigen::Vector3d cur_pos =
    use_imu_as_input ? kf_input.x_.pos : kf_output.x_.pos;

  // A non-finite pose means the filter diverged -> lost.
  if (!cur_pos.allFinite())
  {
    localization_failed_ = true;
    RCLCPP_WARN(
      this->get_logger(),
      "Localization FAILED: pose is non-finite; stopping localization, map->base "
      "TF suppressed until a new initial pose.");
    return false;
  }

  // Post-relocalization fly-away guard (robot stationary right after a re-seed).
  if (relocalized_verify_)
  {
    const double since_seed = lidar_beg_time - relocalized_time_;
    if (since_seed > loc_params_.relocalization_verify_window)
    {
      relocalized_verify_ = false;  // verification window passed -> normal checks
    }
    else
    {
      relocalized_path_ += (cur_pos - last_good_pos_).norm();
      const double drift = (cur_pos - relocalized_pos_).norm();
      if (drift > loc_params_.relocalization_verify_max_drift ||
          relocalized_path_ > loc_params_.relocalization_verify_max_path)
      {
        relocalized_verify_ = false;
        localization_failed_ = true;
        RCLCPP_WARN(
          this->get_logger(),
          "Localization FAILED: relocalization drifted %.2f m from the re-seeded "
          "position / traveled %.2f m in %.1f s; stopping localization, map->base "
          "TF suppressed until a new initial pose.",
          drift, relocalized_path_, since_seed);
        return false;
      }
    }
  }

  // Map support: Euclidean distance from the pose to the nearest map point
  // (exact nearest neighbor on the loaded map via the KD tree).
  PointType pose_pt;
  pose_pt.x = static_cast<float>(cur_pos(0));
  pose_pt.y = static_cast<float>(cur_pos(1));
  pose_pt.z = static_cast<float>(cur_pos(2));
  std::vector<int> nn_idx(1);
  std::vector<float> nn_sqdist(1);
  double nearest = std::numeric_limits<double>::infinity();
  if (map_kdtree_ && map_kdtree_->nearestKSearch(pose_pt, 1, nn_idx, nn_sqdist) > 0)
  {
    nearest = std::sqrt(static_cast<double>(nn_sqdist[0]));
  }

  // Motion: pose displacement rate over a ~1 s window of healthy poses. A
  // per-scan displacement threshold would false-trigger at high scan rates
  // (one small correction step over a 10 ms interval looks like a huge
  // speed); a jump that is impossible over ~1 s means the match is wrong.
  double speed = 0.0;
  {
    // Reference: the oldest buffered pose at least 1 s old, else the re-seed
    // pose (so the post-seed convergence steps are amortized over the window).
    double ref_t = 0.0;
    Eigen::Vector3d ref_p = Eigen::Vector3d::Zero();
    bool have_ref = false;
    for (const auto & h : pos_hist_)
    {
      if (lidar_beg_time - h.first >= 1.0)
      {
        ref_t = h.first;
        ref_p = h.second;
        have_ref = true;
      }
    }
    if (!have_ref && last_good_valid_)
    {
      ref_t = relocalized_time_;
      ref_p = relocalized_pos_;
      have_ref = (lidar_beg_time - ref_t) > 1e-6;
    }
    if (have_ref)
    {
      speed = (cur_pos - ref_p).norm() / (lidar_beg_time - ref_t);
    }
  }

  const bool ok =
    nearest <= loc_params_.max_pose_to_map_dist && speed <= loc_params_.max_pose_speed;

  if (ok)
  {
    last_good_valid_ = true;
    last_good_pos_ = cur_pos;
    last_good_time_ = lidar_beg_time;
    pos_hist_.emplace_back(lidar_beg_time, cur_pos);
    while (!pos_hist_.empty() && pos_hist_.front().first < lidar_beg_time - 2.0)
    {
      pos_hist_.pop_front();
    }
    if (localization_failed_)
    {
      localization_failed_ = false;
      RCLCPP_INFO(
        this->get_logger(), "Localization recovered: nearest map point %.2f m, speed %.2f m/s.",
        nearest, speed);
    }
  }
  else
  {
    localization_failed_ = true;
    RCLCPP_WARN(
      this->get_logger(),
      "Localization FAILED: pose (%.2f, %.2f, %.2f), nearest map point %.2f m (> %.2f) or speed "
      "%.2f m/s (> %.2f); stopping localization, map->base TF suppressed until a new initial pose.",
      cur_pos(0), cur_pos(1), cur_pos(2), nearest, loc_params_.max_pose_to_map_dist, speed,
      loc_params_.max_pose_speed);
  }
  return ok;
}

// ---------------------------------------------------------------------------
// Sensor callbacks / synchronization
// ---------------------------------------------------------------------------

void LaserMappingNode::lidar_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(mtx_buffer_);
  scan_count_++;

  // Express the scan in the base frame before preprocessing.
  sensor_msgs::msg::PointCloud2 cloud_transformed;
  try
  {
    tf2::doTransform(*msg, cloud_transformed, tf_lidar_to_base_);
  }
  catch (const tf2::TransformException & ex)
  {
    RCLCPP_ERROR(this->get_logger(), "LiDAR transform failed: %s", ex.what());
    return;
  }

  const double cur_time = get_time_sec(msg->header.stamp);
  const double preprocess_start_time = omp_get_wtime();

  if (!is_first_lidar_ && cur_time < last_timestamp_lidar)
  {
    std::cerr << "lidar loop back, clear buffer" << std::endl;
    lidar_buffer_.clear();
    time_buffer_.clear();
    merge_accum_.clear();
    merge_count_ = 0;
  }
  if (is_first_lidar_)
  {
    is_first_lidar_ = false;
  }

  PointCloudXYZI::Ptr ptr(new PointCloudXYZI(20000, 1));
  p_pre->process(cloud_transformed, ptr);

  if (frame_merge_num_ > 1)
  {
    // Accumulate frame_merge_num_ consecutive scans into one cloud. The
    // per-point curvature (offset ms w.r.t. the scan's first point) is shifted
    // by the scan's window offset so the merged cloud keeps a continuous
    // timeline for the per-point undistortion.
    if (merge_count_ == 0)
    {
      merge_window_start_ = cur_time;
      merge_accum_.clear();
    }
    const float offset_ms = static_cast<float>((cur_time - merge_window_start_) * 1000.0);
    if (offset_ms > 0.0f)
    {
      for (auto & pt : ptr->points)
      {
        pt.curvature += offset_ms;
      }
    }
    merge_accum_.insert(merge_accum_.end(), ptr->begin(), ptr->end());
    merge_count_++;
    if (merge_count_ >= frame_merge_num_)
    {
      if (merge_accum_.size() > 0)
      {
        PointCloudXYZI::Ptr merged(new PointCloudXYZI());
        merged->swap(merge_accum_);
        lidar_buffer_.emplace_back(merged);
        time_buffer_.emplace_back(merge_window_start_);
      }
      merge_count_ = 0;
    }
  }
  else if (ptr->points.size() > 0)
  {
    lidar_buffer_.emplace_back(ptr);
    time_buffer_.emplace_back(cur_time);
  }
  last_timestamp_lidar = cur_time;
  s_plot11[scan_count_] = omp_get_wtime() - preprocess_start_time;
}

void LaserMappingNode::imu_callback(const sensor_msgs::msg::Imu::ConstSharedPtr msg_in)
{
  // Rotate the IMU measurement from the lidar frame into the base frame using
  // the cached static transform.
  sensor_msgs::msg::Imu imu_transformed = *msg_in;

  tf2::Quaternion q_tf;
  tf2::fromMsg(tf_lidar_to_base_.transform.rotation, q_tf);

  tf2::Quaternion q_imu;
  tf2::fromMsg(msg_in->orientation, q_imu);
  imu_transformed.orientation = tf2::toMsg(q_tf * q_imu);

  tf2::Vector3 gyro(
    msg_in->angular_velocity.x, msg_in->angular_velocity.y, msg_in->angular_velocity.z);
  gyro = tf2::quatRotate(q_tf, gyro);
  imu_transformed.angular_velocity.x = gyro.x();
  imu_transformed.angular_velocity.y = gyro.y();
  imu_transformed.angular_velocity.z = gyro.z();

  tf2::Vector3 accel(
    msg_in->linear_acceleration.x, msg_in->linear_acceleration.y, msg_in->linear_acceleration.z);
  accel = tf2::quatRotate(q_tf, accel);
  imu_transformed.linear_acceleration.x = accel.x();
  imu_transformed.linear_acceleration.y = accel.y();
  imu_transformed.linear_acceleration.z = accel.z();

  auto msg = std::make_shared<sensor_msgs::msg::Imu>(imu_transformed);
  const double t_shifted = get_time_sec(msg_in->header.stamp) - time_diff_lidar_to_imu;
  msg->header.stamp.sec = static_cast<int32_t>(std::floor(t_shifted));
  msg->header.stamp.nanosec =
    static_cast<uint32_t>((t_shifted - std::floor(t_shifted)) * 1e9);

  const double timestamp = get_time_sec(msg->header.stamp);

  std::lock_guard<std::mutex> lock(mtx_buffer_);
  if (timestamp < last_timestamp_imu)
  {
    std::cerr << "imu loop back, clear deque" << std::endl;
    imu_deque_.clear();
  }
  imu_deque_.emplace_back(msg);
  last_timestamp_imu = timestamp;
}

void LaserMappingNode::initial_pose_callback(
  const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
{
  init_pose_ = *msg;
  initial_pose_set_.store(true);
  RCLCPP_INFO(this->get_logger(), "Initial pose received from Rviz and applied.");
}

bool LaserMappingNode::sync_packages(MeasureGroup & meas)
{
  std::lock_guard<std::mutex> lock(mtx_buffer_);

  if (!imu_en)
  {
    if (!lidar_buffer_.empty())
    {
      if (!lidar_pushed_)
      {
        meas.lidar = lidar_buffer_.front();
        meas.lidar_beg_time = time_buffer_.front();
        lose_lid_ = false;
        if (meas.lidar->points.size() < 1)
        {
          cout << "lose lidar" << std::endl;
          lose_lid_ = true;
        }
        else
        {
          double end_time = meas.lidar->points.back().curvature;
          for (auto pt : meas.lidar->points)
          {
            if (pt.curvature > end_time)
            {
              end_time = pt.curvature;
            }
          }
          lidar_end_time = meas.lidar_beg_time + end_time / double(1000);
          meas.lidar_last_time = lidar_end_time;
        }
        lidar_pushed_ = true;
      }

      time_buffer_.pop_front();
      lidar_buffer_.pop_front();
      lidar_pushed_ = false;
      return !lose_lid_;
    }
    return false;
  }

  if (lidar_buffer_.empty() || imu_deque_.empty())
  {
    return false;
  }
  /*** push a lidar scan ***/
  if (!lidar_pushed_)
  {
    lose_lid_ = false;
    meas.lidar = lidar_buffer_.front();
    meas.lidar_beg_time = time_buffer_.front();
    if (meas.lidar->points.size() < 1)
    {
      cout << "lose lidar" << endl;
      lose_lid_ = true;
    }
    else
    {
      double end_time = meas.lidar->points.back().curvature;
      for (auto pt : meas.lidar->points)
      {
        if (pt.curvature > end_time)
        {
          end_time = pt.curvature;
        }
      }
      lidar_end_time = meas.lidar_beg_time + end_time / double(1000);
      meas.lidar_last_time = lidar_end_time;
    }
    lidar_pushed_ = true;
  }

  if (!lose_lid_ && (last_timestamp_imu < lidar_end_time))
  {
    return false;
  }
  if (lose_lid_ && last_timestamp_imu < meas.lidar_beg_time + lidar_time_inte)
  {
    return false;
  }

  if (!lose_lid_ && !imu_pushed_)
  {
    /*** push imu data, and pop from imu buffer ***/
    if (p_imu->imu_need_init_)
    {
      double imu_time = get_time_sec(imu_deque_.front()->header.stamp);
      imu_next_ = *(imu_deque_.front());
      meas.imu.clear();
      while (imu_time < lidar_end_time)
      {
        sensor_msgs::msg::Imu::SharedPtr imu_msg = imu_deque_.front();
        meas.imu.emplace_back(imu_msg);
        imu_last_ = imu_next_;
        imu_deque_.pop_front();
        if (imu_deque_.empty()) break;
        imu_time = get_time_sec(imu_deque_.front()->header.stamp);
        imu_next_ = *(imu_deque_.front());
      }
    }
    imu_pushed_ = true;
  }

  if (lose_lid_ && !imu_pushed_)
  {
    /*** push imu data, and pop from imu buffer ***/
    if (p_imu->imu_need_init_)
    {
      double imu_time = get_time_sec(imu_deque_.front()->header.stamp);
      meas.imu.clear();
      imu_next_ = *(imu_deque_.front());
      while (imu_time < meas.lidar_beg_time + lidar_time_inte)
      {
        sensor_msgs::msg::Imu::SharedPtr imu_msg = imu_deque_.front();
        meas.imu.emplace_back(imu_msg);
        imu_last_ = imu_next_;
        imu_deque_.pop_front();
        if (imu_deque_.empty()) break;
        imu_time = get_time_sec(imu_deque_.front()->header.stamp);
        imu_next_ = *(imu_deque_.front());
      }
    }
    imu_pushed_ = true;
  }

  lidar_buffer_.pop_front();
  time_buffer_.pop_front();
  lidar_pushed_ = false;
  imu_pushed_ = false;
  return true;
}

}  // namespace point_lio
// Point-LIO per-scan pipeline: the ROS1 main() while-loop body, ported to the
// node's timer callback, plus the localization-mode gating (initial pose wait,
// relocalization seeding, failure suppression).

#include "laserMapping.hpp"

#include <omp.h>

#include <pcl_conversions/pcl_conversions.h>

namespace
{
// Debug/averaging state of the runtime_pos_log block (per scan).
int frame_num = 0;
int time_log_counter = 0;
double aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0,
       aver_time_solve = 0, aver_time_propag = 0;
V3D euler_cur;

// Undistorted scan of the current package (global, as in the ROS1 version).
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());

inline double imu_sec(const sensor_msgs::msg::Imu & m)
{
  return get_time_sec(m.header.stamp);
}

template <typename T>
void set_posestamp(T & out)
{
  if (!use_imu_as_input)
  {
    out.position.x = kf_output.x_.pos(0);
    out.position.y = kf_output.x_.pos(1);
    out.position.z = kf_output.x_.pos(2);
    Eigen::Quaterniond q(kf_output.x_.rot);
    out.orientation.x = q.coeffs()[0];
    out.orientation.y = q.coeffs()[1];
    out.orientation.z = q.coeffs()[2];
    out.orientation.w = q.coeffs()[3];
  }
  else
  {
    out.position.x = kf_input.x_.pos(0);
    out.position.y = kf_input.x_.pos(1);
    out.position.z = kf_input.x_.pos(2);
    Eigen::Quaterniond q(kf_input.x_.rot);
    out.orientation.x = q.coeffs()[0];
    out.orientation.y = q.coeffs()[1];
    out.orientation.z = q.coeffs()[2];
    out.orientation.w = q.coeffs()[3];
  }
}

void pointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
  V3D p_body_lidar(pi->x, pi->y, pi->z);
  V3D p_body_imu;
  if (extrinsic_est_en)
  {
    if (!use_imu_as_input)
    {
      p_body_imu = kf_output.x_.offset_R_L_I * p_body_lidar + kf_output.x_.offset_T_L_I;
    }
    else
    {
      p_body_imu = kf_input.x_.offset_R_L_I * p_body_lidar + kf_input.x_.offset_T_L_I;
    }
  }
  else
  {
    p_body_imu = Lidar_R_wrt_IMU * p_body_lidar + Lidar_T_wrt_IMU;
  }
  po->x = p_body_imu(0);
  po->y = p_body_imu(1);
  po->z = p_body_imu(2);
  po->intensity = pi->intensity;
}
}  // namespace

namespace point_lio
{

// ---------------------------------------------------------------------------
// Publishing helpers
// ---------------------------------------------------------------------------

void LaserMappingNode::publish_odometry()
{
  odom_aft_mapped_.header.frame_id = map_frame_;
  odom_aft_mapped_.child_frame_id = base_frame_;
  if (publish_odometry_without_downsample)
  {
    odom_aft_mapped_.header.stamp = stamp_from_sec(time_current);
  }
  else
  {
    odom_aft_mapped_.header.stamp = stamp_from_sec(lidar_end_time);
  }
  set_posestamp(odom_aft_mapped_.pose.pose);

  pub_odom_->publish(odom_aft_mapped_);

  geometry_msgs::msg::TransformStamped trans;
  trans.header.frame_id = map_frame_;
  trans.header.stamp = odom_aft_mapped_.header.stamp;
  trans.child_frame_id = base_frame_;
  trans.transform.translation.x = odom_aft_mapped_.pose.pose.position.x;
  trans.transform.translation.y = odom_aft_mapped_.pose.pose.position.y;
  trans.transform.translation.z = odom_aft_mapped_.pose.pose.position.z;
  trans.transform.rotation = odom_aft_mapped_.pose.pose.orientation;
  tf_broadcaster_->sendTransform(trans);
}

void LaserMappingNode::publish_path()
{
  set_posestamp(msg_body_pose_.pose);
  msg_body_pose_.header.stamp = stamp_from_sec(lidar_end_time);
  msg_body_pose_.header.frame_id = map_frame_;
  path_.poses.emplace_back(msg_body_pose_);
  pub_path_->publish(path_);
}

void LaserMappingNode::publish_frame_world()
{
  if (scan_pub_en)
  {
    int size = feats_down_world->points.size();
    PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));
    for (int i = 0; i < size; i++)
    {
      laserCloudWorld->points[i].x = feats_down_world->points[i].x;
      laserCloudWorld->points[i].y = feats_down_world->points[i].y;
      laserCloudWorld->points[i].z = feats_down_world->points[i].z;
      laserCloudWorld->points[i].intensity = feats_down_world->points[i].intensity;
    }
    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
    laserCloudmsg.header.stamp = stamp_from_sec(lidar_end_time);
    laserCloudmsg.header.frame_id = map_frame_;
    pub_laser_cloud_full_->publish(laserCloudmsg);
  }

  /**************** save map ****************/
  if (pcd_save_en)
  {
    int size = feats_down_world->points.size();
    PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));
    for (int i = 0; i < size; i++)
    {
      laserCloudWorld->points[i].x = feats_down_world->points[i].x;
      laserCloudWorld->points[i].y = feats_down_world->points[i].y;
      laserCloudWorld->points[i].z = feats_down_world->points[i].z;
      laserCloudWorld->points[i].intensity = feats_down_world->points[i].intensity;
    }
    *pcl_wait_save_ += *laserCloudWorld;

    static int scan_wait_num = 0;
    scan_wait_num++;
    if (pcl_wait_save_->size() > 0 && scan_wait_num >= pcd_save_interval)
    {
      pcd_index++;
      std::string all_points_dir(
        std::string(ROOT_DIR) + "PCD/scans_" + std::to_string(pcd_index) + ".pcd");
      pcl::PCDWriter pcd_writer;
      std::cout << "current scan saved to " << all_points_dir << std::endl;
      pcd_writer.writeBinary(all_points_dir, *pcl_wait_save_);
      pcl_wait_save_->clear();
      scan_wait_num = 0;
    }
  }
}

void LaserMappingNode::publish_frame_body()
{
  int size = feats_undistort->points.size();
  PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));
  for (int i = 0; i < size; i++)
  {
    pointBodyLidarToIMU(&feats_undistort->points[i], &laserCloudIMUBody->points[i]);
  }
  sensor_msgs::msg::PointCloud2 laserCloudmsg;
  pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
  laserCloudmsg.header.stamp = stamp_from_sec(lidar_end_time);
  laserCloudmsg.header.frame_id = base_frame_;
  pub_laser_cloud_full_body_->publish(laserCloudmsg);
}

void LaserMappingNode::publish_init_map()
{
  sensor_msgs::msg::PointCloud2 laserCloudmsg;
  pcl::toROSMsg(*init_feats_world_, laserCloudmsg);
  laserCloudmsg.header.stamp = stamp_from_sec(lidar_end_time);
  laserCloudmsg.header.frame_id = map_frame_;
  pub_laser_cloud_map_->publish(laserCloudmsg);
}

void LaserMappingNode::publish_map()
{
  if (!map_raw_)
  {
    return;  // mapping mode: nothing loaded to show yet
  }
  PointCloudXYZI::Ptr map_pub(new PointCloudXYZI());
  down_size_filter_map_.setInputCloud(map_raw_);
  down_size_filter_map_.filter(*map_pub);
  sensor_msgs::msg::PointCloud2 laserCloudmsg;
  pcl::toROSMsg(*map_pub, laserCloudmsg);
  laserCloudmsg.header.stamp = this->get_clock()->now();
  laserCloudmsg.header.frame_id = map_frame_;
  pub_laser_cloud_map_->publish(laserCloudmsg);
}

void LaserMappingNode::save_to_pcd()
{
  pcl::PCDWriter pcd_writer;
  pcd_writer.writeBinary(
    std::string(ROOT_DIR) + "PCD/scans.pcd", *pcl_wait_save_);
}

void LaserMappingNode::map_save_callback(
  const std_srvs::srv::Trigger::Request::SharedPtr,
  std_srvs::srv::Trigger::Response::SharedPtr res)
{
  if (pcd_save_en)
  {
    save_to_pcd();
    res->success = true;
    res->message = "Map saved.";
  }
  else
  {
    res->success = false;
    res->message = "Map save disabled.";
  }
}

// ---------------------------------------------------------------------------
// Main loop (one synchronized package per timer tick)
// ---------------------------------------------------------------------------

void LaserMappingNode::timer_callback()
{
  if (!sync_packages(Measures))
  {
    return;
  }

  if (flg_reset_)
  {
    RCLCPP_WARN(this->get_logger(), "reset when rosbag play back");
    p_imu->Reset();
    feats_undistort.reset(new PointCloudXYZI());
    state_in = state_input();
    {
      Eigen::Matrix<double, 24, 24> P_reset;
      reset_cov(P_reset);
      kf_input.change_P(P_reset);
    }
    state_out = state_output();
    {
      Eigen::Matrix<double, 30, 30> P_reset_out;
      reset_cov_output(P_reset_out);
      kf_output.change_P(P_reset_out);
    }
    flg_first_scan_ = true;
    is_first_frame = true;
    flg_reset_ = false;
    if (loc_params_.enable_localization)
    {
      // Keep the loaded map; require a new /initialpose to re-seed.
      localization_seeded_ = false;
      initial_pose_set_.store(false);
    }
    else
    {
      init_map_ = false;
      ivox_.reset(new IVoxType(ivox_options_));
    }
  }

  // (Re-)localization request: match the current scan against the loaded map
  // and, only on a trusted match, seed the filter. The seed scan is consumed.
  if (loc_params_.enable_localization && initial_pose_set_.load())
  {
    initial_pose_set_.store(false);
    if (relocalize_and_seed())
    {
      return;
    }
    localization_failed_ = true;
    RCLCPP_WARN(
      this->get_logger(),
      "Relocalization did not produce a trusted pose; localization stays "
      "stopped until a new %s is received.",
      loc_params_.initial_pose_topic.c_str());
    return;
  }

  // Localization mode does nothing until the operator's first /initialpose.
  if (loc_params_.enable_localization && !localization_seeded_)
  {
    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 5000,
      "Waiting for initial pose on %s before starting localization.",
      loc_params_.initial_pose_topic.c_str());
    return;
  }

  // Localization was declared lost on an earlier scan: hold the pose and
  // suppress the map->base TF until a new /initialpose re-seeds the filter.
  if (loc_params_.enable_localization && localization_failed_)
  {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 5000,
      "Localization lost (pose off the map or moving too fast); map->base TF "
      "suppressed. Send a new %s to re-seed.", loc_params_.initial_pose_topic.c_str());
    return;
  }

  if (flg_first_scan_)
  {
    first_lidar_time = Measures.lidar_beg_time;
    flg_first_scan_ = false;
    if (first_imu_time < 1)
    {
      first_imu_time = imu_sec(imu_next_);
      printf("first imu time: %f\n", first_imu_time);
    }
    time_current = 0.0;
    if (imu_en)
    {
      kf_input.x_.gravity << VEC_FROM_ARRAY(gravity);
      kf_output.x_.gravity << VEC_FROM_ARRAY(gravity);
      {
        while (Measures.lidar_beg_time > imu_sec(imu_next_))  // needed for a new map
        {
          imu_deque_.pop_front();
          if (imu_deque_.empty())
          {
            break;
          }
          imu_last_ = imu_next_;
          imu_next_ = *(imu_deque_.front());
        }
      }
    }
    else
    {
      kf_input.x_.gravity << VEC_FROM_ARRAY(gravity);
      kf_output.x_.gravity << VEC_FROM_ARRAY(gravity);
      kf_output.x_.acc << VEC_FROM_ARRAY(gravity);
      kf_output.x_.acc *= -1;
      p_imu->imu_need_init_ = false;
    }
    G_m_s2 = std::sqrt(gravity[0] * gravity[0] + gravity[1] * gravity[1] + gravity[2] * gravity[2]);
  }

  double t0, t1, t2, t3, t5, solve_start;
  match_time = 0;
  solve_time = 0;
  propag_time = 0;
  update_time = 0;
  t0 = omp_get_wtime();

  /*** downsample the feature points in a scan ***/
  t1 = omp_get_wtime();
  p_imu->Process(Measures, feats_undistort);
  if (space_down_sample)
  {
    down_size_filter_surf_.setInputCloud(feats_undistort);
    down_size_filter_surf_.filter(*feats_down_body);
    std::sort(feats_down_body->points.begin(), feats_down_body->points.end(), time_list);
  }
  else
  {
    *feats_down_body = *Measures.lidar;
    std::sort(feats_down_body->points.begin(), feats_down_body->points.end(), time_list);
  }
  {
    time_seq = time_compressing<int>(feats_down_body);
    feats_down_size = feats_down_body->points.size();
  }

  if (!p_imu->after_imu_init_)  // mapping mode: initialize gravity from the IMU
  {
    if (!p_imu->imu_need_init_)
    {
      V3D tmp_gravity;
      if (imu_en)
      {
        tmp_gravity = -p_imu->mean_acc / p_imu->mean_acc.norm() * G_m_s2;
      }
      else
      {
        tmp_gravity << VEC_FROM_ARRAY(gravity_init);
        p_imu->after_imu_init_ = true;
      }
      M3D rot_init;
      p_imu->Set_init(tmp_gravity, rot_init);
      kf_input.x_.rot = rot_init;
      kf_output.x_.rot = rot_init;
      kf_output.x_.acc = -rot_init.transpose() * kf_output.x_.gravity;
    }
    else
    {
      return;
    }
  }

  /*** initialize the map (mapping mode only; localization pre-loads the map) ***/
  if (!init_map_)
  {
    feats_down_world->resize(feats_undistort->size());
    for (size_t i = 0; i < feats_undistort->size(); i++)
    {
      pointBodyToWorld(&(feats_undistort->points[i]), &(feats_down_world->points[i]));
    }
    for (size_t i = 0; i < feats_down_world->size(); i++)
    {
      init_feats_world_->points.emplace_back(feats_down_world->points[i]);
    }
    if (init_feats_world_->size() < static_cast<size_t>(init_map_size))
    {
      init_map_ = false;
    }
    else
    {
      ivox_->AddPoints(init_feats_world_->points);
      publish_init_map();
      init_feats_world_.reset(new PointCloudXYZI());
      init_map_ = true;
    }
    return;
  }

  /*** ICP and Kalman filter update ***/
  normvec->resize(feats_down_size);
  feats_down_world->resize(feats_down_size);
  Nearest_Points.resize(feats_down_size);

  t2 = omp_get_wtime();

  /*** iterated state estimation ***/
  crossmat_list.reserve(feats_down_size);
  pbody_list.reserve(feats_down_size);

  for (size_t i = 0; i < feats_down_body->size(); i++)
  {
    V3D point_this(
      feats_down_body->points[i].x, feats_down_body->points[i].y, feats_down_body->points[i].z);
    pbody_list[i] = point_this;
    if (!extrinsic_est_en)
    {
      point_this = Lidar_R_wrt_IMU * point_this + Lidar_T_wrt_IMU;
      M3D point_crossmat;
      point_crossmat << SKEW_SYM_MATRX(point_this);
      crossmat_list[i] = point_crossmat;
    }
  }

  if (!use_imu_as_input)
  {
    bool imu_upda_cov = false;
    effct_feat_num = 0;
    /**** point by point update ****/
    if (time_seq.size() > 0)
    {
      double pcl_beg_time = Measures.lidar_beg_time;
      idx = -1;
      for (k = 0; k < static_cast<int>(time_seq.size()); k++)
      {
        PointType & point_body = feats_down_body->points[idx + time_seq[k]];
        time_current = point_body.curvature / 1000.0 + pcl_beg_time;

        if (is_first_frame)
        {
          if (imu_en)
          {
            while (time_current > imu_sec(imu_next_))
            {
              imu_deque_.pop_front();
              if (imu_deque_.empty()) break;
              imu_last_ = imu_next_;
              imu_next_ = *(imu_deque_.front());
            }
            angvel_avr << imu_last_.angular_velocity.x, imu_last_.angular_velocity.y,
              imu_last_.angular_velocity.z;
            acc_avr << imu_last_.linear_acceleration.x, imu_last_.linear_acceleration.y,
              imu_last_.linear_acceleration.z;
          }
          is_first_frame = false;
          imu_upda_cov = true;
          time_update_last = time_current;
          time_predict_last_const = time_current;
        }
        if (imu_en && !imu_deque_.empty())
        {
          bool last_imu = imu_sec(imu_next_) == imu_sec(*(imu_deque_.front()));
          while (imu_sec(imu_next_) < time_predict_last_const && !imu_deque_.empty())
          {
            if (!last_imu)
            {
              imu_last_ = imu_next_;
              imu_next_ = *(imu_deque_.front());
              break;
            }
            else
            {
              imu_deque_.pop_front();
              if (imu_deque_.empty()) break;
              imu_last_ = imu_next_;
              imu_next_ = *(imu_deque_.front());
            }
          }
          bool imu_comes = time_current > imu_sec(imu_next_);
          while (imu_comes)
          {
            imu_upda_cov = true;
            angvel_avr << imu_next_.angular_velocity.x, imu_next_.angular_velocity.y,
              imu_next_.angular_velocity.z;
            acc_avr << imu_next_.linear_acceleration.x, imu_next_.linear_acceleration.y,
              imu_next_.linear_acceleration.z;

            /*** covariance update ***/
            double dt = imu_sec(imu_next_) - time_predict_last_const;
            kf_output.predict(dt, Q_output_, input_in, true, false);
            time_predict_last_const = imu_sec(imu_next_);

            {
              double dt_cov = imu_sec(imu_next_) - time_update_last;
              if (dt_cov > 0.0)
              {
                time_update_last = imu_sec(imu_next_);
                double propag_imu_start = omp_get_wtime();

                kf_output.predict(dt_cov, Q_output_, input_in, false, true);

                propag_time += omp_get_wtime() - propag_imu_start;
                double solve_imu_start = omp_get_wtime();
                kf_output.update_iterated_dyn_share_IMU();
                solve_time += omp_get_wtime() - solve_imu_start;
              }
            }
            imu_deque_.pop_front();
            if (imu_deque_.empty()) break;
            imu_last_ = imu_next_;
            imu_next_ = *(imu_deque_.front());
            imu_comes = time_current > imu_sec(imu_next_);
          }
        }
        if (flg_reset_)
        {
          break;
        }

        double dt = time_current - time_predict_last_const;
        double propag_state_start = omp_get_wtime();
        if (!prop_at_freq_of_imu)
        {
          double dt_cov = time_current - time_update_last;
          if (dt_cov > 0.0)
          {
            kf_output.predict(dt_cov, Q_output_, input_in, false, true);
            time_update_last = time_current;
          }
        }
        kf_output.predict(dt, Q_output_, input_in, true, false);
        propag_time += omp_get_wtime() - propag_state_start;
        time_predict_last_const = time_current;
        double t_update_start = omp_get_wtime();

        if (feats_down_size < 1)
        {
          printf("No point, skip this scan!\n");
          idx += time_seq[k];
          continue;
        }
        if (!kf_output.update_iterated_dyn_share_modified())
        {
          idx = idx + time_seq[k];
          continue;
        }
        solve_start = omp_get_wtime();

        if (publish_odometry_without_downsample)
        {
          /******* Publish odometry *******/
          publish_odometry();
          if (runtime_pos_log)
          {
            euler_cur = SO3ToEuler(kf_output.x_.rot);
            fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " "
                     << euler_cur.transpose() << " " << kf_output.x_.pos.transpose() << " "
                     << kf_output.x_.vel.transpose() << " " << kf_output.x_.omg.transpose() << " "
                     << kf_output.x_.acc.transpose() << " " << kf_output.x_.gravity.transpose()
                     << " " << kf_output.x_.bg.transpose() << " " << kf_output.x_.ba.transpose()
                     << " " << feats_undistort->points.size() << endl;
          }
        }

        for (int j = 0; j < time_seq[k]; j++)
        {
          PointType & point_body_j = feats_down_body->points[idx + j + 1];
          PointType & point_world_j = feats_down_world->points[idx + j + 1];
          pointBodyToWorld(&point_body_j, &point_world_j);
        }

        solve_time += omp_get_wtime() - solve_start;

        update_time += omp_get_wtime() - t_update_start;
        idx += time_seq[k];
      }
    }
    else
    {
      if (!imu_deque_.empty())
      {
        imu_last_ = imu_next_;
        imu_next_ = *(imu_deque_.front());

        while (imu_sec(imu_next_) > time_current &&
               (imu_sec(imu_next_) < Measures.lidar_beg_time + lidar_time_inte))
        {
          if (is_first_frame)
          {
            while (imu_sec(imu_next_) < Measures.lidar_beg_time + lidar_time_inte)
            {
              imu_deque_.pop_front();
              if (imu_deque_.empty()) break;
              imu_last_ = imu_next_;
              imu_next_ = *(imu_deque_.front());
            }
            angvel_avr << imu_last_.angular_velocity.x, imu_last_.angular_velocity.y,
              imu_last_.angular_velocity.z;
            acc_avr << imu_last_.linear_acceleration.x, imu_last_.linear_acceleration.y,
              imu_last_.linear_acceleration.z;
            imu_upda_cov = true;
            time_update_last = time_current;
            time_predict_last_const = time_current;
            is_first_frame = false;
            break;
          }
          time_current = imu_sec(imu_next_);

          if (!is_first_frame)
          {
            double dt = time_current - time_predict_last_const;
            {
              double dt_cov = time_current - time_update_last;
              if (dt_cov > 0.0)
              {
                kf_output.predict(dt_cov, Q_output_, input_in, false, true);
                time_update_last = time_current;
              }
              kf_output.predict(dt, Q_output_, input_in, true, false);
            }
            time_predict_last_const = time_current;

            angvel_avr << imu_next_.angular_velocity.x, imu_next_.angular_velocity.y,
              imu_next_.angular_velocity.z;
            acc_avr << imu_next_.linear_acceleration.x, imu_next_.linear_acceleration.y,
              imu_next_.linear_acceleration.z;
            kf_output.update_iterated_dyn_share_IMU();
            imu_deque_.pop_front();
            if (imu_deque_.empty()) break;
            imu_last_ = imu_next_;
            imu_next_ = *(imu_deque_.front());
          }
          else
          {
            imu_deque_.pop_front();
            if (imu_deque_.empty()) break;
            imu_last_ = imu_next_;
            imu_next_ = *(imu_deque_.front());
          }
        }
      }
    }
  }
  else
  {
    bool imu_prop_cov = false;
    effct_feat_num = 0;
    if (time_seq.size() > 0)
    {
      double pcl_beg_time = Measures.lidar_beg_time;
      idx = -1;
      for (k = 0; k < static_cast<int>(time_seq.size()); k++)
      {
        PointType & point_body = feats_down_body->points[idx + time_seq[k]];
        time_current = point_body.curvature / 1000.0 + pcl_beg_time;
        if (is_first_frame)
        {
          while (time_current > imu_sec(imu_next_))
          {
            imu_deque_.pop_front();
            if (imu_deque_.empty()) break;
            imu_last_ = imu_next_;
            imu_next_ = *(imu_deque_.front());
          }
          imu_prop_cov = true;

          is_first_frame = false;
          t_last = time_current;
          time_update_last = time_current;
          {
            input_in.gyro << imu_last_.angular_velocity.x, imu_last_.angular_velocity.y,
              imu_last_.angular_velocity.z;
            input_in.acc << imu_last_.linear_acceleration.x, imu_last_.linear_acceleration.y,
              imu_last_.linear_acceleration.z;
            input_in.acc = input_in.acc * G_m_s2 / acc_norm;
          }
        }

        while (time_current > imu_sec(imu_next_))
        {
          imu_deque_.pop_front();

          input_in.gyro << imu_last_.angular_velocity.x, imu_last_.angular_velocity.y,
            imu_last_.angular_velocity.z;
          input_in.acc << imu_last_.linear_acceleration.x, imu_last_.linear_acceleration.y,
            imu_last_.linear_acceleration.z;
          input_in.acc = input_in.acc * G_m_s2 / acc_norm;
          double dt = imu_sec(imu_last_) - t_last;

          double dt_cov = imu_sec(imu_last_) - time_update_last;
          if (dt_cov > 0.0)
          {
            kf_input.predict(dt_cov, Q_input_, input_in, false, true);
            time_update_last = imu_sec(imu_last_);
          }
          kf_input.predict(dt, Q_input_, input_in, true, false);
          t_last = imu_sec(imu_last_);
          imu_prop_cov = true;

          if (imu_deque_.empty()) break;
          imu_last_ = imu_next_;
          imu_next_ = *(imu_deque_.front());
        }
        if (flg_reset_)
        {
          break;
        }
        double dt = time_current - t_last;
        t_last = time_current;
        double propag_start = omp_get_wtime();

        if (!prop_at_freq_of_imu)
        {
          double dt_cov = time_current - time_update_last;
          if (dt_cov > 0.0)
          {
            kf_input.predict(dt_cov, Q_input_, input_in, false, true);
            time_update_last = time_current;
          }
        }
        kf_input.predict(dt, Q_input_, input_in, true, false);

        propag_time += omp_get_wtime() - propag_start;

        double t_update_start = omp_get_wtime();

        if (feats_down_size < 1)
        {
          printf("No point, skip this scan!\n");
          idx += time_seq[k];
          continue;
        }
        if (!kf_input.update_iterated_dyn_share_modified())
        {
          idx = idx + time_seq[k];
          continue;
        }

        solve_start = omp_get_wtime();

        if (publish_odometry_without_downsample)
        {
          /******* Publish odometry *******/
          publish_odometry();
          if (runtime_pos_log)
          {
            euler_cur = SO3ToEuler(kf_input.x_.rot);
            fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " "
                     << euler_cur.transpose() << " " << kf_input.x_.pos.transpose() << " "
                     << kf_input.x_.vel.transpose() << " " << kf_input.x_.bg.transpose() << " "
                     << kf_input.x_.ba.transpose() << " " << kf_input.x_.gravity.transpose() << " "
                     << feats_undistort->points.size() << endl;
          }
        }

        for (int j = 0; j < time_seq[k]; j++)
        {
          PointType & point_body_j = feats_down_body->points[idx + j + 1];
          PointType & point_world_j = feats_down_world->points[idx + j + 1];
          pointBodyToWorld(&point_body_j, &point_world_j);
        }
        solve_time += omp_get_wtime() - solve_start;

        update_time += omp_get_wtime() - t_update_start;
        idx = idx + time_seq[k];
      }
    }
    else
    {
      if (!imu_deque_.empty())
      {
        imu_last_ = imu_next_;
        imu_next_ = *(imu_deque_.front());
        while (imu_sec(imu_next_) > time_current &&
               (imu_sec(imu_next_) < Measures.lidar_beg_time + lidar_time_inte))
        {
          if (is_first_frame)
          {
            while (imu_sec(imu_next_) < Measures.lidar_beg_time + lidar_time_inte)
            {
              imu_deque_.pop_front();
              if (imu_deque_.empty()) break;
              imu_last_ = imu_next_;
              imu_next_ = *(imu_deque_.front());
            }
            imu_prop_cov = true;

            t_last = time_current;
            time_update_last = time_current;
            input_in.gyro << imu_last_.angular_velocity.x, imu_last_.angular_velocity.y,
              imu_last_.angular_velocity.z;
            input_in.acc << imu_last_.linear_acceleration.x, imu_last_.linear_acceleration.y,
              imu_last_.linear_acceleration.z;
            input_in.acc = input_in.acc * G_m_s2 / acc_norm;

            is_first_frame = false;
            break;
          }
          time_current = imu_sec(imu_next_);

          if (!is_first_frame)
          {
            double dt = time_current - t_last;

            double dt_cov = time_current - time_update_last;
            if (dt_cov > 0.0)
            {
              time_update_last = imu_sec(imu_next_);
            }
            t_last = imu_sec(imu_next_);

            input_in.gyro << imu_next_.angular_velocity.x, imu_next_.angular_velocity.y,
              imu_next_.angular_velocity.z;
            input_in.acc << imu_next_.linear_acceleration.x, imu_next_.linear_acceleration.y,
              imu_next_.linear_acceleration.z;
            input_in.acc = input_in.acc * G_m_s2 / acc_norm;
            imu_deque_.pop_front();
            if (imu_deque_.empty()) break;
            imu_last_ = imu_next_;
            imu_next_ = *(imu_deque_.front());
          }
          else
          {
            imu_deque_.pop_front();
            if (imu_deque_.empty()) break;
            imu_last_ = imu_next_;
            imu_next_ = *(imu_deque_.front());
          }
        }
      }
    }
  }

  /******* Publish odometry downsample *******/
  if (!publish_odometry_without_downsample)
  {
    publish_odometry();
  }

  /*** add the feature points to map ***/
  t3 = omp_get_wtime();

  if (feats_down_size > 4 && loc_params_.enable_map_incremental)
  {
    PointVector points_to_add;
    int cur_pts = feats_down_world->size();
    points_to_add.reserve(cur_pts);

    for (int i = 0; i < cur_pts; ++i)
    {
      /* decide if need add to map */
      PointType & point_world = feats_down_world->points[i];
      if (!Nearest_Points[i].empty())
      {
        const PointVector & points_near = Nearest_Points[i];

        Eigen::Vector3f center =
          ((point_world.getVector3fMap() / filter_size_map_min).array().floor() + 0.5) *
          filter_size_map_min;
        bool need_add = true;
        for (size_t readd_i = 0; readd_i < points_near.size(); readd_i++)
        {
          Eigen::Vector3f dis_2_center = points_near[readd_i].getVector3fMap() - center;
          if (fabs(dis_2_center.x()) < 0.5 * filter_size_map_min &&
              fabs(dis_2_center.y()) < 0.5 * filter_size_map_min &&
              fabs(dis_2_center.z()) < 0.5 * filter_size_map_min)
          {
            need_add = false;
            break;
          }
        }
        if (need_add)
        {
          points_to_add.emplace_back(point_world);
        }
      }
      else
      {
        points_to_add.emplace_back(point_world);
      }
    }
    ivox_->AddPoints(points_to_add);
  }

  t5 = omp_get_wtime();

  // Detect localization failure on the updated pose: if it drifted away from
  // the map support or moved faster than a plausible robot, set the failure
  // flag and suppress the TF/odometry for this scan (the hold at the top of
  // this callback keeps it suppressed until a new /initialpose).
  const bool healthy = localization_health_check(Measures.lidar_beg_time);

  /******* Publish points *******/
  if (healthy)
  {
    if (path_en) publish_path();
    if (scan_pub_en || pcd_save_en) publish_frame_world();
    if (scan_pub_en && scan_body_pub_en) publish_frame_body();
  }

  /*** Debug variables Logging ***/
  if (runtime_pos_log)
  {
    frame_num++;
    aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
    {
      aver_time_icp = aver_time_icp * (frame_num - 1) / frame_num + update_time / frame_num;
    }
    aver_time_match = aver_time_match * (frame_num - 1) / frame_num + (match_time) / frame_num;
    aver_time_solve = aver_time_solve * (frame_num - 1) / frame_num + solve_time / frame_num;
    aver_time_propag = aver_time_propag * (frame_num - 1) / frame_num + propag_time / frame_num;
    T1[time_log_counter] = Measures.lidar_beg_time;
    s_plot[time_log_counter] = t5 - t0;
    s_plot2[time_log_counter] = feats_undistort->points.size();
    s_plot3[time_log_counter] = aver_time_consu;
    time_log_counter++;
    printf(
      "[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  "
      "ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f propogate: %0.6f \n",
      t1 - t0, aver_time_match, aver_time_solve, t3 - t1, t5 - t3, aver_time_consu, aver_time_icp,
      aver_time_propag);
    if (!publish_odometry_without_downsample)
    {
      if (!use_imu_as_input)
      {
        euler_cur = SO3ToEuler(kf_output.x_.rot);
        fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " "
                 << euler_cur.transpose() << " " << kf_output.x_.pos.transpose() << " "
                 << kf_output.x_.vel.transpose() << " " << kf_output.x_.omg.transpose() << " "
                 << kf_output.x_.acc.transpose() << " " << kf_output.x_.gravity.transpose() << " "
                 << kf_output.x_.bg.transpose() << " " << kf_output.x_.ba.transpose() << " "
                 << feats_undistort->points.size() << endl;
      }
      else
      {
        euler_cur = SO3ToEuler(kf_input.x_.rot);
        fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " "
                 << euler_cur.transpose() << " " << kf_input.x_.pos.transpose() << " "
                 << kf_input.x_.vel.transpose() << " " << kf_input.x_.bg.transpose() << " "
                 << kf_input.x_.ba.transpose() << " " << kf_input.x_.gravity.transpose() << " "
                 << feats_undistort->points.size() << endl;
      }
    }
    // state dump to pos_log
    {
      V3D rot_ang = SO3ToEuler(use_imu_as_input ? kf_input.x_.rot : kf_output.x_.rot);
      fprintf(fp_, "%lf ", Measures.lidar_beg_time - first_lidar_time);
      fprintf(fp_, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));
      const V3D & pos = use_imu_as_input ? kf_input.x_.pos : kf_output.x_.pos;
      fprintf(fp_, "%lf %lf %lf ", pos(0), pos(1), pos(2));
      fprintf(fp_, "\r\n");
      fflush(fp_);
    }
  }
}

}  // namespace point_lio
