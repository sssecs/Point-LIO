// Point-LIO ROS2 node: Mid-360 localization/mapping against a pre-built map.
//
// Ported from the ROS1 Point-LIO laserMapping.cpp and restructured as a node:
//  - sensor callbacks + buffer synchronization (from li_initialization.cpp)
//  - the per-scan point-wise iterated-EKF pipeline (the former main loop)
//  - static-map localization mode (load PCD into iVox, no map update),
//    Open3D relocalization on /initialpose, and localization failure
//    detection mirroring the FAST_LIO_ROS2 localization node.

#pragma once

#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include <deque>
#include <fstream>
#include <mutex>

#include <pcl/kdtree/kdtree_flann.h>

#include "Estimator.h"
#include "parameters.h"
#include "open3d_localizer.hpp"

namespace point_lio
{

struct LocalizationParams
{
  bool enable_localization = false;    // true: localize against a pre-built map
  bool enable_map_incremental = true;  // false: never modify the loaded map
  bool relocalization_en = true;       // Open3D scan-to-map relocalization on /initialpose
  double leaf_size = 0.1;              // voxel size used to downsample the loaded map [m]
  std::string map_path;                // PCD of the pre-built map (map frame)
  double init_z = 1.2;                 // z-height forced onto the RViz 2D initial pose
  double tf_lookup_timeout = 5.0;      // [s] wait for the lidar->base static TF at startup
  std::string initial_pose_topic = "/initialpose";
  // Localization failure detection (pose must stay near the map and move plausibly).
  bool failure_detect_en = true;
  double max_pose_to_map_dist = 3.0;   // [m] farther than this from the nearest map point => lost
  double max_pose_speed = 3.0;         // [m/s] scan-to-scan displacement rate above this => lost
  // Post-relocalization fly-away guard (robot is stationary right after a re-seed).
  bool relocalization_verify_en = true;
  double relocalization_verify_window = 2.0;
  double relocalization_verify_max_drift = 0.1;
  double relocalization_verify_max_path = 0.2;
  // Open3D relocalization (see open3d_localizer.hpp).
  Open3DParams open3d;
};

class LaserMappingNode : public rclcpp::Node
{
public:
  LaserMappingNode();
  ~LaserMappingNode() override;

private:
  // --- parameters -----------------------------------------------------------
  void load_params();
  LocalizationParams loc_params_;
  std::string map_frame_ = "map";
  std::string base_frame_ = "base_link";
  std::string lidar_frame_ = "livox_frame";

  // --- map / relocalization -------------------------------------------------
  void load_map();             // PCD -> voxel downsample -> iVox (+ relocalizer target)
  bool relocalize_and_seed();  // align current scan with map, seed the EKF states
  bool localization_health_check(double lidar_beg_time);
  void rearm_relocalization(const Eigen::Vector3d & seed_pos, double lidar_beg_time);

  // --- sensor callbacks -----------------------------------------------------
  void lidar_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);
  void imu_callback(const sensor_msgs::msg::Imu::ConstSharedPtr msg);
  void initial_pose_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg);
  bool sync_packages(MeasureGroup & meas);

  // --- main loop ------------------------------------------------------------
  void timer_callback();

  // --- publishing -----------------------------------------------------------
  void publish_odometry();
  void publish_path();
  void publish_frame_world();
  void publish_frame_body();
  void publish_init_map();
  void publish_map();
  void save_to_pcd();
  void map_save_callback(
    const std_srvs::srv::Trigger::Request::SharedPtr req,
    std_srvs::srv::Trigger::Response::SharedPtr res);

  // --- ROS graph ------------------------------------------------------------
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr sub_init_pose_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_laser_cloud_full_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_laser_cloud_full_body_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_laser_cloud_map_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr map_save_srv_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  geometry_msgs::msg::TransformStamped tf_lidar_to_base_;

  // --- buffers (single producer/consumer, guarded by mtx_buffer_) ------------
  std::mutex mtx_buffer_;
  std::deque<PointCloudXYZI::Ptr> lidar_buffer_;
  std::deque<double> time_buffer_;
  std::deque<sensor_msgs::msg::Imu::SharedPtr> imu_deque_;
  bool lidar_pushed_ = false;
  bool imu_pushed_ = false;
  bool lose_lid_ = false;
  int scan_count_ = 0;
  // Accumulation buffer for the relocalization scan, filled only between an
  // /initialpose request and the seed: at high scan rates (e.g. 100 Hz) one
  // sweep is too sparse for a stable scan-to-map ICP match, so
  // reloc_accum_frames_ scans (~100 ms) are collected after the click and
  // matched against the map as one dense cloud. Normal tracking is unaffected
  // and runs at the full scan rate.
  int reloc_accum_frames_ = 10;
  std::deque<PointCloudXYZI::Ptr> reloc_accum_;
  std::atomic<bool> reloc_pending_{false};

  // --- state carried across scans (former main() locals/globals) -------------
  bool init_map_ = false;
  bool flg_first_scan_ = true;
  bool flg_reset_ = false;
  sensor_msgs::msg::Imu imu_last_, imu_next_;

  // --- initial pose / relocalization / failure state --------------------------
  geometry_msgs::msg::PoseWithCovarianceStamped init_pose_;
  std::atomic<bool> initial_pose_set_{false};
  bool localization_seeded_ = false;   // the EKF has been seeded with a trusted pose
  bool localization_failed_ = false;   // lost: suppress TF until a new /initialpose
  Open3DLocalizer localizer_;
  bool localizer_ready_ = false;
  PointCloudXYZI::Ptr map_raw_;        // full-resolution loaded map (map frame)
  // KD tree on the downsampled map for the exact pose-to-map distance check in
  // the health check (the iVox nearest search only covers the voxels adjacent
  // to the query point, so it cannot measure the distance of a pose hovering
  // between map surfaces).
  pcl::KdTreeFLANN<PointType>::Ptr map_kdtree_;
  // failure-detection reference state
  bool last_good_valid_ = false;
  Eigen::Vector3d last_good_pos_ = Eigen::Vector3d::Zero();
  double last_good_time_ = 0.0;
  // Ring buffer of recent healthy poses for the windowed speed check
  // (rate-independent: a per-scan displacement threshold false-triggers at
  // high scan rates, where one small correction step looks like a huge speed).
  std::deque<std::pair<double, Eigen::Vector3d>> pos_hist_;
  // post-relocalization fly-away guard
  bool relocalized_verify_ = false;
  Eigen::Vector3d relocalized_pos_ = Eigen::Vector3d::Zero();
  double relocalized_time_ = 0.0;
  double relocalized_path_ = 0.0;

  // --- misc ------------------------------------------------------------------
  nav_msgs::msg::Path path_;
  nav_msgs::msg::Odometry odom_aft_mapped_;
  geometry_msgs::msg::PoseStamped msg_body_pose_;
  // Last published TF/odom stamp, used to enforce monotonic stamps (a scan
  // with a short sweep span can produce a lidar_end_time slightly older than
  // the previous one; non-monotonic TF stamps make every consumer's tf buffer
  // spam TF_OLD_DATA and can flicker displays).
  double last_pub_stamp_ = -1.0;
  PointCloudXYZI::Ptr init_feats_world_{new PointCloudXYZI()};
  PointCloudXYZI::Ptr pcl_wait_save_{new PointCloudXYZI()};
  pcl::VoxelGrid<PointType> down_size_filter_surf_;
  pcl::VoxelGrid<PointType> down_size_filter_map_;
  FILE * fp_ = nullptr;
  bool is_first_lidar_ = true;
  // Constant process-noise covariances (built once, after the params are read).
  Eigen::Matrix<double, 24, 24> Q_input_;
  Eigen::Matrix<double, 30, 30> Q_output_;
};

}  // namespace point_lio
