// Mid-360 only lidar preprocessor (ROS2).
//
// Point-LIO has been trimmed to run exclusively with the Livox Mid-360
// receiving sensor_msgs/PointCloud2 (as published by livox_ros_driver2).
// Each point carries a per-point timestamp used for scan undistortion.

#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sensor_msgs/msg/point_cloud2.hpp>

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

// Mid-360 PointCloud2 point layout. Keep the field set in sync with the
// driver configuration so pcl::fromROSMsg can fill the struct.
namespace livox_ros
{
struct LivoxPointXyzitlt
{
  float x;            /**< X axis, unit: m */
  float y;            /**< Y axis, unit: m */
  float z;            /**< Z axis, unit: m */
  float intensity;    /**< Intensity */
  uint8_t tag;        /**< Livox point tag */
  uint8_t line;       /**< Laser line id */
  double timestamp;   /**< Per-point timestamp [ns] */
};
}  // namespace livox_ros

POINT_CLOUD_REGISTER_POINT_STRUCT(
  livox_ros::LivoxPointXyzitlt,
  (float, x, x)
  (float, y, y)
  (float, z, z)
  (float, intensity, intensity)
  (uint8_t, tag, tag)
  (uint8_t, line, line)
  (double, timestamp, timestamp))

class Preprocess
{
public:
  Preprocess() = default;

  // Convert a Mid-360 PointCloud2 scan into the Point-LIO point cloud.
  // Point.curvature carries the per-point offset time in ms w.r.t. the first
  // point of the scan (used by the point-wise scan undistortion step).
  void process(const sensor_msgs::msg::PointCloud2 & msg, PointCloudXYZI::Ptr & pcl_out);

  double blind = 0.5;  // [m] points closer than this to the sensor are dropped

private:
  void mid360_handler(const sensor_msgs::msg::PointCloud2 & msg);

  PointCloudXYZI pl_surf_;
};
