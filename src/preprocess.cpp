#include "preprocess.h"

#include <pcl_conversions/pcl_conversions.h>

void Preprocess::process(const sensor_msgs::msg::PointCloud2 & msg, PointCloudXYZI::Ptr & pcl_out)
{
  mid360_handler(msg);
  *pcl_out = pl_surf_;
}

void Preprocess::mid360_handler(const sensor_msgs::msg::PointCloud2 & msg)
{
  pl_surf_.clear();

  pcl::PointCloud<livox_ros::LivoxPointXyzitlt> pl_orig;
  pcl::fromROSMsg(msg, pl_orig);
  const int plsize = static_cast<int>(pl_orig.points.size());
  if (plsize == 0)
  {
    return;
  }
  pl_surf_.reserve(plsize);

  // Livox points are timestamped by the driver, so no scan-yaw reconstruction
  // is needed. Store the offset w.r.t. the first point in curvature [ms].
  const double scan_start_time = pl_orig.points.front().timestamp;
  const double blind2 = blind * blind;

  for (const auto & pt : pl_orig.points)
  {
    if (pt.x * pt.x + pt.y * pt.y + pt.z * pt.z > blind2)
    {
      PointType added_pt;
      added_pt.normal_x = 0.0;
      added_pt.normal_y = 0.0;
      added_pt.normal_z = 0.0;
      added_pt.x = pt.x;
      added_pt.y = pt.y;
      added_pt.z = pt.z;
      added_pt.intensity = pt.intensity;
      added_pt.curvature = static_cast<float>((pt.timestamp - scan_start_time) * 1e-6);

      pl_surf_.push_back(added_pt);
    }
  }
}
