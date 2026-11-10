#include <rclcpp/rclcpp.hpp>

#include "laserMapping.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<point_lio::LaserMappingNode>());
  rclcpp::shutdown();
  return 0;
}
