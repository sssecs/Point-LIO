import os.path

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node


def generate_launch_description():
    package_path = get_package_share_directory('point_lio')
    default_config_path = os.path.join(package_path, 'config')

    use_sim_time = LaunchConfiguration('use_sim_time')
    config_path = LaunchConfiguration('config_path')
    config_file = LaunchConfiguration('config_file')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Use simulation (rosbag) clock if true'
    )
    declare_config_path_cmd = DeclareLaunchArgument(
        'config_path', default_value=default_config_path,
        description='Yaml config file path'
    )
    declare_config_file_cmd = DeclareLaunchArgument(
        'config_file', default_value='mid360.yaml',
        description='Config file'
    )

    point_lio_node = Node(
        package='point_lio',
        executable='pointlio_mapping',
        parameters=[PathJoinSubstitution([config_path, config_file]),
                    {'use_sim_time': use_sim_time}],
        output='screen'
    )

    # The Mid-360 is mounted upside down: rotate it (and its IMU) 180 deg
    # around the x axis into the base frame. The node expresses both the scan
    # and the IMU in base_link using this static TF.
    lidar_to_base = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='lidar_to_base_tf',
        arguments=[
            '0', '0', '0',
            '0', '0', '3.14159265359',
            'base_link',
            'livox_frame'
        ]
    )

    ld = LaunchDescription()
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_config_path_cmd)
    ld.add_action(declare_config_file_cmd)
    ld.add_action(lidar_to_base)
    ld.add_action(point_lio_node)

    return ld
