#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    usb_port = LaunchConfiguration('usb_port', default='/dev/ttyACM0')
    tb3_param_dir = LaunchConfiguration(
        'tb3_param_dir',
        default=os.path.join(get_package_share_directory('wheel'), 'models', 'turtlebot3_waffle.yaml'))
    urdf = os.path.join(get_package_share_directory('wheel'), 'models', 'turtlebot3_waffle.urdf')
    rplidar_param = {
        'serial_port': '/dev/ttyUSB0',
        'serial_baudrate': 256000,
        'frame_id': 'base_scan',
        'inverted': False,
        'angle_compensate': True,
        'scan_mode': 'Sensitivity'}

    return LaunchDescription([
        DeclareLaunchArgument(
           'use_sim_time',
           default_value=use_sim_time,
           description='Use simulation clock if true'),
        DeclareLaunchArgument(
            'usb_port',
            default_value=usb_port,
            description='Connected USB port with OpenCR'),
        DeclareLaunchArgument(
            'tb3_param_dir',
            default_value=tb3_param_dir,
            description='Full path to turtlebot3 parameter file to load'),
        Node(
            package='turtlebot3_node',
            node_executable='turtlebot3_ros',
            parameters=[tb3_param_dir],
            arguments=['-i', usb_port],
            output='screen',
            remappings=[('cmd_vel', 'vel')]),
        Node(
            package='robot_state_publisher',
            node_executable='robot_state_publisher',
            node_name='robot_state_publisher',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
            arguments=[urdf]),
        Node(
            package='rplidar_ros',
            node_executable='rplidar_composition',
            node_name='rplidar_composition',
            output='screen',
            remappings=[('scan', 'laser')],
            parameters=[rplidar_param]),
        Node(
            package='wheel',
            node_executable='filter',
            node_name='wheel_filter',
            output='screen'),
    ])