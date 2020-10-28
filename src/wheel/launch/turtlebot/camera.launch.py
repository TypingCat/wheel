#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    return LaunchDescription([
        Node(
            package='realsense_ros2_camera',
            node_executable='realsense_ros2_camera',
            node_name='realsense_ros2_camera',
            output='screen',
            remappings=[('tf_static', 'tf_realsense')]),
        Node(
            package='wheel_perception',
            node_executable='throttle',
            node_name='wheel_perception_throttle',
            output='screen'),
    ])