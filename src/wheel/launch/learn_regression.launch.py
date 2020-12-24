#!/usr/bin/env python3

import os

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    return LaunchDescription([
        Node(
            package='wheel_navigation',
            node_executable='sim0',
            node_name='wheel_navigation_simulation',
            output='screen'),
        Node(
            package='wheel_navigation',
            node_executable='learn0',
            node_name='wheel_navigation_learning',
            output='screen'),
    ])
