#!/usr/bin/env python3

import os

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    return LaunchDescription([
        Node(
            package='wheel_navigation',
            node_executable='unity',
            node_name='wheel_navigation_unity',
            output='screen'),
    ])
