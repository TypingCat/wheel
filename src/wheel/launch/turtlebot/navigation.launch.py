#!/usr/bin/env python3

import os

from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from nav2_common.launch import RewrittenYaml
from launch.actions import SetEnvironmentVariable
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    autostart = LaunchConfiguration('autostart')
    params_file = LaunchConfiguration('params')
    bt_xml_file = LaunchConfiguration('bt_xml_file')
    param_substitutions = {
        'use_sim_time': use_sim_time,
        'bt_xml_filename': bt_xml_file,
        'autostart': autostart,
    }
    configured_params = RewrittenYaml(
        source_file=params_file,
        rewrites=param_substitutions,
        convert_types=True)

    return LaunchDescription([
        SetEnvironmentVariable(
            'RCUTILS_CONSOLE_STDOUT_LINE_BUFFERED', '1'),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value=use_sim_time,
            description='Use simulation clock if true'),
        DeclareLaunchArgument(
            'autostart',
            default_value='true',
            description='Automatically startup the nav2 stack'),
        DeclareLaunchArgument(
            'params',
            default_value=[os.path.join(
                get_package_share_directory('wheel'), 'config', 'navigation2.yaml')],
            description='Full path to the ROS2 parameters file to use'),
        DeclareLaunchArgument(
            'bt_xml_file',
            default_value=os.path.join(
                get_package_share_directory('wheel'), 'config', 'behavior.xml'),
            description='Full path to the behavior tree xml file to use'),
        Node(
            package='nav2_world_model',
            node_executable='world_model',
            output='screen',
            parameters=[configured_params]),
        Node(
            package='dwb_controller',
            node_executable='dwb_controller',
            output='screen',
            parameters=[configured_params]),
        Node(
            package='nav2_navfn_planner',
            node_executable='navfn_planner',
            node_name='navfn_planner',
            output='screen',
            parameters=[configured_params]),
        Node(
            package='nav2_recoveries',
            node_executable='recoveries_node',
            node_name='recoveries',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}]),
        Node(
            package='nav2_bt_navigator',
            node_executable='bt_navigator',
            node_name='bt_navigator',
            output='screen',
            parameters=[configured_params]),
        Node(
            package='nav2_lifecycle_manager',
            node_executable='lifecycle_manager',
            node_name='lifecycle_manager_control',
            output='screen',
            parameters=[
                {'use_sim_time': use_sim_time},
                {'autostart': autostart},
                {'node_names': ['world_model', 'dwb_controller', 'navfn_planner', 'bt_navigator']}]),
    ])
