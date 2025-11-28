#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('ur_simulation_gz')

    ur_sim_launch = os.path.join(
        pkg_share, 'launch', 'ur_sim_control.launch.py')

    include_ur_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(ur_sim_launch)
    )

    force_motion_node = Node(
        package='ur_controller',                    # Package containing the node
        executable='force_motion_controller',
        name='force_motion_controller_node',       # The name for the ROS 2 node
        output='screen'                            # Display output in the console
    )
    
    inverse_kinematics_node = Node(
        package='ur_controller',
        executable='inverse_kinematics',
        name='inverse_kinematics_node',
        output='screen'
    )

    return LaunchDescription([
        include_ur_sim,
        force_motion_node,
        inverse_kinematics_node
    ])
