"""
Launch the EKF localization.

Launch arguments (name:=value):
    mode:=sim|real    sim also starts Gazebo and the landmark simulator
    fusion:=false|true  false -> ekf_node (3D state), true -> ekf_fusion_node (5D state)
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _setup(context):
    mode = LaunchConfiguration('mode').perform(context)
    fusion = LaunchConfiguration('fusion').perform(context).lower() == 'true'
    if mode not in ('sim', 'real'):
        raise ValueError(f"mode must be 'sim' or 'real', got '{mode}'")

    params = os.path.join(
        get_package_share_directory('ekf_localization'), 'config', f'ekf_{mode}.yaml')
    executable = 'ekf_fusion_node' if fusion else 'ekf_node'

    actions = []
    if mode == 'sim':
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory('sesasr_bringup'), 'launch', 'sim_world.launch.py')),
            launch_arguments={
                'world': 'turtlebot3_world.world', 'x_pose': '-2.0', 'y_pose': '-0.5'}.items()))
        actions.append(Node(
            package='turtlebot3_perception',
            executable='landmark_simulator',
            output='screen',
            parameters=[{
                'frequency_hz': 5.0,
                'field_of_view_deg': 70.0,
                'max_range': 3.0,
                'min_range': 0.3,
                'range_stddev': 0.01,
                'bearing_stddev_deg': 1.0,
                'use_sim_time': True,
            }],
        ))

    actions.append(Node(
        package='ekf_localization',
        executable=executable,
        name=executable,
        output='screen',
        parameters=[params],
    ))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('mode', default_value='sim', description='sim or real'),
        DeclareLaunchArgument(
            'fusion', default_value='false',
            description='Use the 5D state with odometry + IMU fusion'),
        OpaqueFunction(function=_setup),
    ])
