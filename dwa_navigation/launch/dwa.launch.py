"""
Launch the DWA navigator.

Launch arguments (name:=value):
    mode:=sim|real   sim also starts Gazebo with the hard_maze world, where the robot
                     chases the moving goal published by the world
    rviz:=true|false start RViz to see the markers
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
    rviz = LaunchConfiguration('rviz').perform(context).lower() == 'true'
    if mode not in ('sim', 'real'):
        raise ValueError(f"mode must be 'sim' or 'real', got '{mode}'")

    params = os.path.join(
        get_package_share_directory('dwa_navigation'), 'config', f'dwa_{mode}.yaml')

    actions = []
    if mode == 'sim':
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory('sesasr_bringup'), 'launch', 'sim_world.launch.py')),
            launch_arguments={'world': 'hard_maze.world'}.items()))

    actions.append(Node(
        package='dwa_navigation',
        executable='dwa_node',
        name='dwa_navigator',
        output='screen',
        parameters=[params],
    ))

    if rviz:
        actions.append(Node(
            package='rviz2',
            executable='rviz2',
            output='screen',
            parameters=[{'use_sim_time': mode == 'sim'}],
        ))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('mode', default_value='sim', description='sim or real'),
        DeclareLaunchArgument('rviz', default_value='false', description='Start RViz'),
        OpaqueFunction(function=_setup),
    ])
