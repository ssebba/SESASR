"""Start the maze world in Gazebo and, after a short delay, the reactive controller."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim = LaunchConfiguration('use_sim')
    params_file = LaunchConfiguration('params_file')

    default_params = os.path.join(
        get_package_share_directory('reactive_controller'), 'config', 'reactive_controller.yaml')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('sesasr_bringup'), 'launch', 'sim_world.launch.py')),
        launch_arguments={'world': 'maze.world'}.items(),
        condition=IfCondition(use_sim),
    )

    controller = Node(
        package='reactive_controller',
        executable='reactive_controller',
        name='reactive_controller',
        output='screen',
        parameters=[params_file, {'use_sim_time': use_sim}],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim', default_value='true',
            description='Start Gazebo (maze world) and use the simulation clock'),
        DeclareLaunchArgument(
            'params_file', default_value=default_params,
            description='YAML file with the controller parameters'),
        gazebo,
        # Give Gazebo time to spawn the robot before sending commands
        TimerAction(period=8.0, actions=[controller]),
    ])
