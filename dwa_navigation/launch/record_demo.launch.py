"""
Run the DWA in Gazebo without any window and record an animated GIF of the run.

The robot chases the moving goal (goal_box) that the hard_maze world publishes on
/dynamic_goal_pose. Everything stops by itself when the run ends.

Launch arguments (name:=value):
    output:=path.gif          where to save the GIF
    params_file:=path.yaml    planner parameters
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, EmitEvent, IncludeLaunchDescription, RegisterEventHandler)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Frames the path of goal_box: x from -10 to 5 m, y from -3.6 to 7.6 m
CAMERA_CENTER = [-2.5, 2.0]
CAMERA_HEIGHT = 13.0
# gzserver can take more than the default 30 s to offer the spawn service
SPAWN_TIMEOUT = '180'


def generate_launch_description():
    dwa_share = get_package_share_directory('dwa_navigation')

    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('sesasr_bringup'), 'launch', 'sim_world.launch.py')),
        launch_arguments={'world': 'hard_maze.world', 'gui': 'false'}.items(),
    )
    # Looking straight down, with +x to the right of the image and +y to the top
    spawn_camera = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        output='screen',
        arguments=[
            '-entity', 'overhead_camera',
            '-file', os.path.join(dwa_share, 'config', 'overhead_camera.sdf'),
            '-x', str(CAMERA_CENTER[0]), '-y', str(CAMERA_CENTER[1]),
            '-z', str(CAMERA_HEIGHT), '-P', '1.5708', '-Y', '1.5708',
            '-timeout', SPAWN_TIMEOUT,
        ],
    )
    planner = Node(
        package='dwa_navigation',
        executable='dwa_node',
        name='dwa_navigator',
        output='screen',
        parameters=[LaunchConfiguration('params_file'), {'publish_markers': True}],
    )
    recorder = Node(
        package='dwa_navigation',
        executable='demo_recorder',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'output': LaunchConfiguration('output'),
            'camera_center': CAMERA_CENTER,
            'camera_height': CAMERA_HEIGHT,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument('output', default_value='dwa_gazebo.gif'),
        DeclareLaunchArgument(
            'params_file', default_value=os.path.join(dwa_share, 'config', 'dwa_sim.yaml')),
        simulation,
        spawn_camera,
        planner,
        recorder,
        RegisterEventHandler(OnProcessExit(
            target_action=recorder, on_exit=[EmitEvent(event=Shutdown())])),
    ])
