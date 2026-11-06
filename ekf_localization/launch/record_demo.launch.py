"""
Compare the two EKF nodes in Gazebo, without any window, and save a plot of the run.

The robot wanders in turtlebot3_world driven by reactive_controller, while
ekf_node and ekf_fusion_node run side by side on the same data. Everything stops
by itself when the run ends.

Launch arguments (name:=value):
    output:=path.png    where to save the plot
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
import yaml


def generate_launch_description():
    ekf_share = get_package_share_directory('ekf_localization')
    params = os.path.join(ekf_share, 'config', 'ekf_sim.yaml')
    with open(params) as f:
        landmarks = yaml.safe_load(f)['/**']['ros__parameters']['landmarks']

    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('sesasr_bringup'), 'launch', 'sim_world.launch.py')),
        launch_arguments={
            'world': 'turtlebot3_world.world', 'x_pose': '-2.0', 'y_pose': '-0.5',
            'gui': 'false'}.items(),
    )
    # Same settings as ekf.launch.py
    landmark_simulator = Node(
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
    )
    driver = Node(
        package='reactive_controller',
        executable='reactive_controller',
        name='reactive_controller',
        output='screen',
        parameters=[{'use_sim_time': True, 'goal': [100.0, 100.0]}],
    )
    ekf = Node(
        package='ekf_localization',
        executable='ekf_node',
        name='ekf_node',
        output='screen',
        parameters=[params],
    )
    ekf_fusion = Node(
        package='ekf_localization',
        executable='ekf_fusion_node',
        name='ekf_fusion_node',
        output='screen',
        parameters=[params, {'ekf_topic': '/ekf_fusion'}],
    )
    recorder = Node(
        package='ekf_localization',
        executable='ekf_demo_recorder',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'output': LaunchConfiguration('output'),
            'landmarks_x': landmarks['x'],
            'landmarks_y': landmarks['y'],
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument('output', default_value='ekf_gazebo.png'),
        simulation,
        landmark_simulator,
        driver,
        ekf,
        ekf_fusion,
        recorder,
        RegisterEventHandler(OnProcessExit(
            target_action=recorder, on_exit=[EmitEvent(event=Shutdown())])),
    ])
