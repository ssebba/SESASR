"""
Start a Gazebo world of the turtlebot3_gazebo package and spawn the TurtleBot3.

Launch arguments (name:=value):
    world:=file.world     world file in turtlebot3_gazebo/worlds
    x_pose:=0.0           spawn position [m]
    y_pose:=0.0
    gui:=true|false       false runs Gazebo without any window
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

# gzserver can take more than the default 30 s of spawn_entity to offer the service
SPAWN_TIMEOUT = '180'


def generate_launch_description():
    gazebo_launch = os.path.join(get_package_share_directory('gazebo_ros'), 'launch')
    tb3_share = get_package_share_directory('turtlebot3_gazebo')
    model = os.environ.get('TURTLEBOT3_MODEL', 'burger')

    return LaunchDescription([
        DeclareLaunchArgument('world', description='World file in turtlebot3_gazebo/worlds'),
        DeclareLaunchArgument('x_pose', default_value='0.0'),
        DeclareLaunchArgument('y_pose', default_value='0.0'),
        DeclareLaunchArgument('gui', default_value='true'),
        # Everything needed is local: do not wait for the online model database
        SetEnvironmentVariable('GAZEBO_MODEL_DATABASE_URI', ''),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(gazebo_launch, 'gzserver.launch.py')),
            launch_arguments={'world': PathJoinSubstitution(
                [tb3_share, 'worlds', LaunchConfiguration('world')])}.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(gazebo_launch, 'gzclient.launch.py')),
            condition=IfCondition(LaunchConfiguration('gui')),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(tb3_share, 'launch', 'robot_state_publisher.launch.py')),
            launch_arguments={'use_sim_time': 'true'}.items(),
        ),
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            output='screen',
            arguments=[
                '-entity', model,
                '-file', os.path.join(tb3_share, 'models', f'turtlebot3_{model}', 'model.sdf'),
                '-x', LaunchConfiguration('x_pose'),
                '-y', LaunchConfiguration('y_pose'),
                '-z', '0.01',
                '-timeout', SPAWN_TIMEOUT,
            ],
        ),
    ])
