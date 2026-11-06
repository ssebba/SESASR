#!/bin/bash
set -e
source /opt/ros/humble/setup.bash
source /ws/install/setup.bash
source /usr/share/gazebo/setup.sh
exec "$@"
