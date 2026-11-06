# ROS 2 Humble + Gazebo Classic image with this repository and its dependencies built.
#
#   docker build -t sesasr .
#   # Linux host with X11, to see Gazebo and RViz:
#   xhost +local:docker
#   docker run -it --rm --net=host -e DISPLAY=$DISPLAY \
#       -v /tmp/.X11-unix:/tmp/.X11-unix sesasr \
#       ros2 launch dwa_navigation dwa.launch.py mode:=sim rviz:=true
FROM osrf/ros:humble-desktop

ENV DEBIAN_FRONTEND=noninteractive \
    TURTLEBOT3_MODEL=burger

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-vcstool \
        python3-colcon-common-extensions \
        ros-humble-gazebo-ros-pkgs \
        ros-humble-turtlebot3 \
        ros-humble-turtlebot3-msgs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ws
COPY . src/SESASR
RUN vcs import src < src/SESASR/dependencies.repos

# The camera drivers of turtlebot3_perception are only needed on the real robot
RUN apt-get update && rosdep update && \
    rosdep install -y --ignore-src --from-paths src \
        --skip-keys "depthai-ros realsense2_camera perception" \
    && rm -rf /var/lib/apt/lists/*

RUN . /opt/ros/humble/setup.sh && \
    colcon build --symlink-install \
        --packages-up-to reactive_controller ekf_localization dwa_navigation turtlebot3_gazebo

RUN echo 'source /opt/ros/humble/setup.bash' >> /root/.bashrc && \
    echo 'source /ws/install/setup.bash' >> /root/.bashrc && \
    echo 'source /usr/share/gazebo/setup.sh' >> /root/.bashrc

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
