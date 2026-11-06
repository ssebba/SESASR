"""
ROS 2 node running the DWA local planner on a TurtleBot3.

Obstacles come from the LiDAR, the pose from /odom. The goal can be:
    - a pose sent from RViz ("2D Goal Pose", topic /goal_pose),
    - a moving goal published as nav_msgs/Odometry on ``goal_topic``,
    - an AprilTag (``target_tag_id``) seen by the camera, which the robot follows
      keeping ``target_distance_desired`` from it.
"""

import signal

from geometry_msgs.msg import Point, PoseStamped, Twist
from landmark_msgs.msg import LandmarkArray
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker, MarkerArray

from .dwa import DWA
from .utils import yaw_from_quaternion

# Parameters forwarded to the DWA planner and the robot model, with default values
DWA_PARAMS = {
    'sim_time': 4.0,
    'time_granularity': 0.1,
    'v_samples': 20,
    'w_samples': 25,
    'weight_angle': 2.0,
    'weight_vel': 1.5,
    'weight_obs': 2.0,
    'weight_vel_decay': 1.0,
    'weight_target': 2.0,
    'slowdown_distance': 0.40,
    'target_distance_desired': 0.30,
    'velocity_decay': True,
}
ROBOT_PARAMS = {
    'min_lin_vel': 0.0,
    'max_lin_vel': 0.22,
    'min_ang_vel': -2.0,
    'max_ang_vel': 2.0,
    'max_linear_acc': 0.7,
    'max_ang_acc': float(np.pi),
    'radius': 0.15,
}


class DWANavigator(Node):

    def __init__(self):
        super().__init__('dwa_navigator')

        self.declare_parameter('control_frequency', 15.0)  # [Hz]
        self.declare_parameter('collision_tol', 0.20)      # emergency stop distance [m]
        self.declare_parameter('goal_dist_tol', 0.35)      # [m]
        self.declare_parameter('lidar_max_range', 3.5)     # [m]
        self.declare_parameter('num_lidar_ranges', 25)     # LiDAR sectors kept as obstacles
        self.declare_parameter('max_steps', 300)           # timeout, in control steps
        self.declare_parameter('goal_topic', '/dynamic_goal_pose')
        self.declare_parameter('landmarks_topic', '/landmarks')
        self.declare_parameter('target_tag_id', 33)        # AprilTag to follow
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('publish_markers', False)   # RViz visualisation
        for name, value in {**DWA_PARAMS, **ROBOT_PARAMS}.items():
            self.declare_parameter(f'dwa.{name}', value)

        param = self.get_parameter
        control_freq = param('control_frequency').value
        self.collision_tol = param('collision_tol').value
        self.goal_dist_tol = param('goal_dist_tol').value
        self.lidar_max_range = param('lidar_max_range').value
        self.num_lidar_ranges = param('num_lidar_ranges').value
        self.max_steps = param('max_steps').value
        self.target_tag_id = param('target_tag_id').value
        self.odom_frame = param('odom_frame').value
        self.publish_markers = param('publish_markers').value

        self.robot_pose = None   # [x, y, theta]
        self.robot_vel = np.zeros(2)
        self.lidar_scan = None
        self.goal_pose = None    # [x, y]
        self.obstacles = np.empty((0, 2))
        self.navigation_active = False
        self.collided = False    # latched: the robot stays stopped after a collision
        self.step_count = 0
        self.feedback_rate = 50

        dwa_kwargs = {name: param(f'dwa.{name}').value for name in {**DWA_PARAMS, **ROBOT_PARAMS}}
        self.dwa = DWA(
            dt=1.0 / control_freq,
            collision_tol=self.collision_tol,
            goal_dist_tol=self.goal_dist_tol,
            **dwa_kwargs,
        )

        self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(PoseStamped, '/goal_pose', self.rviz_goal_callback, 10)
        self.create_subscription(
            Odometry, param('goal_topic').value, self.goal_callback, 10)
        self.create_subscription(
            LandmarkArray, param('landmarks_topic').value, self.landmarks_callback, 10)

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.cost_pubs = {
            name: self.create_publisher(Float32, f'/dwa_cost/{name}', 10)
            for name in ('heading', 'vel', 'obs', 'target', 'total')
        }
        self.min_range_pub = self.create_publisher(Float32, '/dwa/min_lidar_range', 10)
        self.dist_to_goal_pub = self.create_publisher(Float32, '/dist_to_goal', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/dwa/markers', 10)

        self.create_timer(1.0 / control_freq, self.control_loop)
        self.get_logger().info('DWA navigator ready, waiting for a goal')

    # ---------------------------------------------------------------- callbacks

    def scan_callback(self, msg):
        self.lidar_scan = msg

    def odom_callback(self, msg):
        p = msg.pose.pose
        self.robot_pose = np.array(
            [p.position.x, p.position.y, yaw_from_quaternion(p.orientation)])
        self.robot_vel = np.array([msg.twist.twist.linear.x, msg.twist.twist.angular.z])

    def rviz_goal_callback(self, msg):
        self.set_goal(msg.pose.position.x, msg.pose.position.y)

    def goal_callback(self, msg):
        self.set_goal(msg.pose.pose.position.x, msg.pose.pose.position.y)

    def landmarks_callback(self, msg):
        """Follow the AprilTag ``target_tag_id``: convert range/bearing to the odom frame."""
        if self.robot_pose is None:
            return
        tag = next((lm for lm in msg.landmarks if lm.id == self.target_tag_id), None)
        if tag is None:
            return

        x_r, y_r, theta_r = self.robot_pose
        angle = theta_r + tag.bearing
        # The tag estimate jitters at every detection: do not restart the timeout
        self.set_goal(x_r + tag.range * np.cos(angle), y_r + tag.range * np.sin(angle),
                      reset_timeout_on_move=False)

    def set_goal(self, x, y, reset_timeout_on_move=True):
        goal = np.array([x, y])
        moved = self.goal_pose is None or not np.allclose(goal, self.goal_pose)
        if reset_timeout_on_move and moved:
            self.step_count = 0
        self.goal_pose = goal
        if not self.navigation_active:
            self.navigation_active = True
            self.step_count = 0
            self.get_logger().info(f'New goal: ({x:.2f}, {y:.2f})')

    # ------------------------------------------------------------- control loop

    def control_loop(self):
        if not self.navigation_active or self.goal_pose is None \
                or self.robot_pose is None or self.lidar_scan is None:
            return

        if self.collided or self.check_collision(self.lidar_scan):
            self.collided = True
            self.finish('Task failed: COLLISION', self.get_logger().error)
            return

        ranges, angles = self.filter_lidar_scan(self.lidar_scan)
        self.obstacles = self.lidar_to_odom_frame(ranges, angles)

        dist_to_goal = np.linalg.norm(self.robot_pose[0:2] - self.goal_pose)
        if dist_to_goal < self.goal_dist_tol:
            self.finish('Task succeeded: GOAL REACHED', self.get_logger().info)
            return
        if self.step_count > self.max_steps:
            self.finish('Task failed: TIMEOUT', self.get_logger().warn)
            return

        try:
            u, costs, active_obs, paths, best_idx = self.dwa.compute_cmd(
                self.goal_pose, self.robot_pose, self.obstacles)
        except Exception as e:  # noqa: B902 - never leave the robot moving on errors
            self.get_logger().error(f'DWA error: {e}')
            self.stop_robot()
            return

        cmd = Twist()
        cmd.linear.x = float(u[0])
        cmd.angular.z = float(u[1])
        self.cmd_vel_pub.publish(cmd)
        # The dynamic window is computed around the measured velocity
        self.dwa.robot.vel = self.robot_vel

        for name, pub in self.cost_pubs.items():
            pub.publish(Float32(data=float(costs[name])))
        self.dist_to_goal_pub.publish(Float32(data=float(dist_to_goal)))
        if self.publish_markers:
            self.publish_rviz_markers(active_obs, paths, best_idx)

        if self.step_count % self.feedback_rate == 0:
            self.get_logger().info(
                f'step {self.step_count} | distance to goal: {dist_to_goal:.3f} m | '
                f'obstacles: {len(self.obstacles)}')
        self.step_count += 1

    def finish(self, message, log):
        self.stop_robot()
        self.navigation_active = False
        log(message)

    def stop_robot(self):
        self.cmd_vel_pub.publish(Twist())

    # ------------------------------------------------------------------- LiDAR

    def filter_lidar_scan(self, scan):
        """
        Reduce the scan to ``num_lidar_ranges`` points.

        The scan is split into equal angular sectors and the closest reading of
        each sector is kept, together with its exact angle.
        """
        ranges = np.array(scan.ranges)
        ranges[np.isnan(ranges)] = scan.range_min if scan.range_min > 0 else 0.01
        ranges[np.isinf(ranges)] = scan.range_max
        ranges = np.minimum(ranges, self.lidar_max_range)

        n = self.num_lidar_ranges
        sector_size = len(ranges) // n
        filtered_ranges = np.zeros(n)
        filtered_angles = np.zeros(n)
        for i in range(n):
            start = i * sector_size
            end = start + sector_size if i < n - 1 else len(ranges)
            idx = start + int(np.argmin(ranges[start:end]))
            filtered_ranges[i] = ranges[idx]
            filtered_angles[i] = scan.angle_min + idx * scan.angle_increment
        return filtered_ranges, filtered_angles

    def lidar_to_odom_frame(self, ranges, angles):
        """Convert polar LiDAR points in the robot frame to (N, 2) points in the odom frame."""
        x_r, y_r, theta_r = self.robot_pose
        global_angles = theta_r + angles
        return np.column_stack([
            x_r + ranges * np.cos(global_angles),
            y_r + ranges * np.sin(global_angles),
        ])

    def check_collision(self, scan):
        """Return True if any valid LiDAR reading is closer than ``collision_tol``."""
        ranges = np.array(scan.ranges)
        ranges = ranges[np.isfinite(ranges)]
        if len(ranges) == 0:
            return False
        min_range = float(np.min(ranges))
        self.min_range_pub.publish(Float32(data=min_range))
        if min_range < self.collision_tol:
            self.get_logger().warn(f'Collision detected, min range: {min_range:.3f} m')
            return True
        return False

    # -------------------------------------------------------------------- RViz

    def publish_rviz_markers(self, obstacles, paths, best_idx, path_stride=10):
        """Publish obstacles (red), some candidate paths (grey) and the best path (green)."""
        stamp = self.get_clock().now().to_msg()

        def marker(ns, marker_type, scale, rgba):
            m = Marker()
            m.header.frame_id = self.odom_frame
            m.header.stamp = stamp
            m.ns = ns
            m.type = marker_type
            m.action = Marker.ADD
            m.scale.x = m.scale.y = m.scale.z = scale
            m.color.r, m.color.g, m.color.b, m.color.a = rgba
            m.pose.orientation.w = 1.0
            return m

        obs = marker('dwa_obstacles', Marker.SPHERE_LIST, 0.1, (1.0, 0.0, 0.0, 1.0))
        obs.points = [Point(x=float(o[0]), y=float(o[1])) for o in obstacles]

        candidates = marker('dwa_candidates', Marker.LINE_LIST, 0.01, (0.5, 0.5, 0.5, 0.6))
        for path in paths[::path_stride]:
            for p0, p1 in zip(path[:-1], path[1:]):
                candidates.points.append(Point(x=float(p0[0]), y=float(p0[1])))
                candidates.points.append(Point(x=float(p1[0]), y=float(p1[1])))

        best = marker('dwa_best', Marker.LINE_STRIP, 0.03, (0.0, 1.0, 0.0, 1.0))
        best.points = [Point(x=float(p[0]), y=float(p[1])) for p in paths[best_idx]]

        self.marker_pub.publish(MarkerArray(markers=[obs, candidates, best]))


def main(args=None):
    # Keep the context alive on Ctrl-C, so that the final stop command can be sent
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = DWANavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # A second Ctrl-C must not interrupt the final stop command
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        node.stop_robot()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
