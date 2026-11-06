"""
Reactive obstacle-avoidance controller for the TurtleBot3.

The robot drives forward at constant speed. When the closest obstacle in the
frontal LiDAR sector gets closer than ``threshold``, it stops and rotates in
place towards the side (left or right) with the larger mean free distance, then
resumes driving straight. The node stops once the robot is within
``goal_tolerance`` of the ``goal`` position read from odometry.
"""

import math
import signal

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray


class ReactiveController(Node):

    def __init__(self):
        super().__init__('reactive_controller')

        self.declare_parameter('goal', [6.5, 3.0])
        self.declare_parameter('goal_tolerance', 1.0)
        self.declare_parameter('linear_velocity', 0.2)
        self.declare_parameter('angular_velocity', 0.8)
        self.declare_parameter('control_frequency', 5.0)
        # Half-width of the frontal sector [rad]
        self.declare_parameter('front_sector_half_angle', 0.32)
        # Lateral sectors span [side_sector_min, side_sector_max] on each side [rad]
        self.declare_parameter('side_sector_min', 0.1)
        self.declare_parameter('side_sector_max', 2.0)
        # Distance from the frontal obstacle that triggers a rotation [m]
        self.declare_parameter('threshold', 0.35)

        self.goal = self.get_parameter('goal').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.linear_velocity = self.get_parameter('linear_velocity').value
        self.angular_velocity = self.get_parameter('angular_velocity').value
        self.front_half_angle = self.get_parameter('front_sector_half_angle').value
        self.side_min = self.get_parameter('side_sector_min').value
        self.side_max = self.get_parameter('side_sector_max').value
        self.threshold = self.get_parameter('threshold').value

        self.turning = False
        self.goal_reached = False

        self.min_front_distance = math.inf
        self.mean_left_distance = math.inf
        self.mean_right_distance = math.inf

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        # [front, right, left] minimum distances, useful to plot the behaviour
        self.debug_pub = self.create_publisher(Float32MultiArray, '/min_distances', 10)

        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)

        period = 1.0 / self.get_parameter('control_frequency').value
        self.timer = self.create_timer(period, self.control_loop)

        self.get_logger().info(
            f'Reactive controller started, goal: ({self.goal[0]:.2f}, {self.goal[1]:.2f})')

    def control_loop(self):
        msg = Twist()
        if self.goal_reached:
            self.cmd_pub.publish(msg)
            return

        if self.min_front_distance <= self.threshold:
            # Obstacle ahead: rotate in place towards the freer side
            self.turning = True
            direction = 1.0 if self.mean_left_distance > self.mean_right_distance else -1.0
            msg.angular.z = direction * self.angular_velocity
        else:
            self.turning = False
            msg.linear.x = self.linear_velocity

        self.cmd_pub.publish(msg)

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        if not self.goal_reached and math.hypot(self.goal[0] - x, self.goal[1] - y) < \
                self.goal_tolerance:
            self.goal_reached = True
            self.get_logger().info('Goal reached!')

    def scan_callback(self, msg):
        front, left, right = [], [], []

        for i, distance in enumerate(msg.ranges):
            angle = msg.angle_min + i * msg.angle_increment
            if math.isinf(distance) or math.isnan(distance):
                distance = msg.range_max

            # The TurtleBot3 LDS scans [0, 2*pi): the front sector wraps around 0
            if angle <= self.front_half_angle or angle >= msg.angle_max - self.front_half_angle:
                front.append(distance)
            if self.side_min <= angle <= self.side_max:
                left.append(distance)
            if msg.angle_max - self.side_max <= angle <= msg.angle_max - self.side_min:
                right.append(distance)

        # Freeze the side means while turning so the robot keeps rotating the same way
        if not self.turning:
            self.mean_left_distance = sum(left) / len(left) if left else math.inf
            self.mean_right_distance = sum(right) / len(right) if right else math.inf

        self.min_front_distance = min(front, default=math.inf)

        debug = Float32MultiArray()
        debug.data = [
            float(self.min_front_distance),
            float(min(right, default=math.inf)),
            float(min(left, default=math.inf)),
        ]
        self.debug_pub.publish(debug)

        self.get_logger().debug(
            f'front: {self.min_front_distance:.3f} m | '
            f'left mean: {self.mean_left_distance:.3f} m | '
            f'right mean: {self.mean_right_distance:.3f} m')


def main(args=None):
    # Keep the context alive on Ctrl-C, so that the final stop command can be sent
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = ReactiveController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # A second Ctrl-C must not interrupt the final stop command
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
