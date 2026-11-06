"""Common ROS 2 plumbing for the EKF localization nodes."""

from landmark_msgs.msg import LandmarkArray
from nav_msgs.msg import Odometry
import numpy as np
from rclpy.node import Node

from .ekf import RobotEKF
from .models import landmark_measurement_model, velocity_motion_model
from .utils import normalize_angle, quaternion_from_yaw


class EkfBaseNode(Node):
    """
    Base class: landmark map, prediction timer, landmark update and publishing.

    Subclasses choose the state dimension and subscribe to their extra sensors.
    """

    def __init__(self, node_name, state_dim, defaults):
        super().__init__(node_name)
        self.state_dim = state_dim

        # Default values are the tuning used in the lab; override them in YAML
        self.declare_parameter('prediction_rate', 20.0)
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('landmarks_topic', '/landmarks')
        self.declare_parameter('ekf_topic', '/ekf')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('initial_pose', [0.0, 0.0, 0.0])
        self.declare_parameter('initial_covariance', defaults['initial_covariance'])
        self.declare_parameter('motion_noise', defaults['motion_noise'])
        self.declare_parameter('landmark_noise_std', defaults['landmark_noise_std'])
        self.declare_parameter('landmarks.id', [0])
        self.declare_parameter('landmarks.x', [0.0])
        self.declare_parameter('landmarks.y', [0.0])
        self.declare_parameter('landmarks.z', [0.0])

        self.dt = 1.0 / self.get_parameter('prediction_rate').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.landmarks = self._load_landmarks()

        eval_gux, eval_Gt, eval_Vt = velocity_motion_model(state_dim)
        self.eval_hx_landmark, self.eval_Ht_landmark = landmark_measurement_model(state_dim)

        self.ekf = RobotEKF(state_dim, 2, eval_gux, eval_Gt, eval_Vt)
        self.ekf.mu[:3] = self.get_parameter('initial_pose').value
        self.ekf.Sigma = np.diag(self.get_parameter('initial_covariance').value)

        # alpha: control noise grows with the commanded velocities
        self.alpha = np.array(self.get_parameter('motion_noise').value)
        sigma_r, sigma_b_deg = self.get_parameter('landmark_noise_std').value
        self.Qt_landmark = np.diag([sigma_r ** 2, np.deg2rad(sigma_b_deg) ** 2])

        self.last_u = np.zeros(2)
        self._pred_count = 0

        self.create_subscription(
            Odometry, self.get_parameter('odom_topic').value, self.odom_callback, 10)
        self.create_subscription(
            LandmarkArray, self.get_parameter('landmarks_topic').value,
            self.landmark_callback, 10)
        self.ekf_pub = self.create_publisher(Odometry, self.get_parameter('ekf_topic').value, 10)
        self.create_timer(self.dt, self.prediction_callback)

    def _load_landmarks(self):
        ids = self.get_parameter('landmarks.id').value
        coords = [self.get_parameter(f'landmarks.{axis}').value for axis in 'xyz']
        if not all(len(c) == len(ids) for c in coords):
            raise ValueError('landmarks.id/x/y/z must have the same length')
        landmarks = {int(i): np.array(p) for i, p in zip(ids, zip(*coords))}
        self.get_logger().info(f'Loaded {len(landmarks)} landmarks: {sorted(landmarks)}')
        return landmarks

    def odom_callback(self, msg):
        """Store the measured velocities, used as control input for the prediction."""
        self.last_u = np.array([msg.twist.twist.linear.x, msg.twist.twist.angular.z])

    def prediction_callback(self):
        v, w = self.last_u
        a1, a2, a3, a4 = self.alpha
        self.ekf.Mt = np.diag([a1 * v ** 2 + a2 * w ** 2, a3 * v ** 2 + a4 * w ** 2])
        self.ekf.predict(self.last_u, self.dt)
        self.ekf.mu[2] = normalize_angle(self.ekf.mu[2])

        self._pred_count += 1
        if self._pred_count % round(1.0 / self.dt) == 0:
            x, y, theta = self.ekf.mu[:3]
            self.get_logger().info(
                f'pose: ({x:.3f}, {y:.3f}, {np.degrees(theta):.1f} deg) | '
                f'trace(Sigma): {np.trace(self.ekf.Sigma):.4f}')

        self.publish_state()

    def landmark_callback(self, msg):
        for lm in msg.landmarks:
            m = self.landmarks.get(lm.id)
            if m is None:
                continue
            z = np.array([lm.range, lm.bearing])
            self.ekf.update(
                z,
                self.eval_hx_landmark(self.ekf.mu, m),
                self.eval_Ht_landmark(self.ekf.mu, m),
                self.Qt_landmark,
                angle_idx=1,
            )
            self.ekf.mu[2] = normalize_angle(self.ekf.mu[2])
            self.get_logger().debug(
                f'landmark {lm.id}: range={lm.range:.3f} m, '
                f'bearing={np.degrees(lm.bearing):.1f} deg')

        self.publish_state()

    def publish_state(self):
        """Publish the estimate as nav_msgs/Odometry (pose covariance for x, y, yaw)."""
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.odom_frame
        msg.child_frame_id = self.base_frame

        msg.pose.pose.position.x = float(self.ekf.mu[0])
        msg.pose.pose.position.y = float(self.ekf.mu[1])
        q = quaternion_from_yaw(self.ekf.mu[2])
        (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
         msg.pose.pose.orientation.z, msg.pose.pose.orientation.w) = q

        covariance = np.zeros(36)
        covariance[0] = self.ekf.Sigma[0, 0]
        covariance[7] = self.ekf.Sigma[1, 1]
        covariance[35] = self.ekf.Sigma[2, 2]
        msg.pose.covariance = covariance.tolist()

        if self.state_dim == 5:
            msg.twist.twist.linear.x = float(self.ekf.mu[3])
            msg.twist.twist.angular.z = float(self.ekf.mu[4])

        self.ekf_pub.publish(msg)
