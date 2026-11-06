"""
EKF localization with a 5D state [x, y, theta, v, omega] and sensor fusion.

Prediction: constant-velocity model (v and omega are part of the state).
Corrections:
    - wheel odometry twist  -> [v, omega]
    - IMU gyroscope         -> [omega]
    - known landmarks       -> [range, bearing]
"""

import signal

import numpy as np
import rclpy
from sensor_msgs.msg import Imu

from .ekf_base_node import EkfBaseNode
from .models import imu_gyro_model, odometry_velocity_model


class EkfFusionNode(EkfBaseNode):

    def __init__(self):
        super().__init__('ekf_fusion_node', state_dim=5, defaults={
            'initial_covariance': [0.1, 0.1, float(np.deg2rad(1.0)), 0.1, 0.1],
            'motion_noise': [0.1, 0.05, 0.1, 0.05],
            'landmark_noise_std': [0.1, 1.0],  # [m, deg]
        })

        self.declare_parameter('imu_topic', '/imu')
        self.declare_parameter('odom_noise_std', [0.1, 23.94])  # [m/s, deg/s]
        self.declare_parameter('imu_noise_std', 5.0)            # [deg/s]

        sigma_v, sigma_w_deg = self.get_parameter('odom_noise_std').value
        self.Qt_odom = np.diag([sigma_v ** 2, np.deg2rad(sigma_w_deg) ** 2])
        self.Qt_imu = np.diag([np.deg2rad(self.get_parameter('imu_noise_std').value) ** 2])

        self.eval_h_odom, self.eval_H_odom = odometry_velocity_model()
        self.eval_h_imu, self.eval_H_imu = imu_gyro_model()

        self.create_subscription(
            Imu, self.get_parameter('imu_topic').value, self.imu_callback, 10)
        self.get_logger().info('EKF (5D state, odometry + IMU + landmarks) started')

    def odom_callback(self, msg):
        super().odom_callback(msg)
        mu = self.ekf.mu
        self.ekf.update(self.last_u, self.eval_h_odom(mu), self.eval_H_odom(mu), self.Qt_odom)

    def imu_callback(self, msg):
        z = np.array([msg.angular_velocity.z])
        mu = self.ekf.mu
        self.ekf.update(z, self.eval_h_imu(mu), self.eval_H_imu(mu), self.Qt_imu)


def main(args=None):
    rclpy.init(args=args)
    node = EkfFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
