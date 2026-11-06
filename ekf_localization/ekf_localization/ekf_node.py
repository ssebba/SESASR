"""
EKF localization with a 3D state [x, y, theta].

Prediction: velocity motion model driven by the wheel-odometry twist.
Correction: range-bearing measurements of known landmarks (AprilTags).
"""

import signal

import numpy as np
import rclpy

from .ekf_base_node import EkfBaseNode


class EkfNode(EkfBaseNode):

    def __init__(self):
        super().__init__('ekf_node', state_dim=3, defaults={
            'initial_covariance': [0.05, 0.05, float(np.deg2rad(0.5))],
            'motion_noise': [0.01, float(np.deg2rad(0.5)), 0.01, float(np.deg2rad(0.5))],
            'landmark_noise_std': [0.15, 2.0],  # [m, deg]
        })
        self.get_logger().info('EKF (3D state, landmarks) started')


def main(args=None):
    rclpy.init(args=args)
    node = EkfNode()
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
