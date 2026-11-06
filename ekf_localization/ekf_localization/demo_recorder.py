"""
Log a localization run in Gazebo and save a plot of the estimates vs. the ground truth.

The node samples /ground_truth, /ekf and /ekf_fusion, and after ``duration``
simulated seconds it writes a figure with the trajectories and the position error.
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402, I100, I202
from nav_msgs.msg import Odometry  # noqa: E402, I100
import numpy as np  # noqa: E402, I100
import rclpy  # noqa: E402, I100
from rclpy.executors import ExternalShutdownException  # noqa: E402, I100
from rclpy.node import Node  # noqa: E402, I100

LANDMARK_LABEL = 'landmarks'


class DemoRecorder(Node):

    def __init__(self):
        super().__init__('ekf_demo_recorder')
        self.declare_parameter('output', 'ekf_gazebo.png')
        self.declare_parameter('duration', 120.0)  # simulated seconds
        self.declare_parameter('sample_rate', 5.0)  # [Hz]
        self.declare_parameter('landmarks_x', [0.0])
        self.declare_parameter('landmarks_y', [0.0])

        self.output = self.get_parameter('output').value
        self.duration = self.get_parameter('duration').value
        self.landmarks = (self.get_parameter('landmarks_x').value,
                          self.get_parameter('landmarks_y').value)

        self.latest = {}
        self.log = {'t': [], 'truth': [], 'ekf': [], 'fusion': []}
        self.start_time = None

        for key, topic in (('truth', '/ground_truth'), ('ekf', '/ekf'), ('fusion', '/ekf_fusion')):
            self.create_subscription(
                Odometry, topic, lambda msg, key=key: self.store(key, msg), 10)
        self.create_timer(1.0 / self.get_parameter('sample_rate').value, self.sample)

    def store(self, key, msg):
        p = msg.pose.pose.position
        self.latest[key] = [p.x, p.y]

    def sample(self):
        t = self.get_clock().now().nanoseconds * 1e-9
        if t == 0.0 or len(self.latest) < 3:
            return
        if self.start_time is None:
            self.start_time = t
        self.log['t'].append(t - self.start_time)
        for key in ('truth', 'ekf', 'fusion'):
            self.log[key].append(self.latest[key])
        if t - self.start_time > self.duration:
            self.finish()

    def finish(self):
        t = np.array(self.log['t'])
        truth = np.array(self.log['truth'])
        ekf = np.array(self.log['ekf'])
        fusion = np.array(self.log['fusion'])
        err_ekf = np.linalg.norm(ekf - truth, axis=1)
        err_fusion = np.linalg.norm(fusion - truth, axis=1)
        rmse_ekf = float(np.sqrt(np.mean(err_ekf ** 2)))
        rmse_fusion = float(np.sqrt(np.mean(err_fusion ** 2)))
        self.get_logger().info(
            f'Run finished: {t[-1]:.1f} s, position RMSE ekf_node {rmse_ekf:.3f} m, '
            f'ekf_fusion_node {rmse_fusion:.3f} m, '
            f'max error {err_ekf.max():.3f} m / {err_fusion.max():.3f} m')

        fig, (ax, ax_err) = plt.subplots(
            1, 2, figsize=(13, 5.5), gridspec_kw={'width_ratios': [1.2, 1]})
        ax.scatter(*self.landmarks, marker='s', c='tab:orange', edgecolors='k', s=60,
                   label=LANDMARK_LABEL, zorder=4)
        ax.plot(*truth.T, 'k-', lw=2, label='ground truth')
        ax.plot(*ekf.T, 'b-', lw=1.2, label='ekf_node')
        ax.plot(*fusion.T, 'r--', lw=1.2, label='ekf_fusion_node')
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4)
        ax.set_title('Trajectories in Gazebo')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')

        ax_err.plot(t, err_ekf, 'b-', label=f'ekf_node (RMSE {rmse_ekf:.3f} m)')
        ax_err.plot(t, err_fusion, 'r--', label=f'ekf_fusion_node (RMSE {rmse_fusion:.3f} m)')
        ax_err.set_xlabel('simulated time [s]')
        ax_err.set_ylabel('position error [m]')
        ax_err.set_title('Localization error')
        ax_err.grid(alpha=0.3)
        ax_err.legend()
        fig.tight_layout()

        directory = os.path.dirname(self.output)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fig.savefig(self.output, dpi=150, bbox_inches='tight')
        self.get_logger().info(f'Saved {self.output}')
        raise SystemExit(0)


def main(args=None):
    rclpy.init(args=args)
    node = DemoRecorder()
    code = 1
    try:
        rclpy.spin(node)
    except SystemExit as e:
        code = e.code
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(code)


if __name__ == '__main__':
    main()
