#!/usr/bin/env python3
"""
Offline EKF demo: landmark-based localization vs. pure dead reckoning.

A robot drives a loop among the landmarks used in the Gazebo world. Its noisy
velocity readings are integrated open loop (dead reckoning) and fused with
range-bearing landmark measurements by the EKF. No ROS needed:
    python ekf_localization/scripts/ekf_demo.py --png ekf_demo.png
"""

import argparse
import os
import sys

from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ekf_localization.ekf import RobotEKF  # noqa: E402, I100
from ekf_localization.models import (  # noqa: E402
    landmark_measurement_model, velocity_motion_model)
from ekf_localization.utils import normalize_angle  # noqa: E402

LANDMARKS = np.array([[x, y, 0.0] for x in (-1.1, 0.0, 1.1) for y in (-1.1, 0.0, 1.1)])


def simulate(seed=0, duration=90.0, dt=0.05, fov=np.deg2rad(70), max_range=3.0):
    rng = np.random.default_rng(seed)
    g, G, V = velocity_motion_model(3)
    h, H = landmark_measurement_model(3)

    ekf = RobotEKF(3, 2, g, G, V)
    ekf.mu = np.array([0.0, -1.8, 0.0])
    ekf.Sigma = np.diag([0.05, 0.05, 0.01])
    alpha = np.array([0.05, 0.02, 0.05, 0.02])
    Qt = np.diag([0.05 ** 2, np.deg2rad(2) ** 2])

    truth = ekf.mu.copy()
    odom = ekf.mu.copy()
    log = {'truth': [], 'odom': [], 'ekf': [], 'sigma': []}

    for k in range(int(duration / dt)):
        t = k * dt
        u = np.array([0.2, 0.11 + 0.08 * np.sin(0.3 * t)])
        truth = g(truth, u, dt)
        # wheel odometry: biased and noisy velocity readings
        u_meas = u * [1.05, 0.95] + rng.normal(0, [0.01, 0.02])
        odom = g(odom, u_meas, dt)

        v, w = u_meas
        ekf.Mt = np.diag([alpha[0] * v ** 2 + alpha[1] * w ** 2,
                          alpha[2] * v ** 2 + alpha[3] * w ** 2])
        ekf.predict(u_meas, dt)

        if k % 4 == 0:  # landmark detections at 5 Hz
            for m in LANDMARKS:
                z = h(truth, m)
                if z[0] > max_range or abs(normalize_angle(z[1])) > fov / 2:
                    continue
                z = z + rng.normal(0, [0.05, np.deg2rad(2)])
                ekf.update(z, h(ekf.mu, m), H(ekf.mu, m), Qt, angle_idx=1)
        ekf.mu[2] = normalize_angle(ekf.mu[2])

        log['truth'].append(truth.copy())
        log['odom'].append(odom.copy())
        log['ekf'].append(ekf.mu.copy())
        log['sigma'].append(ekf.Sigma[:2, :2].copy())

    return {key: np.array(val) for key, val in log.items()}


def covariance_ellipse(mean, cov, n_std=3.0, **kwargs):
    vals, vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    width, height = 2 * n_std * np.sqrt(vals[::-1])
    return Ellipse(mean, width, height, angle=angle, fill=False, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--png', help='save the figure instead of showing it')
    args = parser.parse_args()

    log = simulate()
    err_odom = np.linalg.norm(log['odom'][:, :2] - log['truth'][:, :2], axis=1)
    err_ekf = np.linalg.norm(log['ekf'][:, :2] - log['truth'][:, :2], axis=1)
    print(f'final position error  dead reckoning: {err_odom[-1]:.3f} m  '
          f'EKF: {err_ekf[-1]:.3f} m')
    print(f'mean position error   dead reckoning: {err_odom.mean():.3f} m  '
          f'EKF: {err_ekf.mean():.3f} m')

    fig, (ax, ax_err) = plt.subplots(1, 2, figsize=(13, 5.5),
                                     gridspec_kw={'width_ratios': [1.2, 1]})
    ax.scatter(LANDMARKS[:, 0], LANDMARKS[:, 1], marker='s', c='tab:orange',
               edgecolors='k', s=60, label='landmarks', zorder=4)
    ax.plot(*log['truth'][:, :2].T, 'k-', lw=2, label='ground truth')
    ax.plot(*log['odom'][:, :2].T, 'r--', lw=1.5, label='dead reckoning')
    ax.plot(*log['ekf'][:, :2].T, 'b-', lw=1.5, label='EKF')
    for i in range(0, len(log['ekf']), 150):
        ax.add_patch(covariance_ellipse(log['ekf'][i, :2], log['sigma'][i],
                                        color='tab:blue', alpha=0.6))
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=8)
    ax.set_title('Trajectories (3-sigma ellipses)')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')

    t = np.arange(len(err_ekf)) * 0.05
    ax_err.plot(t, err_odom, 'r--', label='dead reckoning')
    ax_err.plot(t, err_ekf, 'b-', label='EKF')
    ax_err.set_xlabel('time [s]')
    ax_err.set_ylabel('position error [m]')
    ax_err.set_title('Localization error')
    ax_err.grid(alpha=0.3)
    ax_err.legend()

    fig.tight_layout()
    if args.png:
        fig.savefig(args.png, dpi=150, bbox_inches='tight')
        print(f'Saved {args.png}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
