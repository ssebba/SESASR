#!/usr/bin/env python3
"""
Visualise the robot poses compatible with a noisy range-bearing landmark measurement.

Run from the repository root (no ROS needed):
    python ekf_localization/scripts/plot_landmark_measurement_model.py
"""

from math import degrees
import os
import sys

import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ekf_localization.probabilistic_models import (  # noqa: E402, I100
    landmark_range_bearing_model, landmark_sm_Ht, sample_pose_from_landmark)

ARROW = '$↑$'


def arrow_marker(theta):
    marker = mpl.markers.MarkerStyle(marker=ARROW)
    marker._transform = marker.get_transform().rotate_deg(degrees(theta) - 90)
    return marker


def main():
    np.random.seed(61)
    x0 = np.array([2.0, 1.0, np.pi / 6])
    landmark = np.array([5.0, 3.0])
    sigma_r, sigma_phi = 0.2, 0.2
    n_samples = 1000

    print('H_t at x0:\n', landmark_sm_Ht(x0, *landmark))

    z = landmark_range_bearing_model(x0, landmark, np.array([sigma_r, sigma_phi]))
    samples = np.array([
        sample_pose_from_landmark(x0, z, landmark, sigma_r, sigma_phi)
        for _ in range(n_samples)
    ])
    print(f'mean={samples.mean(axis=0)}, std={samples.std(axis=0)}')

    plt.figure(figsize=(14, 9))
    ax = plt.subplot(2, 2, 1)
    for s in samples:
        ax.scatter(s[0], s[1], marker=arrow_marker(s[2]), s=40,
                   facecolors='none', edgecolors='r')
    ax.scatter(x0[0], x0[1], marker=arrow_marker(x0[2]), s=100,
               facecolors='none', edgecolors='b')
    ax.scatter(*landmark, c='red', s=250, marker='*', edgecolors='black', zorder=5)
    ax.legend(handles=[
        Line2D([0], [0], marker=ARROW, color='w', label='True pose',
               markeredgecolor='b', markersize=15, linestyle='None'),
        Line2D([0], [0], marker=ARROW, color='w', label='Samples',
               markeredgecolor='red', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='*', color='w', label='Landmark',
               markerfacecolor='red', markeredgecolor='black', markersize=15,
               linestyle='None'),
    ], loc='lower right')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Poses sampled from the measurement model', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    labels = ['x position [m]', 'y position [m]', 'orientation [rad]']
    for idx, label in enumerate(labels):
        ax = plt.subplot(2, 2, idx + 2)
        ax.hist(samples[:, idx], bins=30, alpha=0.7, color='blue',
                edgecolor='black', density=True)
        ax.axvline(samples[:, idx].mean(), color='red', linestyle='--',
                   label=f'samples mean = {samples[:, idx].mean():.3f}')
        ax.axvline(x0[idx], color='green', linestyle='--', label=f'true = {x0[idx]:.3f}')
        ax.set_xlabel(label)
        ax.set_ylabel('density')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('landmark_measurement_model.png', dpi=150, bbox_inches='tight')
    print("Plot saved to 'landmark_measurement_model.png'")
    plt.show()


if __name__ == '__main__':
    main()
