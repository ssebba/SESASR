#!/usr/bin/env python3
"""
Visualise the velocity motion model with high linear vs high angular noise.

Run from the repository root (no ROS needed):
    python ekf_localization/scripts/plot_velocity_motion_model.py
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
    sample_velocity_motion_model)

ARROW = '$↑$'


def arrow_marker(theta):
    marker = mpl.markers.MarkerStyle(marker=ARROW)
    marker._transform = marker.get_transform().rotate_deg(degrees(theta) - 90)
    return marker


def plot_samples(ax_xy, axes_hist, samples, x0, color, title):
    ax_xy.scatter(x0[0], x0[1], marker=arrow_marker(x0[2]), s=100,
                  facecolors='none', edgecolors='b')
    for s in samples:
        ax_xy.scatter(s[0], s[1], marker=arrow_marker(s[2]), s=40,
                      facecolors='none', edgecolors=color)
    ax_xy.set_xlabel('x [m]')
    ax_xy.set_ylabel('y [m]')
    ax_xy.set_title(title, fontweight='bold')
    ax_xy.legend(handles=[
        Line2D([0], [0], marker=ARROW, color='w', label='Initial pose',
               markeredgecolor='b', markersize=15, linestyle='None'),
        Line2D([0], [0], marker=ARROW, color='w', label='Samples',
               markeredgecolor=color, markersize=10, linestyle='None'),
    ], loc='lower right')
    ax_xy.grid(True, alpha=0.3)

    for ax, idx, label in zip(axes_hist, range(3), ['x [m]', 'y [m]', 'theta [rad]']):
        ax.hist(samples[:, idx], bins=30, alpha=0.7, color=color, edgecolor='black')
        ax.axvline(samples[:, idx].mean(), color='red', linestyle='--',
                   label=f'mean = {samples[:, idx].mean():.3f}')
        ax.set_xlabel(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)


def main():
    np.random.seed(25)
    x0 = np.array([0.0, 0.0, 0.0])
    u = np.array([1.0, 0.5])
    dt = 1.0
    n_samples = 500

    noise = {
        'high linear noise': (np.array([0.2, 0.2, 0.02, 0.02, 0.1, 0.1]), 'orange'),
        'high angular noise': (np.array([0.02, 0.02, 0.2, 0.2, 0.1, 0.1]), 'green'),
    }

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for row, (name, (a, color)) in enumerate(noise.items()):
        samples = sample_velocity_motion_model(np.tile(x0, (n_samples, 1)), u, a, dt)
        print(f'{name}: mean={samples.mean(axis=0)}, std={samples.std(axis=0)}')
        plot_samples(axes[row, 0], axes[row, 1:], samples, x0, color,
                     f'Velocity motion model - {name}')

    plt.tight_layout()
    plt.savefig('velocity_motion_model.png', dpi=150, bbox_inches='tight')
    print("Plot saved to 'velocity_motion_model.png'")
    plt.show()


if __name__ == '__main__':
    main()
