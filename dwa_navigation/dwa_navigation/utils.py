"""Helpers for the DWA planner (pure NumPy, no ROS)."""

import math

import numpy as np


class DifferentialDriveRobot:
    """Kinematic limits and state of a circular differential-drive robot."""

    def __init__(self,
                 init_pose=None,
                 max_linear_acc=0.8,             # [m/s^2]
                 max_ang_acc=100 * math.pi / 180,  # [rad/s^2]
                 max_lin_vel=1.0,                # [m/s]
                 min_lin_vel=0.0,                # [m/s]
                 max_ang_vel=3.0,                # [rad/s]
                 min_ang_vel=-3.0,               # [rad/s]
                 radius=0.3):                    # [m]
        self.pose = np.zeros(3) if init_pose is None else np.asarray(init_pose, dtype=float)
        self.vel = np.zeros(2)

        self.max_linear_acc = max_linear_acc
        self.max_ang_acc = max_ang_acc
        self.max_lin_vel = max_lin_vel
        self.min_lin_vel = min_lin_vel
        self.max_ang_vel = max_ang_vel
        self.min_ang_vel = min_ang_vel
        self.radius = radius

        # history of [x, y, theta, v, w], used by the offline demo
        self.trajectory = np.concatenate([self.pose, self.vel]).reshape(1, -1)

    def update_state(self, u, dt):
        """Integrate the unicycle kinematics for one step with command u = [v, w]."""
        self.vel = np.asarray(u, dtype=float)
        x, y, th = self.pose
        self.pose = np.array([
            x + self.vel[0] * math.cos(th) * dt,
            y + self.vel[0] * math.sin(th) * dt,
            th + self.vel[1] * dt,
        ])
        state = np.concatenate([self.pose, self.vel]).reshape(1, -1)
        self.trajectory = np.concatenate([self.trajectory, state], axis=0)
        return self.pose


def calc_nearest_obs(robot_pose, obstacles, obstacle_max_dist=3.0):
    """Return the obstacles closer than ``obstacle_max_dist`` (a local map)."""
    obstacles = np.asarray(obstacles, dtype=float).reshape(-1, 2)
    dist = np.linalg.norm(obstacles - robot_pose[0:2], axis=1)
    return obstacles[dist < obstacle_max_dist]


def normalize_angle(theta):
    """Wrap an angle (scalar or array) to [-pi, pi)."""
    return (np.asarray(theta) + np.pi) % (2.0 * np.pi) - np.pi


def normalize(arr):
    """Min-max normalise an array to [0, 1] (all zeros if the array is constant)."""
    span = np.max(arr) - np.min(arr)
    if np.isclose(span, 0.0):
        return np.zeros_like(arr)
    return (arr - np.min(arr)) / span


def yaw_from_quaternion(q):
    """Return the yaw angle of a quaternion message (fields x, y, z, w)."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y ** 2 + q.z ** 2))
