"""Small math helpers shared by the EKF nodes."""

import numpy as np


def normalize_angle(theta):
    """Wrap an angle (scalar or array) to [-pi, pi)."""
    return (np.asarray(theta) + np.pi) % (2.0 * np.pi) - np.pi


def residual(a, b, angle_idx=None):
    """Return a - b, wrapping the component at ``angle_idx`` (if given) to [-pi, pi)."""
    y = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if angle_idx is not None:
        y[angle_idx] = normalize_angle(y[angle_idx])
    return y


def quaternion_from_yaw(yaw):
    """Return the quaternion [x, y, z, w] of a rotation of ``yaw`` around the z axis."""
    return [0.0, 0.0, float(np.sin(yaw / 2.0)), float(np.cos(yaw / 2.0))]


def yaw_from_quaternion(q):
    """Return the yaw angle of a quaternion message (fields x, y, z, w)."""
    return float(np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y ** 2 + q.z ** 2)))
