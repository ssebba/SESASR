"""
Sampling versions of the probabilistic motion and sensor models.

Used by the offline scripts in ``scripts/`` to visualise the uncertainty of each
model, and by the unit tests to cross-check the symbolic Jacobians.
"""

import math

import numpy as np


def sample_velocity_motion_model(x, u, a, dt):
    """
    Sample the velocity motion model.

    Parameters
    ----------
    x : array_like
        Pose(s) before moving, shape (3,) or (N, 3): [x, y, theta].
    u : array_like
        Velocity command [v, w].
    a : array_like
        Noise parameters [a1, ..., a6] (velocity-dependent variances) or
        [std_v, std_w] (constant standard deviations).
    dt : float
        Prediction interval [s].

    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    u = np.asarray(u, dtype=float)
    a = np.asarray(a, dtype=float)

    sigma = np.ones(3)
    if a.shape == u.shape:
        sigma[:-1] = a
        sigma[-1] = a[1] * 0.5
    else:
        sigma[0] = a[0] * u[0] ** 2 + a[1] * u[1] ** 2
        sigma[1] = a[2] * u[0] ** 2 + a[3] * u[1] ** 2
        sigma[2] = a[4] * u[0] ** 2 + a[5] * u[1] ** 2

    n = x.shape[0]
    # noisy commands model actuation errors and unmodelled dynamics
    v_hat = u[0] + np.random.normal(0, sigma[0], n)
    w_hat = u[1] + np.random.normal(0, sigma[1], n)
    gamma_hat = np.random.normal(0, sigma[2], n)

    r = v_hat / w_hat
    x_prime = x[:, 0] - r * np.sin(x[:, 2]) + r * np.sin(x[:, 2] + w_hat * dt)
    y_prime = x[:, 1] + r * np.cos(x[:, 2]) - r * np.cos(x[:, 2] + w_hat * dt)
    theta_prime = x[:, 2] + w_hat * dt + gamma_hat * dt
    return np.squeeze(np.stack([x_prime, y_prime, theta_prime], axis=-1))


def velocity_mm_Gt(x, u, dt):
    """Jacobian of the velocity motion model w.r.t. the state [x, y, theta]."""
    theta = x[2]
    v, w = u[0], u[1]
    r = v / w
    return np.array([
        [1.0, 0.0, -r * math.cos(theta) + r * math.cos(theta + w * dt)],
        [0.0, 1.0, -r * math.sin(theta) + r * math.sin(theta + w * dt)],
        [0.0, 0.0, 1.0],
    ])


def velocity_mm_Vt(x, u, dt):
    """Jacobian of the velocity motion model w.r.t. the command [v, w]."""
    theta = x[2]
    v, w = u[0], u[1]
    s, c = math.sin(theta), math.cos(theta)
    s_dt, c_dt = math.sin(theta + w * dt), math.cos(theta + w * dt)
    return np.array([
        [(-s + s_dt) / w, dt * v * c_dt / w + v * (s - s_dt) / w ** 2],
        [(c - c_dt) / w, dt * v * s_dt / w - v * (c - c_dt) / w ** 2],
        [0.0, dt],
    ])


def landmark_range_bearing_model(robot_pose, landmark, sigma):
    """
    Sample a range-bearing measurement of a 2D landmark.

    ``robot_pose`` can be a single pose (3,) or a set of particles (N, 3).
    """
    robot_pose = np.atleast_2d(np.asarray(robot_pose, dtype=float))
    n = robot_pose.shape[0]
    r = np.linalg.norm(robot_pose[:, 0:2] - landmark, axis=1) + np.random.normal(0, sigma[0], n)
    phi = (np.arctan2(landmark[1] - robot_pose[:, 1], landmark[0] - robot_pose[:, 0])
           - robot_pose[:, 2] + np.random.normal(0, sigma[1], n))
    return np.squeeze(np.stack([r, phi], axis=-1))


def sample_pose_from_landmark(x0, z, landmark, sigma_r, sigma_phi):
    """
    Sample a robot position consistent with a range-bearing measurement z.

    The heading is assumed known (taken from ``x0``).
    """
    r = z[0] + np.random.normal(0, sigma_r)
    phi = z[1] + np.random.normal(0, sigma_phi)
    theta = x0[2]
    x = landmark[0] - r * math.cos(phi + theta)
    y = landmark[1] - r * math.sin(phi + theta)
    return np.array([x, y, theta])


def landmark_sm_Ht(x, mx, my):
    """Jacobian of the range-bearing model w.r.t. the state [x, y, theta]."""
    dx, dy = mx - x[0], my - x[1]
    q = dx ** 2 + dy ** 2
    d = math.sqrt(q)
    return np.array([
        [-dx / d, -dy / d, 0.0],
        [dy / q, -dx / q, -1.0],
    ])
