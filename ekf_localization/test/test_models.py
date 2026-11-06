"""Unit tests for the motion/measurement models (no ROS required)."""

from ekf_localization.models import (
    imu_gyro_model, landmark_measurement_model, odometry_velocity_model,
    velocity_motion_model)
from ekf_localization.probabilistic_models import (
    landmark_sm_Ht, velocity_mm_Gt, velocity_mm_Vt)
from ekf_localization.utils import normalize_angle, residual
import numpy as np
import pytest


def numerical_jacobian(f, x, eps=1e-6):
    x = np.asarray(x, dtype=float)
    f0 = f(x)
    J = np.zeros((f0.size, x.size))
    for i in range(x.size):
        dx = np.zeros_like(x)
        dx[i] = eps
        J[:, i] = (f(x + dx) - f(x - dx)) / (2 * eps)
    return J


@pytest.mark.parametrize('angle, expected', [
    (0.0, 0.0), (np.pi / 2, np.pi / 2), (3 * np.pi, -np.pi), (-3 * np.pi / 2, np.pi / 2),
])
def test_normalize_angle(angle, expected):
    assert normalize_angle(angle) == pytest.approx(expected)


def test_residual_wraps_bearing():
    y = residual([1.0, np.pi - 0.1], [0.5, -np.pi + 0.1], angle_idx=1)
    assert y == pytest.approx([0.5, -0.2])


@pytest.mark.parametrize('w', [0.5, -1.2])
def test_motion_jacobians_match_analytic_and_numerical(w):
    g, G, V = velocity_motion_model(3)
    mu, u, dt = np.array([1.0, -2.0, 0.3]), np.array([0.4, w]), 0.1

    assert G(mu, u, dt) == pytest.approx(velocity_mm_Gt(mu, u, dt))
    assert V(mu, u, dt) == pytest.approx(velocity_mm_Vt(mu, u, dt))
    assert G(mu, u, dt) == pytest.approx(numerical_jacobian(lambda m: g(m, u, dt), mu))
    assert V(mu, u, dt) == pytest.approx(
        numerical_jacobian(lambda c: g(mu, c, dt), u), abs=1e-6)


def test_motion_model_straight_line_when_omega_is_zero():
    g, G, V = velocity_motion_model(3)
    mu = np.array([0.0, 0.0, np.pi / 2])
    out = g(mu, np.array([1.0, 0.0]), 0.5)
    assert out == pytest.approx([0.0, 0.5, np.pi / 2])
    assert np.all(np.isfinite(G(mu, np.array([1.0, 0.0]), 0.5)))
    assert np.all(np.isfinite(V(mu, np.array([1.0, 0.0]), 0.5)))


def test_5d_model_uses_state_velocities():
    g, G, V = velocity_motion_model(5)
    mu = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
    # The control is ignored by the mean propagation of the 5D model
    out = g(mu, np.array([5.0, 3.0]), 0.1)
    assert out == pytest.approx([0.1, 0.0, 0.0, 1.0, 0.0])
    assert G(mu, np.zeros(2), 0.1).shape == (5, 5)
    assert V(mu, np.zeros(2), 0.1).shape == (5, 2)


def test_landmark_model():
    h, H = landmark_measurement_model(3)
    mu, m = np.array([1.0, 1.0, np.pi / 2]), np.array([1.0, 3.0, 0.0])
    assert h(mu, m) == pytest.approx([2.0, 0.0])
    assert H(mu, m) == pytest.approx(landmark_sm_Ht(mu, m[0], m[1]))
    assert H(mu, m) == pytest.approx(numerical_jacobian(lambda s: h(s, m), mu))


def test_landmark_range_is_planar():
    # turtlebot3_perception publishes sqrt(x^2 + y^2): the tag height must not count
    h, _ = landmark_measurement_model(3)
    mu = np.array([0.0, 0.0, 0.0])
    assert h(mu, np.array([2.0, 0.0, 1.5])) == pytest.approx(h(mu, np.array([2.0, 0.0, 0.0])))
    assert h(mu, np.array([2.0, 0.0, 1.5]))[0] == pytest.approx(2.0)


def test_landmark_model_5d_is_padded():
    _, H = landmark_measurement_model(5)
    J = H(np.array([0.0, 0.0, 0.0, 0.2, 0.1]), np.array([2.0, 1.0, 0.3]))
    assert J.shape == (2, 5)
    assert np.all(J[:, 3:] == 0.0)


def test_linear_sensor_models():
    mu = np.array([1.0, 2.0, 0.3, 0.25, -0.4])
    h_odom, H_odom = odometry_velocity_model()
    h_imu, H_imu = imu_gyro_model()
    assert h_odom(mu) == pytest.approx([0.25, -0.4])
    assert h_imu(mu) == pytest.approx([-0.4])
    assert H_odom(mu).shape == (2, 5)
    assert H_imu(mu).shape == (1, 5)
