"""Behavioural tests for the EKF on a simulated robot (no ROS required)."""

from ekf_localization.ekf import RobotEKF
from ekf_localization.models import (
    imu_gyro_model, landmark_measurement_model, velocity_motion_model)
from ekf_localization.utils import normalize_angle
import numpy as np
import pytest

LANDMARKS = [np.array([x, y, 0.0]) for x in (-2.0, 2.0) for y in (-2.0, 2.0)]
QT = np.diag([0.05 ** 2, np.deg2rad(2) ** 2])


def make_ekf(state_dim=3):
    return RobotEKF(state_dim, 2, *velocity_motion_model(state_dim))


def test_landmark_updates_correct_a_wrong_initial_guess():
    rng = np.random.default_rng(0)
    ekf = make_ekf()
    h, H = landmark_measurement_model(3)

    truth = np.array([0.5, -0.3, 0.2])
    ekf.mu = np.array([0.0, 0.0, 0.0])
    ekf.Sigma = np.diag([1.0, 1.0, 0.3])

    for _ in range(20):
        for m in LANDMARKS:
            z = h(truth, m) + rng.normal(0, [0.05, np.deg2rad(2)])
            ekf.update(z, h(ekf.mu, m), H(ekf.mu, m), QT, angle_idx=1)

    assert ekf.mu[:2] == pytest.approx(truth[:2], abs=0.05)
    assert normalize_angle(ekf.mu[2] - truth[2]) == pytest.approx(0.0, abs=0.05)
    assert np.trace(ekf.Sigma) < 0.01


def test_prediction_grows_covariance_and_stays_symmetric():
    ekf = make_ekf()
    ekf.Sigma = np.eye(3) * 1e-3
    ekf.Mt = np.diag([0.01, 0.01])
    trace0 = np.trace(ekf.Sigma)
    for _ in range(50):
        ekf.predict(np.array([0.2, 0.3]), 0.05)

    assert np.trace(ekf.Sigma) > trace0
    assert ekf.Sigma == pytest.approx(ekf.Sigma.T)
    assert np.all(np.linalg.eigvalsh(ekf.Sigma) > 0)


def test_tracking_beats_dead_reckoning():
    rng = np.random.default_rng(1)
    g, _, _ = velocity_motion_model(3)
    h, H = landmark_measurement_model(3)
    ekf = make_ekf()
    ekf.Sigma = np.eye(3) * 0.01
    dt = 0.05

    truth = np.zeros(3)
    dead_reckoning = np.zeros(3)
    for k in range(600):
        u = np.array([0.2, 0.2])
        truth = g(truth, u, dt)
        u_meas = u * [1.05, 0.95] + rng.normal(0, [0.01, 0.02])
        dead_reckoning = g(dead_reckoning, u_meas, dt)

        ekf.Mt = np.diag([0.01 * u_meas[0] ** 2, 0.01 * u_meas[1] ** 2 + 1e-4])
        ekf.predict(u_meas, dt)
        if k % 4 == 0:
            for m in LANDMARKS:
                z = h(truth, m) + rng.normal(0, [0.05, np.deg2rad(2)])
                ekf.update(z, h(ekf.mu, m), H(ekf.mu, m), QT, angle_idx=1)

    err_ekf = np.linalg.norm(ekf.mu[:2] - truth[:2])
    err_dr = np.linalg.norm(dead_reckoning[:2] - truth[:2])
    assert err_ekf < 0.1
    assert err_ekf < err_dr


def test_scalar_measurement_update():
    ekf = make_ekf(5)
    ekf.Sigma = np.eye(5)
    h, H = imu_gyro_model()
    ekf.update(np.array([0.5]), h(ekf.mu), H(ekf.mu), np.array([[0.01]]))
    # Only the angular-velocity state is observed and corrected
    assert ekf.mu[4] == pytest.approx(0.5 / 1.01)
    assert ekf.mu[:4] == pytest.approx(np.zeros(4))
