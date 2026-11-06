"""Unit tests for the DWA planner (no ROS required)."""

from dwa_navigation.dwa import DWA
from dwa_navigation.utils import calc_nearest_obs, normalize
import numpy as np
import pytest

ROBOT = {
    'max_lin_vel': 0.22, 'min_lin_vel': 0.0, 'max_ang_vel': 2.0, 'min_ang_vel': -2.0,
    'max_linear_acc': 0.7, 'max_ang_acc': np.pi, 'radius': 0.15,
}


def make_dwa(**kwargs):
    params = {
        'dt': 1.0 / 15, 'sim_time': 4.0, 'time_granularity': 0.1, 'v_samples': 20,
        'w_samples': 25, 'weight_angle': 1.0, 'weight_vel': 1.0, 'weight_obs': 1.0,
        'weight_target': 0.0, 'velocity_decay': False, 'collision_tol': 0.1,
        'goal_dist_tol': 0.15, **ROBOT,
    }
    params.update(kwargs)
    return DWA(**params)


def test_dynamic_window_respects_limits():
    dwa = make_dwa()
    min_v, max_v, min_w, max_w = dwa.compute_dynamic_window(np.array([0.0, 0.0]))
    assert min_v == 0.0
    assert max_v == pytest.approx(0.7 / 15)
    assert (min_w, max_w) == pytest.approx((-np.pi / 15, np.pi / 15))

    min_v, max_v, min_w, max_w = dwa.compute_dynamic_window(np.array([0.22, 2.0]))
    assert max_v == 0.22
    assert max_w == 2.0


def test_simulate_paths_straight_and_turning():
    dwa = make_dwa()
    u = np.array([[0.2, 0.0], [0.0, 1.0]])
    paths = dwa.simulate_paths(np.zeros(3), u)
    assert paths.shape == (2, dwa.sim_step, 3)
    steps = dwa.sim_step - 1
    assert paths[0, -1] == pytest.approx([0.2 * dwa.dt * steps, 0.0, 0.0])
    assert paths[1, -1] == pytest.approx([0.0, 0.0, dwa.dt * steps])


def test_colliding_paths_are_penalised():
    dwa = make_dwa()
    paths = dwa.simulate_paths(np.zeros(3), np.array([[0.2, 0.0], [0.2, 1.5]]))
    scores = dwa.score_obstacles(paths, np.array([[0.4, 0.0]]))
    assert scores[0] == -100.0
    assert scores[1] > 0.0


def test_reaches_goal_in_free_space():
    dwa = make_dwa(init_pose=np.zeros(3))
    goal = np.array([1.5, 1.0])
    for _ in range(600):
        if np.linalg.norm(dwa.robot.pose[:2] - goal) < dwa.goal_dist_tol:
            break
        u, *_ = dwa.compute_cmd(goal, dwa.robot.pose, np.empty((0, 2)))
        dwa.robot.update_state(u, dwa.dt)
    assert np.linalg.norm(dwa.robot.pose[:2] - goal) < dwa.goal_dist_tol


def test_swerves_around_obstacle_near_the_straight_line():
    dwa = make_dwa(init_pose=np.zeros(3))
    goal = np.array([2.0, 0.0])
    # Driving straight would pass 0.15 m from the first obstacle (= robot radius).
    # Note: an obstacle exactly on the line is a local minimum where DWA stops.
    obstacles = np.array([[1.0, 0.15], [1.0, 0.3]])
    min_clearance = np.inf
    for _ in range(900):
        pose = dwa.robot.pose
        min_clearance = min(min_clearance, np.min(np.linalg.norm(obstacles - pose[:2], axis=1)))
        if np.linalg.norm(pose[:2] - goal) < dwa.goal_dist_tol:
            break
        u, *_ = dwa.compute_cmd(goal, pose, obstacles)
        dwa.robot.update_state(u, dwa.dt)
    assert np.linalg.norm(dwa.robot.pose[:2] - goal) < dwa.goal_dist_tol
    assert min_clearance >= ROBOT['radius'] + dwa.collision_tol - 1e-3


def test_helpers():
    assert normalize(np.array([1.0, 3.0, 2.0])) == pytest.approx([0.0, 1.0, 0.5])
    assert normalize(np.array([2.0, 2.0])) == pytest.approx([0.0, 0.0])
    near = calc_nearest_obs(np.zeros(3), np.array([[1.0, 0.0], [5.0, 0.0]]), 3.0)
    assert near.tolist() == [[1.0, 0.0]]
    assert calc_nearest_obs(np.zeros(3), np.empty((0, 2))).shape == (0, 2)
