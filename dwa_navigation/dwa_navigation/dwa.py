"""Dynamic Window Approach local planner (pure NumPy, no ROS)."""

import numpy as np

from .utils import calc_nearest_obs, DifferentialDriveRobot, normalize, normalize_angle


class DWA:
    """
    Dynamic Window Approach for a differential-drive robot.

    At every control step the planner:
      1. computes the window of velocities reachable within one step given the
         acceleration limits (dynamic window);
      2. samples ``v_samples x w_samples`` commands inside it and forward
         simulates each one at constant velocity;
      3. scores every trajectory and returns the command of the best one.

    The objective is a weighted sum of min-max normalised terms:
        J = w_angle * heading + w_vel * velocity + w_obs * clearance
            + w_target * target_distance
    """

    def __init__(self,
                 dt=0.1,                       # control period [s]
                 sim_time=2.0,                 # sim_time / time_granularity = rollout steps
                 time_granularity=0.1,
                 v_samples=10,
                 w_samples=20,
                 goal_dist_tol=0.3,            # goal reached below this distance [m]
                 collision_tol=0.3,            # safety margin around the robot [m]
                 weight_angle=0.04,
                 weight_vel=0.2,
                 weight_obs=0.1,
                 weight_vel_decay=0.1,
                 weight_target=0.05,
                 slowdown_distance=1.0,        # velocity decay length near the goal [m]
                 target_distance_desired=0.8,  # preferred distance from the target [m]
                 velocity_decay=True,          # slow down near the goal (target following)
                 obstacle_max_dist=3.0,        # radius of the local obstacle map [m]
                 **robot_params):
        self.dt = dt
        # Each rollout has sim_step points spaced by dt
        self.sim_step = round(sim_time / time_granularity)
        self.robot = DifferentialDriveRobot(**robot_params)
        self.goal_dist_tol = goal_dist_tol
        self.collision_tol = collision_tol

        self.v_samples = v_samples
        self.w_samples = w_samples

        self.weight_angle = weight_angle
        self.weight_vel = weight_vel
        self.weight_obs = weight_obs
        self.weight_vel_decay = weight_vel_decay
        self.weight_target = weight_target
        self.slowdown_distance = slowdown_distance
        self.target_distance_desired = target_distance_desired
        self.velocity_decay = velocity_decay
        self.obstacle_max_dist = obstacle_max_dist

    def compute_cmd(self, goal_pose, robot_pose, obstacles):
        """
        Compute the next velocity command u = [v, w].

        Returns
        -------
        tuple
            ``(u, costs, active_obstacles, paths, best_idx)`` where ``costs``
            holds the weighted score terms of the selected trajectory.

        """
        paths, velocities = self.get_trajectories(robot_pose)
        best_idx, costs, active_obs = self.evaluate_paths(
            paths, velocities, goal_pose, robot_pose, obstacles)
        return velocities[best_idx], costs, active_obs, paths, best_idx

    def get_trajectories(self, robot_pose):
        """Sample the dynamic window and simulate one trajectory per command."""
        min_v, max_v, min_w, max_w = self.compute_dynamic_window(self.robot.vel)
        v_values = np.linspace(min_v, max_v, self.v_samples)
        w_values = np.linspace(min_w, max_w, self.w_samples)

        vv, ww = np.meshgrid(v_values, w_values)
        velocities = np.dstack([vv, ww]).reshape(-1, 2)
        return self.simulate_paths(robot_pose, velocities), velocities

    def simulate_paths(self, pose, u):
        """Forward simulate all the commands u (N, 2) from ``pose``: returns (N, sim_step, 3)."""
        paths = np.zeros((u.shape[0], self.sim_step, pose.shape[0]))
        paths[:, 0] = pose
        for i in range(1, self.sim_step):
            th = paths[:, i - 1, 2]
            paths[:, i, 0] = paths[:, i - 1, 0] + u[:, 0] * np.cos(th) * self.dt
            paths[:, i, 1] = paths[:, i - 1, 1] + u[:, 0] * np.sin(th) * self.dt
            paths[:, i, 2] = th + u[:, 1] * self.dt
        return paths

    def compute_dynamic_window(self, robot_vel):
        """Velocities reachable in one step, clipped to the robot limits."""
        r = self.robot
        min_v = max(robot_vel[0] - self.dt * r.max_linear_acc, r.min_lin_vel)
        max_v = min(robot_vel[0] + self.dt * r.max_linear_acc, r.max_lin_vel)
        min_w = max(robot_vel[1] - self.dt * r.max_ang_acc, r.min_ang_vel)
        max_w = min(robot_vel[1] + self.dt * r.max_ang_acc, r.max_ang_vel)
        return min_v, max_v, min_w, max_w

    def evaluate_paths(self, paths, velocities, goal_pose, robot_pose, obstacles):
        """Score all trajectories and return the index of the best one."""
        nearest_obs = calc_nearest_obs(robot_pose, obstacles, self.obstacle_max_dist)

        heading = normalize(self.score_heading_angle(paths, goal_pose))
        obstacle = normalize(self.score_obstacles(paths, nearest_obs))
        target = normalize(self.score_target_distance(paths, goal_pose))
        if self.velocity_decay:
            vel = normalize(self.score_vel_decay(velocities, paths, goal_pose))
            weight_vel = self.weight_vel_decay
        else:
            vel = normalize(self.score_vel(velocities, paths, goal_pose))
            weight_vel = self.weight_vel

        total = (self.weight_angle * heading + weight_vel * vel
                 + self.weight_obs * obstacle + self.weight_target * target)
        best = int(np.argmax(total))

        costs = {
            'heading': self.weight_angle * heading[best],
            'vel': weight_vel * vel[best],
            'obs': self.weight_obs * obstacle[best],
            'target': self.weight_target * target[best],
            'total': total[best],
        }
        return best, costs, nearest_obs

    def score_heading_angle(self, path, goal_pose):
        """Higher when the final heading of the trajectory points at the goal."""
        last = path[:, -1]
        angle_to_goal = np.arctan2(goal_pose[1] - last[:, 1], goal_pose[0] - last[:, 0])
        return np.pi - np.fabs(normalize_angle(angle_to_goal - last[:, 2]))

    def score_vel(self, u, path, goal_pose):
        """Prefer fast trajectories, plus a bonus for ending close to the goal."""
        dist_to_goal = np.linalg.norm(path[:, -1, 0:2] - goal_pose, axis=-1)
        return u[:, 0] + np.exp(-dist_to_goal / self.goal_dist_tol)

    def score_vel_decay(self, u, path, goal_pose):
        """Prefer fast trajectories far from the goal and slow ones close to it."""
        dist_to_goal = np.linalg.norm(path[:, -1, 0:2] - goal_pose, axis=-1)
        return u[:, 0] - np.exp(-dist_to_goal / self.slowdown_distance)

    def score_obstacles(self, path, obstacles):
        """Clearance from the nearest obstacle (capped at 2 m); colliding paths get -100."""
        score = 2.0 * np.ones(path.shape[0])
        for obs in obstacles:
            dist = np.hypot(path[:, :, 0] - obs[0], path[:, :, 1] - obs[1])
            score = np.minimum(score, np.min(dist, axis=-1))
        score[score < self.robot.radius + self.collision_tol] = -100.0
        return score

    def score_target_distance(self, path, target_pose):
        """Gaussian reward for ending at ``target_distance_desired`` from the target."""
        dist = np.linalg.norm(path[:, -1, 0:2] - target_pose, axis=-1)
        error = np.abs(dist - self.target_distance_desired)
        return np.exp(-error ** 2 / (2 * 0.2 ** 2))
