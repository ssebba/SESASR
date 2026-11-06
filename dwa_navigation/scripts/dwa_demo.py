#!/usr/bin/env python3
"""
Offline DWA demo: a TurtleBot3-like robot reaches a goal through a cluttered map.

No ROS needed. Run from the repository root:
    python dwa_navigation/scripts/dwa_demo.py            # interactive plot
    python dwa_navigation/scripts/dwa_demo.py --gif out.gif
"""

import argparse
import os
import sys

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dwa_navigation.dwa import DWA  # noqa: E402, I100


def make_obstacles(goal, seed=1):
    """Random point obstacles, keeping the start and the goal free."""
    rng = np.random.default_rng(seed)
    obstacles = rng.uniform([0.6, -0.6], [4.2, 3.0], size=(18, 2))
    free = (np.linalg.norm(obstacles - goal, axis=1) > 0.6) & \
        (np.linalg.norm(obstacles, axis=1) > 0.6)
    return obstacles[free]


def run(goal, obstacles, max_steps=900):
    dt = 1.0 / 15
    # Goal-reaching weights (velocity term with goal bonus, no target following)
    dwa = DWA(
        dt=dt, sim_time=4.0, time_granularity=0.1, v_samples=20, w_samples=25,
        weight_angle=1.0, weight_vel=1.0, weight_obs=1.0, weight_target=0.0,
        velocity_decay=False, collision_tol=0.1, goal_dist_tol=0.15,
        init_pose=np.array([0.0, 0.0, 0.0]),
        max_lin_vel=0.22, min_lin_vel=0.0, max_ang_vel=2.0, min_ang_vel=-2.0,
        max_linear_acc=0.7, max_ang_acc=np.pi, radius=0.15,
    )
    frames = []
    for _ in range(max_steps):
        pose = dwa.robot.pose
        if np.linalg.norm(pose[:2] - goal) < dwa.goal_dist_tol:
            print(f'Goal reached in {len(frames) * dt:.1f} s')
            break
        u, _, _, paths, best = dwa.compute_cmd(goal, pose, obstacles)
        frames.append((pose.copy(), paths[::12].copy(), paths[best].copy()))
        dwa.robot.update_state(u, dt)
    else:
        print('Timeout: goal not reached')
    return dwa.robot.trajectory, frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gif', help='save an animated GIF instead of showing the plot')
    parser.add_argument('--png', help='save the final trajectory as PNG')
    args = parser.parse_args()

    goal = np.array([4.5, 2.5])
    obstacles = make_obstacles(goal)
    trajectory, frames = run(goal, obstacles)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(obstacles[:, 0], obstacles[:, 1], c='k', s=40, label='obstacles')
    ax.scatter(*goal, marker='*', s=300, c='gold', edgecolors='k', zorder=5, label='goal')
    ax.set_xlim(-0.5, 5.2)
    ax.set_ylim(-1.2, 3.4)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Dynamic Window Approach')

    (traj_line,) = ax.plot([], [], 'b-', lw=2, label='trajectory')
    (best_line,) = ax.plot([], [], 'g-', lw=2, label='best rollout')
    candidates = [ax.plot([], [], color='0.6', lw=0.6)[0] for _ in range(len(frames[0][1]))]
    robot = plt.Circle((0, 0), 0.15, fc='none', ec='b', lw=1.5)
    ax.add_patch(robot)
    ax.legend(loc='lower right', fontsize=8)

    def draw(i):
        pose, paths, best = frames[i]
        traj_line.set_data(trajectory[:i + 1, 0], trajectory[:i + 1, 1])
        best_line.set_data(best[:, 0], best[:, 1])
        for line, path in zip(candidates, paths):
            line.set_data(path[:, 0], path[:, 1])
        robot.center = (pose[0], pose[1])
        return [traj_line, best_line, robot, *candidates]

    if args.png:
        draw(len(frames) - 1)
        fig.savefig(args.png, dpi=150, bbox_inches='tight')
        print(f'Saved {args.png}')
    if args.gif:
        anim = animation.FuncAnimation(fig, draw, frames=range(0, len(frames), 3), blit=True)
        anim.save(args.gif, writer=animation.PillowWriter(fps=15))
        print(f'Saved {args.gif}')
    if not (args.gif or args.png):
        anim = animation.FuncAnimation(fig, draw, frames=len(frames), interval=1000 / 15)
        plt.show()


if __name__ == '__main__':
    main()
