"""
Record an animated GIF of a DWA run from an overhead camera in Gazebo.

The node saves the camera images and draws on them the trajectory of the robot,
the LiDAR obstacles, the trajectories evaluated by the planner and the moving
goal. It needs no screen capture, so it also works with a headless simulator.
It shuts down after ``duration`` simulated seconds, or earlier if the robot stalls.
"""

import math
import os

from nav_msgs.msg import Odometry
import numpy as np
from PIL import Image as PilImage
from PIL import ImageDraw
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from visualization_msgs.msg import MarkerArray


class DemoRecorder(Node):

    def __init__(self):
        super().__init__('demo_recorder')

        self.declare_parameter('output', 'dwa_gazebo.gif')
        self.declare_parameter('camera_center', [0.0, 0.0])  # camera position [m]
        self.declare_parameter('camera_height', 10.0)  # [m]
        self.declare_parameter('camera_hfov', 1.047)  # [rad]
        self.declare_parameter('frame_period', 0.75)  # simulated seconds between frames
        self.declare_parameter('playback_fps', 10)
        self.declare_parameter('duration', 120.0)  # simulated seconds
        self.declare_parameter('stall_timeout', 40.0)  # simulated seconds without moving

        param = self.get_parameter
        self.output = param('output').value
        self.center = param('camera_center').value
        self.height = param('camera_height').value
        self.hfov = param('camera_hfov').value
        self.frame_period = param('frame_period').value
        self.playback_fps = param('playback_fps').value
        self.duration = param('duration').value
        self.stall_timeout = param('stall_timeout').value

        self.pose = None
        self.goal = None
        self.trajectory = []
        self.obstacles = []
        self.candidates = []
        self.best_path = []
        self.frames = []
        self.dist_to_goal = []
        self.min_range = math.inf
        self.start_time = None
        self.last_frame_time = None
        self.last_progress_time = None
        self.last_progress_pose = None
        self.done = False

        self.create_subscription(Odometry, '/ground_truth', self.pose_callback, 10)
        self.create_subscription(Odometry, '/dynamic_goal_pose', self.goal_callback, 10)
        self.create_subscription(MarkerArray, '/dwa/markers', self.markers_callback, 10)
        self.create_subscription(Float32, '/dist_to_goal', self.dist_callback, 10)
        self.create_subscription(Float32, '/dwa/min_lidar_range', self.range_callback, 10)
        self.create_subscription(
            Image, '/overhead/image_raw', self.image_callback, qos_profile_sensor_data)
        self.create_timer(1.0, self.supervise)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def pose_callback(self, msg):
        p = msg.pose.pose.position
        self.pose = np.array([p.x, p.y])
        if not self.trajectory or np.linalg.norm(self.pose - self.trajectory[-1]) > 0.02:
            self.trajectory.append(self.pose)

    def goal_callback(self, msg):
        p = msg.pose.pose.position
        self.goal = (p.x, p.y)

    def dist_callback(self, msg):
        self.dist_to_goal.append(msg.data)

    def range_callback(self, msg):
        self.min_range = min(self.min_range, msg.data)

    def markers_callback(self, msg):
        for marker in msg.markers:
            points = [(p.x, p.y) for p in marker.points]
            if marker.ns == 'dwa_obstacles':
                self.obstacles = points
            elif marker.ns == 'dwa_candidates':
                self.candidates = points
            elif marker.ns == 'dwa_best':
                self.best_path = points

    def supervise(self):
        """Detect the end of the run."""
        t = self.now()
        if self.done or self.pose is None or not self.frames:
            return
        if self.start_time is None:
            self.start_time = t
            self.last_progress_time = t
            self.last_progress_pose = self.pose
            return

        if np.linalg.norm(self.pose - self.last_progress_pose) > 0.05:
            self.last_progress_pose = self.pose
            self.last_progress_time = t

        if t - self.start_time > self.duration:
            self.finish(True, 'duration reached')
        elif t - self.last_progress_time > self.stall_timeout:
            self.finish(False, f'robot stalled at {np.round(self.pose, 2).tolist()}')

    def world_to_pixel(self, x, y, width, height):
        metres_per_pixel = 2.0 * self.height * math.tan(self.hfov / 2.0) / width
        return (width / 2.0 + (x - self.center[0]) / metres_per_pixel,
                height / 2.0 - (y - self.center[1]) / metres_per_pixel)

    def image_callback(self, msg):
        t = self.now()
        if self.done or t == 0.0:
            return
        if self.last_frame_time is not None and t - self.last_frame_time < self.frame_period:
            return
        self.last_frame_time = t

        data = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        image = PilImage.fromarray(data, 'RGB')
        self.draw_overlay(ImageDraw.Draw(image), msg.width, msg.height)
        self.frames.append(image.convert('P', palette=PilImage.ADAPTIVE, colors=64))

    def draw_overlay(self, draw, width, height):
        def px(point):
            return self.world_to_pixel(point[0], point[1], width, height)

        for a, b in zip(self.candidates[0::2], self.candidates[1::2]):
            draw.line([px(a), px(b)], fill=(170, 170, 170), width=1)
        if len(self.trajectory) > 1:
            draw.line([px(p) for p in self.trajectory], fill=(30, 90, 220), width=2)
        if len(self.best_path) > 1:
            draw.line([px(p) for p in self.best_path], fill=(0, 200, 60), width=2)
        for point in self.obstacles:
            u, v = px(point)
            draw.ellipse([u - 2, v - 2, u + 2, v + 2], fill=(230, 40, 40))
        if self.goal is not None:
            u, v = px(self.goal)
            draw.ellipse([u - 6, v - 6, u + 6, v + 6], outline=(255, 200, 0), width=2)

    def finish(self, success, reason):
        self.done = True
        distances = np.array(self.dist_to_goal) if self.dist_to_goal else np.array([math.nan])
        self.get_logger().info(
            f'Run finished ({reason}): {self.now() - self.start_time:.1f} s simulated, '
            f'{len(self.frames)} frames, distance to goal mean {distances.mean():.2f} m '
            f'max {distances.max():.2f} m, closest LiDAR reading {self.min_range:.3f} m')
        directory = os.path.dirname(self.output)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.frames[0].save(
            self.output, save_all=True, append_images=self.frames[1:],
            duration=round(1000 / self.playback_fps), loop=0, optimize=True)
        self.get_logger().info(f'Saved {self.output}')
        raise SystemExit(0 if success else 1)


def main(args=None):
    rclpy.init(args=args)
    node = DemoRecorder()
    code = 1
    try:
        rclpy.spin(node)
    except SystemExit as e:
        code = e.code
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(code)


if __name__ == '__main__':
    main()
