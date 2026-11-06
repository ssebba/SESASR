# Media

| File | Content | How it is produced |
| --- | --- | --- |
| `dwa_demo.gif`, `dwa_demo.png` | Offline DWA navigation | `python dwa_navigation/scripts/dwa_demo.py --gif docs/media/dwa_demo.gif --png docs/media/dwa_demo.png` |
| `ekf_demo.png` | EKF vs. dead reckoning, offline | `python ekf_localization/scripts/ekf_demo.py --png docs/media/ekf_demo.png` |
| `dwa_gazebo.gif` | DWA in Gazebo | `ros2 launch dwa_navigation record_demo.launch.py output:=docs/media/dwa_gazebo.gif` |
| `ekf_gazebo.png` | Both EKF nodes in Gazebo | `ros2 launch ekf_localization record_demo.launch.py output:=docs/media/ekf_gazebo.png` |

`real_dwa.gif` and `real_reactive.gif` are phone videos of the real robot, converted to
320 px, 8 fps and 5x speed.
