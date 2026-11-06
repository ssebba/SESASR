# sesasr_bringup

Starts a Gazebo world of the `turtlebot3_gazebo` package and spawns the TurtleBot3. All
the simulation launch files in this repository use it.

```bash
ros2 launch sesasr_bringup sim_world.launch.py world:=hard_maze.world
ros2 launch sesasr_bringup sim_world.launch.py world:=maze.world gui:=false   # no windows
```

| Argument | Default | Description |
| --- | --- | --- |
| `world` | required | world file in `turtlebot3_gazebo/worlds` |
| `x_pose`, `y_pose` | `0.0` | spawn position [m] |
| `gui` | `true` | `false` runs Gazebo without any window |

`gzserver` can take more than 30 s to start, which is the default timeout of
`spawn_entity`: when that happens the robot never appears. This launch file uses a longer
timeout and does not wait for the online model database.
