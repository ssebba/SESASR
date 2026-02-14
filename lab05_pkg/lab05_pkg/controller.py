import rclpy
from rclpy.node import Node
from rclpy.timer import Timer
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from tf_transformations import euler_from_quaternion
import numpy as np
from .dwa import DWA
from .plot_utils import plot_obstacles, plot_trajectories, plot_velocities
import matplotlib.pyplot as plt
from std_msgs.msg import Float32
from landmark_msgs.msg import LandmarkArray
import math
from rclpy.qos import qos_profile_sensor_data


from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

class DWANavigator(Node):
    def __init__(self):
        super().__init__('dwa_navigator')
        
        # Node parameters
        # self.declare_parameter('control_frequency', 15)  # 15 Hz
        self.declare_parameter('collision_tol', 0.20)  # 15-25 cm
        # self.declare_parameter('robot_radius', 0.15)  # TurtleBot3 Burger #con 0.15 va bene
        self.declare_parameter('lidar_max_range', 3.5)
        self.declare_parameter('num_lidar_ranges', 25)
        self.declare_parameter('goal_dist_tol', 0.35)  # 30 cm
        
        # self.control_freq = self.get_parameter('control_frequency').value
        self.collision_tol = self.get_parameter('collision_tol').value
        # self.robot_radius = self.get_parameter('robot_radius').value
        self.lidar_max_range = self.get_parameter('lidar_max_range').value
        self.num_lidar_ranges = self.get_parameter('num_lidar_ranges').value
        self.goal_dist_tol = self.get_parameter('goal_dist_tol').value
        
        #self.counter_obstacles = 0

        self.stop_flag = False
        self.timeout_flag = False

        # Subscribers
        self.scan_subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile_sensor_data
        )
        
        self.odom_subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        
        self.goal_subscription = self.create_subscription(
            Odometry,
            '/dynamic_goal_pose',  # o /camera/landmarks per real robot
            self.goal_callback,
            10
        )

        self.landmarks_subscription = self.create_subscription(
            LandmarkArray,
            #'/camera/landmarks',
            '/landmarks',
            self.real_goal_callback,
            10
        )
        
        # Publisher
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # Publisher to visualize parameters in RViz
        # self.marker_publisher = self.create_publisher(
        #     Marker,
        #     '/dwa_obstacles_markers',
        #     10
        # )

        # self.active_obs_publisher = self.create_publisher(
        #     Marker,
        #     '/dwa_near_obstacle_marker',
        #     10
        # )

        # Publisher for cost function
        self.pub_cost_heading = self.create_publisher(Float32, '/dwa_cost/heading', 10)
        self.pub_cost_vel = self.create_publisher(Float32, '/dwa_cost/vel', 10)
        self.pub_cost_obs = self.create_publisher(Float32, '/dwa_cost/obs', 10)
        self.pub_cost_total = self.create_publisher(Float32, '/dwa_cost/total', 10)

        # Publisher for LiDAR min range
        self.pub_min_lidar_range = self.create_publisher(Float32, '/dwa/min_lidar_range', 10)

        # Publisher per la distanza dal goal
        self.pub_dist_to_goal = self.create_publisher(Float32, '/dist_to_goal', 10)

        # Publisher for paths
        #self.candidate_paths_pub = self.create_publisher(Marker, '/dwa_visual/candidates', 10)
        #self.optimal_path_pub = self.create_publisher(Marker, '/dwa_visual/optimal', 10)
        
        # Timer for the control loop
        period = 1.0 / 15
        self.timer = self.create_timer(period, self.control_loop_callback)
        
        # Robot and obstacles state
        self.robot_pose = np.array([0.0, 0.0, 0.0])  # [x, y, theta]
        self.robot_vel = np.array([0.0, 0.0])  # [v, w]
        self.lidar_scan = None
        self.goal_pose = None  # [x, y]
        self.obstacles = []  # Lista di ostacoli nel global frame
        
        # DWA instance 
        self.dwa = DWA(
            # dt=1.0/self.control_freq,       # con sim_time=3.0, v_samples=15, w_samples=25 funziona discretamente
            # sim_time=5.0,
            # time_granularity=0.1,
            # v_samples=10,
            # w_samples=20,
            # weight_angle=0.06,
            # weight_vel=0.2,
            # #weight_obs=0.04,
            # weight_obs=0.04,
            # collision_tol=self.collision_tol,
            # goal_dist_tol=self.goal_dist_tol,
            # # Nuovi parametri per il Task 2
            # weight_vel_decay=0.15,
            # weight_target=0.20,
            # slowdown_distance=1.0,
            # target_distance_desired=0.3,
            dt=1.0/15,       # con sim_time=3.0, v_samples=15, w_samples=25 funziona discretamente
            sim_time=4.0,
            time_granularity=0.1,
            v_samples=20,
            w_samples=25,
            weight_angle=2.0,
            #weight_vel=1.5,       # (ignorato se uso score_vel_decay)
            #weight_obs=0.04,
            weight_obs=2.0,
            collision_tol=self.collision_tol,
            goal_dist_tol=self.goal_dist_tol,
            # Nuovi parametri per il Task 2
            weight_vel_decay=1.0,          # era 0.15
            weight_target=2.0,
            slowdown_distance=0.40,
            target_distance_desired=0.30,
            
            # Robot parameters
            min_lin_vel=0.0,
            max_lin_vel=0.22,
            min_ang_vel=-2.0,
            max_ang_vel=2.0,
            max_linear_acc=0.7,     # era 0.5
            max_ang_acc=math.pi,
            radius=0.15,
            obstacles_map=self.obstacles,
            init_pose=self.robot_pose

        )
        
        # Metrics and state
        self.step_count = 0
        self.feedback_rate = 50
        self.navigation_active = False
        
        self.get_logger().info('DWA Navigator inizializzato')
    
    def scan_callback(self, msg: LaserScan):
        """Callback for LiDAR readings"""
        self.lidar_scan = msg
    
    def odom_callback(self, msg: Odometry):
        """Callback for robot odometry"""
        # Extract position
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        # Extract orientation (quaternione -> euler)
        quat = msg.pose.pose.orientation
        _, _, theta = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
        
        self.robot_pose = np.array([x, y, theta])
        
        # Estrai velocità
        v = msg.twist.twist.linear.x
        w = msg.twist.twist.angular.z
        self.robot_vel = np.array([v, w])
    
    def goal_callback(self, msg: Odometry):
        """Callback for received goal"""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        if self.navigation_active:
            if x!= self.goal_pose[0] or y!= self.goal_pose[1]: #reset step count if the goal has moved
                self.step_count=0

        self.goal_pose = np.array([x, y])
        
        if not self.navigation_active:
            self.navigation_active = True
            self.step_count = 0
            self.get_logger().info(f'Goal ricevuto: [{x:.2f}, {y:.2f}]')

        # self.counter_obstacles += 1
        # if self.counter_obstacles % 5 == 0:
        #     #plt.show()
        #     self.get_logger().info(f'Ostacoli aggiornati: {len(self.obstacles)}')
        #     #plot_obstacles(np.array(self.obstacles), plt.gca())
    
    def filter_lidar_scan(self, scan: LaserScan) -> np.ndarray:
        """
        Filters LiDAR readings and reduces range #
        
        Input: LaserScan message
        Output: Array of filtered ranges (num_lidar_ranges,)
        """
        ranges = np.array(scan.ranges)
        
        # 1. Remove NaN e Inf
        ranges[np.isnan(ranges)] = scan.range_min if scan.range_min > 0 else 0.01 # Assegna max per NaN  (prima era scan.range_max)
        ranges[np.isinf(ranges)] = scan.range_max  # Assegna max per Inf
        
        # 2. Limit to max_range
        ranges[ranges > self.lidar_max_range] = self.lidar_max_range
        
        # 3. Reduce ranges # from 270 to num_lidar_ranges
        # Pick the minimum for each angular sector
        total_ranges = len(ranges)
        #indices = np.linspace(0, total_ranges - 1, self.num_lidar_ranges, dtype=int)
        sector_size = total_ranges // self.num_lidar_ranges

        filtered_ranges = np.zeros(self.num_lidar_ranges)
        filtered_angles = np.zeros(self.num_lidar_ranges)
        
        
        # for i in range(self.num_lidar_ranges):
        #     start_idx = i * sector_size
        #     end_idx = start_idx + sector_size if i < self.num_lidar_ranges - 1 else total_ranges
        #     filtered_ranges[i] = np.min(ranges[start_idx:end_idx])

        for i in range(self.num_lidar_ranges):
            start_idx = i * sector_size
            # Gestione dell'ultimo settore per coprire eventuali resti
            end_idx = start_idx + sector_size if i < self.num_lidar_ranges - 1 else total_ranges
            
            # Extract the sector
            sector = ranges[start_idx:end_idx]
            
            # Find the index related to the min value of the sector
            min_relative_idx = np.argmin(sector)
            
            # Compute the absolute index in the original array
            min_absolute_idx = start_idx + min_relative_idx
            
            # Save the min value
            filtered_ranges[i] = sector[min_relative_idx]
            
            # Compute the exact angle using the absolute index
            # angle = angle_min + index * angle_increment
            filtered_angles[i] = scan.angle_min + min_absolute_idx * scan.angle_increment

        
        return filtered_ranges, filtered_angles
    
    # def lidar_to_obstacles_global_frame(self, filtered_ranges: np.ndarray, 
    #                                     scan: LaserScan) -> np.ndarray:
    #     """
    #     Converts the filtered LiDAR reading in the global frame
        
    #     Input: 
    #         filtered_ranges: Array di range filtrati
    #         scan: LaserScan message originale
            
    #     Output: Array of obstacles in the global frame (num_obstacles, 2)
    #     """
    #     if self.robot_pose is None:
    #         return np.array([])
        
    #     robot_x, robot_y, robot_theta = self.robot_pose
        
    #     # Compute the angles for the filtered ranges
    #     total_ranges = len(scan.ranges)
    #     angle_indices = np.linspace(0, total_ranges - 1, self.num_lidar_ranges, dtype=int)
        
    #     # Angles in the robot local frame
    #     angles_local = scan.angle_min + angle_indices * scan.angle_increment
        
    #     obstacles = []
        
    #     for i, (angle_local, range_val) in enumerate(zip(angles_local, filtered_ranges)):
    #         # Converti da local frame del robot a local frame del robot (coordinate x-y)
    #         x_local = range_val * np.cos(angle_local)
    #         y_local = range_val * np.sin(angle_local)
            
    #         # Convert from the robot local frame to the global one
    #         # Rotation + Translation
    #         cos_theta = np.cos(robot_theta)
    #         sin_theta = np.sin(robot_theta)
            
    #         x_global = robot_x + x_local * cos_theta - y_local * sin_theta
    #         y_global = robot_y + x_local * sin_theta + y_local * cos_theta
            
    #         obstacles.append([x_global, y_global])
        
    #     return np.array(obstacles)

    def lidar_to_obstacles_global_frame(self, filtered_ranges: np.ndarray, 
                                        filtered_angles: np.ndarray) -> np.ndarray:
        """
       Converts the filtered LiDAR readings in the global frame using the exact angles.
        
        Input: 
            filtered_ranges: Array of distances
            filtered_angles: Array of exact corresponding angles (above computed)
        """
        if self.robot_pose is None:
            return np.array([])
        
        robot_x, robot_y, robot_theta = self.robot_pose
        
        obstacles = []
        
        # Now e use the exact angles passed to the function
        for range_val, angle_local in zip(filtered_ranges, filtered_angles):
            
            # Compute local coordinates 
            x_local = range_val * np.cos(angle_local)
            y_local = range_val * np.sin(angle_local)
            
            # Global transformation (Rotation + Translation)
            cos_theta = np.cos(robot_theta)
            sin_theta = np.sin(robot_theta)
            
            x_global = robot_x + x_local * cos_theta - y_local * sin_theta
            y_global = robot_y + x_local * sin_theta + y_local * cos_theta
            
            obstacles.append([x_global, y_global])
        
        return np.array(obstacles)
    
    def check_collision(self, scan: LaserScan) -> bool:
        """
        Check if the robot is too close to an obstacle
        
        Input: LaserScan message
        Output: True if the collision is close
        """
        ranges = np.array(scan.ranges)
        
        # Remove NaN and Inf
        ranges = ranges[~np.isnan(ranges) & ~np.isinf(ranges)]
        
        if len(ranges) == 0:
            return False
        
        min_range = np.min(ranges)
        #collision_threshold = self.robot_radius + self.collision_tol
        collision_threshold = self.collision_tol
        #collision_threshold = 0.01

        # Publish of LiDAR min range
        self.pub_min_lidar_range.publish(Float32(data=float(min_range)))
        
        if min_range < collision_threshold:
            self.get_logger().warn(f'Collisione rilevata! Min range: {min_range:.3f}m')
            self.stop_flag = True
            #return True
        
        return False
    


    # def publish_obstacles_markers(self, obstacles):
    #     """
    #     Publish the obstacles as Marker for Rviz.
    #     Input: obstacles (np.ndarray) [N, 2] global coordinates x,y
    #     """
    #     if len(obstacles) == 0:
    #         return

    #     marker = Marker()
    #     marker.header.frame_id = "odom"  # Frame in which i computed the coordinates
    #     marker.header.stamp = self.get_clock().now().to_msg()
    #     marker.ns = "dwa_obstacles"
    #     marker.id = 0
    #     marker.type = Marker.SPHERE_LIST  # Disegna una lista di sfere
    #     marker.action = Marker.ADD
        
    #     # Sizes of the spheres (0.1 metri = 10 cm)
    #     marker.scale.x = 0.1
    #     marker.scale.y = 0.1
    #     marker.scale.z = 0.1
        
    #     # Color (Rosso brillante, non trasparente)
    #     marker.color.r = 1.0
    #     marker.color.g = 0.0
    #     marker.color.b = 0.0
    #     marker.color.a = 1.0  # Alpha (1 = opaco)

    #     marker.lifetime.sec = 0  # 0 significa "rimani per sempre finché non sovrascritto"
    #     marker.lifetime.nanosec = 200000000 # Oppure breve durata (0.2s) se aggiorni veloce

    #     # Add the points to the list
    #     for obs in obstacles:
    #         p = Point()
    #         p.x = float(obs[0])
    #         p.y = float(obs[1])
    #         p.z = 0.0  # Assumiamo ostacoli a terra
    #         marker.points.append(p)

    #     self.marker_publisher.publish(marker)

    # def publish_active_obstacles(self, obstacles):
    #     """
    #     Publish only the obstacles used by DWA (in BLU/CIANO)
    #     """
    #     if len(obstacles) == 0:
    #         return

    #     marker = Marker()
    #     marker.header.frame_id = "odom"
    #     marker.header.stamp = self.get_clock().now().to_msg()
    #     marker.ns = "dwa_active"
    #     marker.id = 1  # ID diverso
    #     marker.type = Marker.SPHERE_LIST
    #     marker.action = Marker.ADD
        
    #     # Li facciamo leggermente più grandi per vederli "sopra" a quelli rossi
    #     marker.scale.x = 0.15 
    #     marker.scale.y = 0.15
    #     marker.scale.z = 0.15
        
    #     # Colore BLU CIANO (R=0, G=1, B=1)
    #     marker.color.r = 0.0
    #     marker.color.g = 1.0
    #     marker.color.b = 1.0
    #     marker.color.a = 0.8  # Leggermente trasparenti

    #     for obs in obstacles:
    #         p = Point()
    #         p.x = float(obs[0])
    #         p.y = float(obs[1])
    #         p.z = 0.0
    #         marker.points.append(p)

    #     self.active_obs_publisher.publish(marker)

    # def publish_candidate_paths(self, paths):
    #     """
    #     Pubblica tutte le traiettorie valutate come una nuvola di linee grigie.
    #     """
    #     if paths is None or len(paths) == 0:
    #         return

    #     marker = Marker()
    #     marker.header.frame_id = "odom"
    #     marker.header.stamp = self.get_clock().now().to_msg()
    #     marker.ns = "dwa_candidates"
    #     marker.id = 0
    #     marker.type = Marker.LINE_LIST  # Usa LINE_LIST per efficienza
    #     marker.action = Marker.ADD
        
    #     # Linee sottili
    #     marker.scale.x = 0.05 
        
    #     # Colore Grigio semi-trasparente
    #     marker.color.r = 0.5
    #     marker.color.g = 0.5
    #     marker.color.b = 0.5
    #     marker.color.a = 0.8  # Trasparenza per non coprire troppo la mappa

    #     # Aggiungi i punti: Per ogni path, aggiungi segmenti (P0-P1, P1-P2, ...)
    #     # Ottimizzazione vettoriale: prendiamo tutti i punti tranne l'ultimo e tutti tranne il primo
    #     # Ma per chiarezza usiamo un doppio loop semplice (tanto sono poche centinaia)
        
    #     # Iteriamo su ogni singola traiettoria
    #     for path in paths:
    #         # path ha shape (sim_steps, 3) -> (x, y, theta)
    #         for i in range(len(path) - 1):
    #             p_start = Point()
    #             p_start.x = float(path[i, 0])
    #             p_start.y = float(path[i, 1])
                
    #             p_end = Point()
    #             p_end.x = float(path[i+1, 0])
    #             p_end.y = float(path[i+1, 1])
                
    #             marker.points.append(p_start)
    #             marker.points.append(p_end)

    #     self.candidate_paths_pub.publish(marker)

    # def publish_optimal_path(self, path):
    #     """
    #     Evidenzia la traiettoria scelta in VERDE.
    #     """
    #     marker = Marker()
    #     marker.header.frame_id = "odom"
    #     marker.header.stamp = self.get_clock().now().to_msg()
    #     marker.ns = "dwa_optimal"
    #     marker.id = 1
    #     marker.type = Marker.LINE_STRIP
    #     marker.action = Marker.ADD
        
    #     # Linea più spessa
    #     marker.scale.x = 0.05
        
    #     # Colore VERDE acceso
    #     marker.color.r = 0.0
    #     marker.color.g = 1.0
    #     marker.color.b = 0.0
    #     marker.color.a = 1.0

    #     for i in range(len(path)):
    #         p = Point()
    #         p.x = float(path[i, 0])
    #         p.y = float(path[i, 1])
    #         marker.points.append(p)

    #     self.optimal_path_pub.publish(marker)



    def real_goal_callback(self, msg: LandmarkArray):
        """
        Manages the goal for the Task 3 (Real Robot con AprilTag).
        Converts Range/Bearing into global coordinates (Odom frame).
        """
        # 1. Verify if there are rilevated landmarks
        if len(msg.landmarks) == 0:
            #If i loose the tag, I can decide to stop 
            # Se perdi il tag, puoi decidere di fermarti o mantenere l'ultimo goal noto.
            # Per sicurezza, se non vediamo nulla per un po', potremmo voler fermare il robot.
            return
        

        # 2. Pick the first tag seen (or filter by specific ID if there are more tags)
        #tag = msg.landmarks[0]
        tag = None

        target_id = 33  # ID del tag da inseguire

        for landmark in msg.landmarks:
            if landmark.id == target_id:
                tag = landmark
                break
        
        if tag == None:
            return
        # Raw data from msg
        r = tag.range     # Distanza in metri
        beta = tag.bearing   # Angolo in radianti
        
        # 3. Verify that the robot pose is known
        if self.robot_pose is None:
            return

        # Extract the actual robot pose (computed in odom_callback)
        #rx, ry, r_theta = self.robot_pose

        # # 4. Compute the global coordinates (Frame Odom)
        # # The absolute angle pointing to the tag is: OrientamentoRobot + BearingTag
        # global_angle = r_theta + b
        
        # # Formule of polar projection -> global cartesian
        # goal_x = rx + r * np.cos(global_angle)
        # goal_y = ry + r * np.sin(global_angle)

        # # 5. update the goal
        # self.goal_pose = np.array([goal_x, goal_y])
        
        x_r, y_r, theta_r = self.robot_pose
        x_rel = r * np.cos(beta)
        y_rel = r * np.sin(beta)
        cos_th = np.cos(theta_r)
        sin_th = np.sin(theta_r)
        x_g = x_r + x_rel * cos_th - y_rel * sin_th
        y_g = y_r + x_rel * sin_th + y_rel * cos_th
        self.goal_pose=np.array([x_g, y_g])
        # Turn on the navigation if is set OFF
        if not self.navigation_active:
            self.navigation_active = True
            self.step_count = 0
            self.get_logger().info(f'AprilTag rilevato! Inseguimento iniziato: Range={r:.2f}m')



    def control_loop_callback(self):
        """
        Callback of the timer for the DWA control loop
        Executed at 15 Hz
        """
        # Verify if the goal and sensor data are available
        if self.goal_pose is None or self.robot_pose is None or self.lidar_scan is None:
            return
        
        if not self.navigation_active:
            return
        
        # 1. Check for collision
        if self.check_collision(self.lidar_scan) or self.stop_flag == True:
            self.stop_robot()
            self.navigation_active = False
            self.get_logger().error('Task fallito: COLLISIONE')
            return
        
        # 2. Filter LiDAR and convert into the global frame
        #filtered_ranges = self.filter_lidar_scan(self.lidar_scan)
        filtered_ranges, filtered_angles = self.filter_lidar_scan(self.lidar_scan)
        #self.obstacles = self.lidar_to_obstacles_global_frame(filtered_ranges, self.lidar_scan)
        self.obstacles = self.lidar_to_obstacles_global_frame(filtered_ranges, filtered_angles)

        # Publish obstacles as Marker per Rviz
        #self.publish_obstacles_markers(self.obstacles)
        
        
        
        # 3. Compute distance to the goal
        dist_to_goal = np.linalg.norm(self.robot_pose[0:2] - self.goal_pose)
        
        # 4. Verify if the goal is reached
        if dist_to_goal < self.goal_dist_tol:  # goal_dist_tol
            self.stop_robot()
            self.navigation_active = False
            self.get_logger().info('Task riuscito: GOAL RAGGIUNTO')
            return
        
        # 5. Verify timeout
        if self.step_count > 300 or self.timeout_flag == True:
            self.stop_robot()
            self.timeout_flag = True
            self.navigation_active = False
            self.get_logger().warn('Task fallito: TIMEOUT')
            return
        
        # 6. Compute DWA command
        try:
            u, costs, active_obs, all_paths, opt_idx = self.dwa.compute_cmd(self.goal_pose, self.robot_pose, self.obstacles)

            # === COSTS PUBBLICATION=== 
            self.pub_cost_heading.publish(Float32(data=float(costs['heading'])))
            self.pub_cost_vel.publish(Float32(data=float(costs['vel'])))
            self.pub_cost_obs.publish(Float32(data=float(costs['obs'])))
            self.pub_cost_total.publish(Float32(data=float(costs['total'])))

            #=== ACTIVE OBSTACLES USED BY DWA PUBBLICATION ===
            #self.publish_active_obstacles(active_obs)

            #=== CANDIDATE TRAJECTORIES PUBBLICATION  ===
            #self.publish_candidate_paths(all_paths)
            #=== OPTIMAL TRAJECTORY PUBBLICATION ===
            #self.publish_optimal_path(all_paths[opt_idx])
        
        except Exception as e:
            self.get_logger().error(f'Errore nel calcolo DWA: {str(e)}')
            self.stop_robot()
            return
        
        # 7. CMD VEL PUBBLICATION
        cmd_msg = Twist()
        cmd_msg.linear.x = u[0]
        cmd_msg.angular.z = u[1]
        self.cmd_vel_publisher.publish(cmd_msg)
        
        # Update vel in DWA
        self.dwa.robot.vel = self.robot_vel
        
        # 8. Periodic feedback
        if self.step_count % self.feedback_rate == 0:
            
            self.get_logger().info(
                f'Step {self.step_count} | '
                f'Dist: {dist_to_goal:.3f}m | '
                #f'Cmd: v={u:.3f}, w={u:.3f} | '
                f'Ostacoli: {len(self.obstacles)}'
            )

            dist_msg = Float32()
            dist_msg.data = float(dist_to_goal)
            self.pub_dist_to_goal.publish(dist_msg)
            
        
        self.step_count += 1
    
    def stop_robot(self):
        """Stop the robot"""
        cmd_msg = Twist()
        cmd_msg.linear.x = 0.0
        cmd_msg.angular.z = 0.0
        self.cmd_vel_publisher.publish(cmd_msg)
        self.get_logger().info('Robot fermato')





def main(args=None):
    rclpy.init(args=args)
    node = DWANavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
