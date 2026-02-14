import numpy as np
from .utils import Differential_drive_robot, normalize_angle, normalize, calc_nearest_obs

# DWA
class DWA():
    def __init__(self,
                 dt = 0.1,                  # time interval for simulation
                 sim_time = 2.0,            # total simulation time, it defines how far in the future to simulate
                 time_granularity = 0.1,    # time step for trajectory simulation, it defines the resolution of the simulation of each trajectory
                 v_samples = 10,            # number of linear velocity samples 
                 w_samples = 20,            # number of angular velocity samples (in this case it means that i'll try 200 different velocities values (v*w))
                 goal_dist_tol = 0.3,       # tolerance to consider the goal reached
                 collision_tol = 0.3,       # tolerance to consider a collision with an obstacle
                 weight_angle = 0.04,
                 weight_vel = 0.2,
                 weight_obs = 0.1,
                 weight_vel_decay = 0.1,      
                 weight_target = 0.05,        
                 slowdown_distance = 1.0,     
                 target_distance_desired = 0.8, 
                 obstacles_map = None,      # A list of obstacles poses in the map # TO DO: accept a gridmap and use local map
                 **kwargs
                 ):

        self.dt = dt
        self.sim_step = round(sim_time / time_granularity)      # it's the number of steps in the future to simulate for each trajectory (look ahead horizon) (in this case it's 20 steps)
        self.robot = Differential_drive_robot(**kwargs)         # i'm using this class to uodate the robot state (in simulation and real application i'll probably use the odometry and not this)
        self.obstacles = obstacles_map 
        self.goal_dist_tol = goal_dist_tol
        self.collision_tol = collision_tol

        self.v_samples = v_samples
        self.w_samples = w_samples

        # define weight
        self.weight_angle = weight_angle
        self.weight_vel = weight_vel
        self.weight_obs = weight_obs
        self.weight_vel_decay = weight_vel_decay
        self.weight_target = weight_target
        self.slowdown_distance = slowdown_distance
        self.target_distance_desired = target_distance_desired

        self.obstacle_max_dist = 3 # like a local costmap size
        self.max_num_steps = 300
        self.feedback_rate = 50
        self.obst_tolerance = 0.5

    
    
    #######################################
    ######## CORE OF THE PROGRAM ##########
    #######################################

    # def go_to_pose(self, goal_pose):        # in my ROS implementation this will be replaced by an action request (long term task)
    #     """
    #     Assign a target goal to the DWA controller.
    #     Compute commands until goal is reached.
    #     Provide intermediate feedback on the navigation task.
    #     """
    #     if goal_pose is list:                           
    #         goal_pose = np.array(goal_pose)

    #     success = False
    #     dist_to_goal = np.linalg.norm(self.robot.pose[0:2] - goal_pose)

    #     print("Initial distance to goal: ", dist_to_goal)
    #     print("Initial Robot pose: ", self.robot.pose)

    #     steps = 1
    #     while steps <= self.max_num_steps:                                  # in ROS this infinite loop can't be used, it can be replaced by a timer that works at 15 hz
    #         # 1. Check if Goal reached
    #         dist_to_goal = np.linalg.norm(self.robot.pose[0:2] - goal_pose) # np.linalg.norm calculate euclidean distance ||v||
    #         if dist_to_goal < self.goal_dist_tol:
    #             success = True
    #             print("Goal reached!")
    #             break

    #         # 2. Get new observations for obstacles is needed
    #         # no lidar used here
    #         # TO DO: update self.obstacles with new obstacle observations

    #         # 3. Compute command for the robot with DWA controller
    #         u = self.compute_cmd(goal_pose, self.robot.pose, self.obstacles)

    #         # 4. Send the command or update the robot pose
    #         pose = self.robot.update_state(u, self.dt)          # in gazebo/real simulation this will be replaced by odometry update subscriber
            
    #         # 5. Provide intermediate task feedback
    #         if steps % self.feedback_rate == 0:
    #             dist_to_goal = np.linalg.norm(self.robot.pose[0:2] - goal_pose)
    #             print("Current distance to goal ", dist_to_goal, " at step ", steps)
    #             print("Current Robot pose: ", pose)

    #         steps += 1

    #     if steps > self.max_num_steps:
    #         print("Timeout! Goal not reached.")

    #     return success, self.robot.trajectory
    




    ########################################
    ##### Actual implementation of DWA #####
    ########################################

    
    def compute_cmd(self, goal_pose, robot_state, obstacles):
        """
        Compute the next velocity command u=(v,w) according to the DWA algorithm.
        The velocity leading to the highest scored trajectory is selected.
        """
        # create path
        paths, velocities = self.get_trajectories(robot_state)

        # evaluate path
        opt_idx, best_costs, active_obs = self.evaluate_paths(paths, velocities, goal_pose, robot_state, obstacles)     # select the specific velocity that lead to the optimal path
        u = velocities[opt_idx]
        return u, best_costs, active_obs, paths, opt_idx
       
    def get_trajectories(self, robot_pose): # select different velocities and simulate the trajectories
        """
        Get all the reachable velocities inside the dynamic window
        Simulate all trajectory with velocity values u=(v,w)
        Return: 
        - simulated paths : np.ndarray,  shape: (n_paths, sim_step, state_dim)
        - velocities : np.ndarray, shape (n_paths, 2)
        """
        # calculate reachable range of velocity and angular velocity in the dynamic window
        min_lin_vel, max_lin_vel, min_ang_vel, max_ang_vel = self.compute_dynamic_window(self.robot.vel)
        
        v_values = np.linspace(min_lin_vel, max_lin_vel, self.v_samples)        # np.linspace generates an array of evenly spaced values between min and max
        w_values = np.linspace(min_ang_vel, max_ang_vel, self.w_samples)

        # list of all paths and velocities
        n_paths = w_values.shape[0]*v_values.shape[0]                           # calculates the total number of paths/trajectories to simulate
        sim_paths = np.zeros((n_paths, self.sim_step, robot_pose.shape[0]))     # 3D array to store all simulated paths
        velocities = np.zeros((n_paths, 2))

        # evaluate all velocities and angular velocities combinations    
        vv, ww = np.meshgrid(v_values, w_values)
        velocities = np.dstack([vv,ww]).reshape(n_paths, 2)
        sim_paths = self.simulate_paths(n_paths, robot_pose, velocities)

        return sim_paths, velocities
    
    def simulate_paths(self, n_paths, pose, u):
        """
        Simulate trajectory at constant velocity u=(v,w)
        """
        sim_paths = np.zeros((n_paths, self.sim_step, pose.shape[0]))   # [paths × steps × (x,y,θ)]
        sim_paths[:, 0] = pose.copy()                                   # All paths start from current robot pose

        for i in range(1, self.sim_step):                               # Iteratively propagate each path using differential drive kinematics, for each time step
                                                                        # for each cycle, update each path position based on previous position and velocity command for 20 time in the future
            sim_paths[:, i, 0] = sim_paths[:, i - 1, 0] + u[:, 0] * np.cos(sim_paths[:, i - 1, 2]) * self.dt    # x = x + v*cos(θ)*dt
            sim_paths[:, i, 1] = sim_paths[:, i - 1, 1] + u[:, 0] * np.sin(sim_paths[:, i - 1, 2]) * self.dt    # y = y + v*sin(θ)*dt
            sim_paths[:, i, 2] = sim_paths[:, i - 1, 2] + u[:, 1] * self.dt                                     # θ = θ + w*dt

        return sim_paths

    def compute_dynamic_window(self, robot_vel): # calculates the max and min linear and angular velocities
        """
        Calculate the dynamic window composed of reachable linear velocity and angular velocity according to robot's kinematic limits.
        """
        # linear velocity
        min_vel = robot_vel[0] - self.dt * self.robot.max_linear_acc
        max_vel = robot_vel[0] + self.dt * self.robot.max_linear_acc
        # minimum
        if min_vel < self.robot.min_lin_vel:
            min_vel = self.robot.min_lin_vel
        # maximum
        if max_vel > self.robot.max_lin_vel:
            max_vel = self.robot.max_lin_vel

        # angular velocity
        min_ang_vel = robot_vel[1] - self.dt * self.robot.max_ang_acc
        max_ang_vel = robot_vel[1] + self.dt * self.robot.max_ang_acc
        # minimum
        if min_ang_vel < self.robot.min_ang_vel:
            min_ang_vel = self.robot.min_ang_vel
        # maximum
        if max_ang_vel > self.robot.max_ang_vel:
            max_ang_vel = self.robot.max_ang_vel

        return min_vel, max_vel, min_ang_vel, max_ang_vel


    def evaluate_paths(self, paths, velocities, goal_pose, robot_pose, obstacles):
        """
        Evaluate the simulated paths using the objective function.
        J = w_h * heading + w_v * vel + w_o * obst_dist
        """
        # detect nearest obstacle
        nearest_obs = calc_nearest_obs(robot_pose, obstacles)       # for each trajectory i find the nearest obstacles to consider for the scoring

        # Compute the scores for the generated path
        # (1) heading_angle and goal distance
        score_heading_angles = self.score_heading_angle(paths, goal_pose)   # if the angle is small the score is high (i'm facing the goal)
        # (2) velocity
        # funzione per il task 1
        #score_vel = self.score_vel(velocities, paths, goal_pose) # in the code the score is vel + exp(-dist_to_goal/tol), higher the velocity higher the score
        # funzione per il task 2
        score_vel = self.score_vel_decay(velocities, paths, goal_pose)
        # (3) obstacles
        score_obstacles = self.score_obstacles(paths, nearest_obs)

        #(4) target distance (task 2)
        score_target = self.score_target_distance(paths, goal_pose)

        # Scores Normalization (the different scores will have a value between 0 and 1)
        # This is a min-max normalization, if all values are the same the score will be 0 for all trajectories, if they are different they will be scaled between 0 (for the lowest) and 1 (for the highest)
        score_heading_angles = normalize(score_heading_angles)
        score_vel = normalize(score_vel)
        score_obstacles = normalize(score_obstacles)
        score_target = normalize(score_target)

        # norm_heading = normalize(score_heading_angles)
        # norm_vel = normalize(score_vel)
        # norm_obs = normalize(score_obstacles)

        

        # Compute the idx of the optimal path according to the overall score
        # opt_idx = np.argmax(np.sum(
        #     np.array([score_heading_angles, score_vel, score_obstacles])
        #     * np.array([[self.weight_angle, self.weight_vel, self.weight_obs]]).T,
        #     axis=0,
        #))
        total_score = (score_heading_angles * self.weight_angle +   # total score for each trajectory weighted sum of the 3 scores
                       score_vel * self.weight_vel_decay +
                       score_obstacles * self.weight_obs + score_target * self.weight_target
        )
        opt_idx = np.argmax(total_score)

        best_costs = {
            'heading': score_heading_angles[opt_idx] * self.weight_angle,
            'vel': score_vel[opt_idx] * self.weight_vel,
            'obs': score_obstacles[opt_idx] * self.weight_obs,
            'total': total_score[opt_idx]
        }



        try:
            return opt_idx, best_costs, nearest_obs
        except:
            raise Exception("Not possible to find an optimal path")

    def score_heading_angle(self, path, goal_pose):
        """
        Go towards the target objective: score trajectory according to the heading angle to the goal
        """
        last_x = path[:, -1, 0]
        last_y = path[:, -1, 1]
        last_th = path[:, -1, 2]

        # calculate angle
        angle_to_goal = np.arctan2(goal_pose[1] - last_y, goal_pose[0] - last_x)

        # calculate score
        score_angle = angle_to_goal - last_th
        score_angle = np.fabs(normalize_angle(score_angle))
        score_angle = np.pi - score_angle

        return score_angle

    def score_vel(self, u, path, goal_pose):
        """
        Maximum velocity objective: score trajectory according to the forward velocity. When the robot is near the goal, slow down.
        """

        vel = u[:,0] # linear velocity value is the score itself (to maximize)
        dist_to_goal = np.linalg.norm(path[:, -1, 0:2] - goal_pose, axis=-1)
        score = vel + np.exp(-dist_to_goal / self.goal_dist_tol)
        return score



    def score_vel_decay(self, u, path, goal_pose):
        """
        Variant for Task 2: high velocity far away from the goal,
        small one close to the goal.
        """
        vel = u[:, 0]
        dist_to_goal = np.linalg.norm(path[:, -1, 0:2] - goal_pose, axis=-1)

        # increasing penalty when dist_to_goal < slowdown_distance
        decay_term = np.exp(-dist_to_goal/ (self.slowdown_distance))
        score = vel - decay_term
        return score



    ### IBRID VERSION###

    # def score_vel_decay(self, u, path, goal_pose):
    #     """
    #     FUNZIONE IBRIDA (La migliore per il Task 2)
    #     Combina l'attrazione verso il goal (il tuo snippet) 
    #     con la penalità di velocità (il requisito).
    #     """
    #     # 1. Dati
    #     vel = u[:, 0]
    #     dist_to_goal = np.linalg.norm(path[:, -1, 0:2] - goal_pose, axis=-1)
    #     threshold = self.slowdown_distance
        
    #     # 2. CALCOLO DEL TERMINE VELOCITÀ "TRUCCATO"
    #     # Partiamo dalla velocità reale
    #     vel_score = vel.copy()
        
    #     # Identifichiamo chi è nella zona di rallentamento
    #     mask_close = dist_to_goal < threshold
        
    #     if np.any(mask_close):
    #         # Calcoliamo fattore penalità (0=bordo, 1=goal)
    #         closeness = (threshold - dist_to_goal[mask_close]) / threshold
            
    #         # Applichiamo il "Decreasing Scoring Term" richiesto dal report.
    #         # Riduciamo il valore della velocità percepita.
    #         # Se closeness è 1 (sul goal), la velocità viene penalizzata tantissimo.
    #         # Moltiplico per 2.0 per essere sicuro che diventi negativa se corre troppo.
    #         penalty = vel[mask_close] * closeness * 2.0
            
    #         vel_score[mask_close] -= penalty

    #     # 3. CALCOLO DEL BONUS VICINANZA (Dal tuo snippet)
    #     # Questo assicura che il robot voglia avvicinarsi al goal
    #     proximity_bonus = np.exp(-dist_to_goal / self.goal_dist_tol)
        
    #     # 4. SCORE TOTALE
    #     # Sommiamo la velocità (penalizzata se vicino) + il bonus di essere vicino
    #     return vel_score + proximity_bonus


    def score_obstacles(self, path, obstacles):
        """
        Obstacle avoidance objective: score trajectory according to the distance to the nearest obstacle.
        """
        score_obstacle = 2.0*np.ones((path.shape[0]))

        for obs in obstacles:
            dx = path[:, :, 0] - obs[0]
            dy = path[:, :, 1] - obs[1]
            dist = np.hypot(dx, dy)

            min_dist = np.min(dist, axis=-1)
            score_obstacle[min_dist < score_obstacle] = min_dist[min_dist < score_obstacle]
        
            # collision with obstacle
            score_obstacle[score_obstacle < self.robot.radius + self.collision_tol] = -100 # heavy penalty for collision
               
        return score_obstacle

    
    def score_target_distance(self, path, target_pose):
        """
        Term for Task 2: it prefers trajectories which held the target to the desired distance
        """
        final_pos = path[:, -1, 0:2]
        dist_to_target = np.linalg.norm(final_pos - target_pose, axis=-1)
        error = np.abs(dist_to_target - self.target_distance_desired)

        # max score when error ≈ 0, it decreases when robot is moving away
        score = np.exp(-error**2 / (2 * 0.2**2))
        return score





