#!/usr/bin/env python3
"""
EKF Localization Node - TASK 2: Extended State with Sensor Fusion
Stato: [x, y, theta, v, omega] (5D)
Sensori: Odometry + IMU + Landmark
"""

import rclpy
from rclpy.node import Node
import numpy as np
import sympy
from sympy import symbols, Matrix

# Import RobotEKF
from .ekf import RobotEKF

# Import message types
from nav_msgs.msg import Odometry
from landmark_msgs.msg import LandmarkArray
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist
from .utils import normalize_angle, residual
OMEGA_THRESHOLD = 1e-6

class EKFLocalizationNodeReal(Node):
    def __init__(self):
        super().__init__('ekf_localization_node_real')

        self.get_logger().info("="*60)
        self.get_logger().info("TASK 2: EKF with Extended State (5D) + Sensor Fusion")
        self.get_logger().info("="*60)

        # ================================================================
        # STEP 1: MOTION MODEL SYMBOLS (5D)
        # ================================================================
        x, y, theta, v, w, dt = symbols('x y theta v w dt')
        R = v / w
        beta = theta + w * dt

        # Velocity motion model (5D: predict v and omega as well)
        gux = Matrix([
            x - R * sympy.sin(theta) + R * sympy.sin(beta),  # x'
            y + R * sympy.cos(theta) - R * sympy.cos(beta),  # y'
            theta + w * dt,                                    # theta'
            v,                                                 # v' (constant)
            w                                                  # w' (constant)
        ])

        # Jacobians (5x5 and 5x2)
        Gt = gux.jacobian(Matrix([x, y, theta, v, w]))
        Vt = gux.jacobian(Matrix([v, w]))

        # Convert to numerical functions
        self.eval_gux_sym = sympy.lambdify((x, y, theta, v, w, dt), gux, 'numpy')
        self.eval_Gt_sym = sympy.lambdify((x, y, theta, v, w, dt), Gt, 'numpy')
        self.eval_Vt_sym = sympy.lambdify((x, y, theta, v, w, dt), Vt, 'numpy')

        # Motion model for straight motion (omega ~ 0)
        gux_straight = Matrix([
            x + v * sympy.cos(theta) * dt,
            y + v * sympy.sin(theta) * dt,
            theta,
            v,
            w
        ])

        Gt_straight = gux_straight.jacobian(Matrix([x, y, theta, v, w]))
        Vt_straight = gux_straight.jacobian(Matrix([v, w]))

        self.eval_gux_straight = sympy.lambdify((x, y, theta, v, w, dt), gux_straight, 'numpy')
        self.eval_Gt_straight = sympy.lambdify((x, y, theta, v, w, dt), Gt_straight, 'numpy')
        self.eval_Vt_straight = sympy.lambdify((x, y, theta, v, w, dt), Vt_straight, 'numpy')

        # ================================================================
        # Wrapper for Motion Model
        # ================================================================
        def eval_gux_wrapper(mu, u, sigma_u, dt):
            """
            Wrapper for g(u,x) - 5D state
            Receive: mu=[x,y,θ,v,ω], u=[v_cmd,ω_cmd], sigma_u, dt
            """
            v_cmd, w_cmd = u

            if abs(w_cmd) < OMEGA_THRESHOLD:
                result = self.eval_gux_straight(mu[0], mu[1], mu[2], mu[3], mu[4], dt)
            else:
                result = self.eval_gux_sym(mu[0], mu[1], mu[2], mu[3], mu[4], dt)

            return np.array(result).flatten()

        def eval_Gt_wrapper(x_val, y_val, theta_val, v_state, w_state, v_cmd, w_cmd, dt_val):
            """
            Wrapper for G_t - 5D state
            Receive 8 arguments: (x, y, θ, v_state, ω_state, v_cmd, ω_cmd, dt)
            Use: (x, y, θ, v_cmd, ω_cmd, dt) to compute Gt
            """
            if abs(w_cmd) < OMEGA_THRESHOLD:
                return np.array(self.eval_Gt_straight(x_val, y_val, theta_val, v_cmd, w_cmd, dt_val))
            else:
                return np.array(self.eval_Gt_sym(x_val, y_val, theta_val, v_cmd, w_cmd, dt_val))

        def eval_Vt_wrapper(x_val, y_val, theta_val, v_state, w_state, v_cmd, w_cmd, dt_val):
            """
            Wrapper for V_t - 5D state
            Receive 8 arguments, use (x, y, θ, v_cmd, ω_cmd, dt)
            """
            if abs(w_cmd) < OMEGA_THRESHOLD:
                return np.array(self.eval_Vt_straight(x_val, y_val, theta_val, v_cmd, w_cmd, dt_val))
            else:
                return np.array(self.eval_Vt_sym(x_val, y_val, theta_val, v_cmd, w_cmd, dt_val))

        self.eval_gux = eval_gux_wrapper
        self.eval_Gt = eval_Gt_wrapper
        self.eval_Vt = eval_Vt_wrapper


        # ================================================================
        # STEP 2: MEASUREMENT MODEL - LANDMARK (updated for 5D)
        # ================================================================
        mx, my, mz = symbols('mx my mz')  # Landmark position (3D, mz=0 in our case)

        hx_landmark = Matrix([
            sympy.sqrt((mx - x)**2 + (my - y)**2 + (mz)**2), #z1 equals to 0 in our case
            sympy.atan2((my - y), (mx - x)) - theta
        ])

        # Jacobian with respect to the 5D state
        Ht_landmark = hx_landmark.jacobian(Matrix([x, y, theta, v, w]))

        self.eval_hx_landmark_sym = sympy.lambdify((x, y, theta, v, w, mx, my, mz), hx_landmark, 'numpy') 
        self.eval_Ht_landmark_sym = sympy.lambdify((x, y, theta, v, w, mx, my, mz), Ht_landmark, 'numpy')

        def eval_hx_landmark_wrapper(x_val, y_val, theta_val, v_val, w_val, mx_val, my_val, mz_val):
            result = self.eval_hx_landmark_sym(x_val, y_val, theta_val, v_val, w_val, mx_val, my_val, mz_val)
            return np.array(result).flatten()

        def eval_Ht_landmark_wrapper(x_val, y_val, theta_val, v_val, w_val, mx_val, my_val, mz_val):
            return np.array(self.eval_Ht_landmark_sym(x_val, y_val, theta_val, v_val, w_val, mx_val, my_val, mz_val))

        self.eval_hx_landmark = eval_hx_landmark_wrapper
        self.eval_Ht_landmark = eval_Ht_landmark_wrapper


        # ================================================================
        # STEP 3: MEASUREMENT MODEL - ODOMETRY (new for Task 2)
        # ================================================================
        # measure linear and angular velocity from odometry
        h_odom = Matrix([v, w])
        Ht_odom = h_odom.jacobian(Matrix([x, y, theta, v, w]))

        self.eval_h_odom_sym = sympy.lambdify((x, y, theta, v, w), h_odom, 'numpy')
        self.eval_Ht_odom_sym = sympy.lambdify((x, y, theta, v, w), Ht_odom, 'numpy')

        def eval_h_odom_wrapper(x_val, y_val, theta_val, v_val, w_val):
            result = self.eval_h_odom_sym(x_val, y_val, theta_val, v_val, w_val)
            return np.array(result).flatten()

        def eval_Ht_odom_wrapper(x_val, y_val, theta_val, v_val, w_val):
            return np.array(self.eval_Ht_odom_sym(x_val, y_val, theta_val, v_val, w_val))

        self.eval_h_odom = eval_h_odom_wrapper
        self.eval_Ht_odom = eval_Ht_odom_wrapper

        # ================================================================
        # STEP 4: MEASUREMENT MODEL - IMU (new for Task 2)
        # ================================================================
        # measure angular velocity from IMU
        h_imu = Matrix([w])
        Ht_imu = h_imu.jacobian(Matrix([x, y, theta, v, w]))

        self.eval_h_imu_sym = sympy.lambdify((x, y, theta, v, w), h_imu, 'numpy')
        self.eval_Ht_imu_sym = sympy.lambdify((x, y, theta, v, w), Ht_imu, 'numpy')

        def eval_h_imu_wrapper(x_val, y_val, theta_val, v_val, w_val):
            result = self.eval_h_imu_sym(x_val, y_val, theta_val, v_val, w_val)
            return np.array(result).flatten()

        def eval_Ht_imu_wrapper(x_val, y_val, theta_val, v_val, w_val):
            return np.array(self.eval_Ht_imu_sym(x_val, y_val, theta_val, v_val, w_val))

        self.eval_h_imu = eval_h_imu_wrapper
        self.eval_Ht_imu = eval_Ht_imu_wrapper

        # ================================================================
        # STEP 5: CREATE EKF INSTANCE (5D)
        # ================================================================
        self.ekf = RobotEKF(
            dim_x=5,  # ← State 5D: [x, y, theta, v, omega]
            dim_u=2,  # ← Command 2D: [v, omega]
            eval_gux=self.eval_gux,
            eval_Gt=self.eval_Gt,
            eval_Vt=self.eval_Vt,
        )

        # Initial state (5D)
        #self.ekf.mu = np.array([-2.0, -0.5, 0.0, 0.0, 0.0])         # simulation pose
        self.ekf.mu = np.array([0.0, 0.77, 0.0, 0.0, 0.0])         # real pose (maybe)
        self.ekf.Sigma = np.diag([0.1, 0.1, np.deg2rad(1), 0.1, 0.1])

        # Process noise (2x2, only for v and omega in control)
        alpha = np.array([0.1, 0.05, 0.1, 0.05])
        self.ekf.Mt = np.diag([alpha[0], alpha[2]])
        self.alpha = alpha

        # Measurement noise   (the higher the value, the less the sensor is trusted)
        self.Qt_landmark = np.diag([0.1**2, np.deg2rad(1)**2])    # range, bearing
        self.Qt_odom = np.diag([0.1**2, np.deg2rad(10)])        # v, omega (odom)
        self.Qt_imu = np.diag([np.deg2rad(5)**2])                 # omega (IMU)

        #Load landmark map
        self.landmarks = {                       # real landmarks
            0: np.array([1.20, 1.68, 0.16]),
            1: np.array([1.68, -0.05, 0.18]),
            2: np.array([3.72, 0.14, 0.22]),
            3: np.array([3.75, 1.37, 0.21]),
            4: np.array([2.48, 1.25, 0.22]),
            5: np.array([4.80, 1.87, 0.24]),
            6: np.array([2.18, 1.00, 0.24]),
            7: np.array([2.94, 2.70, 0.73])
        }


        # self.landmarks = {                          # gazebo landmarks
        #     11: np.array([-1.1, -1.1, 0.5]),
        #     12: np.array([-1.1,  0.0, 1.5]),
        #     13: np.array([-1.1,  1.1, 0.5]),
        #     21: np.array([ 0.0, -1.1, 1.0]),
        #     22: np.array([ 0.0,  0.0, 0.75]),
        #     23: np.array([ 0.0,  1.1, 0.3]),
        #     31: np.array([ 1.1, -1.1, 1.5]),
        #     32: np.array([ 1.1,  0.0, 1.0]),
        #     33: np.array([ 1.1,  1.1, 0.0])
        # }

        # Variables for prediction
        self.last_cmd_vel = np.array([0.0, 0.0])
        self.dt = 0.05  # 50ms → 20Hz
        self._pred_count = 0

        # ================================================================
        # STEP 6: ROS2 SUBSCRIBERS AND PUBLISHERS
        # ================================================================
        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)

        self.imu_sub = self.create_subscription(
            Imu, '/imu', self.imu_callback, 10)

        self.landmark_sub = self.create_subscription(
            LandmarkArray, 
            '/camera/landmarks',                       # for real simulation
            #'/landmarks', 
            self.landmark_callback, 10)

        # Publisher
        self.ekf_pub = self.create_publisher(Odometry, '/ekf', 10)

        # Timer per prediction
        self.timer = self.create_timer(self.dt, self.prediction_callback)

        self.get_logger().info("EKF Node (Task 2) initialized!")
        # self.get_logger().info(f"State dimension: {self.ekf.dim_x}D")
        self.get_logger().info("="*60)

    # ================================================================
    # PREDICTION CALLBACK
    # ================================================================
    def prediction_callback(self):
        """Prediction step at 20 Hz"""
        u = self.last_cmd_vel
        sigma_u = np.array([0.1, 0.1])

        v, omega = u
        a1, a2, a3, a4 = self.alpha
        self.ekf.Mt = np.diag([
            a1 * v**2 + a2 * omega**2,
            a3 * v**2 + a4 * omega**2
        ])

        # Prediction
        self.ekf.predict(u, sigma_u, g_extra_args=(self.dt,))

        # normalize theta
        self.ekf.mu[2] = normalize_angle(self.ekf.mu[2])

        # log every 20 predictions (every 1 second)
        self._pred_count += 1
        if self._pred_count % 20 == 0:
            x, y, theta, v_est, w_est = self.ekf.mu
            cov_trace = np.trace(self.ekf.Sigma)
            self.get_logger().info(
                f"[PRED] Pos: ({x:.3f}, {y:.3f}, {np.degrees(theta):.1f}°) | "
                f"Vel: v={v_est:.2f}m/s, ω={np.degrees(w_est):.1f}°/s | "
                f"Σ_trace: {cov_trace:.4f}"
            )

        self.publish_ekf_state()

    # ================================================================
    # ODOMETRY CALLBACK (Update with velocity)
    # ================================================================
    def odom_callback(self, msg):
        """
        1. Save velocity for prediction (as in Task 1)
        2. UPDATE with velocity measurement (new in Task 2)
        """
        v_meas = msg.twist.twist.linear.x
        w_meas = msg.twist.twist.angular.z

        # Save for prediction
        self.last_cmd_vel = np.array([v_meas, w_meas])

        # UPDATE with h_odom
        z_odom = np.array([v_meas, w_meas])

        self.ekf.update(
            z=z_odom,
            eval_hx=self.eval_h_odom,
            eval_Ht=self.eval_Ht_odom,
            Qt=self.Qt_odom,
            Ht_args=(*self.ekf.mu,),
            hx_args=(*self.ekf.mu,),
            residual=lambda z, z_hat, **kw: z - z_hat, # ← Lambda!
            angle_idx=None
        )

    # ================================================================
    # IMU CALLBACK (Update with angular velocity)
    # ================================================================
    def imu_callback(self, msg):
        """UPDATE with angular velocity measurement from IMU"""
        w_meas_imu = msg.angular_velocity.z

        z_imu = np.array([w_meas_imu])

        self.ekf.update(
            z=z_imu,
            eval_hx=self.eval_h_imu,
            eval_Ht=self.eval_Ht_imu,
            Qt=self.Qt_imu,
            Ht_args=(*self.ekf.mu,),
            hx_args=(*self.ekf.mu,),
            residual=lambda z, z_hat, **kw: z - z_hat,  # ← Lambda!
            angle_idx=None
        )

    # ================================================================
    # LANDMARK CALLBACK (Update with landmark)
    # ================================================================
    def landmark_callback(self, msg):
        """UPDATE with landmark (as in Task 1, but state is 5D)"""
        num_landmarks = len(msg.landmarks)
        if num_landmarks == 0:
            return

        self.get_logger().info(
            f"[UPDATE] Received {num_landmarks} landmark(s)",
            throttle_duration_sec=1.0
        )

        for lm in msg.landmarks:
            if lm.id not in self.landmarks:
                continue

            m = self.landmarks[lm.id]
            z = np.array([lm.range, lm.bearing])

            mu_before = self.ekf.mu.copy()

            self.ekf.update(
                z=z,
                eval_hx=self.eval_hx_landmark,
                eval_Ht=self.eval_Ht_landmark,
                Qt=self.Qt_landmark,
                Ht_args=(*self.ekf.mu, *m),
                hx_args=(*self.ekf.mu, *m),
                residual=residual,
                angle_idx=1
            )

            delta = self.ekf.mu - mu_before
            self.get_logger().info(
                f"  → Landmark ID={lm.id}: "
                f"range={z[0]:.3f}m, bearing={np.degrees(z[1]):.1f}° | "
                f"Correction: Δx={delta[0]:.4f}, Δy={delta[1]:.4f}, "
                f"Δθ={np.degrees(delta[2]):.2f}°"
            )

        self.publish_ekf_state()

    # ================================================================
    # PUBLISH EKF STATE
    # ================================================================
    def publish_ekf_state(self):
        """Publish EKF state as Odometry message"""
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_link'

        # Position
        msg.pose.pose.position.x = float(self.ekf.mu[0])
        msg.pose.pose.position.y = float(self.ekf.mu[1])
        msg.pose.pose.position.z = 0.0

        # Orientation
        theta = float(self.ekf.mu[2])
        quat = self.quaternion_from_euler(0, 0, theta)
        msg.pose.pose.orientation.x = quat[0]
        msg.pose.pose.orientation.y = quat[1]
        msg.pose.pose.orientation.z = quat[2]
        msg.pose.pose.orientation.w = quat[3]

        # Covariance (only x, y, theta)
        covariance_matrix = np.zeros(36)
        covariance_matrix[0] = self.ekf.Sigma[0, 0]
        covariance_matrix[7] = self.ekf.Sigma[1, 1]
        covariance_matrix[35] = self.ekf.Sigma[2, 2]
        msg.pose.covariance = covariance_matrix.tolist()

        # Velocity
        msg.twist.twist.linear.x = float(self.ekf.mu[3])
        msg.twist.twist.angular.z = float(self.ekf.mu[4])

        self.ekf_pub.publish(msg)

    def quaternion_from_euler(self, roll, pitch, yaw):
        """Euler → Quaternion"""
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return [x, y, z, w]


def main(args=None):
    rclpy.init(args=args)
    try:
        node = EKFLocalizationNodeReal()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()