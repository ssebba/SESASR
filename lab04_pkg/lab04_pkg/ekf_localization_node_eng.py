import rclpy
from rclpy.node import Node
import numpy as np
import sympy
from sympy import symbols, Matrix
from .probabilistic_models import (
    sample_velocity_motion_model,
    landmark_range_bearing_model,
    velocity_mm_simpy,                  #     This creates eval_gux, eval_Gt, eval_Vt
    landmark_sm_simpy                   #    This creates eval_hx, eval_Ht
)


# Import RobotEKF
from .ekf import RobotEKF

# Import message types
from nav_msgs.msg import Odometry
from landmark_msgs.msg import LandmarkArray
from geometry_msgs.msg import Twist
from .utils import normalize_angle, residual
OMEGA_THRESHOLD = 1e-6 
class EKFLocalizationNode(Node):
    def __init__(self):
        super().__init__('ekf_localization_node')

        x, y, theta, v, w, dt = symbols('x y theta v w dt')
        R = v / w
        beta = w * dt + theta

        # Velocity motion model
        gux = Matrix([
            x - R * sympy.sin(theta) + R * sympy.sin(beta),
            y + R * sympy.cos(theta) - R * sympy.cos(beta),
            theta + w * dt
        ])

        # Jacobians
        Gt = gux.jacobian(Matrix([x, y, theta]))
        Vt = gux.jacobian(Matrix([v, w]))

        # Convert to numerical functions
        self.eval_gux_sym = sympy.lambdify((x, y, theta, v, w, dt), gux, 'numpy')
        self.eval_Gt_sym = sympy.lambdify((x, y, theta, v, w, dt), Gt, 'numpy')
        self.eval_Vt_sym = sympy.lambdify((x, y, theta, v, w, dt), Vt, 'numpy')


        gux_straight = Matrix([
            x + v * sympy.cos(theta) * dt,
            y + v * sympy.sin(theta) * dt,
            theta
        ])

        Gt_straight = gux_straight.jacobian(Matrix([x, y, theta]))
        Vt_straight = gux_straight.jacobian(Matrix([v, w]))

        self.eval_gux_straight = sympy.lambdify((x, y, theta, v, w, dt), gux_straight, 'numpy')
        self.eval_Gt_straight = sympy.lambdify((x, y, theta, v, w, dt), Gt_straight, 'numpy')
        self.eval_Vt_straight = sympy.lambdify((x, y, theta, v, w, dt), Vt_straight, 'numpy')
        # ekf.predict() calls:
        # - eval_gux(mu, u, sigma_u, dt)          ← 4 arguments
        # - eval_Gt(*args, *g_extra_args)         ← args=(*mu, *u), g_extra_args=(dt,)
        #   = eval_Gt(x, y, theta, v, w, dt)      ← 6 arguments separated!


        
        def eval_gux_wrapper(mu, u, sigma_u, dt):
            """
            Wrapper for g(u,x) with handling for the case ω ≈ 0.
            """
            v_val, w_val = u
            
            # Case ω ≈ 0: straight motion
            if abs(w_val) < OMEGA_THRESHOLD:
                result = self.eval_gux_straight(mu[0], mu[1], mu[2], v_val, w_val, dt)
            else:
                result = self.eval_gux_sym(mu[0], mu[1], mu[2], v_val, w_val, dt)
            
            return np.array(result).flatten()


        def eval_Gt_wrapper(x_val, y_val, theta_val, v_val, w_val, dt_val):
            """
            Wrapper for G_t with handling for the case ω ≈ 0.
            """
            # Case ω ≈ 0: straight motion
            if abs(w_val) < OMEGA_THRESHOLD:
                return np.array(self.eval_Gt_straight(x_val, y_val, theta_val, v_val, w_val, dt_val))
            else:
                return np.array(self.eval_Gt_sym(x_val, y_val, theta_val, v_val, w_val, dt_val))


        def eval_Vt_wrapper(x_val, y_val, theta_val, v_val, w_val, dt_val):
            """
            Wrapper for V_t with handling for the case ω ≈ 0.
            """
            # Case ω ≈ 0: straight motion
            if abs(w_val) < OMEGA_THRESHOLD:
                return np.array(self.eval_Vt_straight(x_val, y_val, theta_val, v_val, w_val, dt_val))
            else:
                return np.array(self.eval_Vt_sym(x_val, y_val, theta_val, v_val, w_val, dt_val))


        self.eval_gux = eval_gux_wrapper
        self.eval_Gt = eval_Gt_wrapper
        self.eval_Vt = eval_Vt_wrapper

        self.eval_gux = eval_gux_wrapper
        self.eval_Gt = eval_Gt_wrapper
        self.eval_Vt = eval_Vt_wrapper
        self._pred_count = 0

        mx, my, mz = symbols('mx my mz')

        # Landmark measurement model
        hx = Matrix([
            sympy.sqrt((mx - x)**2 + (my - y)**2 + (mz)**2),
            sympy.atan2((my - y), (mx - x)) - theta
        ])

        # Landmark measurement jacobian
        Ht = hx.jacobian(Matrix([x, y, theta]))

        # Convert to numerical functions
        self.eval_hx_sym = sympy.lambdify((x, y, theta, mx, my, mz), hx, 'numpy')
        self.eval_Ht_sym = sympy.lambdify((x, y, theta, mx, my, mz), Ht, 'numpy')


            # ekf.update() calls:
        # - eval_hx(*hx_args) where hx_args=(mu, lmark, sigma_z)
        #   = eval_hx(mu, lmark, sigma_z)  ← 3 arguments!
        # - eval_Ht(*Ht_args) where Ht_args=(*mu, *lmark)
        #   = eval_Ht(x, y, theta, mx, my)  ← 5 arguments separated!

        def eval_hx_wrapper(x_val, y_val, theta_val, mx_val, my_val, mz_val):
            """
            Wrapper for h(x): receive 5 arguments SEPARATI.

            ekf.update() calls: eval_hx(*hx_args)
            where hx_args = (*self.ekf.mu, *m) = (x, y, theta, mx, my)
            """
            result = self.eval_hx_sym(x_val, y_val, theta_val, mx_val, my_val, mz_val)
            return np.array(result).flatten()


    

      

        def eval_Ht_wrapper(x_val, y_val, theta_val, mx_val, my_val, mz_val):
            """Wrapper per H_t: receive 5 arguments SEPARATI!"""
            return np.array(self.eval_Ht_sym(x_val, y_val, theta_val, mx_val, my_val, mz_val))

        self.eval_hx = eval_hx_wrapper
        self.eval_Ht = eval_Ht_wrapper
            

       

        #Create EKF instance
        self.ekf = RobotEKF(
            dim_x=3,
            dim_u=2,
            eval_gux=self.eval_gux,
            eval_Gt=self.eval_Gt,
            eval_Vt=self.eval_Vt,
            )
        
        #Initial state
        #self.ekf.mu = np.array([0.0, 0.77, 0.0]) real
        self.ekf.mu = np.array([-2.0, -0.5, 0.0])  # simulated
        self.ekf.Sigma = np.diag([0.05, 0.05, np.deg2rad(0.5)])

        #Process noise for the motion model, used in prediction
        alpha = np.array([0.01, np.deg2rad(0.5), 0.01, np.deg2rad(0.5)])
        # self.ekf.Mt = np.diag([
        #     0.01,  # variance for linear velocity
        #     0.01,  # variance for angular velocity
        # ])
        self.alpha = alpha

        #Measurement noise for landmark update, used in correction
        self.Qt = np.diag([
            0.15**2,  # variance for range
            np.deg2rad(2)**2,  # variance for bearing
        ])


        #Load landmark map
        self.landmarks = {
            
            11: np.array([-1.1, -1.1, 0.5]),
            12: np.array([-1.1,  0.0, 1.5]),
            13: np.array([-1.1,  1.1, 0.5]),
            21: np.array([ 0.0, -1.1, 1.0]),
            22: np.array([ 0.0,  0.0, 0.75]),
            23: np.array([ 0.0,  1.1, 0.3]),
            31: np.array([ 1.1, -1.1, 1.5]),
            32: np.array([ 1.1,  0.0, 1.0]),
            33: np.array([ 1.1,  1.1, 0.0])

            # 0: np.array([1.20, 1.68, 0.16]),
            # 1: np.array([1.68, -0.05, 0.18]),
            # 2: np.array([3.72, 0.14, 0.22]),
            # 3: np.array([3.75, 1.37, 0.21]),
            # 4: np.array([2.48, 1.25, 0.22]),
            # 5: np.array([4.80, 1.87, 0.24]),
            # 6: np.array([2.18, 1.00, 0.24]),
            # 7: np.array([2.94, 2.70, 0.73])
        }

        #Variables for /odom
        self.last_cmd_vel=np.array([0.0,0.0])
        self.dt=0.05 #50ms --> 20Hz


        #Subscribers
        self.odom_sub=self.create_subscription(Odometry,'/odom',self.odom_callback,10) #Measures v and omega
        self.landmark_sub=self.create_subscription(LandmarkArray,'/landmarks',self.landmark_callback,10) #Measures landmarks (range and bearing)

        #Publisher
        self.ekf_pub=self.create_publisher(Odometry,'/ekf',10) #Publishes EKF estimated state
        self.timer=self.create_timer(self.dt,self.prediction_callback) #20Hz

    def prediction_callback(self):
        """performs the EKF prediction step at each 50ms using velocity motion model
        """
        u=self.last_cmd_vel
        sigma_u=np.array([0.1,0.1]) #std dev for v
        
        v,omega = u
        a1,a2,a3,a4=self.alpha
        self.ekf.Mt = np.diag([ 
            a1 * v**2 + a2 * omega**2,
            a3 * v**2 + a4 * omega**2
    ])
        # Perform prediction step
        self.ekf.predict(u, sigma_u, g_extra_args=(self.dt,))
        self.ekf.mu[2] = normalize_angle(self.ekf.mu[2])

        if not hasattr(self, '_pred_count'):
            self._pred_count = 0
    
        self._pred_count += 1
        if self._pred_count % 20 == 0:  # every 20 predictions (1 second)
            x, y, theta = self.ekf.mu
            cov_trace = np.trace(self.ekf.Sigma)  # Sum of variances
            self.get_logger().info(
                f"[PREDICTION] Pos: ({x:.3f}, {y:.3f}, {np.degrees(theta):.1f}°) | "
                f"Vel: (v={u[0]:.2f}, ω={np.degrees(u[1]):.1f}°/s) | "
                f"Σ_trace: {cov_trace:.4f}"
        )
        #publish ekf state
        self.publish_ekf_state()

    def odom_callback(self,msg):
        """Callback for /odom topic to get the last command velocities
        """
        v=msg.twist.twist.linear.x
        w=msg.twist.twist.angular.z
        self.last_cmd_vel=np.array([v,w])

    def landmark_callback(self,msg):
        """Callback for /landmarks topic to perform EKF update step using landmark measurements
        """
        num_landmarks=len(msg.landmarks)
        if num_landmarks==0:
            return
        self.get_logger().info(f"Received {num_landmarks} landmarks for EKF update.")


        for lm in msg.landmarks:
            landmark_id=lm.id
            #Verifiy if landmark is in the map
            if landmark_id in self.landmarks:
                m=self.landmarks[landmark_id]
                r_measured=lm.range
                phi_measured=lm.bearing
                z=np.array([r_measured,phi_measured])
            
                mu_before = self.ekf.mu.copy()
                #Perform EKF update step
                self.ekf.update(
                    z=z,
                    eval_hx=self.eval_hx,
                    eval_Ht=self.eval_Ht,
                    Qt=self.Qt,
                    Ht_args=(*self.ekf.mu, *m),
                    hx_args=(*self.ekf.mu, *m),
                    residual = residual,
                    angle_idx=1 #bearing is the second element

                )

                delta = self.ekf.mu - mu_before
                self.get_logger().info(
                    f"  → Landmark ID={landmark_id}: "
                    f"range={z[0]:.3f}m, bearing={np.degrees(z[1]):.1f}° | "
                    f"Correction: Δx={delta[0]:.4f}m, Δy={delta[1]:.4f}m, "
                    f"Δθ={np.degrees(delta[2]):.2f}°"
                    f"new position: x={self.ekf.mu[0]:.3f}, y={self.ekf.mu[1]:.3f}, theta={np.degrees(self.ekf.mu[2]):.1f}°"
                )
        #publish ekf state
        self.publish_ekf_state()

    def publish_ekf_state(self):
        """Publish the EKF estimated state as an Odometry message
        """
        msg=Odometry()
        msg.header.stamp=self.get_clock().now().to_msg()
        msg.header.frame_id='odom'
        msg.child_frame_id='base_link'

        msg.pose.pose.position.x=float(self.ekf.mu[0])
        msg.pose.pose.position.y=float(self.ekf.mu[1])
        msg.pose.pose.position.z=0.0

        #Convert yaw to quaternion
        theta=float(self.ekf.mu[2])
        quat = self.quaternion_from_euler(0, 0, theta)
        msg.pose.pose.orientation.x=quat[0]
        msg.pose.pose.orientation.y=quat[1]
        msg.pose.pose.orientation.z=quat[2]
        msg.pose.pose.orientation.w=quat[3]

        covariance_matrix = np.zeros((36))
        covariance_matrix[0] = self.ekf.Sigma[0, 0]  # x variance
        covariance_matrix[7] = self.ekf.Sigma[1, 1]  # y variance
        covariance_matrix[35] = self.ekf.Sigma[2, 2]  # theta variance
        msg.pose.covariance = covariance_matrix.tolist()

        self.ekf_pub.publish(msg)
    
    def quaternion_from_euler(self, roll, pitch, yaw):
        """Euler → Quaternion."""
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
        node = EKFLocalizationNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
