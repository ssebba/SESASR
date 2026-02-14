import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import tf_transformations
import math
from std_msgs.msg import Float32MultiArray
from rclpy.qos import qos_profile_sensor_data

class controller(Node):
    
    def __init__(self):
        super().__init__("controller")
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)        
        self.debug_publisher_ = self.create_publisher(Float32MultiArray, '/min_distances', 10)  # for testing purposes

        self.declare_parameter('goal', [6.5, 3.0, 0.5])
        self.declare_parameter('max_vel_ang', [0.0, 0.0, 0.2])
        self.declare_parameter('max_vel_lin', [0.2, 0.0, 0.0])
        self.declare_parameter('freq', 5)
        self.declare_parameter('max_angle', [0.324, 0.5, 1.74])
        self.declare_parameter('threshold', 0.35)

        self.goal = self.get_parameter('goal').get_parameter_value().double_array_value
        self.parameters = {
            "max_vel_ang": self.get_parameter('max_vel_ang').get_parameter_value().double_array_value,
            "max_vel_lin": self.get_parameter('max_vel_lin').get_parameter_value().double_array_value,
            "freq": self.get_parameter('freq').get_parameter_value().integer_value,
            "max_angle": self.get_parameter('max_angle').get_parameter_value().double_array_value,
            "threshold": self.get_parameter('threshold').get_parameter_value().double_value
        }
        
        self.flag = 0
        self.goal_flag = 0
        
        self.min_centre_distance = 10
        self.min_right_distance = 10
        self.min_left_distance = 10

        self.mean_left_distance  = 10
        self.mean_right_distance = 10

        self.timer_period = float(1/(self.parameters["freq"]))

        self.distances={}
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10 )
        
        self.subscription = self.create_subscription(
            LaserScan, '/scan',self.scan_callback,qos_profile_sensor_data)
    
    def timer_callback(self):
        msg = Twist()
        if self.goal_flag == 0:
            if self.flag == 0:
                msg.linear.x = self.parameters["max_vel_lin"][0]
            if self.min_centre_distance <= self.parameters["threshold"]:
                self.flag=1
                if self.mean_left_distance > self.mean_right_distance:
                    msg.angular.z = self.parameters["max_vel_ang"][2]
                else:
                    msg.angular.z = -self.parameters["max_vel_ang"][2]
            else:
                self.flag=0
        else:
            self.get_logger().info(f"arrived to goal!!!")

        self.publisher_.publish(msg)

    def odom_callback(self, msg):
        x,y,z = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        qx,qy,qz,qw = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        _, _, yaw = tf_transformations.euler_from_quaternion([qx, qy, qz, qw])

        d = math.sqrt((self.goal[0]-x)**2+(self.goal[1]-y)**2)
        if d < 1.0:
            self.goal_flag = 1

    
    def scan_callback(self, msg):
        max_range = msg.range_max
        min_range = msg.range_min
        angle_min = msg.angle_min
        angle_max = msg.angle_max
        angle_increment = msg.angle_increment
        distances_centre = {}
        distances_left = {}
        distances_right = {}
        angle_first_sect=self.parameters["max_angle"][0]
        min_angle_second_sect=self.parameters["max_angle"][1]
        max_angle_second_sect=self.parameters["max_angle"][2]

        for i in range(len(msg.ranges)):
            angle = angle_min + i * angle_increment
            distance = msg.ranges[i]
            if angle <= angle_first_sect or angle >= angle_max-angle_first_sect:
                distances_centre[angle] = distance
            
            if angle >= min_angle_second_sect and angle <= max_angle_second_sect:
                if distance == float('inf') or math.isnan(distance):
                    distances_left[angle] = max_range
                    #distances_left[angle] = min_range
                else:
                    distances_left[angle] = distance

            if angle >= angle_max-(min_angle_second_sect+max_angle_second_sect) and angle <= angle_max-min_angle_second_sect:
                if distance == float('inf') or math.isnan(distance):
                    distances_right[angle] = max_range
                    #distances_left[angle] = min_range
                else:
                    distances_right[angle] = distance

        if self.flag == 0:
            if len(distances_right) > 0:
                self.mean_right_distance = sum(distances_right.values()) / len(distances_right)
            else:
                self.mean_right_distance = float('inf')
            
            if len(distances_left) > 0:
                self.mean_left_distance = sum(distances_left.values()) / len(distances_left)
            else:
                self.mean_left_distance = float('inf')
        
        self.min_centre_distance = min(distances_centre.values())
        self.min_right_distance = min(distances_right.values())
        self.min_left_distance = min(distances_left.values())

        msg_pub = Float32MultiArray()
        msg_pub.data = [self.min_centre_distance, self.min_right_distance, self.min_left_distance]
        self.debug_publisher_.publish(msg_pub)

        self.get_logger().info(f"centre_distance: {self.min_centre_distance:.4f}, right mean: {self.mean_right_distance:.4f}, left mean: {self.mean_left_distance:.4f}")
        


def main(args=None):
    rclpy.init(args=args)

    contr = controller()

    rclpy.spin(contr)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    contr.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()