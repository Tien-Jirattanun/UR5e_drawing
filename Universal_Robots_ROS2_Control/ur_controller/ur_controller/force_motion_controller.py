import rclpy
from rclpy.node import Node

import numpy as np
import pinocchio as pin
from enum import Enum
import numpy as np
from scipy.spatial.transform import Rotation as R

from std_msgs.msg import Float64
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener, TransformException
# --- visualization rviz ---
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import math
from collections import deque

class StateMachine(Enum):
    CALIBRATION = 0
    INIT_POSITION = 1

class ForceMotionControl(Node):

    def __init__(self):
        super().__init__('force_motion_controller_node')

        self.iteration = 0
        self.state = StateMachine.INIT_POSITION;
        
        # --- marker publisher ---
        self.publisher_ = self.create_publisher(
            Marker, 
            'visualization_marker', 
            10
        )
        
        self.path_history = [] 
        self.max_history_points = 100000
        # ------------------------

        # --- end effector pose subscriber ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.target_frame = 'base_link'
        self.source_frame = 'end_effector'
        
        self.curr_pos = np.zeros(3, dtype=np.float64)
        # default quaternion (x,y,z,w) - identity rotation
        self.curr_rot = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        
        self.end_pos_publisher_ = self.create_publisher(
            Float64MultiArray,
            '/end_effector_pos',
            10
        )
        # ------------------------------------
        
        # --- Wrench Subscriber ---
        self.wrench_subscription_ = self.create_subscription(
            WrenchStamped,                  
            '/force_torque',                
            self.wrench_callback,          
            10                               
        )
        self.wrench_publisher_ = self.create_publisher(
            Float64MultiArray,
            '/wrench_data',
            10
        )
        self.wrench_subscription_  
        self.ft_calibration_offset = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.wrench_data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # -------------------------
        
        # --- desire position aqquire ---
        self.ik_pos_subscription_ = self.create_subscription(
            Float64MultiArray,
            'joint_desire_pos',
            self.ik_pos_callback,
            10)
        self.ik_pos_subscription_  # prevent unused variable warning

        self.ik_vel_subscription_ = self.create_subscription(
            Float64MultiArray,
            'joint_desire_vel',
            self.ik_vel_callback,
            10)
        self.ik_vel_subscription_  # prevent unused variable warning

        self.ik_acc_subscription_ = self.create_subscription(
            Float64MultiArray,
            'joint_desire_acc',
            self.ik_acc_callback,
            10)
        self.ik_acc_subscription_  # prevent unused variable warning

        self.desire_pos = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0], dtype=np.float64)
        self.desire_vel = np.zeros(6, dtype=np.float64)
        self.desire_acc = np.zeros(6, dtype=np.float64)
        # -------------------------------

        # --- pinocchio robot model and effort publisher ---
        self.robot_model = pin.buildModelsFromUrdf("/home/tien/ur5_ws/src/Universal_Robots_ROS2_Description/urdf/ur.urdf")[0]
        self.data = pin.Data(self.robot_model)
        self.sum_force_error = np.zeros(6)

        self.joint_effort_publisher_ = self.create_publisher(
            Float64MultiArray,
            '/effort_controller/commands',
            10
        )
        # ---------------------------------------------------

        # --- joint states subscriber ---
        self.joint_state_subscriptions_ = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            9
        )

        self.joint_state_position = np.zeros(self.robot_model.nv)
        self.joint_state_velocity = np.zeros(self.robot_model.nv)
        self.joint_state_effort = np.zeros(self.robot_model.nv)
        # ------------------------------
        
        # --- rmse publisher ---
        self.rmse_publisher_ = self.create_publisher(
            Float64MultiArray,
            'pos_rmse',
            10
        )
        self.error_buffer = deque(maxlen=10000)  
        self.rmse = [0.0] * 6
        self.final_rmse = 0.0
        self.joint_limits_upper = np.array([6.28, 6.28, 3.14, 6.28, 6.28, 6.28], dtype=np.float64)
        self.joint_limits_lower = np.array([-6.28, -6.28, -3.14, -6.28, -6.28, -6.28], dtype=np.float64)
        self.joint_ranges = self.joint_limits_upper - self.joint_limits_lower
        self.final_rmse_pct = np.zeros(6, dtype=np.float64)
        # ----------------------

        # --- Timer callback ---
        timer_period = 0.001  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        # ----------------------
        
    def setup_marker(self):
        marker = Marker()
        marker.header.frame_id = "world" 
        marker.ns = "continuous_line"
        marker.id = 0
        marker.type = Marker.LINE_STRIP  
        marker.action = Marker.ADD      
        
        marker.scale.x = 0.005
        
        marker.color.a = 1.0  
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        
        marker.pose.orientation.w = 1.0
        
        return marker

    def ik_pos_callback(self, msg):
        self.desire_pos = np.array(msg.data, dtype=np.float64)

    def ik_vel_callback(self, msg):
        self.desire_vel = np.array(msg.data, dtype=np.float64)

    def ik_acc_callback(self, msg):
        self.desire_acc = np.array(msg.data, dtype=np.float64)

    def joint_state_callback(self, msg):
        self.joint_state_position = np.array(msg.position, dtype=np.float64)
        self.joint_state_velocity = np.array(msg.velocity, dtype=np.float64)
        self.joint_state_effort = np.array(msg.effort, dtype=np.float64)

    def wrench_callback(self, msg):
        
        # filter
        alpha = 0.85
        
        # Force
        self.wrench_data[0] = (alpha * self.wrench_data[0]) + ((1 - alpha) * (msg.wrench.force.x - self.ft_calibration_offset[0]))
        self.wrench_data[1] = (alpha * self.wrench_data[1]) + ((1 - alpha) * (msg.wrench.force.y - self.ft_calibration_offset[1]))
        self.wrench_data[2] = (alpha * self.wrench_data[2]) + ((1 - alpha) * (msg.wrench.force.z - self.ft_calibration_offset[2]))
        # Torque
        self.wrench_data[3] = (alpha * self.wrench_data[3]) + ((1 - alpha) * (msg.wrench.torque.x - self.ft_calibration_offset[3]))
        self.wrench_data[4] = (alpha * self.wrench_data[4]) + ((1 - alpha) * (msg.wrench.torque.y - self.ft_calibration_offset[4]))
        self.wrench_data[5] = (alpha * self.wrench_data[5]) + ((1 - alpha) * (msg.wrench.torque.z - self.ft_calibration_offset[5]))
        
    def end_effector_pose(self):
        if not self.tf_buffer.can_transform(self.target_frame, self.source_frame, rclpy.time.Time()):
            self.get_logger().info(
                f'Waiting for transform from {self.source_frame} to {self.target_frame}...')
            return

        try:
            t = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.source_frame,
                rclpy.time.Time()
            )
        except TransformException as ex:
            self.get_logger().error(
                f'Could not transform {self.source_frame} to {self.target_frame}: {ex}')
            return
        
        # tip_link pose in base_link
        self.curr_pos = np.array([t.transform.translation.x,
                                  t.transform.translation.y,
                                  t.transform.translation.z], dtype=np.float64)

        # orientation as quaternion (x,y,z,w)
        self.curr_rot = np.array([t.transform.rotation.x,
                                  t.transform.rotation.y,
                                  t.transform.rotation.z,
                                  t.transform.rotation.w], dtype=np.float64)
        
        # self.get_logger().info("End eff pos x:{curr_pos}")
        
        # self.get_logger().info("End Effector Position: x: {:.3f}, y: {:.3f}, z: {:.3f}".format(
        #     self.end_x, self.end_y, self.end_z)) 

    def timer_callback(self):

        # --- TF2 ---
        self.end_effector_pose() 
        # -----------

        # --- joint effort publisher and torque calculation ---
        if np.allclose(self.joint_state_position, 0.0):
            self.get_logger().warn("Waiting for valid joint states...")
            return 
        if self.joint_state_position is np.zeros(self.robot_model.nv):
            return
        # -----------------------------------------------------

        # --- State Machine ---        
        if self.state == StateMachine.CALIBRATION:
            desire_pos = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0], dtype=np.float64)
            desire_force = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64) 
            if self.iteration > 1000: 
                self.state = StateMachine.INIT_POSITION
                self.iteration = 0
            self.iteration += 1
        elif self.state == StateMachine.INIT_POSITION: 
            desire_pos = self.desire_pos 
            desire_vel = self.desire_vel
            desire_acc = self.desire_acc
            desire_force = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64) 
        
        # --------------------- 
        
        # --- Controller ---
        # PD Motion Control
        motion_kp = 60
        motion_kd = 10
        
        a = np.zeros(self.robot_model.nv) 
        errors = np.zeros(self.robot_model.nv)
        
        # Motion Control
        for i in range(self.robot_model.nv):
            a[i] = desire_acc[i] + (motion_kp * (desire_pos[i] - self.joint_state_position[i])) + (motion_kd * (desire_vel[i] - self.joint_state_velocity[i]))
            errors[i] = desire_pos[i] - self.joint_state_position[i]
        
        self.error_buffer.append(errors)
        
        if len(self.error_buffer) > 0:
            all_errors = np.array(list(self.error_buffer))  # shape: (buffer_len, nv)
            self.rmse = np.sqrt(np.mean(np.square(all_errors), axis=0))  # RMSE per joint
            self.final_rmse = np.sqrt(np.mean(np.square(all_errors)))  # overall RMSE
         
        self.final_rmse = np.mean(self.rmse)/(6.28*2)
        rmse_percentage = (self.rmse / self.joint_ranges) * 100.0
        self.final_rmse_pct = (self.final_rmse / np.mean(self.joint_ranges)) * 100.0

        
        motion_control = pin.rnea(self.robot_model, self.data, self.joint_state_position , self.joint_state_velocity, a)

        # torque = motion_control + force_control
        torque = motion_control
        
        # ------------------

        # --- rviz visualization Publisher ---
        current_point = Point(x=float(self.curr_pos[0]), y=float(self.curr_pos[1]), z=float(self.curr_pos[2]))
       
        # if self.curr_pos[2] == 0.0: 
        if self.curr_pos[2] < 0.002:
            self.path_history.append(current_point)
        
        if len(self.path_history) > self.max_history_points:
            self.path_history.pop(0)
        
        marker = self.setup_marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        
        marker.points = self.path_history
        
        self.publisher_.publish(marker)
        # ------------------------------------
        
        # --- rmse publisher ---
        rmse_msg = Float64MultiArray()
        rmse_msg.data = np.array([self.final_rmse, self.final_rmse_pct],dtype=np.float64).tolist()
        self.rmse_publisher_.publish(rmse_msg) 
        # ----------------------
        
        # --- end effector Publisher ---
        end_pos_msg = Float64MultiArray()
        end_pos_msg.data = self.curr_pos.tolist()
        self.end_pos_publisher_.publish(end_pos_msg)
        # ------------------------------

        # --- Joint effort Publisher ---
        joint_effort_msg = Float64MultiArray()
        joint_effort_msg.data = torque.tolist()
        self.joint_effort_publisher_.publish(joint_effort_msg)
        # ------------------------------
        
        # --- Wrench Publisher ---
        wrench_msg = Float64MultiArray()
        wrench_msg.data = self.wrench_data
        self.wrench_publisher_.publish(wrench_msg)
        # ------------------------


def main(args=None):
    rclpy.init(args=args)

    force_motion_control = ForceMotionControl()

    rclpy.spin(force_motion_control)

    force_motion_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
