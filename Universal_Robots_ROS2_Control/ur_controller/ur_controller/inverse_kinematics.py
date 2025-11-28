import numpy as np
from ur_analytic_ik import ur5e
import pinocchio as pin
import pandas as pd
from collections import deque

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


import pandas as pd

class UR5eAnalyticIK:

    def __init__(self, urdf_path):
        self.tool_extension = -0.059

        self.robot = pin.buildModelFromUrdf(urdf_path)
        self.data = self.robot.createData()

        # End-effector frame ID
        self.ee_name = "tool0"
        self.ee_id = self.robot.getFrameId(self.ee_name)
        
        # self.get_logger().info("info")

        np.set_printoptions(precision=4, suppress=True, floatmode="fixed")

    def build_transform(self, p, R):
        T = np.eye(4)
        T[:3, 3] = p
        T[:3, :3] = R
        return T

    def apply_tool_extension(self, T):
        if self.tool_extension == 0.0:
            return T

        T_tool = np.eye(4)
        T_tool[2, 3] = self.tool_extension
        return T @ T_tool

    def solve_ik(self, p, R, elbow_up_only=True):
        T = self.build_transform(p, R)
        T_ext = self.apply_tool_extension(T)

        solutions = ur5e.inverse_kinematics(T_ext)

        if elbow_up_only:
            solutions = [q for q in solutions if q[1] < 0]

        return solutions, T_ext

    def jacobian(self, q):
        pin.computeJointJacobians(self.robot, self.data, q)
        pin.updateFramePlacements(self.robot, self.data)
        J = pin.getFrameJacobian(
            self.robot, self.data, self.ee_id, pin.ReferenceFrame.LOCAL
        )
        return J

    def solve_joint_velocities(self, q, ee_velocity):
        J = self.jacobian(q)
        Jp = J[:3, :]
        qdot = np.linalg.pinv(Jp) @ ee_velocity
        return qdot

    def solve_joint_accelerations(self, q, qdot, ee_acc, dt=1e-4):
        J = self.jacobian(q)

        q_next = q.copy()
        q_next += qdot * dt
        J_next = self.jacobian(q_next)

        Jdot = (J_next - J) / dt

        Jp = J[:3, :]
        Jpdot = Jdot[:3, :]

        qddot = np.linalg.pinv(Jp) @ (ee_acc - Jpdot @ qdot)
        return qddot

    def forward_kinematics(self, q):
        return ur5e.forward_kinematics(*q)


class JointSpacePublisher(Node):

    def __init__(self):
        super().__init__("joint_space_node")
        self.joint_publisher_ = self.create_publisher(
            Float64MultiArray, "joint_desire_pos", 10
        )
        self.vel_publisher_ = self.create_publisher(
            Float64MultiArray, "joint_desire_vel", 10
        )
        self.acc_publisher_ = self.create_publisher(
            Float64MultiArray, "joint_desire_acc", 10
        )
        
        self.wrench_subscriber_ = self.create_subscription(
            Float64MultiArray, "wrench_data", self.wrench_callback,10
        )
        self.wrench_subscriber_
        self.wrench_data = np.zeros(6, dtype=np.float64)
        
        self.end_pos_subscriber_ = self.create_subscription(
            Float64MultiArray, '/end_effector_pos', self.end_pos_callback, 10
        )
        self.end_pos_subscriber_
        self.end_pos_value = np.array([0.0, 0.3, 1.0], dtype=np.float64)
        self.error_buffer = deque([0.0] * 20, maxlen=20000)
        self.sum_error = 0.0
        
        urdf_path = "/home/tien/ur5_ws/src/Universal_Robots_ROS2_Description/urdf/ur.urdf"

        self.P = np.array([0.42, 0.42, 0.01])
        self.R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        
        self.last_q = None

        df = pd.read_csv("/home/tien/ur5_ws/src/Universal_Robots_ROS2_Control/ur_controller/ur_controller/Final_Trajectory_Arch_Full-2.csv")
        self.positions = df[["x", "y", "z"]].to_numpy()
        self.velocities = df[["vx", "vy", "vz"]].to_numpy()
        self.accelerations = df[["ax", "ay", "az"]].to_numpy()

        self.ik = UR5eAnalyticIK(urdf_path)
        self.solution_index = 0  # track which solution to use

        self.idx = 0
        self.hold_steps = int(15.0 / 0.02)
        
        self.iteration = 0

        timer_period = 0.008
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
    def wrench_callback(self,msg):
        self.wrench_data = np.array(msg.data, dtype=np.float64)
        # self.get_logger().info(f"Wrench data: {self.wrench_data}")
        
    def end_pos_callback(self, msg):
        self.end_pos_value = np.array(msg.data, dtype=np.float64)

    def timer_callback(self):
        if self.idx < self.hold_steps:
            row = 0
            # self.get_logger().info("hold")
        else:
            row = self.idx - self.hold_steps     
        
        if row >= len(self.positions):
            # self.get_logger().info("done")
            row = len(self.positions) - 1   
        else:
            pass
            # self.get_logger().info("run")
        
        self.P = self.positions[row]
        V = self.velocities[row]
        A = self.accelerations[row]
      
        self.get_logger().info(f"end_pos_z : {self.end_pos_value[2]}")
        if self.end_pos_value[2] <= 0.004:
            K_p = 0.01
            K_i = 0.015

            # new instantaneous error
            err = 0.0 - float(self.wrench_data[2])

            self.error_buffer.append(err)

            self.sum_error = float(sum(self.error_buffer))

            integral = self.sum_error * 0.008
            
            self.P[2] = -1.0 * (K_p * err + K_i * integral)
            


            self.get_logger().info(f"z-offset : {self.P[2]:.6f}  err={err:.6f}  integral={integral:.6f}") 

        self.R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])

        solutions, T_target = self.ik.solve_ik(self.P, self.R, elbow_up_only=True)
        
        if len(solutions) == 0:
            self.get_logger().info(f"fail at row {row}: P={self.P}, R={self.R}")
        
        if self.last_q is None:
            chosen = solutions[0]
        else:
            distances = [np.linalg.norm(q - self.last_q) for q in solutions]
            chosen = solutions[np.argmin(distances)]

        self.last_q = chosen
        solutions = chosen

        qdot = self.ik.solve_joint_velocities(solutions, V)
        qddot = self.ik.solve_joint_accelerations(solutions, qdot, A)

        self.idx += 1

        msg = Float64MultiArray()
        msg.data = solutions.tolist()
        self.joint_publisher_.publish(msg)

        vel_msg = Float64MultiArray()
        vel_msg.data = qdot.tolist()
        self.vel_publisher_.publish(vel_msg)

        acc_msg = Float64MultiArray()
        acc_msg.data = qddot.tolist()
        self.acc_publisher_.publish(acc_msg)


def main(args=None):

    rclpy.init(args=args)

    joint_space_publisher = JointSpacePublisher()

    rclpy.spin(joint_space_publisher)

    joint_space_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
