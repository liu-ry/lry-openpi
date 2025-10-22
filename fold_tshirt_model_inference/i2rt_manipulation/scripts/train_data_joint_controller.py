#!/usr/bin/env python3

import numpy as np
import time
import rclpy
import signal
import matplotlib.pyplot as plt
from rclpy.node import Node
from i2rt.robots.get_robot import get_yam_robot 
from i2rt_interfaces.msg import RobotStateStamped
from i2rt.robots.motor_chain_robot import MotorChainRobot

filename = open('action_robot.txt', 'w')
DT = 0.001
FPS = 30

robot0_initial_pos = [0, 0, 0, 0, 0, 0, 0]
robot1_initial_pos = [0, 0, 0, 0, 0, 0, 0]

# robot0_initial_pos = [-0.3122, 0.2386, 0.4580, -0.3992, 0.0708, 0.0971, 0.9892]
# robot1_initial_pos = [-0.0608, 0.3859, 0.3805, -0.2470, -0.0429, 0.0334, 0.9518]

class JointControler(Node):
    def __init__(self):
        super().__init__('joint_controller')
        self.get_logger().info("Action Controller started!")
        self.index = 0

        # Initialize can communication and control
        self.robot0 = get_yam_robot(channel="can0")
        self.robot1 = get_yam_robot(channel="can1")
        # self.robot0_initial_pos = self.robot0.get_joint_pos()
        # self.robot1_initial_pos = self.robot1.get_joint_pos()
        self.robots_sync_move(self.robot0, robot0_initial_pos, self.robot1, robot1_initial_pos, 2)

        self.filtered_action = np.zeros(14)
        self.alpha = 0.2                       
        self.subscription = self.create_subscription(RobotStateStamped, "/i2rt_robot/actions", self.action_listener_callback, 1)
        
        self.fig1, self.ax1 = plt.subplots()
        self.ax1.set_title("Received Action: Left vs Right Arm")
        self.ax1.set_xlabel("Joint Index")
        self.ax1.set_ylabel("Joint Value")

        # 初始化左右两条曲线
        self.left_line, = self.ax1.plot([], [], 'b-o', label="Left Arm")
        self.right_line, = self.ax1.plot([], [], 'r-s', label="Right Arm")

        self.ax1.set_xticks(range(7))  # 每臂 7 个关节
        self.ax1.legend()

    def robots_sync_move(
            self,
            robot1: MotorChainRobot, 
            target_joint_pos1: np.ndarray, 
            robot2: MotorChainRobot, 
            target_joint_pos2: np.ndarray,
            time_interval_s: float = 10.0,
    ):
        joint_pos1 = np.array(robot1.get_joint_pos())
        joint_pos2 = np.array(robot2.get_joint_pos())

        # print("joint_pos1", joint_pos1)
        # print("joint_pos2", joint_pos2)

        target_joint_pos1 = np.array(target_joint_pos1)
        target_joint_pos2 = np.array(target_joint_pos2)

        assert len(joint_pos1) == len(target_joint_pos1)
        assert len(joint_pos2) == len(target_joint_pos2)

        steps = int(time_interval_s / DT)  # 50 steps over time_interval_s
        for i in range(steps + 1):
            alpha = i / steps  
            target_pos1 = (1 - alpha) * joint_pos1 + alpha * target_joint_pos1
            target_pos2 = (1 - alpha) * joint_pos2 + alpha * target_joint_pos2
            
            t1 = time.time_ns()
            robot1.command_joint_pos(target_pos1)
            t2 = time.time_ns()
            robot2.command_joint_pos(target_pos2)
            time.sleep(time_interval_s / steps)


    def action_listener_callback(self, msg:RobotStateStamped):
        try:
            raw_action = np.array(msg.data)
            self.get_logger().info(f"\nRobot action: {raw_action}")
            self.robots_sync_move(self.robot0, raw_action[:7], self.robot1, raw_action[7:], 1)
            left_joints = raw_action[:7]
            right_joints = raw_action[7:]

            self.left_line.set_data(range(7), left_joints)
            self.right_line.set_data(range(7), right_joints)
            self.ax1.relim()
            self.ax1.autoscale_view()
            plt.pause(0.01)

        except Exception as e:
            self.get_logger().error(f"Failed to execute robot ACTION: {e}")    
            self.robots_sync_move(self.robot0, robot0_initial_pos, self.robot1, robot1_initial_pos, 2)

    def stop_action(self):
        self.subscription.destroy()
        self.timer.destroy()

        self.robot0.move_joints(np.array(robot0_initial_pos), 5)
        self.robot1.move_joints(np.array(robot1_initial_pos), 5)

def main(args=None):
    rclpy.init(args=args)
    joint_controller = JointControler()
    try:
        rclpy.spin(joint_controller)
    except KeyboardInterrupt:
        pass
    finally:
        joint_controller.stop_action()
        joint_controller.destroy_node()
        filename.close()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
