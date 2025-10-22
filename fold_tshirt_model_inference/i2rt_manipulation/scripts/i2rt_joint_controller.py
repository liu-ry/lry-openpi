#!/usr/bin/env python3

import numpy as np
import time
import rclpy
import signal
from rclpy.node import Node
import matplotlib.pyplot as plt
from i2rt_interfaces.msg import RobotStateStamped
from i2rt.robots.get_robot import get_yam_robot 
from i2rt.robots.motor_chain_robot import MotorChainRobot

filename_action = open('action_robot.txt', 'w')
filename_joints = open('joint_robot.txt', 'w')
DT = 0.001
FPS = 30

# robot0_initial_pos = [0, 0, 0, 0, 0, 0, 0]
# robot1_initial_pos = [0, 0, 0, 0, 0, 0, 0]
# robot0_initial_pos = [-0.41265633, 1.65976721, 1.19957297, -0.16881489, -0.28838839, -0.32064847, 1]
# robot1_initial_pos = [0.26622448, 1.2807813, 0.87574855, -0.11957365, 0.06896747, 0.46482717, 1]
# robot0_initial_pos = [-0.3122, 0.2386, 0.4580, -0.3992, 0.0708, 0.0971, 0.9892]
# robot1_initial_pos = [-0.0608, 0.3859, 0.3805, -0.2470, -0.0429, 0.0334, 0.9518]

robot0_initial_pos = [-0.3637, 0.5304, 1.1278, -1.1538, -0.1169, 0.0677, 0.6272]
robot1_initial_pos = [-0.1326, 0.5488, 1.0172, -0.7700, -0.0952, 0.0666, 0.4958]

class JointControler(Node):
    def __init__(self):
        super().__init__('joint_controller')
        self.publisher = self.create_publisher(RobotStateStamped, '/i2rt_robot/states', 1)
        self.get_logger().info("JointStatePublisher started!")
        self.index = 0

        # self.fig, self.ax1 = plt.subplots()
        # self.ax1.set_title("Joint States: 1-7 Left, 8-14 Right")
        # self.ax1.set_xlabel("Joint Index")
        # self.ax1.set_ylabel("Joint Value")

        # # self.left_line, = self.ax1.plot([], [], 'b-o', label="Left Arm")
        # # self.right_line, = self.ax1.plot([], [], 'r-s', label="Right Arm")
        # # self.line, = self.ax1.plot([], [], 'g-', label="Action")
        #         # 设置绘图数据
        # self.joint_lines = self.ax1.plot([], [], label="Joint States")[0]
        # self.ax1.legend()

        self.robot0 = get_yam_robot(channel="can0")
        self.robot1 = get_yam_robot(channel="can1")
        self.robots_sync_move(self.robot0, robot0_initial_pos, self.robot1, robot1_initial_pos, 2)

        self.filtered_action = np.zeros(14)
        self.alpha = 0.2                       

        self.timer = self.create_timer(1/FPS, self.timer_callback)     
        self.get_logger().info("\n ==== ACTION EXECUTION STARTED! ====\n")
        self.subscription = self.create_subscription(RobotStateStamped, "/i2rt_robot/action", self.action_listener_callback, 1)

    def timer_callback(self):
        try:
            msg = RobotStateStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            robot0_joint_pose = self.robot0.get_joint_pos().tolist()
            robot1_joint_pose = self.robot1.get_joint_pos().tolist()

            msg.data = robot0_joint_pose + robot1_joint_pose
            filename_joints.write(str(msg.data) + '\n')
            self.get_logger().info(f"\n===Joint Pose===: {msg.data}")
            self.publisher.publish(msg)
            # self.joint_lines.set_data(range(len(msg.data)), msg.data)
            # self.ax1.relim()
            # self.ax1.autoscale_view()
            # plt.pause(0.01)

        except Exception as e:
            self.get_logger().error(f"Failed to read joint state: {e}")

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

        steps = int(time_interval_s / DT)  
        for i in range(steps + 1):
            alpha = i / steps  
            target_pos1 = (1 - alpha) * joint_pos1 + alpha * target_joint_pos1
            target_pos2 = (1 - alpha) * joint_pos2 + alpha * target_joint_pos2
            
            # t1 = time.time_ns()
            robot1.command_joint_pos(target_pos1)
            # t2 = time.time_ns()
            robot2.command_joint_pos(target_pos2)
            time.sleep(time_interval_s / steps)


    def action_listener_callback(self, msg:RobotStateStamped):
        try:
            raw_action = np.array(msg.data)
            filename_action.write(str(raw_action) + '\n')
            self.get_logger().info(f"\nRobot action: {raw_action}")

            self.filtered_action = self.alpha * raw_action + (1 - self.alpha) * self.filtered_action
            self.robots_sync_move(self.robot0, self.filtered_action[:7], self.robot1, self.filtered_action[7:], 1)

            filename_action.flush()
            self.index += 1
            self.robots_sync_move(self.robot0, raw_action[:7], self.robot1, raw_action[7:], 1)
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
        filename_action.close()
        filename_joints.close()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
