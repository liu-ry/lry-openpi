#!/usr/bin/env python3

import numpy as np
import time
import rclpy
from rclpy.node import Node
from i2rt_interfaces.msg import RobotStateStamped
from i2rt.robots.get_robot import get_yam_robot 
from i2rt.robots.motor_chain_robot import MotorChainRobot

filename_action = open('inference_action.txt', 'w')
filename_joints = open('robot_joint_states.txt', 'w')

DT = 0.001
FPS = 30 
# robot0_initial_pos = [0,0,0,0,0,0,0]
# robot1_initial_pos = [0,0,0,0,0,0,0]
# robot0_initial_pos = [-0.0029, 0.0036, 0.1173, -0.1345, -0.2333, 0.1120, 0.4853]
# robot1_initial_pos = [-0.1402, 0.0021, 0.1806, -0.2146, 0.2729, -0.0605, 0.5897]
robot0_initial_pos = [-0.3637, 0.5304, 1.1278, -1.1538, -0.1169, 0.0677, 0.6272]
robot1_initial_pos = [-0.1326, 0.5488, 1.0172, -0.7700, -0.0952, 0.0666, 0.4958]

MAX_DELTA_PER_STEP = 0.05  
ALPHA = 0.2  

class JointController(Node):
    def __init__(self):
        super().__init__('joint_controller')
        self.get_logger().info(" JointController Started")

        self.robot0 = get_yam_robot(channel="can0")
        self.robot1 = get_yam_robot(channel="can1")
        self.robot0.command_joint_pos(robot0_initial_pos)
        self.robot1.command_joint_pos(robot1_initial_pos)

        self.current_target_action = np.array(robot0_initial_pos + robot1_initial_pos, dtype=np.float32)
        self.smoothed_action = self.current_target_action.copy()

        self.publisher = self.create_publisher(RobotStateStamped, '/i2rt_robot/states', 1)

        self.subscription = self.create_subscription(
            RobotStateStamped, 
            "/i2rt_robot/action", 
            self.action_listener_callback, 
            1
        )

        self.timer = self.create_timer(1.0 / FPS, self.control_loop)
        self.get_logger().info(" Non-blocking control loop running at {:.1f}Hz".format(FPS))

    def action_listener_callback(self, msg: RobotStateStamped):
        action = np.array(msg.data, dtype=np.float32)
        if action.shape[0] != 14:
            self.get_logger().warn(f"Received invalid action shape: {action.shape}")
            return

        delta = action - self.current_target_action
        delta = np.clip(delta, -MAX_DELTA_PER_STEP, MAX_DELTA_PER_STEP)
        self.current_target_action = self.current_target_action + delta

        t = time.time()
        filename_action.write(f"[{t:.2f}] {self.current_target_action.tolist()}\n")
        filename_action.flush()

    def control_loop(self):
        try:
            joint0 = np.array(self.robot0.get_joint_pos(), dtype=np.float32)
            joint1 = np.array(self.robot1.get_joint_pos(), dtype=np.float32)
            current_joint = np.concatenate([joint0, joint1]) 
           
            self.smoothed_action = (
                ALPHA * self.current_target_action + (1 - ALPHA) * self.smoothed_action
            )

            delta = self.smoothed_action - current_joint
            delta = np.clip(delta, -MAX_DELTA_PER_STEP, MAX_DELTA_PER_STEP)
            next_joint_command = current_joint + delta

            left_target = next_joint_command[:7]
            right_target = next_joint_command[7:]   

            self.robot0.command_joint_pos(left_target)
            self.robot1.command_joint_pos(right_target)

            joint_msg = RobotStateStamped()
            joint_msg.header.stamp = self.get_clock().now().to_msg()
            # joint0 = self.robot0.get_joint_pos().tolist()
            # joint1 = self.robot1.get_joint_pos().tolist()
            # joint_msg.data = joint0 + joint1
            joint_msg.data = next_joint_command.tolist()
            self.publisher.publish(joint_msg)

            t = time.time()
            # filename_joints.write(f"[{t:.2f}] {joint0 + joint1}\n")
            # filename_joints.flush()
            filename_joints.write(f"[{t:.2f}] {next_joint_command.tolist()}\n")
            filename_joints.flush()
        except Exception as e:
            self.get_logger().error(f"Control loop error: {e}")

    def stop(self):
        self.get_logger().info("Stopping and resetting robots.")
        self.robot0.move_joints(np.array(robot0_initial_pos), 5)
        self.robot1.move_joints(np.array(robot1_initial_pos), 5)
        self.timer.cancel()

def main(args=None):
    rclpy.init(args=args)
    controller = JointController()
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()
        controller.destroy_node()
        filename_action.close()
        filename_joints.close()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
