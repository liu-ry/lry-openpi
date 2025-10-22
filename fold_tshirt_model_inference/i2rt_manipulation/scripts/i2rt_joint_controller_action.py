#!/usr/bin/env python3

import numpy as np
import time
import rclpy
import signal
from rclpy.node import Node
from rclpy.action import ActionServer
import matplotlib.pyplot as plt
from i2rt_interfaces.msg import RobotStateStamped
from i2rt_interfaces.action import ExecuteTrajectory
from i2rt.robots.get_robot import get_yam_robot 
from i2rt.robots.motor_chain_robot import MotorChainRobot

filename = open('action_robot.txt', 'w')

DT = 0.001
FPS = 30
# robot0_initial_pos = [0, 0, 0, 0, 0, 0, 0]
# robot1_initial_pos = [0, 0, 0, 0, 0, 0, 0]

robot0_initial_pos = [-0.3122, 0.2386, 0.4580, -0.3992, 0.0708, 0.0971, 0.9892]
robot1_initial_pos = [-0.0608, 0.3859, 0.3805, -0.2470, -0.0429, 0.0334, 0.9518]

#publish at 30Hz
class JointControlerAction(Node):
    def __init__(self):
        super().__init__('joint_controller')
        self.publisher = self.create_publisher(RobotStateStamped, '/i2rt_robot/states', 1)
        self.get_logger().info("JointStatePublisher started!")
        self.index = 0

        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Joint States: Left vs Right Arm")
        self.ax.set_xlabel("Joint Index")
        self.ax.set_ylabel("Joint Value")

        self.left_line, = self.ax.plot([], [], 'b-o', label="Left Arm")
        self.right_line, = self.ax.plot([], [], 'r-s', label="Right Arm")

        self.ax.set_xticks(range(7))  # 每臂 7 个关节
        self.ax.legend()

        self.subscription = self.create_subscription(RobotStateStamped, "/i2rt_robot/action", self.action_listener_callback, 1)

        # Initialize can communication and control
        self.robot0 = get_yam_robot(channel="can0")
        self.robot1 = get_yam_robot(channel="can1")
        self.robot0_initial_pos = self.robot0.get_joint_pos()
        self.robot1_initial_pos = self.robot1.get_joint_pos()

        self.filtered_action = np.zeros(14)
        self.alpha = 0.2                       
        self.get_logger().info("Action Controller Started!")

        self.timer = self.create_timer(1/30, self.timer_callback)     #publish at 30Hz
        # self.timer_callback()

        self.__action_server = ActionServer(
            self,
            ExecuteTrajectory,
            '/i2rt_robot/execute_trajectory',
            self.execute_callback)

    def timer_callback(self):
        try:
            msg = RobotStateStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            # msg.header.frame_id = "base_link" 
            msg.data = self.robot0.get_joint_pos().tolist() + self.robot1.get_joint_pos().tolist()
            # print("[Joint Pose]: ", msg.data)
            self.get_logger().info(f"Joint Pose: {msg.data}")
            self.publisher.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f"Failed to read joint state from Yam robots: {e}")


    def robots_sync_move(
            self,
            robot1: MotorChainRobot, 
            target_joint_pos1: np.ndarray, 
            robot2: MotorChainRobot, 
            target_joint_pos2:np.ndarray,
            time_interval_s: float = 10.0
            ):
        joint_pos1 = np.array(robot1.get_joint_pos())
        joint_pos2 = np.array(robot2.get_joint_pos())

        target_joint_pos1 = np.array(target_joint_pos1)
        target_joint_pos2 = np.array(target_joint_pos2)

        assert len(joint_pos1) == len(target_joint_pos1)
        assert len(joint_pos2) == len(target_joint_pos2)

        steps =  time_interval_s/DT 
        for i in range(steps + 1):
            alpha = i / steps  # Interpolation factor
            target_pos1 = (1 - alpha) * joint_pos1 + alpha * target_joint_pos1
            target_pos2 = (1 - alpha) * joint_pos2 + alpha * target_joint_pos2
            # t1 = time.time_ns()
            robot1.command_joint_pos(target_pos1)
            # t2 = time.time_ns()
            robot2.command_joint_pos(target_pos2)
            time.sleep(time_interval_s / steps)

    async def execute_callback(self, goal_handle):
        self.get_logger().info("Received Trajectory Goal")
        for msg in goal_handle.request.trajectory:
            try:
                raw_action = np.array(msg.data)
                filename.write(f"({self.index})" +str(raw_action) + '\n')
                filename.flush()
                self.index += 1

                # self.filtered_action = self.alpha * raw_action + (1 - self.alpha) * self.filtered_action
                self.get_logger().info(f"Robot action: {self.raw_action}")
                self.robots_sync_move(self.robot0, self.raw_action[:7], self.robot1, self.raw_action[7:], 1)

                goal_handle.publish_feedback(ExecuteTrajectory.Feedback(
                    current_status=f"Executed step {self.index}"
                ))
            except Exception as e:
                self.get_logger().error(f"Failed to execute robot ACTION: {e}")    
                self.robots_sync_move(self.robot0, self.robot0_initial_pos, self.robot1, self.robot1_initial_pos, 1)
        
        goal_handle.succeed()
        return ExecuteTrajectory.Result(success=True)

    def stop_action(self):
        self.subscription.destroy()
        self.timer.destroy()

        self.robot0.move_joints(self.robot0_initial_pos, 5)
        self.robot1.move_joints(self.robot1_initial_pos, 5)

def main(args=None):
    rclpy.init(args=args)
    joint_controller = JointControlerAction()
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
