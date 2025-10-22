#!/usr/bin/env python3

import numpy as np
import time
import rclpy
from rclpy.node import Node

from i2rt.robots.get_robot import get_yam_robot
from i2rt_interfaces.msg import RobotStateStamped
from i2rt_interfaces.srv import InferencePowlicy  # 你需要先在 .srv 文件中定义 Inference.srv

class JointController(Node):
    def __init__(self):
        super().__init__('joint_controller')

        # === Robot communication setup ===
        self.robot0 = get_yam_robot(channel="can0")
        self.robot1 = get_yam_robot(channel="can1")
        self.robot0_initial_pos = self.robot0.get_joint_pos()
        self.robot1_initial_pos = self.robot1.get_joint_pos()

        self.filtered_action = np.zeros(14)
        self.alpha = 0.2

        # === ROS Interfaces ===
        self.publisher = self.create_publisher(RobotStateStamped, '/i2rt_robot/states', 1)
        self.subscription = self.create_subscription(
            RobotStateStamped,
            "/i2rt_robot/action",
            self.action_listener_callback,
            1
        )

        self.client = self.create_client(InferencePolicy, 'policy_infer')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Waiting for policy_infer service...')

        self.get_logger().info("JointController started!")

        # 30Hz publishing state
        self.timer = self.create_timer(1/30.0, self.timer_callback)

        # 控制是否等待服务响应再推下一步
        self.waiting_for_action = False

    def timer_callback(self):
        try:
            msg = RobotStateStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.data = self.robot0.get_joint_pos().tolist() + self.robot1.get_joint_pos().tolist()
            self.publisher.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish joint state: {e}")

    def action_listener_callback(self, msg: RobotStateStamped):
        try:
            raw_action = np.array(msg.data)
            self.filtered_action = self.alpha * raw_action + (1 - self.alpha) * self.filtered_action

            # 翻转右臂动作方向（如适用）
            # right_arm_flip_mask = np.array([-1, -1, -1, -1, -1, -1])
            # self.filtered_action[7:13] = right_arm_flip_mask * self.filtered_action[7:13]

            # 执行动作指令
            # self.robot0.command_joint_pos(self.filtered_action[:7])
            # self.robot1.command_joint_pos(self.filtered_action[7:])

            self.get_logger().info(f"[Executed Action]: {np.round(self.filtered_action, 3)}")

            # === Request next policy inference ===
            self.request_policy_inference()

        except Exception as e:
            self.get_logger().error(f"Failed to execute robot ACTION: {e}")
            # self.robot0.command_joint_pos(self.robot0_initial_pos)
            # self.robot1.command_joint_pos(self.robot1_initial_pos)

    def request_policy_inference(self):
        req = InferencePolicy.Request()  # 空请求，只是触发
        future = self.client.call_async(req)

        def done_cb(fut):
            try:
                response = fut.result()
                self.get_logger().info("Policy inference requested successfully.")
            except Exception as e:
                self.get_logger().error(f"Failed to call policy_infer service: {e}")

        future.add_done_callback(done_cb)

    def stop_action(self):
        self.subscription.destroy()
        self.timer.destroy()
        self.robot0.move_joints(self.robot0_initial_pos, 5)
        self.robot1.move_joints(self.robot1_initial_pos, 5)


def main(args=None):
    rclpy.init(args=args)
    node = JointController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_action()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
