import numpy as np
import i2rt.robots
# from i2rt.robots.robot import Robot
# from i2rt.robots.utils import GripperForceLimiter, GripperType, JointMapper
# from i2rt.robots.motor_chain_robot import get_yam_robot
# from i2rt.motor_drivers.dm_driver import (
#     CanInterface,
#     DMChainCanInterface,
#     EncoderChain,
#     PassiveEncoderReader,
#     ReceiveMode,
# )
# from i2rt.robots.motor_chain_robot import MotorChainRobot
# from i2rt.robots.utils import GripperType

from i2rt.robots.get_robot import get_yam_robot
from i2rt.utils.utils import override_log_level

# Get a robot instance
robot0 = get_yam_robot(channel="can0")
robot1 = get_yam_robot(channel="can1")
# Get the current joint positions
joint_pos0 = robot0.get_joint_pos()
joint_pos1 = robot1.get_joint_pos()

print("Robot 0 joint_pos: ===", joint_pos0)
print("Robot 1 joint_pos: ===", joint_pos1)
# # Command the robot to move to a new joint position

# 如果gripper输出为1, 则映射为0.3，如果输出为0, 则可以映射为0.05
# target_pos1 = joint_pos1
# target_pos1[4] = 0.5
# target_pos0 = joint_pos0
# target_pos0[4] = 0.2
# target_pos0[1]= -0.02866052
# # target_pos0[6] = 0.1
# # target_pos = np.array([0, 0, 0, 0, 0, 0, 1])
# target_pos1 = joint_pos1
# target_pos1[5] = -1.4
# target_pos1[6] = 0.1a
# # # Command the robot to move to the target position
# robot0.command_joint_pos(target_pos0)
# robot0.command_joint_pos(target_pos0)


