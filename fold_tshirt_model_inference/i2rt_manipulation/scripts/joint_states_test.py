from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.motor_chain_robot import MotorChainRobot
import numpy as np
import signal
import time

close = False
def signal_handler(sig, frame):
    global close
    close = True

signal.signal(signal.SIGINT, signal_handler)

def main():
    robot0 = get_yam_robot(channel="can0")  # 控制左臂（can0 是左臂）
    robot1 = get_yam_robot(channel="can1")  # 控制左臂（can1 是右臂）

    robot0.command_joint_pos(np.array([0,0,0,0,0,0,0]))
    robot1.command_joint_pos(np.array([0,0,0,0,0,0,0]))

    base_pos0 = np.array(robot0.get_joint_pos())  # 当前初始状态
    base_pos1 = np.array(robot1.get_joint_pos())  # 当前初始状态
    print(f"[INFO] Robot 0 Initial joint pos: {base_pos0}")
    print(f"[INFO] Robot 1 Initial joint pos: {base_pos1}")

    joint_idx = 2      # 改为你要控制的 joint index（0~6）
    delta = 0.05       # 每步增加的角度，单位是 rad
    steps = 30         # 步数
    interval = 0.2     # 每步等待时间（秒）

    print(f"[INFO] Moving joint {joint_idx} by {delta * steps:.3f} rad total")

    for i in range(steps):
        if close:
            break

        target0 = base_pos0.copy()
        target0[joint_idx] += delta * (i + 1)
        target1 = base_pos1.copy()
        target1[joint_idx] += delta * (i + 1)

        robot0.command_joint_pos(target0)
        robot1.command_joint_pos(target1)
        time.sleep(interval)

        current_pos0 = robot0.get_joint_pos()
        current_pos1 = robot1.get_joint_pos()
        pos_str0 = np.array2string(current_pos0, suppress_small=True, precision=4, floatmode='fixed')
        pos_str1 = np.array2string(current_pos1, suppress_small=True, precision=4, floatmode='fixed')
        print(f"[{i+1}/{steps}] joint_pos0 = {pos_str0}")
        print(f"[{i+1}/{steps}] joint_pos1 = {pos_str1}")
        print("[INFO] Resetting arm to initial position...")

    # # 获取当前关节状态
    current_pos0 = np.array(robot0.get_joint_pos())
    current_pos1 = np.array(robot1.get_joint_pos())
    steps = 50
    interval = 0.05  # 每步间隔秒数
    for i in range(steps + 1):
        alpha = i / steps
        target0 = (1 - alpha) * current_pos0 + alpha * base_pos0
        target1 = (1 - alpha) * current_pos1 + alpha * base_pos1
        robot0.command_joint_pos(target0)
        robot1.command_joint_pos(target1)
        time.sleep(interval)
    robot0.close()
    robot1.close()

if __name__ == '__main__':
    main()
