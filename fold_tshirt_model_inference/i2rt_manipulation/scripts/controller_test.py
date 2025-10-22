from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.motor_chain_robot import MotorChainRobot
import numpy as np
import signal
import time


def robots_sync_move(
        robot1: MotorChainRobot, 
        target_joint_pos1: np.ndarray, 
        robot2: MotorChainRobot, 
        target_joint_pos2:np.ndarray,
        time_interval_s: float = 2.0
        ):
    joint_pos1 = robot1.get_joint_pos()
    assert len(joint_pos1) == len(target_joint_pos1)

    joint_pos2 = robot2.get_joint_pos()
    assert len(joint_pos2) == len(target_joint_pos2)

    steps = 50  # 50 steps over time_interval_s
    for i in range(steps + 1):
        alpha = i / steps  # Interpolation factor
        target_pos1 = (1 - alpha) * joint_pos1 + alpha * target_joint_pos1
        target_pos2 = (1 - alpha) * joint_pos2 + alpha * target_joint_pos2
        t1 = time.time_ns()
        robot1.command_joint_pos(target_pos1)
        t2 = time.time_ns()
        # print(f'time cost: {(t2 - t1) / 1e6} ms')
        robot2.command_joint_pos(target_pos2)
        time.sleep(time_interval_s / steps)

close = False

def signal_handler(sig, frame):
    global close
    close = True

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)

    robot_l = get_yam_robot(channel="can0")
    base_pos_l = robot_l.get_joint_pos()
    # base_pos_l = np.array([0, 0.0, 0.0, 0.0, 0.628, 0.0, 0.0])
    robot_r = get_yam_robot(channel="can1")
    base_pos_r = robot_r.get_joint_pos()
    # base_pos_r = np.array([0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    while not close:
        try:
            print('r: read, w: write:')
            op = input().strip().split()
            if op[0] == 'r':
                joint_pos_l = robot_l.get_joint_pos()
                pos_str_l = np.array2string(joint_pos_l, suppress_small=True, precision=4, floatmode='fixed')
                print(f'joint_pos_l=[{pos_str_l}], shape:{joint_pos_l.shape}')
                joint_pos_r = robot_r.get_joint_pos()
                pos_str_r = np.array2string(joint_pos_r, suppress_small=True, precision=4, floatmode='fixed')
                print(f'joint_pos_r=[{pos_str_r}], shape:{joint_pos_r.shape}')
                continue
            elif op[0] == 'w':
                # left 
                joint_pos_l = robot_l.get_joint_pos()
                pos_str_l = np.array2string(joint_pos_l, suppress_small=True, precision=4, floatmode='fixed')
                print(f'joint_pos_l=[{pos_str_l}], shape:{joint_pos_l.shape}\nwrite target_pos_l:')
                target_pos_l = np.array(list(map(float, input().strip().split())))

                # # right
                joint_pos_r = robot_r.get_joint_pos()
                pos_str_r = np.array2string(joint_pos_r, suppress_small=True, precision=4, floatmode='fixed')
                print(f'joint_pos_r=[{pos_str_r}], shape:{joint_pos_r.shape}\nwrite target_pos_r:')
                target_pos_r = np.array(list(map(float, input().strip().split())))

                robots_sync_move(robot_l, target_pos_l, robot_r, target_pos_r)
                # robot_l.command_joint_pos(target_pos_l)
                # robot_r.command_joint_pos(target_pos_r)
            else:
                print('Unknown command, exit.')
                break
        except Exception as e:
            print(e)
            break
        
    robots_sync_move(robot_l, base_pos_l, robot_r, base_pos_r)
    robot_l.close()
    robot_r.close()
