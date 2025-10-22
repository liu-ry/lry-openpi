import dataclasses
import logging
import sys
import time 
import cv2
import matplotlib.pyplot as plt
from openpi_client import websocket_client_policy as _websocket_client_policy
import numpy as np
from i2rt.robots.get_robot import get_yam_robot 
from i2rt.robots.motor_chain_robot import MotorChainRobot
import pyrealsense2 as rs

DT = 0.05
FPS = 30
TIMEOUT_MS = 100

duration = 2.0  # seconds
robot_steps = 100
robot0_initial_pos = [0, 0, 0, 0, 0, 0, 0]
robot1_initial_pos = [0, 0, 0, 0, 0, 0, 0]

# robot0_initial_pos = [-0.3122, 0.2386, 0.4580, -0.3992, 0.0708, 0.0971, 0.9892]
# robot1_initial_pos = [-0.0608, 0.3859, 0.3805, -0.2470, -0.0429, 0.0334, 0.9518]

# robot0_initial_pos = [-0.60635538,  1.05802243,  1.01186389, -0.62581064, -0.30079347, -0.26684215, 0.99223379 ]   
# robot1_initial_pos = [0.32101167,  0.84821088,  0.50679026, -0.15850309, -0.21992065, 0.42019532,  0.90369968 ]

# robot0_initial_pos = [-0.0525, 0.0525, 0.5156, -0.5037, -0.0109, -0.0071, 0.0069]
# robot1_initial_pos = [-0.0223, 0.0463, 0.6262, -0.6472, -0.0250, 0.0143, 0.0029]

class RobotActionExecution:
    def __init__(self):
        print("Initializing ActionPublisher...")
        self.server_ip = "192.168.3.5"
        self.server_port = 8040 #8020
        # self.server_ip = "106.13.38.237"
        # self.server_port = 8000
        self.ax1, self.ax2, self.ax3 = None, None, None
        self.init_action_joint_plot()
        self.init_camera()

        self.client = _websocket_client_policy.WebsocketClientPolicy(
            host=self.server_ip,
            port=self.server_port
        )
        # self.task_instruction = "fold tshirt pile and stacking"
        self.task_instruction = "fold the tshirt"
        self.frame_idx = 0
        self.run_frame_idx = 0

        self.robot0 = get_yam_robot(channel="can_follower_l")
        self.robot1 = get_yam_robot(channel="can_follower_r")
        self.robots_sync_move(self.robot0, robot0_initial_pos, self.robot1, robot1_initial_pos, 2)

    def init_camera(self):
        self.camera_names = ['cam_left_wrist', 'cam_right_wrist', 'cam_high']
        self.camera_sns = ['230322276470', '230322271636', '315122272182']

        self.cam_dict = dict(zip(self.camera_sns, self.camera_names))

        print("\n\n===STARTING REALSENSE PUBLISHER===\n\n")
        self.ctx = rs.context()
        devices = self.ctx.query_devices()
        self.device_ids = [d.get_info(rs.camera_info.serial_number) for d in devices]
        
        self.pipes = [rs.pipeline() for _ in range(len(self.camera_sns))]
        self.cfgs = [rs.config() for _ in range(len(self.camera_sns))]
        self.profiles = []
        self.depth_scales = []
        self.frame_index = 0 
        self.setup_cameras()
        self.wait_for_frames()
        print("\n\n===STARTING REALSENSE PUBLISHER===\n\n")

    def setup_cameras(self):
        try:
            for cam_name, sn, pipe, cfg in zip(self.camera_names, self.camera_sns, self.pipes, self.cfgs):
                print(f"{cam_name} {sn}")
                cfg.enable_device(sn)
                print(f"Enabled device at {FPS}")
                cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, FPS)
                profile = pipe.start(cfg)
                device = profile.get_device()
                self.profiles.append(profile)

        except Exception as e:
            print(f"Error starting {cam_name}, {sn}: {e}")

    def wait_for_frames(self):
        print('\nWAITING FOR FRAMES\n')
        for _ in range(3):
            for pipe, cam_name in zip(self.pipes, self.camera_names):
                t = time.time()
                try:
                    pipe.wait_for_frames()                    
                    print(f"{cam_name} waited {time.time() - t}s")
                except:
                    print(f"{cam_name} waited too long: {time.time() - t}s\n\n")
                    raise Exception

    def get_frames(self):
        print('\nWAITING FOR FRAMES\n')

        frame_dict = {}
        for cam_name, pipe in zip(self.camera_names, self.pipes):
            try:
                frameset = pipe.wait_for_frames(timeout_ms=TIMEOUT_MS)
            except Exception as e:
                print(f"\n\n=={cam_name} failed== {e}\n")
                [pipe.stop() for pipe in self.pipes]
                return frame_dict
            color_frame = np.array(frameset.get_color_frame().get_data())
            frame_dict[cam_name] = color_frame
        return frame_dict

    def init_action_joint_plot(self):
        self.fig3, self.ax3 = plt.subplots()
        self.ax3.set_title("Joint State  vs Action")
        self.ax3.set_xlabel("Joint Index")
        self.ax3.set_ylabel("Joint Value")
        self.left_line3, = self.ax3.plot([], [], 'b-o', label="Joint State")
        self.right_line3, = self.ax3.plot([], [], 'r-s', label="Action")
        self.ax3.set_xticks(range(14))  
        self.ax3.legend()

    def run(self):
        while True:         #self.frame_idx < self.total_frames:
            self.construct_and_publish_action()
            # time.sleep(3) 

    def construct_and_publish_action(self):#, timeout: float = 0.8):
        t1 = time.time()
        # time_cam0 = time.time()
        frame_dict = self.get_frames()
        if frame_dict != {}:
            real_left_image = frame_dict[self.camera_names[0]]
            real_right_image = frame_dict[self.camera_names[1]]
            real_top_image = frame_dict[self.camera_names[2]]

        top_image = real_top_image
        left_image = real_left_image
        right_image = real_right_image

        # top_image = cv2.cvtColor(top_image, cv2.COLOR_BGR2RGB)
        # left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
        # right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)

        top_image = cv2.resize(top_image, (224, 224))
        left_image = cv2.resize(left_image, (224, 224))
        right_image = cv2.resize(right_image, (224, 224))
        
        time_cam1 = time.time()
        print(f"Image preprocess cost: {(time_cam1 - t1)*1000} ms")

        time_joint_state0 = time.time()
        robot0_joint_pose = self.robot0.get_joint_pos() 
        robot1_joint_pose = self.robot1.get_joint_pos() 
        time_joint_state1 = time.time()
        print(f"Joint state cost begin cost: {(time_joint_state1 - t1)*1000} ms")
        print(f"Joint state cost only: {(time_joint_state1 - time_joint_state0)*1000} ms")

        real_state = np.concatenate([robot0_joint_pose, robot1_joint_pose], axis=0)
        observation = {
            "images": {
                "cam_high": top_image,
                "cam_left_wrist": left_image,
                "cam_right_wrist": right_image
            },
            "state": real_state.astype(np.float32),  
            "prompt": self.task_instruction,
        }

        t4 = time.time()
        result = self.client.infer(observation)
        t5 = time.time()        
        print(f"client.infer cost: {(t5 - t4)*1000} ms")
        action_chunk = result["actions"]

        infer_cost_time = result["server_timing"]["infer_ms"]
        print(f"infer cost: {infer_cost_time} ms")
        # action = action_chunk[0]
        # action_target1 = np.array(action[:7])
        # action_target2 = np.array(action[7:])
        t6 = time.time()
        robot0_current_joint_pose = self.robot0.get_joint_pos() # left
        robot1_current_joint_pose = self.robot1.get_joint_pos() # right

        for idx, action in enumerate(action_chunk):
            action_target1 = np.array(action[:7])  # left arm
            action_target2 = np.array(action[7:])  # right arm

            # for i in range(robot_steps + 1):
            #     alpha = i / robot_steps
            #     pos1 = (1 - alpha) * robot0_current_joint_pose + alpha * action_target1
            #     pos2 = (1 - alpha) * robot1_current_joint_pose + alpha * action_target2
            self.robot0.command_joint_pos(action_target1)
            self.robot1.command_joint_pos(action_target2)
            time.sleep(0.05)  # 控制频率
            # time.sleep(duration / robot_steps)  # 控制频率
        t7 = time.time()
        print(f"Robot execution time cost: {(t7 - t6)*1000} ms")

        combined = cv2.hconcat([top_image, left_image, right_image])
        h, w, _ = combined.shape
        combined = cv2.resize(combined, (w * 3 , h*3))
        combined = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        cv2.imshow("3 Realsense Camera", combined)

        key = cv2.waitKey(1)  
        if key == ord('q'):
            self.robots_sync_move(self.robot0, [0, 0, 0, 0, 0, 0, 0], self.robot1, [0, 0, 0, 0, 0, 0, 0], 5)
            sys.exit()

        current_state = np.concatenate([robot0_current_joint_pose, robot1_current_joint_pose], axis=0)
        cost = time.time() - t1
        print(f"construct_and_publish_action cost: {(cost)*1000} ms")

    def robots_sync_move(
            self,
            robot1: MotorChainRobot, 
            target_joint_pos1: np.ndarray, 
            robot2: MotorChainRobot, 
            target_joint_pos2: np.ndarray,
            time_interval_s: float = 10.0,
    ):
        t1 = time.time()
        joint_pos1 = np.array(robot1.get_joint_pos())
        joint_pos2 = np.array(robot2.get_joint_pos())

        target_joint_pos1 = np.array(target_joint_pos1)
        target_joint_pos2 = np.array(target_joint_pos2)

        assert len(joint_pos1) == len(target_joint_pos1)
        assert len(joint_pos2) == len(target_joint_pos2)

        steps = int(time_interval_s / DT)  # 50 steps over time_interval_s
        for i in range(steps + 1):
            alpha = i / steps  
            target_pos1 = (1 - alpha) * joint_pos1 + alpha * target_joint_pos1
            target_pos2 = (1 - alpha) * joint_pos2 + alpha * target_joint_pos2
            
            robot1.command_joint_pos(target_pos1)
            robot2.command_joint_pos(target_pos2)
            time.sleep(time_interval_s / steps)
        t2 = time.time()        
        print(f"robots_sync_move cost: {(t2 - t1)*1000} ms")

def main():
    pi0_controller = RobotActionExecution()
    try:
        while True:
            pi0_controller.run()#construct_and_publish_action()
    except KeyboardInterrupt:
        pi0_controller.robots_sync_move(pi0_controller.robot0, robot0_initial_pos, pi0_controller.robot1, robot1_initial_pos, 3)

if __name__ == "__main__":
    main()
