import dataclasses
import logging 
import tyro
import cv2
import matplotlib.pyplot as plt

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
from openpi_client import image_tools
from PIL import Image
import numpy as np
import os

@dataclasses.dataclass
class Args:
    host: str = "192.168.3.5"
    port: int = 8040                    #8030            #8020

    # host: str = "106.13.38.237"
    # port: int = 8000

    action_horizon: int = 25
    num_episodes: int = 1
    max_episode_steps: int = 1000

def load_data(folder):
    left_joint_pos = np.load(os.path.join(folder, "left-joint_pos.npy"))
    right_joint_pos = np.load(os.path.join(folder, "right-joint_pos.npy"))
    left_gripper_pos = np.load(os.path.join(folder, "left-gripper_pos.npy"))
    right_gripper_pos = np.load(os.path.join(folder, "right-gripper_pos.npy"))
    left_action_pos = np.load(os.path.join(folder, "action-left-pos.npy"))
    right_action_pos = np.load(os.path.join(folder, "action-right-pos.npy"))
    timestamps = np.load(os.path.join(folder, "timestamp.npy"))

    state = np.concatenate([left_joint_pos, left_gripper_pos, right_joint_pos , right_gripper_pos], axis=1)
    action = np.concatenate([
        left_action_pos,
        right_action_pos,
    ], axis=1)

    def extract_frames(video_path, num_frames):
        cap = cv2.VideoCapture(video_path)
        frames = []
        for i in range(num_frames):
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
        cap.release()
        return frames

    top_images = extract_frames(os.path.join(folder, "top_camera-images-rgb.mp4"), len(timestamps))
    left_images = extract_frames(os.path.join(folder, "left_camera-images-rgb.mp4"), len(timestamps))
    right_images = extract_frames(os.path.join(folder, "right_camera-images-rgb.mp4"), len(timestamps))
    print(f'Found {len(timestamps)} frames')
    timestamps = [0.033 * i for i in range(len(timestamps))]
    timestamps = np.array(timestamps)

    return state, action, timestamps, top_images, left_images, right_images


def main(args: Args) -> None:
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )

    # task_instruction = "fold the tshirt"
    task_instruction = "fold tshirt pile and stacking"
    logging.info(f"Server metadata: {ws_client_policy.get_server_metadata()}")
    metadata = ws_client_policy.get_server_metadata()
    print("==========metadata ===========", metadata)
    print("policy client create!")

    num_steps = 100
    action_interval = 1       

    fig, ax1 = plt.subplots()
    ax1.set_title("Action: Ground Truth vs Predicted")
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Joint Value")
    left_line, = ax1.plot([], [], 'b-o', label="Ground Truth")
    right_line, = ax1.plot([], [], 'r-s', label="Predicted")
    ax1.set_xticks(range(14))  # 每臂 7 个关节
    ax1.legend()    
    
    data_folder = './example_data/episode_1kRxdi6F-RmKIH7vRZz-EwHvscSOffpof0QH-aCKitY.npy.mp4/'
    # data_folder = '/home/baifj/Documents/example_data/OpenPi0_FoldingTshirt_Dataset/episode_--z7_oAfpmSA8KCpeoL-BVB0A4bWpJxRu8Q5dyaNX-M.npy.mp4'
    # data_folder = '/home/baifj/Documents/example_data/OpenPi0_FoldingTshirt_Dataset/episode_EHkZL56odqHZr-ZevdJYgkWjSi1SpYyGKR3ymjNPnhY.npy.mp4'
    # data_folder = '/home/baifj/Documents/example_data/OpenPi0_FoldingTshirt_Dataset/episode_-_ysLIcC4yO7lssYCiH3jMG0EU02RDmQ6YNpx0fqmxI.npy.mp4'
    states, actions, timestamps, top_images, left_images, right_images = load_data(data_folder)
    length = states.shape[0]

    # step = 0
    # step = step + 10
    # states[:10] = 0
    for step in range(length):
        state = states[step]
        print(f"initial state: {state}")
        gt_action = actions[step]
        timestamp = timestamps[step]
        top_image = top_images[step]
        left_image = left_images[step]
        right_image = right_images[step]

        observation = {
            "images": {
                "cam_high": top_image,
                "cam_left_wrist": left_image,
                "cam_right_wrist": right_image
            },
            "state": state.astype(np.float32), 
            "prompt": task_instruction,
        }

        try:
            if step % action_interval == 0:
                result = ws_client_policy.infer(observation)
                action_chunk = result["actions"]
                action = action_chunk[0]
                print(f"==========")
                print(f"  GT action :\n {gt_action}")
                print(f"  Pred action :\n {action}")

                left_line.set_data(range(14), gt_action)
                right_line.set_data(range(14), action)
                ax1.relim()
                ax1.autoscale_view()
                plt.pause(0.01)

        except Exception as e:
            print(f"[Client Error] Inference failed: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(Args)
