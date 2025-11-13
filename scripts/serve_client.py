import dataclasses
import logging
import tyro
import cv2

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
from openpi_client import image_tools
from PIL import Image
import numpy as np
import os
# from yam_env_wrapper import YamRobotEnvWrapper

@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8000

    action_horizon: int = 25
    num_episodes: int = 1
    max_episode_steps: int = 1000



# 加载所有数据
def load_data(folder):
    # 关节状态和动作
    left_joint_pos = np.load(os.path.join(folder, "left-joint_pos.npy"))
    right_joint_pos = np.load(os.path.join(folder, "right-joint_pos.npy"))
    left_gripper_pos = np.load(os.path.join(folder, "left-gripper_pos.npy"))
    right_gripper_pos = np.load(os.path.join(folder, "right-gripper_pos.npy"))
    left_action_pos = np.load(os.path.join(folder, "action-left-pos.npy"))
    right_action_pos = np.load(os.path.join(folder, "action-right-pos.npy"))
    timestamps = np.load(os.path.join(folder, "timestamp.npy"))

    # right_joint_pos = right_joint_pos[:, ::-1]

    # 合并状态和动作
    state = np.concatenate([left_joint_pos, left_gripper_pos, right_joint_pos , right_gripper_pos], axis=1)
    action = np.concatenate([
        left_action_pos,
        right_action_pos,
    ], axis=1)

    # 加载视频帧
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

    # Demo task instruction
    task_instruction = "fold the tshirt"

    logging.info(f"Server metadata: {ws_client_policy.get_server_metadata()}")
    metadata = ws_client_policy.get_server_metadata()
    print("metadata ===========", metadata)
    print("policy client create!")

    num_steps = 100
    action_interval = 5         #每5个step取一次新的action_chunk
    data_folder = '/sun/code/openpi/data/episode_--R7bh1vTjSt6NBH9IWT9DCIDfdL4JlFUlB04DUMsos.npy.mp4/'
    data_folder = '/data/vitai/folding_tshirt/episode_EHkZL56odqHZr-ZevdJYgkWjSi1SpYyGKR3ymjNPnhY.npy.mp4/'
    states, actions, timestamps, top_images, left_images, right_images = load_data(data_folder)
    length = states.shape[0]

    for step in range(100):
        state = states[step]
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
            "state": state.astype(np.float32),  # shape: (14,)
            "prompt": task_instruction,
        }


        try:
            if step % action_interval == 0:
                result = ws_client_policy.infer(observation)
                action_chunk = result["actions"]
                # print(f"[Client] Received action_chunk with shape: {action_chunk.shape}")
                num_horizons = action_chunk.shape[0]
                # print(f"[Client] Received last action_chunk: {action_chunk[num_horizons - 1]}")
                action = action_chunk[step % action_interval]
                print(f"==========")
                print(f"  GT action :\n {gt_action}")
                print(f"Pred action :\n {action}")




        except Exception as e:
            print(f"[Client Error] Inference failed: {e}")





    # for step in range(num_steps):
    #     img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    #     cam_high = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    #     cam_left_wrist = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    #     cam_right_wrist = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    #     state = np.random.randn(14).astype(np.float32)
    #
    #     observation = {
    #         "images": {
    #             "cam_high": image_tools.convert_to_uint8(image_tools.resize_with_pad(cam_high, 224, 224)).transpose(2, 0, 1),
    #             "cam_left_wrist": image_tools.convert_to_uint8(image_tools.resize_with_pad(cam_left_wrist, 224, 224)).transpose(2, 0, 1),
    #             "cam_right_wrist": image_tools.convert_to_uint8(image_tools.resize_with_pad(cam_right_wrist, 224, 224)).transpose(2, 0, 1),
    #         },
    #         "state": state.astype(np.float32),  # shape: (14,)
    #         "prompt": task_instruction,
    #     }
    #
    #     # print(f"[Client] Built observation keys: {list(observation.keys())}")
    #
    #     try:
    #         if step % action_interval == 0:
    #             result = ws_client_policy.infer(observation)
    #             action_chunk = result["actions"]
    #             print(f"[Client] Received action_chunk with shape: {action_chunk.shape}")
    #             num_horizons = action_chunk.shape[0]
    #             print(f"[Client] Received last action_chunk: {action_chunk[num_horizons - 1]}")
    #     except Exception as e:
    #         print(f"[Client Error] Inference failed: {e}")
    #
    #     action = action_chunk[step % action_interval]
    #     print(f"Step {step}: Executing action {action}")
    #     state += 0.01

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    # tyro.cli(main)
    # a = Args()
    main(Args)
