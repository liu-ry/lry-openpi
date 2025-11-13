import os
import shutil
import json
import numpy as np
from glob import glob
# from moviepy.editor import VideoFileClip
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
import tyro
import cv2
from pathlib import Path

def extract_video_to_frames(video_path):
    """将视频文件转换为帧列表（使用 opencv）"""
    frames = []
    # 目标分辨率 (高度, 宽度)，注意 cv2.resize 的尺寸参数是 (width, height)
    target_height, target_width = 480, 640
    target_channels = 3 
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件：{video_path}")
    
    # 逐帧读取
    while True:
        ret, frame = cap.read()  # ret 表示是否读取成功，frame 是帧数据（BGR 格式）
        if not ret:
            break  # 读取完毕
        
        # 转换为 RGB 格式（可选，根据需求是否需要）
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 检测分辨率和通道数
        h, w, c = frame_rgb.shape
        if (h, w, c) != (target_height, target_width, target_channels):
            # 调整尺寸为 (target_width, target_height)，注意 cv2.resize 的参数顺序是 (宽, 高)
            frame_rgb = cv2.resize(frame_rgb, (target_width, target_height))
            # 确保通道数为 3（若原始图像为灰度图，resize 后仍可能为单通道，这里强制转换）
            if frame_rgb.ndim == 2:
                frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_GRAY2RGB)       
        
        frames.append(frame_rgb)  # 存储 RGB 格式的帧
    
    # 释放资源
    cap.release()
    return frames


def main(root_dir: str = "/data/vitai_vtla_dataset/raw_dataset/blue_cylinder_pick_and_place", 
        output_dir: str = "/data/vitai_vtla_dataset/converted_dataset/converted_blue_cylinder_pick_and_place",
        push_to_hub: bool = False):
    # 处理过程中错误的个数
    error_num = 0
    succ_num = 0
    output_path = Path(output_dir)
    # 清理现有输出目录
    if output_path.exists():
        # 提示用户确认，说明即将删除的目录
        user_input = input(
            f"目录 '{output_path}' 已存在，是否删除该目录及其所有内容？[y/N] "
        ).strip().lower()  # 转为小写，方便判断
        
        # 仅当用户输入 'y' 或 'yes' 时执行删除
        if user_input in ("y", "yes"):
            print(f"正在删除目录: {output_path}")
            shutil.rmtree(output_path)
            print("删除完成")
        else:
            print("已取消删除以及数据转换操作")
            return  
    # 检查root_dir是否存在
    if not os.path.exists(root_dir):
        raise ValueError(f"根目录不存在: {root_dir}")

    # 创建LeRobot数据集，定义要存储的特征
    dataset = LeRobotDataset.create(
        repo_id=output_dir,
        robot_type="yam",  # 替换为您的机器人类型
        fps=30,  # 根据实际数据帧率调整
        features={
            # 相机图像特征
            "observation.images.cam_high": {
                "dtype": "image",
                "shape": (480, 640, 3),  # 根据实际图像尺寸调整
                "names": ["height", "width", "channel"],
            },
            "observation.images.cam_left_wrist": {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.cam_right_wrist": {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            # 关节和动作特征
            "observation.state": {
                "dtype": "float32",
                "shape": (14,),
                "names": [     
                    "left_waist",
                    "left_shoulder",
                    "left_elbow",
                    "left_forearm_roll",
                    "left_wrist_angle",
                    "left_wrist_rotate",
                    "left_gripper",               
                    "right_waist",
                    "right_shoulder",
                    "right_elbow",
                    "right_forearm_roll",
                    "right_wrist_angle",
                    "right_wrist_rotate",
                    "right_gripper"
                    ],
            },
            "actions": {
                "dtype": "float32",
                "shape": (14,),
                "names": [
                    "left_waist",
                    "left_shoulder",
                    "left_elbow",
                    "left_forearm_roll",
                    "left_wrist_angle",
                    "left_wrist_rotate",
                    "left_gripper",               
                    "right_waist",
                    "right_shoulder",
                    "right_elbow",
                    "right_forearm_roll",
                    "right_wrist_angle",
                    "right_wrist_rotate",
                    "right_gripper"
                    ],
            # },
            # "timestamp": {
            #     "dtype": "float64",
            #     "shape": (1,),
            #     "names": ["timestamp"],
            }
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # 遍历root_dir下的所有子文件夹
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        # 仅判断是否为文件夹（排除文件）
        if os.path.isdir(item_path):
            episodes_dir = item_path
            # 这里可以添加对每个episodes_dir的处理逻辑
            print(f"找到episodes目录: {episodes_dir}")

            # 定义数据目录路径
            # episodes_dir = root_dir
            if not os.path.exists(episodes_dir):
                raise ValueError(f"Episodes目录不存在: {episodes_dir}")

            # 获取所有episode文件夹
            episode_folders = sorted(glob(os.path.join(episodes_dir, "episode_*")))
            print(f"找到 {len(episode_folders)} 个episode")

            # 处理每个episode
            for episode_idx, episode_folder in enumerate(episode_folders):
                print(f"处理 episode {episode_idx + 1}/{len(episode_folders)}: {os.path.basename(episode_folder)}")

                try:
                    # 加载元数据
                    metadata_path = os.path.join(episode_folder, "metadata.json")
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)

                    # 加载关节和动作数据
                    left_joint_pos = np.load(os.path.join(episode_folder, "left-joint_pos.npy"))
                    right_joint_pos = np.load(os.path.join(episode_folder, "right-joint_pos.npy"))
                    left_gripper_pos = np.load(os.path.join(episode_folder, "left-gripper_pos.npy"))
                    right_gripper_pos = np.load(os.path.join(episode_folder, "right-gripper_pos.npy"))
                    action_left_pos = np.load(os.path.join(episode_folder, "action-left-pos.npy"))
                    action_right_pos = np.load(os.path.join(episode_folder, "action-right-pos.npy"))
                    timestamps = np.load(os.path.join(episode_folder, "timestamp.npy"))

                    # 加载视频并转换为帧
                    video_files = {
                        "observation.images.cam_high": "top_camera-images-rgb.mp4",
                        "observation.images.cam_left_wrist": "left_camera-images-rgb.mp4",
                        "observation.images.cam_right_wrist": "right_camera-images-rgb.mp4",
                    }

                    # 加载所有视频帧
                    video_frames = {}
                    for key, filename in video_files.items():
                        video_path = os.path.join(episode_folder, filename)
                        video_frames[key] = extract_video_to_frames(video_path)

                    # 检查所有数据的长度是否一致
                    lengths = [
                        len(left_joint_pos),
                        len(right_joint_pos),
                        len(action_left_pos),
                        len(action_right_pos),
                        len(timestamps),
                        len(video_frames["observation.images.cam_high"])
                    ]
                    if len(set(lengths)) > 1:
                        print(f"警告: 数据长度不一致 {lengths}，跳过此episode")
                        continue

                    # 添加每一帧到数据集中
                    num_steps = len(timestamps)
                    for step_idx in range(num_steps):
                        # 合并关节数据为state（左关节在前，右关节在后）
                        state = np.concatenate([
                            left_joint_pos[step_idx],
                            left_gripper_pos[step_idx],
                            right_joint_pos[step_idx],
                            right_gripper_pos[step_idx]
                        ])
                        # 判断是否为 float64 或其他类型，若不是 float32 则转换
                        current_dtype = state.dtype
                        if current_dtype != np.float32:
                            state = state.astype(np.float32)
                            # print(f"已将 state 从 {current_dtype} 转换为 float32")

                        # 合并动作数据为actions（左动作在前，右动作在后）
                        actions = np.concatenate([
                            action_left_pos[step_idx],
                            action_right_pos[step_idx]
                        ])
                        # 判断是否为 float64 或其他类型，若不是 float32 则转换
                        current_dtype = actions.dtype
                        if current_dtype != np.float32:
                            actions = actions.astype(np.float32)
                            # print(f"已将 actions 从 {current_dtype} 转换为 float32")
                        
                        frame_data = {
                            # 相机图像
                            "observation.images.cam_high": video_frames["observation.images.cam_high"][step_idx],
                            "observation.images.cam_left_wrist": video_frames["observation.images.cam_left_wrist"][step_idx],
                            "observation.images.cam_right_wrist": video_frames["observation.images.cam_right_wrist"][step_idx],
                            
                            # 关节和动作数据
                            "observation.state": state,
                            "actions": actions,
                            # "timestamp": timestamps[step_idx],
                            
                            # 从元数据添加任务信息
                            "task": metadata.get("task_name"),
                        }
                        dataset.add_frame(frame_data)

                    # 保存当前episode
                    dataset.save_episode()
                    succ_num = succ_num + 1
                    print(f" {episode_folder} 处理成功， 总计成功处理并保存了 {succ_num} 个")

                except Exception as e:
                    error_num = error_num + 1
                    print(f"{str(e)} 总计错误了 {error_num} 个")
                    continue

    # 可选：推送到Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["fold tshirt", "robot_manipulation"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )

    print("数据集转换完成！")


if __name__ == "__main__":
    tyro.cli(main)