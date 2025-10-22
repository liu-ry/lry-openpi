import os
import shutil
import json
import numpy as np
from glob import glob
# from moviepy.editor import VideoFileClip
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
import tyro
import cv2

REPO_NAME = "/data/vitai_vtla_dataset/converted_dataset/clean_pepsi_trash"  # 输出数据集名称，用于Hugging Face Hub


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
            # print(f"{video_path}: 当前尺寸与预期的图像尺寸不匹配，即将进行分辨率的resize")
            # 调整尺寸为 (target_width, target_height)，注意 cv2.resize 的参数顺序是 (宽, 高)
            frame_rgb = cv2.resize(frame_rgb, (target_width, target_height))
            # 确保通道数为 3（若原始图像为灰度图，resize 后仍可能为单通道，这里强制转换）
            if frame_rgb.ndim == 2:
                frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_GRAY2RGB)       
        
        frames.append(frame_rgb)  # 存储 RGB 格式的帧
    
    # 释放资源
    cap.release()
    return frames


def main(root_dir: str = "/data/vitai_vtla_dataset/raw_dataset", *, push_to_hub: bool = False):
    error_num = 0
    succ_num = 0
    # 清理现有输出目录
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)
    # 创建LeRobot数据集，定义要存储的特征
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="yam",  # 替换为您的机器人类型
        fps=30,  # 根据实际数据帧率调整
        features={
            # 相机图像特征
            "img_high_camera": {
                "dtype": "image",
                "shape": (480, 640, 3),  # 根据实际图像尺寸调整
                "names": ["height", "width", "channel"],
            },
            "img_left_camera": {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "img_right_camera": {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "left1_warped-images": {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "left2_warped-images": {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "right1_warped-images": {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "right2_warped-images": {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            # 关节和动作特征
            "state": {
                "dtype": "float32",
                "shape": (14,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (14,),
                "names": ["actions"],
            },
            # "timestamp": {
            #     "dtype": "float32",
            #     "shape": (1,),
            #     "names": ["timestamp"],
            # }
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )
    
    # 定义数据目录路径
    for subdir_name in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir_name)
    
        if not os.path.exists(subdir_path):
            raise ValueError(f"Episodes目录不存在: {subdir_path}")
        # 确保是目录（排除文件）
        if not os.path.isdir(subdir_path):
            continue
        
        print(f"进入文件夹：{subdir_path}")


        # 获取所有episode文件夹
        episode_folders = sorted(glob(os.path.join(subdir_path, "episode_*")))
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
                left_joint_pos = np.load(os.path.join(episode_folder, "left_joint_pos.npy"))
                right_joint_pos = np.load(os.path.join(episode_folder, "right_joint_pos.npy"))
                action_left_pos = np.load(os.path.join(episode_folder, "action_left_pos.npy"))
                action_right_pos = np.load(os.path.join(episode_folder, "action_right_pos.npy"))
                timestamps = np.load(os.path.join(episode_folder, "timestamp.npy"))

                # 加载视频并转换为帧
                video_files = {
                    "img_high_camera": "img_high_camera-images-rgb.mp4",
                    "img_left_camera": "img_left_camera-images-rgb.mp4",
                    "img_right_camera": "img_right_camera-images-rgb.mp4",
                    "left1_warped-images": "left1_warped-images-rgb.mp4",
                    "left2_warped-images": "left2_warped-images-rgb.mp4",
                    "right1_warped-images": "right1_warped-images-rgb.mp4",
                    "right2_warped-images": "right2_warped-images-rgb.mp4"
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
                    len(video_frames["img_high_camera"])
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
                        right_joint_pos[step_idx]
                    ])

                    # 合并动作数据为actions（左动作在前，右动作在后）
                    actions = np.concatenate([
                        action_left_pos[step_idx],
                        action_right_pos[step_idx]
                    ])
                    frame_data = {
                        # 相机图像
                        "img_high_camera": video_frames["img_high_camera"][step_idx],
                        "img_left_camera": video_frames["img_left_camera"][step_idx],
                        "img_right_camera": video_frames["img_right_camera"][step_idx],
                        "left1_warped-images": video_frames["left1_warped-images"][step_idx],
                        "left2_warped-images": video_frames["left2_warped-images"][step_idx],
                        "right1_warped-images": video_frames["right1_warped-images"][step_idx],
                        "right2_warped-images": video_frames["right2_warped-images"][step_idx],
                        
                        # 关节和动作数据
                        "state": state,
                        "actions": actions,
                        # "timestamp": timestamps[step_idx],
                        
                        # 从元数据添加任务信息
                        "task": metadata.get("task_name", "prompt"),
                    }
                    dataset.add_frame(frame_data)

                # 保存当前episode
                dataset.save_episode()
                succ_num = succ_num + 1
                print(f" {episode_folder} 处理成功， 总计成功处理并保存了 {succ_num} 个")

            except Exception as e:
                error_num = error_num + 1
                print(f"处理episode时出错 {episode_folder}: {str(e)} ,总计错误了 {error_num} 个")
                continue

    # 可选：推送到Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["clean_table", "robot_manipulation"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )

    print("数据集转换完成！")


if __name__ == "__main__":
    tyro.cli(main)