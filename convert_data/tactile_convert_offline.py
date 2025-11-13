#!/usr/bin/env python3
# coding=utf-8
"""
Description  : Example:深度估计
"""
from datetime import datetime
import cv2
import numpy as np
import os
from glob import glob
import tyro
from pyvitaisdk import GF225, VTSDeviceBaseConfig

def debounce(O: np.ndarray, C: np.ndarray):
    D = C - O

    K1 = np.where(abs(D[:, 0]) + abs(D[:, 1]) < 3, 0, 1)
    K2 = np.where(abs(D[:, 2]) < 0.03, 0, 1)

    C[:, 0] = O[:, 0] + K1 * D[:, 0]
    C[:, 1] = O[:, 1] + K1 * D[:, 1]
    C[:, 2] = O[:, 2] + K2 * D[:, 2]

def extract_video_to_frames(video_path, target_height=240, target_width=240):
    """将视频文件转换为帧列表（使用 opencv）"""
    frames = []
    # 目标分辨率 (高度, 宽度)，注意 cv2.resize 的尺寸参数是 (width, height)
    # target_height, target_width = 480, 640
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

def get_depth(frames: list = None, save_path: str = None):

    config = VTSDeviceBaseConfig(name='mock', SN='GF2259043B8C5', index=0)
    gf225 = GF225(config=config)

    bg = frames[0]
    gf225.set_background(bg)
    save = False
    length = len(frames)
    index = 0
    np_frames = []
    while index < length:
        frame = frames[index]
        index += 1
        if gf225.is_background_init():
            gf225.recon3d(frame)
            depth_map = gf225.get_depth_map() * 255 # 这里的depth，不是0-255的图，而是实际的深度值，后面使用时需注意
            np_frames.append(np.asarray(depth_map))
            # cv2.imshow(f"depth_map", depth_map)
            # cv2.imshow(f"diff image", cv2.subtract(frame, bg))

            frame_copy = frame.copy()
            # cv2.imshow(f"warped_frame", frame_copy)

            if save:
                formatted_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
                # cv2.imwrite(os.path.join(folder, f"frame_{formatted_now}.png"), frame)
                # cv2.imwrite(os.path.join(folder, f"depth_map_{formatted_now}.png"), depth_map)

    np_frames = np.array(np_frames)
    rgb_frames = np.repeat(np_frames[..., np.newaxis], 3, axis=-1)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, 30.0, (240, 240))
    for frame in rgb_frames:
        bgr_frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)
    out.release()

    print(f"成功保存深度数据到: {save_path}")

def tracking(frames: list = None, save_path: str = None):

    config = VTSDeviceBaseConfig(name='mock', SN='GF2259043B8C5', index=0)
    gf225 = GF225(config=config, marker_size=9)
    if len(frames) == 0:
        print("frames长度为0, 无法进行tracking处理")
        return
    gf225.set_background(frames[0])

    length = len(frames)
    index = 0
    np_frames = []
    o_v = None
    while index < length:

        warped_frame = frames[index]
        # cv2.imshow("image", warped_frame)
        index += 1

        if not gf225.is_inited_marker():
            gf225.init_marker(warped_frame)

        c_v = gf225.get_xyz_vector(warped_frame)
        if o_v is None:
            o_v = c_v
        debounce(o_v, c_v) # 减少marker点抖动

        delta_v = c_v - o_v
        np_frames.append(np.asarray(delta_v, dtype=np.float32))

    final_frames = np.stack(np_frames, axis=0)
    np.save(save_path, final_frames)
    print(f"成功保存tracking数据到: {save_path}")

def main(root_dir: str = "/home/lry/src/ViTai-SDK-Release/data/2025_11_05/test", 
        single_dir: bool = False):
    count = 0
    error_num = 0
    for subdir_name in os.listdir(root_dir):
        if single_dir:
            if count > 0:
                break
            subdir_path = root_dir
            count = count + 1
        else:
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
                left1_path = os.path.join(episode_folder, "left1_warped-images-rgb.mp4")
                left2_path = os.path.join(episode_folder, "left2_warped-images-rgb.mp4")
                right1_path = os.path.join(episode_folder, "right1_warped-images-rgb.mp4")
                right2_path = os.path.join(episode_folder, "right2_warped-images-rgb.mp4")
                left1_frames = extract_video_to_frames(left1_path)
                left2_frames = extract_video_to_frames(left2_path)
                right1_frames = extract_video_to_frames(right1_path)
                right2_frames = extract_video_to_frames(right2_path)
                get_depth(left1_frames, os.path.join(episode_folder, "left1_depth.mp4"))
                get_depth(left2_frames, os.path.join(episode_folder, "left2_depth.mp4"))
                get_depth(right1_frames, os.path.join(episode_folder, "right1_depth.mp4"))
                get_depth(right2_frames, os.path.join(episode_folder, "right2_depth.mp4"))
                tracking(left1_frames, os.path.join(episode_folder, "left1_tracking.npy"))
                tracking(left2_frames, os.path.join(episode_folder, "left2_tracking.npy"))
                tracking(right1_frames, os.path.join(episode_folder, "right1_tracking.npy"))
                tracking(right2_frames, os.path.join(episode_folder, "right2_tracking.npy"))

            except Exception as e:
                error_num = error_num + 1
                print(f"处理episode时出错 {episode_folder}: {str(e)} ,总计错误了 {error_num} 个")
                continue   

if __name__ == "__main__":
    tyro.cli(main)
