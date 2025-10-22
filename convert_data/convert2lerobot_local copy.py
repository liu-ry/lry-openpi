# -*- coding: utf-8 -*-
import os
import time
import argparse
import warnings

import numpy as np
import cv2
import json
from datasets import Features, Image, Value, Sequence, Dataset
import psutil
import ffmpeg

SOURCE_DIR = '/data/vitai/'
# SOURCE_DIR = '/mnt/pi0/'

TARGET_DIR = '/data/vitai_fold_tshirt/data/chunk-000'
TARGET_DIR = '/home/sun/Desktop/datasets/episode_202509121725'
TASK_TYPES = ['folding_tshirt_pile_and_stacking', 'folding_tshirt']




def safe_save_to_parquet(data, features, output_path, min_memory_gb=6, max_retries=10, retry_interval=10):
    """
    安全保存数据到 Parquet 文件（自动检查内存）

    参数:
        data: 要保存的数据（字典或数据集）
        features: 数据集特征
        output_path: 输出 Parquet 文件路径
        min_memory_gb: 要求的最小剩余内存(GB)
        max_retries: 最大重试次数
        retry_interval: 每次重试间隔(秒)
    """
    free_mem_gb = 0
    for attempt in range(max_retries):
        # 检查剩余内存
        free_mem_gb = psutil.virtual_memory().available / (1024 ** 3)

        if free_mem_gb < min_memory_gb:
            warning_msg = (
                f"Attempt {attempt + 1}/{max_retries}: "
                f"Only {free_mem_gb:.2f}GB free memory (< {min_memory_gb}GB). "
                f"Waiting {retry_interval}s..."
            )
            warnings.warn(warning_msg)
            time.sleep(retry_interval)
            continue

        try:
            # 转换并保存数据集
            dataset = Dataset.from_dict(data, features=features)
            dataset.to_parquet(output_path)
            print(f"Successfully saved to {output_path}")
            return True

        except MemoryError:
            warnings.warn(f"MemoryError occurred, retrying... ({attempt + 1}/{max_retries})")
            time.sleep(retry_interval)

    # 所有重试失败后
    error_msg = (
        f"Failed after {max_retries} attempts. "
        f"Final free memory: {free_mem_gb:.2f}GB (required: {min_memory_gb}GB)"
    )
    raise MemoryError(error_msg)



def load_data(folder):

    # 关节状态和动作
    left_joint_pos = np.load(os.path.join(folder, "left_joint_pos.npy"))
    right_joint_pos = np.load(os.path.join(folder, "right_joint_pos.npy"))
    # left_gripper_pos = np.load(os.path.join(folder, "left-gripper_pos.npy"))
    # right_gripper_pos = np.load(os.path.join(folder, "right-gripper_pos.npy"))
    left_action_pos = np.load(os.path.join(folder, "action_left_pos.npy"))
    right_action_pos = np.load(os.path.join(folder, "action_right_pos.npy"))
    timestamps = np.load(os.path.join(folder, "timestamp.npy"))

    # 合并状态和动作
    # state = np.concatenate([left_joint_pos, left_gripper_pos, right_joint_pos, right_gripper_pos], axis=1)
    state = np.concatenate([left_joint_pos, right_joint_pos], axis=1)
    action = np.concatenate([
        left_action_pos,
        right_action_pos,
    ], axis=1)

    def extract_frames_ffmpeg(video_path, num_frames):
        try:
            # 使用ffmpeg.probe获取视频信息
            probe = ffmpeg.probe(video_path)
            video_stream = next(
                (stream for stream in probe['streams'] if stream['codec_type'] == 'video'),
                None
            )
            if not video_stream:
                raise ValueError("No video stream found in the file")

            # 计算抽帧间隔（均匀分布）
            interval = 1
            total_frames = int(video_stream.get('nb_frames'))  # 默认假设100帧
            interval = max(1, total_frames // num_frames)
            print(f'total frames: {total_frames}, num_frame {num_frames}, interval: {interval}')
            # 使用FFmpeg抽取帧到内存（PNG格式）
            out, _ = (
                ffmpeg.input(video_path)
                .filter('select', f'not(mod(n,{interval}))')
                .output('pipe:', format='image2pipe', vcodec='png', vframes=num_frames)
                .run(capture_stdout=True, quiet=True)
            )

            # 分割二进制流为单帧（通过PNG文件头标识）
            frames = []
            start = 0
            while start < len(out):
                png_header = out.find(b'\x89PNG', start)
                if png_header == -1:
                    break
                end = out.find(b'\x89PNG', png_header + 1)
                if end == -1:
                    end = len(out)
                frames.append(out[png_header:end])
                start = end

            return frames[:num_frames]

        except ffmpeg.Error as e:
            print(f"FFmpeg error: {e.stderr.decode('utf8')}")
            return []
        except Exception as e:
            print(f"Error: {str(e)}")
            return []

    # 加载视频帧
    def extract_frames(video_path, num_frames):
        print(f'Extracting {num_frames} frames from {video_path}')
        cap = cv2.VideoCapture(video_path)
        frames = []
        for i in range(num_frames):
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                ret2, png_buffer = cv2.imencode(".png", frame)
                if ret2:
                    png_bytes = png_buffer.tobytes()
                    frames.append({'bytes': png_bytes, 'path': f'frame_{i:06d}.png'})
        cap.release()
        return frames

    num_frames = len(timestamps)

    timestamps = [0.033 * i for i in range(num_frames)] # 30fps
    timestamps = np.array(timestamps)
    top_images = extract_frames(os.path.join(folder, "img_high_camera-images-rgb.mp4"), num_frames)
    left_images = extract_frames(os.path.join(folder, "img_left_camera-images-rgb.mp4"), num_frames)
    right_images = extract_frames(os.path.join(folder, "img_right_camera-images-rgb.mp4"), num_frames)

    left1_warped = extract_frames(os.path.join(folder, "left1_warped-images-rgb.mp4"), num_frames)
    left2_warped = extract_frames(os.path.join(folder, "left2_warped-images-rgb.mp4"), num_frames)
    right1_warped = extract_frames(os.path.join(folder, "right1_warped-images-rgb.mp4"), num_frames)
    right2_warped = extract_frames(os.path.join(folder, "right2_warped-images-rgb.mp4"), num_frames)
    print(f'Found {len(timestamps)} frames')

    return (state, action, timestamps, top_images, left_images, right_images,
            left1_warped, left2_warped, right1_warped, right2_warped)


def build_parquet(src_dir, episode_idx, task_index, prompt=""):
    """构建Parquet文件"""
    try:
        # 1. 加载数据
        print(f"load data from {src_dir}")
        t1 = time.time()
        (state, action, timestamps, top_images, left_images, right_images,
            left1_warped, left2_warped, right1_warped, right2_warped) = load_data(src_dir)
        t2 = time.time()
        print(f"load data took {t2 - t1} seconds")
        length = state.shape[0]

        # 2. 定义Features schema
        features = Features({
            "observation.state": Sequence(feature=Value(dtype='float64'), length=state.shape[1]),
            "action": Sequence(feature=Value(dtype='float64'), length=action.shape[1]),
            "observation.images.cam_high": Image(),
            "observation.images.cam_left_wrist": Image(),
            "observation.images.cam_right_wrist": Image(),
            "timestamp": Value("float64"),
            "frame_index": Value("int64"),
            "index": Value("int64"),
            "episode_index": Value("int64"),
            "task_index": Value("int64"),
            "prompt": Value("string"),
            "observation.images.left1_warped": Image(),
            "observation.images.left2_warped": Image(),
            "observation.images.right1_warped": Image(),
            "observation.images.right2_warped": Image(),
        })

        # 3. 准备数据
        data = {
            "observation.state": state.tolist(),
            "action": action.tolist(),
            "observation.images.cam_high": top_images,
            "observation.images.cam_left_wrist": left_images,
            "observation.images.cam_right_wrist": right_images,
            "timestamp": timestamps.tolist(),
            "frame_index": list(range(length)),
            "index": list(range(length)),
            "episode_index": [episode_idx] * length,
            "task_index": [task_index] * length,
            "prompt": [prompt] * length,
            "observation.images.left1_warped": left1_warped,
            "observation.images.left2_warped": left2_warped,
            "observation.images.right1_warped": right1_warped,
            "observation.images.right2_warped": right2_warped,
        }

        # 4. 创建临时Parquet文件
        print(f"create temporal features schema")
        parquet_path = os.path.join(TARGET_DIR, f"episode_{episode_idx:06d}.parquet")
        safe_save_to_parquet(data, features, parquet_path)


    except Exception as e:
        print(f"Error processing episode {episode_idx}: {str(e)}")
        raise


def get_all_episodes():
    """获取所有episode文件夹并排序"""
    all_episodes = []

    for task_type in TASK_TYPES:
        full_path = os.path.join(SOURCE_DIR, task_type)
        if os.path.exists(full_path):
            for entry in os.listdir(full_path):
                if entry.startswith("episode_") and os.path.isdir(os.path.join(full_path, entry)):
                    all_episodes.append({
                        'prefix': os.path.join(full_path, entry),
                        'task_type': task_type
                    })
    import random
    random.seed(42)  # 设置固定种子
    random.shuffle(all_episodes)
    return all_episodes



def process_episodes(all_episodes, start_idx, end_idx):
    """处理指定范围内的episodes"""
    # all_episodes = get_all_episodes()

    if end_idx > len(all_episodes):
        end_idx = len(all_episodes)

    print(f"Processing episodes {start_idx} to {end_idx - 1} (total {len(all_episodes)})")

    task_indices = {}  # 记录每个task_type对应的index

    for i in range(start_idx, end_idx):
        episode_info = all_episodes[i]
        episode_prefix = episode_info['prefix']
        task_type = episode_info['task_type']

        # 初始化或获取task_index
        if task_type not in task_indices:
            task_indices[task_type] = len(task_indices)
        task_index = task_indices[task_type]

        print(f"\nProcessing {episode_prefix} ({i + 1}/{len(all_episodes)})...")

        try:

            t_convert = time.time()
            build_parquet(
                src_dir=episode_prefix,
                episode_idx=i,
                task_index=task_index,
                prompt=task_type.replace("ing_", " ").replace("_", " "),
            )

            print(f"Processed in {time.time() - t_convert:.2f}s")

        except Exception as e:
            print(f"Error processing {episode_prefix}: {str(e)}")
            continue





if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0, help='Start index of episodes to process')
    parser.add_argument('--end', type=int, default=None, help='End index of episodes to process (exclusive)')
    args = parser.parse_args()

    # 获取所有episodes数量
    # all_episodes = get_all_episodes()
    #
    # if args.end is None:
    #     args.end = len(all_episodes)
    #
    # print(f"Total episodes: {len(all_episodes)}")
    # # 处理指定范围的数据
    # process_episodes(all_episodes, args.start, args.end)
    src = '/home/sun/Desktop/datasets/episode_202509121725'
    build_parquet(src, 0, 0)


