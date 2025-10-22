# -*- coding: utf-8 -*-
import os
import time
import tempfile
import argparse
import numpy as np
import cv2
import json
import boto3
from botocore.exceptions import ClientError
from datasets import Features, Image, Value, Sequence, Dataset

# S3配置
S3_BUCKET = 'vitai-pi0-data-bucket'
S3_SOURCE_PREFIX = 'xdof/'
S3_TARGET_PREFIX = 'xdof_converted/chunk-000/'
TASK_TYPES = ['folding_tshirt_pile_and_stacking', 'folding_tshirt']
LOCAL_TEMP_DIR = tempfile.mkdtemp()

# 初始化S3客户端
s3 = boto3.client('s3')


def download_s3_folder(s3_prefix, local_dir):
    """递归下载S3文件夹到本地"""
    paginator = s3.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=S3_BUCKET, Prefix=s3_prefix):
        if 'Contents' not in result:
            continue
        for obj in result['Contents']:
            if obj['Key'].endswith('/'):
                continue
            local_path = os.path.join(local_dir, os.path.relpath(obj['Key'], s3_prefix))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            try:
                s3.download_file(S3_BUCKET, obj['Key'], local_path)
            except ClientError as e:
                print(f"Error downloading {obj['Key']}: {e}")


def upload_to_s3(local_path, s3_key):
    """上传文件到S3"""
    try:
        s3.upload_file(local_path, S3_BUCKET, s3_key)
        print(f"Uploaded {s3_key} successfully")
    except ClientError as e:
        print(f"Error uploading {s3_key}: {e}")


def load_data(folder):

    # 关节状态和动作
    left_joint_pos = np.load(os.path.join(folder, "left-joint_pos.npy"))
    right_joint_pos = np.load(os.path.join(folder, "right-joint_pos.npy"))
    left_gripper_pos = np.load(os.path.join(folder, "left-gripper_pos.npy"))
    right_gripper_pos = np.load(os.path.join(folder, "right-gripper_pos.npy"))
    left_action_pos = np.load(os.path.join(folder, "action-left-pos.npy"))
    right_action_pos = np.load(os.path.join(folder, "action-right-pos.npy"))
    timestamps = np.load(os.path.join(folder, "timestamp.npy"))

    # 合并状态和动作
    state = np.concatenate([left_joint_pos, left_gripper_pos, right_joint_pos, right_gripper_pos], axis=1)
    action = np.concatenate([
        left_action_pos,
        right_action_pos,
    ], axis=1)

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


    # timestamps = [0.033 * i for i in range(len(timestamps))] # 30fps
    # timestamps = np.array(timestamps)
    top_images = extract_frames(os.path.join(folder, "top_camera-images-rgb.mp4"), len(timestamps))
    left_images = extract_frames(os.path.join(folder, "left_camera-images-rgb.mp4"), len(timestamps))
    right_images = extract_frames(os.path.join(folder, "right_camera-images-rgb.mp4"), len(timestamps))
    print(f'Found {len(timestamps)} frames')



    return state, action, timestamps, top_images, left_images, right_images
    # return state[:limit], action[:limit], timestamps[:limit], top_images[:limit], left_images[:limit], right_images[:limit]


def build_and_upload_parquet(src_dir, episode_idx, task_index, prompt=""):
    """构建Parquet文件并直接上传到S3"""
    try:
        # 1. 加载数据
        print(f"load data from {src_dir}")
        t1 = time.time()
        state, action, timestamps, top_images, left_images, right_images = load_data(src_dir)
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
            "prompt": Value("string")
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
            "prompt": [prompt] * length
        }

        # 4. 创建临时Parquet文件
        print(f"create temporal features schema")
        temp_parquet = os.path.join(LOCAL_TEMP_DIR, f"temp_{episode_idx}.parquet")
        dataset = Dataset.from_dict(data, features=features)
        dataset.to_parquet(temp_parquet)

        # 5. 上传到S3
        s3_key = f"{S3_TARGET_PREFIX}episode_{episode_idx:06d}.parquet"
        print(f"send parquet to {s3_key}")
        t3 = time.time()
        upload_to_s3(temp_parquet, s3_key)
        t4 = time.time()
        print(f"upload took {t4 - t3} seconds")

        # 6. 清理临时文件
        print(f"remove {temp_parquet}")
        os.remove(temp_parquet)

        return len(timestamps)  # Return the episode length

    except Exception as e:
        print(f"Error processing episode {episode_idx}: {str(e)}")
        raise


def get_all_episodes():
    """获取所有episode文件夹并排序"""
    all_episodes = []

    for task_type in TASK_TYPES:
        s3_task_prefix = os.path.join(S3_SOURCE_PREFIX, task_type)

        # 获取该任务类型下的所有episode文件夹
        paginator = s3.get_paginator('list_objects_v2')
        for result in paginator.paginate(Bucket=S3_BUCKET, Prefix=f'{s3_task_prefix}/episode_', Delimiter='/'):
            for prefix in result.get('CommonPrefixes', []):
                all_episodes.append({
                    'prefix': prefix['Prefix'],
                    'task_type': task_type
                })

    # 按照prefix排序
    all_episodes.sort(key=lambda x: x['prefix'])
    return all_episodes


def write_episodes_jsonl(episodes_data, output_path):
    """写入episodes.jsonl文件"""
    with open(output_path, 'a') as f:
        json_line = json.dumps(episodes_data)
        f.write(json_line + '\n')
    print(f"Saved episodes metadata to {output_path}")


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

        # 创建本地临时目录
        local_episode_dir = os.path.join(LOCAL_TEMP_DIR, f"episode_{i}")
        os.makedirs(local_episode_dir, exist_ok=True)

        try:
            # 1. 下载S3数据到本地
            t_start = time.time()
            download_s3_folder(episode_prefix, local_episode_dir)
            print(f"Downloaded in {time.time() - t_start:.2f}s")

            # 2. 转换并上传
            t_convert = time.time()
            episode_length = build_and_upload_parquet(
                src_dir=local_episode_dir,
                episode_idx=i,
                task_index=task_index,
                prompt=task_type.replace("ing_", " ").replace("_", " "),
            )

            # 添加episode元数据
            episodes_data ={
                "episode_index": i,
                "tasks": [task_type.replace("ing_", " ").replace("_", " ")],
                "length": episode_length
            }
            print(f"episodes_data: {episodes_data}")
            # 写入episodes.jsonl文件
            jsonl_path = os.path.join(".", "episodes.jsonl")
            write_episodes_jsonl(episodes_data, jsonl_path)

            print(f"Processed in {time.time() - t_convert:.2f}s")

            # 3. 清理临时文件
            for f in os.listdir(local_episode_dir):
                os.remove(os.path.join(local_episode_dir, f))
            os.rmdir(local_episode_dir)


        except Exception as e:
            print(f"Error processing {episode_prefix}: {str(e)}")
            continue





if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0, help='Start index of episodes to process')
    parser.add_argument('--end', type=int, default=None, help='End index of episodes to process (exclusive)')
    args = parser.parse_args()

    # 确保目标S3路径存在
    try:
        s3.put_object(Bucket=S3_BUCKET, Key=S3_TARGET_PREFIX)
    except ClientError as e:
        print(f"Error creating target prefix: {e}")

    # 获取所有episodes数量
    all_episodes = get_all_episodes()
    if args.end is None:
        args.end = len(all_episodes)

    print(f"Total episodes: {len(all_episodes)}")
    print(f"Processing range: {args.start} to {args.end - 1}")

    # 处理指定范围的数据
    process_episodes(all_episodes, args.start, args.end)

    # 清理临时目录
    try:
        os.rmdir(LOCAL_TEMP_DIR)
    except OSError:
        pass

    print("\nSelected episodes processed and uploaded to S3 successfully!")