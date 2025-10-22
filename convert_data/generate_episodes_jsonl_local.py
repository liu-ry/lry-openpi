# -*- coding: utf-8 -*-
# @Time    : 7/5/25 1:50 PM
# @Author  : sunbin
# @File    : generate_episodes_jsonl.py
# @Software: PyCharm
import os
import time

import numpy as np
import json


SOURCE_DIR = '/data/vitai/'
# SOURCE_DIR = '/mnt/pi0/'

TASK_TYPES = ['folding_tshirt_pile_and_stacking', 'folding_tshirt']


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


def write_episodes_jsonl(episodes_data, output_path):
    """写入episodes.jsonl文件"""
    with open(output_path, 'a') as f:
        json_line = json.dumps(episodes_data)
        f.write(json_line + '\n')
    print(f"Saved episodes metadata to {output_path}")


def generate_episodes_jsonl():
    all_episodes = get_all_episodes()
    count = len(all_episodes)
    count = 4000
    print(f"generating episodes.jsonl (total {count})")
    total_frames = 0
    for i in range(count):
        episode_info = all_episodes[i]
        episode_prefix = episode_info['prefix']
        task_type = episode_info['task_type']
        print(f"\nProcessing {episode_prefix} ({i + 1}/{count})...")
        t_start = time.time()
        print(f"Downloaded in {time.time() - t_start:.2f}s")
        timestamps = np.load(os.path.join(episode_prefix, "timestamp.npy"))
        print(f'len(timestamps) {len(timestamps)}')

        # 添加episode元数据
        episodes_data = {
            "episode_index": i,
            "tasks": [task_type.replace("ing_", " ").replace("_", " ")],
            "length": len(timestamps)
        }
        total_frames += len(timestamps)
        print(f"episodes_data: {episodes_data}")
        # 写入episodes.jsonl文件
        jsonl_path = os.path.join(".", "episodes.jsonl")
        write_episodes_jsonl(episodes_data, jsonl_path)


    print(f"Total frames: {total_frames}, ~ {total_frames / 30 / 3600} h.")



if __name__ == '__main__':
    generate_episodes_jsonl()