# -*- coding: utf-8 -*-
# @Time    : 8/6/25 9:26 AM
# @Author  : sunbin
# @File    : find_missing.py
# @Software: PyCharm
import os
"""
find the missing episodes in the folder
"""

def find_missing_episodes(folder_path):
    # 获取所有 .parquet 文件
    files = [f for f in os.listdir(folder_path) if f.endswith('.parquet')]

    # 提取数字部分并转换为整数
    numbers = []
    for f in files:
        try:
            num = int(f.split('_')[1].split('.')[0])
            numbers.append(num)
        except (IndexError, ValueError):
            continue  # 忽略不符合命名规则的文件

    # 排序数字
    numbers.sort()

    # 检查连续性
    missing = []
    for i in range(1, len(numbers)):
        prev_num = numbers[i - 1]
        curr_num = numbers[i]
        if curr_num != prev_num + 1:
            # 发现缺失的数字
            missing.extend(range(prev_num + 1, curr_num))

    # 输出缺失的文件名
    if missing:
        print("发现不连续的文件名，缺失以下文件：")
        for num in missing:
            print(f"episode_{num:06d}.parquet")
    else:
        print("文件名是连续的，没有缺失文件。")


# 示例调用
folder_path = "/data/yam_fold_tshirt/data/chunk-000"  # 替换为你的文件夹路径
find_missing_episodes(folder_path)