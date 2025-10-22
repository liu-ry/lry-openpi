#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import os
import multiprocessing
from multiprocessing import Pool
import pyarrow.parquet as pq

SOURCE_DIR = '/data/pi0/xdof_converted/chunk-000'
SOURCE_DIR = '/data2/xdof/data/chunk-000'
NUM_PROCESSES = 1  # 总进程数
FILES_PER_PROCESS = 4000  # 每个进程处理的文件数


def get_all_parquet():
    """获取所有parquet文件并排序"""
    all_parquet = []
    for filename in os.listdir(SOURCE_DIR):
        if filename.endswith('.parquet'):
            all_parquet.append(os.path.join(SOURCE_DIR, filename))
    all_parquet.sort()  # 按文件名排序
    return all_parquet


def check_parquet_file(file_path):
    """快速检查Parquet文件是否损坏"""
    try:
        # 仅打开文件元数据，不读取整个文件
        pq.ParquetFile(file_path)
        return (file_path, True, None)
    except Exception as e:
        return (file_path, False, str(e))


def process_files(file_list):
    """处理文件列表，校验每个文件"""
    process_id = multiprocessing.current_process().name
    print(f"进程 {process_id} 开始校验 {len(file_list)} 个文件")

    results = []
    for idx, file_path in enumerate(file_list):
        print(f"进程 {process_id} 正在校验文件 {idx + 1}/{len(file_list)}: {file_path}")
        result = check_parquet_file(file_path)
        results.append(result)

        # 打印校验结果
        status = "完好" if result[1] else f"损坏: {result[2]}"
        print(f"进程 {process_id} 文件 {file_path}: {status}")

    return results


def main():
    # 获取所有parquet文件并排序
    all_parquet_files = get_all_parquet()
    total_files = len(all_parquet_files)
    print(f"找到 {total_files} 个parquet文件")

    # 验证文件总数
    if total_files != NUM_PROCESSES * FILES_PER_PROCESS:
        print(
            f"警告: 文件总数 {total_files} 不等于 {NUM_PROCESSES}×{FILES_PER_PROCESS}={NUM_PROCESSES * FILES_PER_PROCESS}")

    # 将文件分成多个块
    file_chunks = []
    for i in range(0, total_files, FILES_PER_PROCESS):
        file_chunks.append(all_parquet_files[i:i + FILES_PER_PROCESS])

    # 创建进程池并并行处理
    with Pool(processes=NUM_PROCESSES) as pool:
        results = pool.map(process_files, file_chunks)

    # 汇总结果
    all_results = [item for sublist in results for item in sublist]
    valid_files = [r for r in all_results if r[1]]
    invalid_files = [r for r in all_results if not r[1]]

    print(f"\n校验完成:")
    print(f"完好文件: {len(valid_files)}")
    print(f"损坏文件: {len(invalid_files)}")

    if invalid_files:
        print("\n损坏文件列表:")
        for file_path, _, error in invalid_files:
            print(f"- {file_path}: {error}")


if __name__ == '__main__':
    main()