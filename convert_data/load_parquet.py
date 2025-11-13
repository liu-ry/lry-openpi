# -*- coding: utf-8 -*-
# @Time    : 7/10/25 4:39 PM
# @Author  : sunbin
# @File    : load_parquet.py
# @Software: PyCharm
import time

from datasets import load_dataset, Dataset, Features, Image, Value, Sequence
import pyarrow.parquet as pq



def load_parquet():
    parquet_path = '/data/vitai_fold_tshirt/data/chunk-000/episode_003458.parquet'

    loop = 1
    for _ in range(loop):
        dataset = load_dataset(
            "parquet",
            data_files=parquet_path,
            split="train"
        )


        # 打印所有timestamp值
        print("Timestamps:")
        for timestamp in dataset["timestamp"]:
            print(timestamp)
        print(dataset)


def load_parquet_pandas():
    import pandas as pd
    parquet_path = '/data/vitai_vtla_dataset/converted_dataset/converted_clean_table_dataset_with_tactile/data/chunk-000/episode_000000.parquet'
    # 读取单个 Parquet 文件
    df = pd.read_parquet(parquet_path)
    print(f"数据形状: {df.shape}")

    last_row = df.iloc[-1]
    print("最后一行数据:")
    print(last_row)

if __name__ == '__main__':
    load_parquet_pandas()