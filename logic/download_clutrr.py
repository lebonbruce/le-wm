"""下载 CLUTRR 数据到本地"""
import os
import pandas as pd

os.makedirs('/app/logic/clutrr_data', exist_ok=True)

base = 'https://huggingface.co/datasets/CLUTRR/v1/resolve/refs%2Fconvert%2Fparquet/gen_train234_test2to10'
train = pd.read_parquet(f'{base}/train/0000.parquet')
test = pd.read_parquet(f'{base}/test/0000.parquet')

train.to_parquet('/app/logic/clutrr_data/train.parquet')
test.to_parquet('/app/logic/clutrr_data/test.parquet')
print(f'Saved train={len(train)} test={len(test)}')
