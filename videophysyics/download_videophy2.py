#!/usr/bin/env python
"""下载 VIDEOPHY2 测试集 prompts 并导出为 CSV"""
import os
import csv
from datasets import load_dataset

SAVE_DIR = "./videophy_data"
os.makedirs(SAVE_DIR, exist_ok=True)

print("Downloading videophysics/videophy2_test from HuggingFace...")
ds = load_dataset("videophysics/videophy2_test")

# 打印数据集结构
print(f"\nDataset: {ds}")
for split in ds:
    print(f"\nSplit: {split}, Size: {len(ds[split])}")
    print(f"Columns: {ds[split].column_names}")
    print(f"First 3 samples:")
    for i, item in enumerate(ds[split]):
        if i >= 3:
            break
        print(f"  [{i}] {item}")

# 导出为 CSV
split = list(ds.keys())[0]
data = ds[split]
columns = data.column_names

csv_path = os.path.join(SAVE_DIR, "videophy2_test.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    for item in data:
        writer.writerow(item)

print(f"\nSaved {len(data)} prompts to {csv_path}")
print(f"Columns: {columns}")
