"""
文件名：cleanup_processed_folders.py
概述：
作者：https://github.com/HomerCode
日期：2025/10/3 上午10:16
"""

# coding:UTF-8

import shutil
import os

"""
说明：
 - 

函数：
 - 

类：
 - 
"""

# 1. 定义所有需要删除的 'processed' 文件夹路径
folders_to_delete = [
    './data/PyG/Cora/processed',
    './data/PyG/CiteSeer/processed',
    './data/PyG/PubMed/processed',
    './data/PyG/roman_empire/processed',
    './data/PyG/amazon_ratings/processed',
    './data/PyG/minesweeper/processed',
    './data/PyG/tolokers/processed',
    './data/PyG/questions/processed',
    './data/PyG/Computers/processed',
    './data/PyG/Photo/processed',
    './data/PyG/CS/processed',
    './data/PyG/Physics/processed',
    './data/PyG/CoraFull/cora/processed',
    './data/PyG/CoraFull/citeseer/processed',
    './data/PyG/CoraFull/pubmed/processed',
    './data/File/Squirrel-filtered/processed',
    './data/File/Chameleon-filtered/processed',
    './data/File/Cornell/processed',
    './data/File/Texas/processed',
    './data/File/Wisconsin/processed',
    './data/File/Actor/processed',
    './data/PyG/LINKX/genius/processed'
]

print("--- 开始删除指定的 processed 文件夹 ---")

# 2. 遍历列表，逐个删除
deleted_count = 0
skipped_count = 0

for folder_path in folders_to_delete:
    # 检查路径是否存在并且是一个文件夹
    if os.path.isdir(folder_path):
        try:
            shutil.rmtree(folder_path)
            print(f"[已删除] - {folder_path}")
            deleted_count += 1
        except OSError as e:
            print(f"[错误] - 删除 {folder_path} 失败: {e}")
    else:
        print(f"[跳过] - 路径不存在或不是文件夹: {folder_path}")
        skipped_count += 1

print("\n--- 清理完成 ---")
print(f"总计: {deleted_count} 个文件夹被删除，{skipped_count} 个文件夹被跳过。")
