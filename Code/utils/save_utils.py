"""
文件名：save_utils.py
概述：提供所有与文件保存相关的辅助函数，包括为每次运行创建专属结果目录。
"""

# coding:UTF-8

import os
import torch
import time
import logging

log = logging.getLogger('app')


def create_run_results_dir(config_filename: str, project_root: str = '.') -> str:
    """
    为本次实验运行创建并返回一个唯一的、带时间戳的结果目录。

    Args:
        config_filename (str): 实验配置的文件名，将用于目录命名。
        project_root (str): 项目的根目录路径。

    Returns:
        str: 本次运行专属的结果保存目录的绝对路径。
    """
    # 1. 获取当前时间戳
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

    # 2. 从配置文件名中提取基础名 (例如 'config_0001.yaml' -> 'config_0001')
    config_basename = os.path.splitext(os.path.basename(config_filename))[0]

    # 3. 拼接成唯一的目录名
    run_dir_name = f"{timestamp}_{config_basename}"

    # 4. 在 'results' 目录下创建这个专属子目录
    #    os.path.join 能智能地处理路径拼接
    run_results_dir = os.path.join(project_root, 'results', run_dir_name)
    os.makedirs(run_results_dir, exist_ok=True)

    return run_results_dir


def save_model(model_state_dict: dict, config: dict, run_results_dir: str):
    """
    将模型的状态字典保存到指定的专属运行结果目录中。

    Args:
        model_state_dict (dict): 要保存的模型的 state_dict。
        config (dict): 实验配置，用于生成具有区分度的文件名。
        run_results_dir (str): 本次运行的专属结果目录路径 (由 create_run_results_dir 生成)。
    """
    try:
        # 1. 构建具体的文件名
        dataset_name = config.get('dataset', {}).get('name', 'unknown_dataset')
        filename = f"best_model_on_{dataset_name}.pt"

        # 2. 拼接为最终的路径
        save_path = os.path.join(run_results_dir, filename)

        # 3. 执行保存操作
        torch.save(model_state_dict, save_path)
        log.info(f"最佳模型权重已成功保存至: {save_path}")

    except Exception as e:
        log.error(f"保存模型时发生错误: {e}")
