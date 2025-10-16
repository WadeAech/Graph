"""
文件名：main.py
概述：实验的顶层入口和“协调器”。
      - 负责解析参数、加载配置、准备所有实验组件。
      - 包含多轮次/多种子实验循环。
      - 在循环中创建并启动Trainer。
      - 在所有实验结束后，进行结果的统计分析并保存最佳模型。
作者：https://github.com/HomerCode
日期：2025/9/16 下午3:00
"""

# coding:UTF-8

import torch
import argparse
import numpy as np
import logging

from scripts.model import get_model
from scripts.train import Trainer
from utils.logger_utils import setup_logging
from utils.data_utils import load_dataset, print_dataset_info, calculate_num_splits, set_seed
from utils.config_utils import load_config
from utils.save_utils import create_run_results_dir, save_model
from utils.model_utils import optimize_with_optuna, print_model_architecture
from utils.metrics_utils import GraphMetrics


def main():
    project_root = '.'

    parser = argparse.ArgumentParser(description="GNN 节点分类实验协调器")
    parser.add_argument('--config', type=str, default='config_0003_01.yaml', help='指定要使用的配置文件名')
    parser.add_argument('--seeds', type=int, nargs='+',
                        default=[524950006, 543913760, 225264556, 835912288, 323678811,
                                 249869747, 230826611, 511767234, 202660586, 824107918],
                        help='设置多个随机种子，由 https://www.avast.com/zh-cn/random-password-generator#pc 生成的9位数强密码')
    parser.add_argument('--device', type=str, default='cuda:0', help='指定计算设备 (e.g., "cpu", "cuda:0")')
    args = parser.parse_args()

    # 加载config配置
    config = load_config(args.config, project_root)

    # 创建本次实验的输出目录
    run_results_dir = create_run_results_dir(args.config, project_root)

    # --- 立刻使用该目录来配置日志系统 ---
    setup_logging(log_dir=run_results_dir)

    # --- 现在，我们可以通过名称获取已配置好的logger ---
    log = logging.getLogger('app')
    log_raw = logging.getLogger('raw')

    log_raw.info(f'实验环境设置')  # config, seeds, device
    log_raw.info('==============================================================')
    log.info(f"0、本次实验的结果将保存于: {run_results_dir}")

    # 加载seeds设置，若实验者不设置随机种子，则使用默认种子
    seeds_to_run = args.seeds if args.seeds is not None else config['training']['seeds']
    log.info(f"1、将使用以下随机种子进行 {len(seeds_to_run)} 轮实验: {seeds_to_run}")

    # 加载device配置
    if args.device == 'cuda:0':
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    log.info(f"2、使用设备: {device}")
    log_raw.info('==============================================================\n')

    log_raw.info(f'数据集设置')  # dataset
    log_raw.info('==============================================================')
    dataset = load_dataset(config['dataset']['name'])
    print_dataset_info(config['dataset']['name'])
    data = dataset[0].to(device)
    log.info(f"数据集 {dataset.name} 加载并移动到 {device} 成功")
    log_raw.info('==============================================================\n')

    val_accs_list = []
    test_accs_list = []
    model_states_list = []

    log_raw.info(f'模型训练')
    log_raw.info('==============================================================')
    # --- 为每轮实验创建全新的模型和优化器，确保独立性 ---
    model = get_model(
        config=config['model'],
        num_features=dataset.num_node_features,
        num_classes=dataset.num_classes
    ).to(device)
    print_model_architecture(model)


if __name__ == '__main__':
    main()
