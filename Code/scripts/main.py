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
    parser.add_argument('--config', type=str, default='config_0004_01.yaml', help='指定要使用的配置文件名')
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

    log_raw.info(f'度量计算')
    log_raw.info('==============================================================')
    metrics_calculator = GraphMetrics(data)
    metrics_calculator.calculate_all()
    log_raw.info('==============================================================\n')

    log_raw.info(f'超参数搜索')
    log_raw.info('==============================================================')
    # grid_search(args, dataset, project_root)
    # optimize_with_optuna(args, dataset, project_root)
    # return 0
    log_raw.info('==============================================================\n')

    log_raw.info(f'模型训练')
    log_raw.info('==============================================================')
    num_splits = calculate_num_splits(data)
    num_experiments = num_splits
    if num_splits == 1:
        num_experiments = len(args.seeds)
        for i, seed in enumerate(args.seeds):
            log.info(f'第 {i + 1}/{len(args.seeds)} 轮实验 | 随机种子: {seed}')

            # --- 为每轮实验设置新的随机种子 ---
            set_seed(seed)
            log.info(f"已设置随机种子: {seed}")

            # --- 为每轮实验创建全新的模型和优化器，确保独立性 ---
            model = get_model(
                config=config['model'],
                num_features=dataset.num_node_features,
                num_classes=dataset.num_classes
            ).to(device)
            print_model_architecture(model)

            train_config = config['training']
            if train_config['optimizer'].lower() == 'adam':
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=train_config['lr'], weight_decay=train_config['wd']
                )
            else:
                raise ValueError(f"不支持的优化器: {train_config['optimizer']}")

            # --- 实例化并启动单次实验引擎 ---
            trainer = Trainer(model, optimizer, data, device, train_config)
            single_run_results = trainer.run()

            # --- 收集该轮实验的结果 ---
            if single_run_results and single_run_results['best_model_state']:
                val_accs_list.append(single_run_results['best_val_acc'])
                test_accs_list.append(single_run_results['test_acc_at_best_val'])
                model_states_list.append(single_run_results['best_model_state'])
            else:
                log.warning(f"种子 {seed} 的实验未能成功返回结果，将跳过。")
        log_raw.info('==============================================================\n')
    else:
        for split_idx in range(num_splits):
            log.info(f'第 {split_idx + 1}/{num_splits} 组数据划分')

            # 1. 为当前划分准备数据
            data_for_this_run = data.clone()
            data_for_this_run.train_mask = data.train_mask[:, split_idx]
            data_for_this_run.val_mask = data.val_mask[:, split_idx]
            data_for_this_run.test_mask = data.test_mask[:, split_idx]

            # 2. 设置种子并重新初始化模型/优化器
            #    确保每次划分的起点（模型权重）都完全相同
            set_seed(args.seeds[0])

            model = get_model(config['model'], num_features=data.num_node_features, num_classes=dataset.num_classes).to(
                device)
            print_model_architecture(model)
            train_config = config['training']
            optimizer = torch.optim.Adam(
                model.parameters(), lr=train_config['lr'], weight_decay=train_config['wd']
            )

            # 3. 实例化并启动训练引擎
            trainer = Trainer(model, optimizer, data_for_this_run, device, train_config)
            single_run_results = trainer.run()

            # 4. 收集该轮划分的结果
            if single_run_results and single_run_results['best_model_state']:
                val_accs_list.append(single_run_results['best_val_acc'])
                test_accs_list.append(single_run_results['test_acc_at_best_val'])
                model_states_list.append(single_run_results['best_model_state'])
            else:
                log.warning(f"划分 {split_idx + 1} 的实验未能成功返回结果，将跳过。")
            # --- 结果聚合与分析 (模仿 code.py 的风格) ---
        log_raw.info(f'==============================================================\n')
        if not test_accs_list:
            log.error("所有划分的实验均未能成功，无法进行总结分析。")
            return None

    log_raw.info(f'训练结果的聚合与分析')
    log_raw.info('==============================================================')

    if not test_accs_list:
        log.error("所有实验均未能成功，无法进行总结分析。")
        return

    # --- 计算并打印统计数据 ---
    test_accs_np = np.array(test_accs_list)
    mean_acc = np.mean(test_accs_np)
    std_acc = np.std(test_accs_np)
    if num_splits > 1:
        log.info(f"在 {num_splits} 个数据划分上，测试集准确率:")
    else:
        log.info(f"在 {num_experiments} 个不同随机种子上，测试集准确率:")
    log.info(f"均值 (Mean): {mean_acc:.4f}")
    log.info(f"标准差 (Std): {std_acc:.4f}")

    # --- 找到并保存最佳模型 ---
    best_run_idx = np.argmax(val_accs_list)
    best_val_acc = val_accs_list[best_run_idx]
    corresponding_test_acc = test_accs_list[best_run_idx]
    best_model_state = model_states_list[best_run_idx]

    log.info(f"表现最佳的一轮是在验证集上取得: {best_val_acc:.4f} (对应测试集准确率: {corresponding_test_acc:.4f})")

    save_model(best_model_state, config, run_results_dir)
    # 保存结果
    log_raw.info(f'==============================================================\n')


if __name__ == '__main__':
    main()
