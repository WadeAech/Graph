"""
文件名：model_utils.py
概述：
作者：https://github.com/HomerCode
日期：2025/9/14 下午4:13
"""

# coding:UTF-8

import os
# import yaml
import logging
import copy
import torch
import optuna
from ruamel.yaml import YAML

from scripts.model import get_model, GenericGNN
from scripts.train import Trainer
from .config_utils import load_config
from utils.data_utils import calculate_num_splits, set_seed

# from itertools import product

"""
说明：
 - 

函数：
 - 

类：
 - 
"""

log = logging.getLogger('app')
log_raw = logging.getLogger('raw')


def print_model_architecture(model: GenericGNN):
    """
    加载配置，创建模型，并打印其结构。

    Args:
        config_filename (str): 存放在 './configs/' 目录下的 YAML 配置文件名。
        :param model:
    """
    # --- 打印模型结构 ---
    log.info(model)


def show_model_architecture(model: GenericGNN):
    log_raw.info(f'模型架构信息')
    log_raw.info('==============================================================')
    print_model_architecture(model)
    log_raw.info('==============================================================\n')


# def grid_search(args, dataset, project_root):
#     base_config = load_config(args.config, project_root)
#
#     # --- 准备用于搜索的数据 (只使用一组固定划分) ---
#     log.info("正在准备用于网格搜索的固定数据划分...")
#
#     data = dataset[0].to(args.device)
#
#     data_for_search = data.clone()
#     num_splits = calculate_num_splits(data)
#     if num_splits > 1:
#         log.info(f"数据集有多组划分，将只使用第1组 (索引0) 进行搜索。")
#         data_for_search.train_mask = data.train_mask[:, 0]
#         data_for_search.val_mask = data.val_mask[:, 0]
#         data_for_search.test_mask = data.test_mask[:, 0]
#     log.info("数据准备完成。")
#
#     # --- 1. 定义要搜索的超参数“网格” ---
#     # 您可以在这里自由定义想要尝试的参数组合
#     param_grid = {
#         'lr': [0.001, 0.005, 0.01, 0.05, 0.1],
#         'wd': [0, 5e-7, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
#         'dropout': [0, 0.1, 0.3, 0.5, 0.7],
#         'hidden_dim': [64, 128, 256, 521]
#     }
#     log.info("定义的超参数网格:")
#     log.info(param_grid)
#
#     # --- 2. 生成所有可能的参数组合 ---
#     keys, values = zip(*param_grid.items())
#     param_combinations = [dict(zip(keys, v)) for v in product(*values)]
#     num_trials = len(param_combinations)
#     log.info(f"网格搜索开始，总共需要执行 {num_trials} 次试验...")
#
#     best_val_acc = -1.0
#     best_params = None
#
#     # --- 3. 遍历每一种组合，执行实验 ---
#     for i, params in enumerate(param_combinations):
#         log_raw.info(f'--- 试验 {i + 1}/{num_trials} | 参数: {params} ---')
#
#         config = copy.deepcopy(base_config)
#         config['training']['lr'] = params['lr']
#         config['training']['wd'] = params['wd']
#         config['model']['dropout'] = params['dropout']
#         for layer_conf in config['model']['layers'][:-1]:
#             layer_conf['hidden_dim'] = params['hidden_dim']
#
#         # 使用您指定的固定种子执行一次实验
#         hpo_seed = 999999999
#         results = run_single_experiment(config, data_for_search, args.device, hpo_seed, dataset.num_node_features,
#                                         dataset.num_classes)
#
#         if results:
#             current_val_acc = results.get('best_val_acc', 0.0)
#             log.info(f"试验 {i + 1}/{num_trials} 完成。验证集准确率: {current_val_acc:.4f}")
#
#             if current_val_acc > best_val_acc:
#                 best_val_acc = current_val_acc
#                 best_params = params
#                 log.info(f"  -> 发现新的最佳参数组合！")
#         else:
#             log.warning(f"试验 {i + 1}/{num_trials} 失败，跳过。")
#
#     # --- 4. 打印最佳结果 ---
#     log_raw.info('--------------- 网格搜索结束 ---------------')
#     if best_params:
#         log.info("最佳验证集准确率 (Best Val Acc): {:.4f}".format(best_val_acc))
#         log.info("找到的最佳超参数 (Best Hyperparameters): ")
#         log.info(best_params)
#
#         # --- 【核心修改点】 ---
#         # 直接读取原始配置文件，更新内容，然后写回
#
#         # 1. 获取原始配置文件的路径
#         original_config_path = os.path.join(project_root, 'configs', args.config)
#         log.info(f"正在将最佳参数写回到原始配置文件: {original_config_path}")
#
#         try:
#             # 2. 读取原始文件的全部内容
#             with open(original_config_path, 'r', encoding='utf-8') as f:
#                 # 使用 yaml.safe_load 来读取，得到一个字典
#                 config_to_update = yaml.safe_load(f)
#
#             # 3. 在该字典中更新找到的最佳参数
#             config_to_update['training']['lr'] = best_params.get('lr')
#             config_to_update['training']['wd'] = best_params.get('wd')
#             config_to_update['model']['dropout'] = best_params.get('dropout')
#             if 'hidden_dim' in best_params:
#                 for layer_conf in config_to_update['model']['layers'][:-1]:
#                     layer_conf['hidden_dim'] = best_params['hidden_dim']
#
#             # 4. 将更新后的字典写回到同一个文件中 (w模式会覆盖)
#             with open(original_config_path, 'w', encoding='utf-8') as f:
#                 # 使用 yaml.dump 将字典写回
#                 yaml.dump(config_to_update, f, allow_unicode=True)
#
#             log.info("配置文件更新成功！")
#
#         except Exception as e:
#             log.error(f"更新配置文件时发生错误: {e}")
#         # --- 修改结束 ---
#
#     else:
#         log.error("所有试验均未能成功，未能找到最佳参数。")
#
#
# def run_single_experiment(config: dict, data_split, device: torch.device, seed: int, num_node_features, num_classes):
#     """
#     执行一次独立的、完整的实验运行。
#     这个函数是所有实验流程（评估、HPO）的最小执行单元。
#
#     参数:
#         config (dict): 完整的配置字典。
#         data_split: 用于本次运行的、已准备好的Data对象（只含单组掩码）。
#         device (torch.device): 计算设备。
#         seed (int): 用于本次运行的随机种子。
#
#     返回:
#         dict or None: 包含本次运行核心结果的字典，如果运行失败则返回None。
#                       例如: {'best_val_acc': ..., 'test_acc_at_best_val': ..., 'best_model_state': ...}
#     """
#     log.info(f"--- 启动单次运行 | 随机种子: {seed} ---")
#
#     # 1. 设置当前运行的随机种子
#     set_seed(seed)
#
#     # 2. 创建全新的模型和优化器，确保每次运行的独立性
#     try:
#         model = get_model(
#             config=config['model'],
#             num_features=num_node_features,
#             num_classes=num_classes
#         ).to(device)
#
#         train_config = config['training']
#         if train_config['optimizer'].lower() == 'adam':
#             optimizer = torch.optim.Adam(
#                 model.parameters(), lr=train_config['lr'], weight_decay=train_config['wd']
#             )
#         else:
#             raise ValueError(f"不支持的优化器: {train_config['optimizer']}")
#     except Exception as e:
#         log.error(f"在种子 {seed} 下创建模型或优化器时失败: {e}")
#         return None
#
#     # 3. 实例化并启动训练引擎
#     try:
#         trainer = Trainer(model, optimizer, data_split, device, train_config)
#         results = trainer.run()
#         return results
#     except Exception as e:
#         log.error(f"在种子 {seed} 下的训练过程中发生错误: {e}")
#         return None


def optimize_with_optuna(args, dataset, project_root):
    base_config = load_config(args.config, project_root)

    # --- 数据准备部分 (保持不变) ---
    log.info("正在准备用于HPO的固定数据划分...")
    data = dataset[0].to(args.device)
    data_for_search = data.clone()
    num_splits = calculate_num_splits(data)
    if num_splits > 1:
        log.info(f"数据集有多组划分，将只使用第1组 (索引0) 进行搜索。")
        data_for_search.train_mask = data.train_mask[:, 0]
        data_for_search.val_mask = data.val_mask[:, 0]
        data_for_search.test_mask = data.test_mask[:, 0]
    log.info("数据准备完成。\n")

    # 1. 定义 objective 函数
    def objective(trial: optuna.Trial):
        # --- 建议超参数 (保持不变) ---
        params = {
            'lr': trial.suggest_loguniform('lr', 0.01, 0.1),
            'wd': trial.suggest_loguniform('wd', 1e-6, 1e-2),
            'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256, 512]),
            # --- 这里是我们新搜索的参数 ---
            'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.7, step=0.05),
        }

        log.info(f'--- Optuna Trial {trial.number} | 参数: {params} ---')

        # --- 更新配置 ---
        config = copy.deepcopy(base_config)
        config['training']['lr'] = params['lr']
        config['training']['wd'] = params['wd']

        # 更新 hidden_dim (保持不变)
        # 注意: 这里的 [:-1] 逻辑可能需要根据您的新模型调整，但我们先聚焦 dropout
        # 新的、正确的逐层更新方式:
        for layer_conf in config['model']['layers']:
            # 我们现在更新所有计算层的hidden_dim
            if layer_conf['type'].lower() in ['gcn', 'gat', 'linear']:
                layer_conf['hidden_dim'] = params['hidden_dim']
            if layer_conf['type'].lower() == 'dropout':
                layer_conf['dropout_rate'] = params['dropout_rate']

        # --- 执行单次实验 (保持不变) ---
        hpo_seed = 999999999
        results = run_single_experiment_for_optuna(config, data_for_search, args.device, hpo_seed,
                                                   dataset.num_node_features, dataset.num_classes, trial)
        if results:
            return results.get('best_val_acc', 0.0)
        return 0.0

    # 2. 创建并运行 Study (保持不变)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=30)
    sampler = optuna.samplers.TPESampler(seed=12345)
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=30)

    # 3. 打印最佳结果并写回配置文件
    log_raw.info('--------------- Optuna 优化结束 ---------------')
    best_trial = study.best_trial
    best_params = best_trial.params
    log.info(f"最佳验证集准确率: {best_trial.value:.4f}")
    log.info(f"找到的最佳超参数: {best_params}")

    original_config_path = os.path.join(project_root, 'configs', args.config)
    log.info(f"正在将最佳参数写回到: {original_config_path}")

    try:
        yaml = YAML()
        yaml.preserve_quotes = True
        with open(original_config_path, 'r', encoding='utf-8') as f:
            config_to_update = yaml.load(f)

        # 更新 lr 和 wd (保持不变)
        config_to_update['training']['lr'] = best_params.get('lr')
        config_to_update['training']['wd'] = best_params.get('wd')

        # 更新 hidden_dim (保持不变)
        if 'hidden_dim' in best_params:
            for layer_conf_inner in config_to_update['model']['layers']:
                if layer_conf_inner.get('type').lower() in ['gcn', 'gat', 'linear']:
                    layer_conf_inner['hidden_dim'] = best_params['hidden_dim']

        # 新的、正确的逐层更新方式:
        if 'dropout_rate' in best_params:
            for layer_conf_inner in config_to_update['model']['layers']:
                if layer_conf_inner.get('type', '').lower() == 'dropout':
                    layer_conf_inner['dropout_rate'] = best_params['dropout_rate']

        with open(original_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_to_update, f)
        log.info("配置文件更新成功！")

    except Exception as e:
        log.error(f"更新配置文件时发生错误: {e}")


def run_single_experiment_for_optuna(config: dict, data_split, device: torch.device, seed: int,
                                     num_node_features, num_classes, trial: optuna.Trial = None):
    # ... (函数内部前半部分不变，创建 model 和 optimizer) ...
    log.info(f"--- 启动单次运行 | 随机种子: {seed} ---")

    # 1. 设置当前运行的随机种子
    set_seed(seed)

    # 2. 创建全新的模型和优化器，确保每次运行的独立性
    try:
        model = get_model(
            config=config['model'],
            num_features=num_node_features,
            num_classes=num_classes
        ).to(device)

        train_config = config['training']
        if train_config['optimizer'].lower() == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(), lr=train_config['lr'], weight_decay=train_config['wd']
            )
        else:
            raise ValueError(f"不支持的优化器: {train_config['optimizer']}")
    except Exception as e:
        log.error(f"在种子 {seed} 下创建模型或优化器时失败: {e}")
        return None

    try:
        # 在这里将 trial 传递给 Trainer
        trainer = Trainer(model, optimizer, data_split, device, train_config, trial)
        results = trainer.run()
        return results
    except Exception as e:
        log.error(f"在种子 {seed} 下的训练过程中发生错误: {e}")
        return None
