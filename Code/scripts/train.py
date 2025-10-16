"""
文件名：train.py
概述：包含核心训练和评估逻辑的“训练引擎”。
      - Trainer类被设计为执行一次完整的、单一的实验。
      - 它与文件加载、命令行参数等完全解耦。
作者：https://github.com/HomerCode
日期：2025/9/16 下午2:53
"""

# coding:UTF-8

import optuna
import torch
import torch.nn as nn
import copy
import logging

log = logging.getLogger('app')


class Trainer:
    """
    训练引擎类，负责执行一次完整的训练、验证和测试流程。
    """

    def __init__(self, model: nn.Module, optimizer, data, device: torch.device, config: dict,
                 trial: optuna.Trial = None):
        """
        初始化训练引擎。

        Args:
            model (nn.Module): 实例化的PyTorch模型。
            optimizer: 配置好的优化器。
            data: PyG的图数据对象。
            device (torch.device): 计算设备 (CPU/GPU)。
            config (dict): 包含训练参数的配置字典 (例如, config['training'])。
        """
        self.model = model
        self.optimizer = optimizer
        self.data = data
        self.device = device
        self.config = config
        self.criterion = nn.CrossEntropyLoss()

        # 初始化用于跟踪单次实验最佳性能的指标
        self.best_val_acc = 0.0
        self.corresponding_test_acc = 0.0
        self.corresponding_model_state = None

        self.patience = 300
        self.patience_counter = 0

        self.trial = trial  # 保存 trial 对象

    def _train_one_epoch(self) -> float:
        """
        执行一个完整的训练周期（epoch）。

        Returns:
            float: 该周期的平均训练损失。
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        try:
            out = self.model(self.data.x, self.data.edge_index)
            loss = self.criterion(out[self.data.train_mask], self.data.y[self.data.train_mask])
            loss.backward()
            self.optimizer.step()
            return loss.item()
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                torch.cuda.empty_cache()
                log.error("CUDA内存不足，清空缓存后仍无法继续")
            raise e

    @torch.no_grad()
    def _evaluate_on_split(self, mask: torch.Tensor) -> dict:
        """
        在给定的数据划分（由mask定义）上评估模型。

        Args:
            mask (torch.Tensor): 用于指定数据划分的布尔掩码。

        Returns:
            dict: 包含损失（loss）和准确率（acc）的字典。
        """
        self.model.eval()
        try:
            out = self.model(self.data.x, self.data.edge_index)
            loss = self.criterion(out[mask], self.data.y[mask])

            pred = out.argmax(dim=1)
            correct = (pred[mask] == self.data.y[mask]).sum()
            acc = int(correct) / int(mask.sum())

            return {'loss': loss.item(), 'acc': acc}
        except RuntimeError as e:
            log.error(f"评估过程中发生错误: {e}")
            raise e

    def run(self) -> dict:
        """
        执行完整的训练流程，并返回本次实验的最佳结果。

        Returns:
            dict: 包含本次实验的核心结果，包括：
                  'best_val_acc': 最佳的验证集准确率。
                  'test_acc_at_best_val': 取得最佳验证集准确率时，对应的测试集准确率。
                  'corresponding_model_state': 最佳模型的state_dict。
        """
        log.info("--------------- 开始执行单次实验 ---------------")
        if self.trial:
            log.info(f"本次实验由 Optuna 管理 (Trial {self.trial.number})")
        log.info(f"早停机制已启用, Patience 值设置为: {self.patience}")

        for epoch in range(1, self.config['epochs'] + 1):

            # --- 训练和评估 ---
            train_loss = self._train_one_epoch()
            val_results = self._evaluate_on_split(self.data.val_mask)  # val_results 是字典，现在只使用了acc，里面还有 loss，可以扩展更多
            test_results = self._evaluate_on_split(self.data.test_mask)

            # --- 日志记录 ---
            if epoch % 10 == 0 or epoch == 1:
                log.info(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
                         f"Val Acc: {val_results['acc']:.4f} | Test Acc: {test_results['acc']:.4f}")

            # --- 检查并保存最佳模型 ---
            # 根据我们讨论的结果，实时更新最佳验证集准确率、对应的测试集准确率和模型状态
            if val_results['acc'] > self.best_val_acc:
                self.best_val_acc = val_results['acc']
                self.corresponding_test_acc = test_results['acc']
                # 使用copy.deepcopy确保状态字典被完全复制，而不是一个引用
                self.corresponding_model_state = copy.deepcopy(self.model.state_dict())
                log.info(f"  -> Epoch {epoch:03d} 发现新最佳模型! "
                         f"验证集准确率: {self.best_val_acc:.4f}, "
                         f"对应测试集准确率: {self.corresponding_test_acc:.4f}")
                self.patience_counter = 0
            else:
                # 验证集准确率没有提升，增加计数器
                self.patience_counter += 1

                # 检查是否触发早停
            if self.patience_counter >= self.patience:
                log.info(f"验证集准确率连续 {self.patience} 个 epoch 未提升，触发早停！")
                log.info(f"训练在第 {epoch} 个 epoch 停止。")
                break  # 跳出训练循环

            if self.trial:
                # 1. 向 Optuna 汇报当前 epoch 的验证集准确率
                self.trial.report(val_results['acc'], epoch)

                # 2. 检查 Optuna 是否建议剪枝
                if self.trial.should_prune():
                    log.info(f"Trial {self.trial.number} 在 Epoch {epoch} 被剪枝。\n")
                    # 如果被剪枝，我们也需要返回一个结果，但可以是一个不完整的、表示失败的结果
                    # 这里我们直接返回当前的最佳结果即可
                    return {
                        'best_val_acc': self.best_val_acc,
                        'test_acc_at_best_val': self.corresponding_test_acc,
                        'best_model_state': self.corresponding_model_state
                    }

        log.info("--------------- 单次实验结束 ---------------")
        log.info(
            f"本次实验 最佳验证集准确率: {self.best_val_acc:.4f} 对应的测试集准确率: {self.corresponding_test_acc:.4f}\n")

        # 返回本次实验的核心结果，供 main.py 收集
        return {
            'best_val_acc': self.best_val_acc,
            'test_acc_at_best_val': self.corresponding_test_acc,
            'best_model_state': self.corresponding_model_state
        }
