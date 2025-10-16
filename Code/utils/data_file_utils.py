"""
文件名：data_npz_utils.py
概述：
作者：https://github.com/HomerCode
日期：2025/9/17 下午6:31
"""

# coding:UTF-8

import os
import numpy as np
import shutil
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.utils import to_undirected

import logging
import scipy.io as sio  # <-- 引用 scipy 來读取 .mat
import pandas as pd
import json

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


class NpzDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None, force_reload=True):
        self.name = name.lower()
        if force_reload:
            processed_dir = os.path.join(root, self.name, 'processed')
            if os.path.exists(processed_dir):
                log.warning(f"检测到 force_reload=True，正在删除旧缓存: {processed_dir}")
                shutil.rmtree(processed_dir)
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def filename_map(self):
        return {
            'squirrel-filtered': 'squirrel_filtered.npz',
            'squirrel-directed': 'squirrel_directed.npz',
            'chameleon': 'chameleon.npz'
            # 在这里添加更多数据集...
        }

    @property
    def raw_file_names(self):
        filename = self.filename_map.get(self.name, f'{self.name}.npz')
        return [filename]

    @property
    def processed_file_names(self):
        # 修正：返回值必须是列表
        return ['data.pt']

    def download(self):
        log.info(f"请确保您的数据集文件位于 '{self.raw_dir}' 文件夹中。")

    def process(self):
        npz_path = self.raw_paths[0]
        with np.load(npz_path, allow_pickle=True) as data:
            node_features = torch.from_numpy(data['node_features'])
            node_labels = torch.from_numpy(data['node_labels'])

            # --- 核心点：直接加载边，无需任何转换 ---
            edges = torch.from_numpy(data['edges']).t().contiguous()
            edges = to_undirected(edges)
            log.info(f"加载的边数量: {edges.size(1)}")

            # --- 不再需要 to_undirected(...) 这一步 ---

            # --- 核心修正点：加载并强制标准化掩码为 [N, k] ---
            def standardize_mask_to_N_k(mask_name):
                if mask_name in data:
                    mask_np = data[mask_name]
                    # 检查是否是二维掩码且维度是 [k, N]
                    # (启发式判断：划分数k通常远小于节点数N)
                    if mask_np.ndim == 2 and mask_np.shape[0] < mask_np.shape[1]:
                        log.info(f"检测到 '{mask_name}' 的形状为 {mask_np.shape} (k, N)，将转置为 (N, k)...")
                        mask_np = mask_np.T  # <--- 执行转置
                        log.info(f"  -> 转置后形状: {mask_np.shape}")
                    return torch.from_numpy(mask_np)
                return None

            train_mask = standardize_mask_to_N_k('train_masks')
            val_mask = standardize_mask_to_N_k('val_masks')
            test_mask = standardize_mask_to_N_k('test_masks')
            # --- 修改结束 ---

        pyg_data = Data(
            x=node_features, edge_index=edges, y=node_labels,
            train_mask=train_mask, val_mask=val_mask, test_mask=test_mask
        )

        torch.save(self.collate([pyg_data]), self.processed_paths[0])
        log.info(f"数据集 '{self.name}' 已成功处理并缓存在 '{self.processed_dir}'！")


class MatDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        """
        通用的 MAT 数据集加载器。

        参数:
            root (str): 数据集专属的根目录 (例如 './data/File/Film')。
            name (str): 数据集的规范名称 (例如 'film')。
        """
        self.name = name.lower()
        super().__init__(root, transform, pre_transform)
        # 从已处理的缓存文件中加载数据
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def filename_map(self):
        """
        将数据集的规范名称映射到具体的.mat文件名。
        """
        # --- 修改点 1: 更新文件名映射 ---
        return {
            'film': 'film.mat'
            # 未来可以继续在这里添加更多 .mat 格式的数据集
        }

    @property
    def raw_file_names(self):
        """根据数据集名称动态返回原始文件名。"""
        filename = self.filename_map.get(self.name, f'{self.name}.mat')
        return [filename]

    @property
    def processed_file_names(self):
        """定义处理后的缓存文件名。"""
        return 'data.pt'

    def download(self):
        """如果原始文件不存在，PyG会调用此方法。我们用它来提示用户。"""
        log.info(f"请确保您的 .mat 数据集文件位于 '{self.raw_dir}' 文件夹中。")

    def process(self):
        """
        处理原始数据的核心逻辑。
        只有在processed/目录下找不到缓存文件时，此方法才会被执行一次。
        """
        # --- 修改点 2: 将 NPZ 读取逻辑换成 MAT 读取逻辑 ---
        mat_path = self.raw_paths[0]

        # 使用scipy库读取.mat文件
        mat_data = sio.loadmat(mat_path)
        log.info(f"成功读取 MAT 文件: {os.path.basename(mat_path)}")

        # 根据我们之前讨论的键名提取数据
        edge_index_np = mat_data['edge_index']
        node_feat_np = mat_data['node_feat']
        label_np = mat_data['label']

        # 格式转换 (NumPy -> PyTorch Tensor)，并确保类型和形状正确
        x = torch.from_numpy(node_feat_np).to(torch.float)
        edge_index = torch.from_numpy(edge_index_np).to(torch.long)
        y = torch.from_numpy(label_np).to(torch.long).squeeze()

        # 创建 PyG 的 Data 对象
        pyg_data = Data(x=x, edge_index=edge_index, y=y)

        # (可选) 检查节点分裂的掩码是否存在
        # if 'train_mask' in mat_data:
        #     pyg_data.train_mask = torch.from_numpy(mat_data['train_mask']).to(torch.bool)
        # if 'val_mask' in mat_data:
        #     pyg_data.val_mask = torch.from_numpy(mat_data['val_mask']).to(torch.bool)
        # if 'test_mask' in mat_data:
        #     pyg_data.test_mask = torch.from_numpy(mat_data['test_mask']).to(torch.bool)

        # 将处理好的 Data 对象保存为PyTorch二进制缓存文件
        torch.save(self.collate([pyg_data]), self.processed_paths[0])
        log.info(f"数据集 '{self.name}' 已成功处理并缓存在 '{self.processed_dir}'！")


class MatCSCDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def filename_map(self):
        # --- 扩展文件名映射 ---
        return {
            'film': 'film.mat',
            'deezer-europe': 'deezer-europe.mat'  # <-- 新增数据集
            # 未来可以继续在这里添加
        }

    @property
    def raw_file_names(self):
        filename = self.filename_map.get(self.name, f'{self.name}.mat')
        return [filename]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        log.info(f"请确保您的 .mat 数据集文件位于 '{self.raw_dir}' 文件夹中。")

    def process(self):
        mat_path = self.raw_paths[0]
        mat_data = sio.loadmat(mat_path)
        log.info(f"成功读取 MAT 文件: {os.path.basename(mat_path)}")
        log.info(f"文件包含的键: {list(mat_data.keys())}")

        # --- 核心修改：根据不同的键来选择不同的处理逻辑 ---
        if 'edge_index' in mat_data and 'node_feat' in mat_data:
            # --- 分支1：处理旧的COO格式 (例如 film.mat) ---
            log.info("检测到'edge_index'和'node_feat'键，按COO格式处理...")
            x = torch.from_numpy(mat_data['node_feat']).to(torch.float)
            edge_index = torch.from_numpy(mat_data['edge_index']).to(torch.long)
            y = torch.from_numpy(mat_data['label']).to(torch.long).squeeze()

        elif 'A' in mat_data and 'features' in mat_data:
            # --- 分支2：处理新的CSC稀疏矩阵格式 (例如 deezer-europe.mat) ---
            log.info("检测到'A'和'features'，按稀疏邻接矩阵格式处理...")

            # 1. 处理稀疏特征矩阵 -> 密集特征张量 x
            features_sparse = mat_data['features']
            x = torch.from_numpy(features_sparse.toarray()).to(torch.float)

            # 2. 处理稀疏邻接矩阵 -> edge_index 张量
            adj_matrix_sparse = mat_data['A']
            edge_index, _ = from_scipy_sparse_matrix(adj_matrix_sparse)

            # 3. 处理标签 y (与之前相同)
            y = torch.from_numpy(mat_data['label']).to(torch.long).squeeze()

        else:
            raise ValueError(
                f"无法识别的 .mat 文件格式！文件中缺少 'edge_index'/'node_feat' 或 'A'/'features' 这些必需的键。")

        # 创建 PyG Data 对象
        pyg_data = Data(x=x, edge_index=edge_index, y=y)

        # 保存为缓存文件 (这部分逻辑通用)
        torch.save(self.collate([pyg_data]), self.processed_paths[0])
        log.info(f"数据集 '{self.name}' 已成功处理并缓存在 '{self.processed_dir}'！")


class CsvJsonDataset(InMemoryDataset):
    # __init__, raw_file_names, processed_file_names, download 方法保持不变
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['musae_DE_edges.csv', 'musae_DE_features.json', 'musae_DE_target.csv']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        log.info(f"请确保您的.csv和.json数据集文件位于 '{self.raw_dir}' 文件夹中。")

    def process(self):
        """
        处理原始的csv和json文件，转换成PyG的Data对象。
        【已更新以处理无标题行的CSV文件】
        """
        # --- 1. 读取节点特征 --- (这部分逻辑不变)
        features_path = self.raw_paths[1]
        with open(features_path, 'r') as f:
            features_json = json.load(f)

        node_ids = sorted([int(nid) for nid in features_json.keys()])
        num_nodes = len(node_ids)
        all_feature_indices = set()
        for feature_list in features_json.values():
            all_feature_indices.update(feature_list)
        num_features = max(all_feature_indices) + 1 if all_feature_indices else 0
        x = torch.zeros((num_nodes, num_features), dtype=torch.float)
        for i, node_id in enumerate(node_ids):
            feature_indices = features_json[str(node_id)]
            if feature_indices:
                x[i, feature_indices] = 1.0

        # --- 2. 读取节点标签 (musae_DE_target.csv) ---
        target_path = self.raw_paths[2]

        # --- 核心修改点 ---
        # 1. 告诉pandas文件没有标题行 (header=None)
        target_df = pd.read_csv(target_path, header=None)
        # 2. 手动为列命名，假设第一列是'id'，第二列是'target'
        target_df.columns = ['id', 'target']
        # --- 修改结束 ---

        # 确保标签与节点顺序一致
        target_df = target_df.sort_values('id')
        y = torch.tensor(target_df['target'].values, dtype=torch.long)

        # --- 3. 读取边 (musae_DE_edges.csv) --- (逻辑不变)
        edges_path = self.raw_paths[0]
        # 同样，我们假设edges文件也没有标题行
        edges_df = pd.read_csv(edges_path, header=None)
        edge_index = torch.tensor(edges_df.values, dtype=torch.long).t().contiguous()

        # --- 后续代码不变 ---
        pyg_data = Data(x=x, edge_index=edge_index, y=y)

        if pyg_data.num_nodes != len(target_df):
            log.warning(f"特征文件和标签文件中的节点数量不匹配！")

        torch.save(self.collate([pyg_data]), self.processed_paths[0])
        log.info(f"数据集 '{self.name}' 已成功处理并缓存在 '{self.processed_dir}'！")
