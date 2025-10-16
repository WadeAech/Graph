"""
文件名：data_dgl_utils.py
概述：
作者：https://github.com/HomerCode
日期：2025/9/22 上午9:33
"""

# coding:UTF-8

import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import from_dgl
# import dgl
import logging
import os

"""
说明：
 - 

函数：
 - 

类：
 - 
"""

log = logging.getLogger('app')

"""
文件名：data_dgl_utils.py
概述：提供一个通用的数据集加载器，用于处理DGL库中的数据集。
      - 支持将DGL原始数据下载到项目本地目录。
      - 支持通过 '_dgl' 后缀别名来调用。
"""

# coding:UTF-8

import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import from_dgl
# import dgl
import logging
import os

log = logging.getLogger('app')


class DGLDataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        """
        参数:
            root (str): PyG缓存的根目录 (例如 './data/DGL/cora_dgl')。
            name (str): 数据集的别名 (例如 'cora_dgl')。
        """
        # --- 关键改动 1: 解析名称 ---
        # 在初始化时就处理名称，例如 'cora_dgl' -> 'cora'
        self.name_alias = name.lower()
        if self.name_alias.endswith('_dgl'):
            self.name_base = self.name_alias[:-4]
        else:
            self.name_base = self.name_alias

        self._dgl_dataset_class = self._get_dgl_class()
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def _get_dgl_class(self):
        """根据基础名称查找对应的DGL数据集类。"""
        dgl_class_map = {
            'cora': dgl.data.CoraGraphDataset,
            'citeseer': dgl.data.CiteseerGraphDataset,
            'pubmed': dgl.data.PubmedGraphDataset,
            'flickr': dgl.data.FlickrDataset,
        }
        if self.name_base not in dgl_class_map:
            raise ValueError(f"不支持的DGL数据集基础名称: '{self.name_base}'。")
        return dgl_class_map[self.name_base]

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        """触发DGL的数据集下载，并指定下载到我们自己的raw目录。"""
        log.info(f"为数据集 '{self.name_base}' 触发DGL下载器，将保存至: {self.raw_dir}")
        # --- 关键改动 2: 传入 raw_dir 参数 ---
        self._dgl_dataset_class(raw_dir=self.raw_dir)

    def process(self):
        """处理的核心逻辑：从我们自己的raw目录加载DGL数据 -> 转换 -> 校准 -> 保存。"""
        # --- 关键改动 2: 传入 raw_dir 参数 ---
        # 实例化DGL数据集，让它从我们指定的raw_dir加载数据
        dgl_dataset = self._dgl_dataset_class(raw_dir=self.raw_dir)
        dgl_graph = dgl_dataset[0]

        # --- 后续转换和校准逻辑不变 ---
        pyg_data = from_dgl(dgl_graph)

        if hasattr(pyg_data, 'feat'):
            pyg_data.x = pyg_data.feat
            del pyg_data.feat

        if hasattr(pyg_data, 'label'):
            pyg_data.y = pyg_data.label
            del pyg_data.label

        torch.save(self.collate([pyg_data]), self.processed_paths[0])
        log.info(f"DGL数据集 '{self.name_base}' 已成功处理并缓存在PyG目录 '{self.processed_dir}'！")
