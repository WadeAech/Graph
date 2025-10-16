"""
文件名：data_utils.py
概述：
作者：https://github.com/HomerCode
日期：2025/9/11 下午2:57
"""

# coding:UTF-8

import os

import logging
import numpy as np
# import pandas as pd
import networkx as nx
import scipy.io as sio
import scipy.sparse as sp
import random
from tqdm import tqdm
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
# import dgl
# import dgl.data

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import Compose, LargestConnectedComponents
from torch_geometric.datasets import (
    Planetoid,
    HeterophilousGraphDataset,
    WebKB,
    Actor,
    Flickr,
    Amazon,
    Coauthor,
    WikiCS,
    AttributedGraphDataset,
    LINKXDataset,
    Twitch,
    CitationFull,
    KarateClub,
)
from torch_geometric.utils import to_networkx, degree  # , from_dgl
from torch_geometric.transforms import BaseTransform, RandomNodeSplit
from .data_file_utils import NpzDataset, MatDataset, MatCSCDataset, CsvJsonDataset
from .data_dgl_utils import DGLDataset

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

# 【控制中心】
# 定义数据集名称到其文件夹名称的映射
DATASET_FOLDER_MAP_NPZ = {
    'squirrel_directed': 'Squirrel-directed',
    'squirrel': 'Squirrel',
    'squirrel_filtered': 'Squirrel-filtered',
    'squirrel_filtered_directed': 'Squirrel-filtered-directed',

    'chameleon_directed': 'Chameleon-directed',
    'chameleon': 'Chameleon',
    'chameleon_filtered': 'Chameleon-filtered',
    'chameleon_filtered_directed': 'Chameleon-filtered-directed',
    'cornell': 'Cornell',
    'texas': 'Texas',
    'texas_4_classes': 'Texas-4-classes',
    'wisconsin': 'Wisconsin',
    'actor': 'Actor',
    'corafull': 'CoraFull',

    # 在这里添加更多数据集...
    # 'my-new-dataset': 'MyNewDataset-Folder'
}

DATASET_FOLDER_MAP_MAT = {
    'film': 'Film',
    'genius-v2': 'Genius-v2',
}
DATASET_FOLDER_MAP_MAT_CSC = {
    'deezer-europe': 'Deezer-europe',
    # 'penn94': 'Penn94'
}
DATASET_FOLDER_MAP_CSV_JSON = {
    'twitch-de': 'Twitch-DE'
}
OGB_NODE_DATASETS = ['ogbn-arxiv', 'ogbn-products', 'ogbn-mag']

NPZ_DATASETS = list(DATASET_FOLDER_MAP_NPZ.keys())

# 使用一个全局变量来确保补丁只被应用一次
_PATCH_APPLIED = False


class GenerateMultiSplits(BaseTransform):
    """
    一个自定义的transform，用于为数据对象生成并附加多组随机划分掩码。

    将生成 train_mask, val_mask, test_mask 属性，
    其形状为 [num_nodes, num_splits]。
    """

    def __init__(self, num_splits=10, train_ratio=0.6, val_ratio=0.2):
        self.num_splits = num_splits
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        # 测试集比例将是 1.0 - train_ratio - val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio

    def __call__(self, data):
        # --- 关键修正点：在开始处理前，先清理掉data对象上任何可能存在的旧掩码 ---
        # 这确保了RandomNodeSplit总是在一个“干净”的对象上工作
        if hasattr(data, 'train_mask'):
            del data.train_mask
        if hasattr(data, 'val_mask'):
            del data.val_mask
        if hasattr(data, 'test_mask'):
            del data.test_mask
        # --- 清理结束 ---

        train_masks, val_masks, test_masks = [], [], []

        for i in range(self.num_splits):
            # 现在 splitter 接收到的 data.clone() 是不带任何掩码属性的
            splitter = RandomNodeSplit(
                split='random',
                num_val=self.val_ratio,
                num_test=self.test_ratio
            )
            split_data = splitter(data.clone())

            train_masks.append(split_data.train_mask)
            val_masks.append(split_data.val_mask)
            test_masks.append(split_data.test_mask)

        # 按照 [num_nodes, num_splits] 的约定俗成格式进行堆叠
        data.train_mask = torch.stack(train_masks, dim=1)
        data.val_mask = torch.stack(val_masks, dim=1)
        data.test_mask = torch.stack(test_masks, dim=1)

        return data


class GenerateFixedPerClassSplits(BaseTransform):
    """
    一个自定义transform，用于生成按“每类节点数”固定的多组划分。
    适用于Amazon, Coauthor等需要固定训练/验证集大小的数据集。
    """

    def __init__(self, num_splits=10, num_train_per_class=20, num_val_per_class=30):
        self.num_splits = num_splits
        self.num_train_per_class = num_train_per_class
        self.num_val_per_class = num_val_per_class

    def __call__(self, data):
        # 清理旧掩码，确保一个干净的起点
        if hasattr(data, 'train_mask'):
            del data.train_mask
        if hasattr(data, 'val_mask'):
            del data.val_mask
        if hasattr(data, 'test_mask'):
            del data.test_mask

        train_masks, val_masks, test_masks = [], [], []

        for i in range(self.num_splits):
            # 关键：使用num_train_per_class和num_val_per_class参数
            splitter = RandomNodeSplit(
                split='train_rest',  # 剩余部分全部作为测试集
                num_train_per_class=self.num_train_per_class,
                num_val_per_class=self.num_val_per_class
            )
            split_data = splitter(data.clone())

            train_masks.append(split_data.train_mask)
            val_masks.append(split_data.val_mask)
            test_masks.append(split_data.test_mask)

        # 统一为 [N, k] 格式
        data.train_mask = torch.stack(train_masks, dim=1)
        data.val_mask = torch.stack(val_masks, dim=1)
        data.test_mask = torch.stack(test_masks, dim=1)

        return data


def set_seed(seed: int):
    """设置随机种子以确保实验的可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 确保 cudnn 的确定性，可能会牺牲一些性能
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def apply_torch_load_patch():
    """
    应用一个“猴子补丁”来修复 torch.load 函数
    在新版 PyTorch (>=2.6) 中 weights_only 的默认值变化问题。

    这个补丁会全局性地修改 torch.load 的行为，
    使其默认使用 weights_only=False，从而兼容那些
    尚未更新以适应此变化的旧库（如 ogb）或代码。
    """
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    log.info("正在应用 torch.load 的猴子补丁以兼容旧版库...")

    # 1. 保存原始的 torch.load 函数
    _original_torch_load = torch.load

    # 2. 定义一个新的、我们自己的 load 函数
    def _patched_torch_load(f, *args, **kwargs):
        """
        一个包装函数，它在调用原始的 torch.load 之前，
        自动将 'weights_only' 设置为 False。
        """
        # 强制设置 weights_only=False，除非用户在调用时明确指定了其他值
        kwargs.setdefault('weights_only', False)
        return _original_torch_load(f, *args, **kwargs)

    # 3. 用我们的新函数替换掉 PyTorch 中原始的 load 函数
    #    这样，当任何代码（包括第三方库）在内部调用 torch.load 时，
    #    它实际上调用的是我们这个版本
    torch.load = _patched_torch_load

    # 【核心修改】移除了下面这行导致问题的代码
    # if hasattr(torch.serialization, '_load'):
    #     torch.serialization._load = _patched_torch_load

    _PATCH_APPLIED = True
    log.info("torch.load 补丁应用成功！")


def load_dataset(name):
    dataset = None  # 初始化 dataset 防止未定义错误

    old_transforms = Compose([
        LargestConnectedComponents(),
        RandomNodeSplit(split='train_rest', num_val=0.2, num_test=0.2)
    ])

    none_transforms = Compose([])

    transform_622 = GenerateMultiSplits(num_splits=10, train_ratio=0.6, val_ratio=0.2)
    transform_622 = Compose([LargestConnectedComponents(), transform_622])
    transform_522 = GenerateMultiSplits(num_splits=10, train_ratio=0.5, val_ratio=0.25)
    transform_522 = Compose([LargestConnectedComponents(), transform_522])
    transform_23o = GenerateMultiSplits(num_splits=10, train_ratio=0.2, val_ratio=0.2)
    transform_23o = Compose([LargestConnectedComponents(), transform_23o])

    try:
        if name in ['Cora', 'CiteSeer', 'PubMed']:
            dataset = Planetoid(root='./data/PyG', name=name, transform=old_transforms, force_reload=True)
        elif name in ['Roman-empire', 'Amazon-ratings', 'Minesweeper', 'Tolokers', 'Questions']:
            dataset = HeterophilousGraphDataset(root='./data/PyG', name=name, transform=none_transforms,
                                                force_reload=True)
        elif name in ['Cornell', 'Texas', 'Wisconsin']:
            dataset = WebKB(root='./data/PyG', name=name, transform=none_transforms)
        elif name in ['Actor']:
            dataset = Actor(root=f"./data/PyG/Actor", transform=none_transforms)
        elif name in ['Flickr']:
            dataset = Flickr(root=f"./data/PyG/Flickr", transform=transform_522)
        elif name in ['CoraFull', 'CiteSeerFull', 'PubMedFull', 'Cora_ML', 'DBLP']:
            if name == 'CoraFull':
                name = 'Cora'
            elif name == 'CiteSeerFull':
                name = 'CiteSeer'
            elif name == 'PubMedFull':
                name = 'PubMed'
            else:
                name = name
            dataset = CitationFull(root='./data/PyG/CoraFull', name=name, transform=transform_622, force_reload=True)
        elif name in ['Computers', 'Photo']:
            dataset = Amazon(root='./data/PyG', name=name, transform=transform_622, force_reload=True)
        elif name in ['CS', 'Physics']:
            dataset = Coauthor(root='./data/PyG', name=name, transform=transform_622, force_reload=True)
        elif name in ['WikiCS']:
            dataset = WikiCS(root='./data/PyG/WikiCS', transform=old_transforms)
        elif name in ['BlogCatalog']:
            dataset = AttributedGraphDataset(root='./data/PyG/AttributedGraphDataset', name=name,
                                             transform=none_transforms)
        elif name in ['genius']:
            temp_transform = Compose([T.ToUndirected(), transform_622])
            dataset = LINKXDataset(root='./data/PyG/LINKX', name=name, transform=temp_transform, force_reload=True)
        elif name in ['DE', 'EN']:
            dataset = Twitch(root='./data/PyG/Twitch', name=name, transform=none_transforms)
        elif name in NPZ_DATASETS:
            # 1. 从映射中获取正确的文件夹名
            folder_name = DATASET_FOLDER_MAP_NPZ[name]
            # 2. 构建到该数据集专属根目录的完整路径
            dataset_path = os.path.join('./data/File', folder_name)
            # 3. 将完整路径和规范名称传给 NpzDataset
            if name in ['squirrel_filtered', 'chameleon_filtered', 'cornell', 'texas', 'wisconsin', 'actor']:
                dataset = NpzDataset(root=dataset_path, name=name, transform=transform_622, force_reload=True)
            else:
                dataset = NpzDataset(root=dataset_path, name=name, transform=transform_622)
            return dataset
        elif name in ['KarateClub']:
            dataset = KarateClub(transform=none_transforms)
        elif name.endswith('_dgl'):
            # root目录现在也使用别名，确保每个数据集的缓存独立
            root_dir = f'./data/DGL/{name}'
            log.info(f"检测到 '_dgl' 后缀，将使用 DGLDataset 加载器。")
            dataset = DGLDataset(root=root_dir, name=name)
        elif name in OGB_NODE_DATASETS:
            log.info(f"检测到OGB数据集: {name}，开始加载...")
            root_dir = f'./data/OGB'

            # 1. 正常实例化OGB数据集对象。
            #    这会从磁盘加载缓存（无论是否带有掩码）。
            dataset = PygNodePropPredDataset(name=name, root=root_dir)

            # 2. 直接检查数据集内部持有的 .data 属性
            #    这是确保我们检查和修改的是同一个对象的关键
            if not hasattr(dataset.data, 'train_mask') or dataset.data.train_mask is None:
                log.info("未在缓存中检测到布尔掩码，开始从OGB官方索引进行转换...")

                data_object_to_modify = dataset.data
                split_idx = dataset.get_idx_split()
                train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']

                num_nodes = data_object_to_modify.num_nodes
                train_mask = torch.zeros(num_nodes, dtype=torch.bool)
                val_mask = torch.zeros(num_nodes, dtype=torch.bool)
                test_mask = torch.zeros(num_nodes, dtype=torch.bool)

                train_mask[train_idx] = True
                val_mask[valid_idx] = True
                test_mask[test_idx] = True

                # 3. 直接在数据集的内部 .data 对象上附加属性
                data_object_to_modify.train_mask = train_mask
                data_object_to_modify.val_mask = val_mask
                data_object_to_modify.test_mask = test_mask

                # 4. 重新保存这个被修改过的、现在带有掩码的数据集
                processed_path = dataset.processed_paths[0]
                log.info(f"转换完成，正在将带有掩码的新数据覆写回缓存文件: {processed_path}")
                # dataset.collate 会处理 dataset.data
                torch.save(dataset.collate([dataset.data]), processed_path)
                log.info("缓存覆写成功！")
            else:
                log.info("检测到已存在的布尔掩码，直接使用缓存。")

            # 2. 【核心修正】检查并修正y的维度
            #    这个检查只在需要时执行一次
            data = dataset.data
            if hasattr(data, 'y') and data.y is not None and data.y.dim() > 1:
                log.info(f"检测到 OGB 数据集的标签 y 是二维的 (形状: {data.y.shape})，正在进行修正...")

                # 使用 .squeeze() 将 [N, 1] 形状的张量“压扁”成 [N]
                data.y = data.y.squeeze()

                log.info(f"  -> 修正后形状: {data.y.shape}")

                # 3. 将修正后的数据覆写回缓存，一劳永逸
                processed_path = dataset.processed_paths[0]
                log.info(f"正在将修正后的数据覆写回缓存文件: {processed_path}")
                torch.save(dataset.collate([data]), processed_path)
                log.info("缓存覆写成功！")

            # 5. 返回这个在内存中也确保被更新了的dataset对象
            return dataset

        elif name in DATASET_FOLDER_MAP_MAT:
            # 1. 从映射中获取正确的文件夹名
            folder_name = DATASET_FOLDER_MAP_MAT[name]
            # 2. 构建到该数据集专属根目录的完整路径
            dataset_path = os.path.join('./data/File', folder_name)
            # 3. 将完整路径和规范名称传给 MatDataset
            dataset = MatDataset(root=dataset_path, name=name, transform=transform_622)
        elif name in DATASET_FOLDER_MAP_MAT_CSC:
            folder_name = DATASET_FOLDER_MAP_MAT_CSC[name]
            dataset_path = os.path.join('./data/File', folder_name)
            dataset = MatCSCDataset(root=dataset_path, name=name)
            return dataset
        elif name in DATASET_FOLDER_MAP_CSV_JSON:
            folder_name = DATASET_FOLDER_MAP_CSV_JSON[name]
            dataset_path = os.path.join('./data/File', folder_name)
            dataset = CsvJsonDataset(root=dataset_path, name=name)

        if dataset is None:
            print("数据集不存在")
            raise NotImplementedError

        return dataset
    except Exception as e:
        log.error(f"加载数据集 '{name}' 时发生错误: {e}")


def print_dataset_info(dataset_name='Cora'):
    """
    按照指定格式和顺序，清晰地打印数据集的各项核心指标。
    """
    # --- 1. 数据加载与预计算 ---
    dataset = load_dataset(dataset_name)

    data = dataset[0]

    in_degree = degree(data.edge_index[1], num_nodes=data.num_nodes)
    out_degree = degree(data.edge_index[0], num_nodes=data.num_nodes)
    edge_analysis = analyze_edge_types(data)

    # 【修改处】接收第三个返回值：边的数量
    wcc_num, wcc_nodes, wcc_edges, other_nodes, other_edges = calculate_connected_component(data)

    # --- 2. 格式化输出 ---
    log.info(f'数据集: {dataset_name}')
    log.info(f"图中节点的数量: {data.num_nodes}")
    log.info(f"图中边的数量: {data.num_edges}")

    log.info("边的类型构成:")
    log.info(f"  - 自环: {edge_analysis['self_loops']}")
    log.info(f"  - 双向边 (对): {edge_analysis['reciprocal_pairs']}")
    log.info(f"  - 纯单向边: {edge_analysis['unidirectional_edges']}")

    log.info(f"节点的特征维度: {data.num_node_features}")
    log.info(f"节点标签的类别数量: {dataset.num_classes}")
    log.info(f"是否为有向图: {data.is_directed()}")

    log.info(f"图中的弱连通分量数: {wcc_num}")
    log.info(f"(弱)最大连通分量中节点的数量: {wcc_nodes}")
    log.info(f"(弱)最大连通分量中边的数量: {wcc_edges}")

    # 【新增行】打印其余分量的信息
    if wcc_num > 1:
        log.info(f"  - 其余 {wcc_num - 1} 个连通分量中节点的总数: {other_nodes}")
        log.info(f"  - 其余 {wcc_num - 1} 个连通分量中边的总数: {other_edges}")

    isolated_nodes_count = (in_degree + out_degree == 0).sum().item()
    log.info(f"图中是否包含孤立节点: {data.has_isolated_nodes()}，数量为: {isolated_nodes_count}")

    log.info("度的分布情况:")
    log.info(f"  - 源节点 (入度为0, 出度>0): {(in_degree == 0).logical_and(out_degree > 0).sum().item()}")
    log.info(f"  - 汇节点 (出度为0, 入度>0): {(out_degree == 0).logical_and(in_degree > 0).sum().item()}")

    log.info(f"图中是否包含自环边: {data.has_self_loops()}，数量为: {edge_analysis['self_loops']}")
    log.info(f'-------------------------')

    log.info(f'节点特征矩阵:\n\t{data.x.shape}\n\t{data.x}')
    log.info(f'邻接列表:\n\t{data.edge_index.shape}\n\t{data.edge_index}')
    log.info(f'节点标签:\n\t{data.y.shape}\n\t{data.y}')
    if hasattr(data, 'train_mask'):
        log.info(f'训练集掩码:\n\t{data.train_mask.shape}\n\t{data.train_mask}')
    else:
        log.info('训练集掩码: N/A (数据集中未提供)')

    if hasattr(data, 'val_mask'):
        log.info(f'验证集掩码:\n\t{data.val_mask.shape}\n\t{data.val_mask}')
    else:
        log.info('验证集掩码: N/A (数据集中未提供)')

    if hasattr(data, 'test_mask'):
        log.info(f'测试集掩码:\n\t{data.test_mask.shape}\n\t{data.test_mask}')
    else:
        log.info('测试集掩码: N/A (数据集中未提供)')


def show_dataset_info(dataset='Cora'):
    log_raw.info(f'数据集信息')
    log_raw.info('==============================================================')
    print_dataset_info(dataset)
    log_raw.info('==============================================================\n')


def analyze_edge_types(data):
    """精确分析有向图的边类型构成"""
    num_total_edges = data.num_edges
    is_self_loop = (data.edge_index[0] == data.edge_index[1])
    num_self_loops = torch.sum(is_self_loop).item()

    edges_set = set()
    edge_index_t = data.edge_index.t()
    for i in range(edge_index_t.size(0)):
        u, v = edge_index_t[i].tolist()
        if u != v:
            edges_set.add((u, v))

    num_reciprocal_edges = 0
    for u, v in edges_set:
        if (v, u) in edges_set:
            num_reciprocal_edges += 1

    num_reciprocal_pairs = num_reciprocal_edges // 2
    num_unidirectional = num_total_edges - num_self_loops - (num_reciprocal_pairs * 2)

    return {
        "self_loops": num_self_loops,
        "reciprocal_pairs": num_reciprocal_pairs,
        "unidirectional_edges": num_unidirectional
    }


def calculate_connected_component(data):
    """
    计算弱连通分量，并通过直接遍历统计LCC和其他分量的信息。

    返回:
        - 分量总数 (num_components)
        - LCC 的节点数 (num_nodes_in_lcc)
        - LCC 的边数 (num_edges_in_lcc)
        - 其余所有分量合并后的节点总数 (num_nodes_in_others)
        - 其余所有分量合并后的边总数 (num_edges_in_others)
    """
    g = to_networkx(data, to_undirected=True)

    if g.number_of_nodes() == 0:
        return 0, 0, 0, 0, 0

    # 1. 获取所有连通分量
    connected_components = list(nx.connected_components(g))
    num_components = len(connected_components)

    if num_components == 0:
        return 0, 0, 0, 0, 0

    # 2. 直接统计 LCC 的信息 (逻辑不变)
    largest_component_nodes = max(connected_components, key=len)
    num_nodes_in_lcc = len(largest_component_nodes)
    lcc_subgraph = g.subgraph(largest_component_nodes)
    num_edges_in_lcc = lcc_subgraph.number_of_edges()

    # 3. 【核心修改】通过直接遍历和累加，来统计其余所有分量
    num_nodes_in_others = 0
    num_edges_in_others = 0

    # 识别出除了LCC之外的其他所有分量
    other_components = [c for c in connected_components if c != largest_component_nodes]

    # 如果存在其他分量，则遍历它们
    if other_components:
        # 使用tqdm来显示处理进度，以防分量过多时耗时较长
        for component_nodes in tqdm(other_components, desc="正在统计其余连通分量", leave=False):
            # 累加节点数
            num_nodes_in_others += len(component_nodes)
            # 创建子图并累加边数
            subgraph = g.subgraph(component_nodes)
            num_edges_in_others += subgraph.number_of_edges()

    return num_components, num_nodes_in_lcc, num_edges_in_lcc, num_nodes_in_others, num_edges_in_others


def calculate_num_splits(data):
    """
    一个生成器函数，用于迭代数据对象中包含的所有数据划分。

    它能自动检测数据是包含单组划分还是多组划分。
    在每次迭代中，它会yield一个元组，包含(划分索引, 为该划分准备好的Data对象)。

    参数:
        data (Data): 包含一个或多个划分掩码的PyG Data对象。

    Yields:
        tuple[int, Data]: (当前划分的索引, 准备好的Data对象)。
    """
    if hasattr(data, 'train_mask') and isinstance(data.train_mask, torch.Tensor):
        # 如果张量是二维的，我们认为它包含多组划分
        if data.train_mask.dim() > 1:
            # 根据您的声明，维度是 [划分数, 节点数]，所以我们取第0维的大小
            log.info(f"数据集 {data} 包含 {data.train_mask.shape[1]} 个划分。")
            return data.train_mask.shape[1]

        # 如果train_mask不存在，或者是一维的，都统一视为只有1组划分
    return 1


def standardize_mask_shape(data: Data) -> Data:
    """
    检查Data对象中的掩码，如果其形状为[N, k]，则将其转置为[k, N]。

    参数:
        data (Data): 需要检查和标准化的PyG Data对象。

    返回:
        Data: 经过标准化处理后的Data对象。
    """
    # 检查train_mask属性是否存在且是一个二维的PyTorch张量
    if hasattr(data, 'train_mask') and isinstance(data, Data) and data.train_mask.dim() > 1:
        # PyG约定俗成的格式是[N, k]，但我们的框架需要[k, N]
        # 如果行数(N)大于列数(k)，说明它很可能是[N, k]格式，需要转置
        if data.train_mask.shape[0] > data.train_mask.shape[1]:
            log.info(f"检测到掩码维度为 {data.train_mask.shape}，正在转置以匹配 [num_splits, num_nodes] 格式...")

            # 使用 .t() 方法进行转置
            data.train_mask = data.train_mask.t().contiguous()
            if hasattr(data, 'val_mask'):
                data.val_mask = data.val_mask.t().contiguous()
            if hasattr(data, 'test_mask'):
                data.test_mask = data.test_mask.t().contiguous()

            log.info(f"  -> 转置后维度为: {data.train_mask.shape}")

    return data


def print_npz_file_info(dataset_name: str, base_path: str = './data/File'):
    """
    加载并打印指定 NPZ 数据集文件的内部键值对信息。

    此函数自动通过 DATASET_FOLDER_MAP 配置来定位文件。
    它假设 .npz 文件名与 dataset_name 相同。

    Args:
        dataset_name (str): 数据集的逻辑名称 (例如 'squirrel_directed')。
        base_path (str): 存放所有数据集文件夹的根目录。
    """

    log.info(f'npz文件: {dataset_name}')

    # 1. 从配置中查找文件夹名称
    folder_name = DATASET_FOLDER_MAP_NPZ.get(dataset_name)
    if not folder_name:
        log.error(f"错误: 在 DATASET_FOLDER_MAP_NPZ 中未找到数据集 '{dataset_name}' 的配置。")
        log_raw.info('==============================================================\n')
        return

    # 2. 构建完整的文件路径
    # 假设 .npz 文件名与 dataset_name 相同，并位于 'raw' 子目录中
    file_name = f"{dataset_name}.npz"
    file_path = os.path.join(base_path, folder_name, 'raw', file_name)
    # 【核心修改】在打印前，对路径进行标准化，统一所有斜杠
    file_path = os.path.normpath(file_path)

    # 3. 加载并打印文件内容
    try:
        # 【修正】添加 encoding='latin1' 来兼容旧版本 NumPy/Python 保存的文件
        with np.load(file_path, allow_pickle=True, encoding='latin1') as data:
            data_dict = {key: data[key] for key in data.files}
    except FileNotFoundError:
        log.error(f"文件未找到: {file_path}")
        return
    except Exception as e:
        log.error(f"加载文件 '{file_path}' 时发生错误: {e}")
        return

    # 打印摘要信息
    log.info(f"文件路径: '{file_path}'")
    log.info(f"包含 {len(data_dict)} 个数组 (键): {', '.join(data_dict.keys())}")
    log.info("-" * 25)

    # 逐一打印每个数组的详细信息
    for key, array in data_dict.items():
        log.info(f"  键名 (Key): '{key}'")
        try:
            log.info(f"    - 数据类型 (dtype): {array.dtype}")
            log.info(f"    - 形状 (shape): {array.shape}")
        except AttributeError:
            log.warning(f"    - 无法获取键 '{key}' 的详细信息 (可能不是标准的 NumPy 数组)。")


def show_npz_file_info(dataset_name: str):
    log_raw.info(f'npz文件信息')
    log_raw.info('==============================================================')
    print_npz_file_info(dataset_name)
    log_raw.info('==============================================================\n')


def print_mat_file_info(dataset_name: str, base_path: str = './data/File'):
    """
    加载并打印指定 MAT 数据集文件的内部变量信息。

    此函数通过 DATASET_FOLDER_MAP_MAT 配置来定位文件。
    它假设 .mat 文件名与 dataset_name 相同。

    Args:
        dataset_name (str): 数据集的逻辑名称 (例如 'cora_mat')。
        base_path (str): 存放所有数据集文件夹的根目录。
    """
    log.info(f"MAT 文件: {dataset_name}")

    # 1. 从配置中查找文件夹名称
    folder_name = DATASET_FOLDER_MAP_MAT.get(dataset_name)
    if not folder_name:
        log.error(f"错误: 在 DATASET_FOLDER_MAP_MAT 中未找到数据集 '{dataset_name}' 的配置。")
        return

    # 2. 构建完整的文件路径
    file_name = f"{dataset_name}.mat"
    file_path = os.path.join(base_path, folder_name, 'raw', file_name)
    file_path = os.path.normpath(file_path)  # 路径标准化

    # 3. 加载并解析文件内容
    try:
        mat_data = sio.loadmat(file_path)
        # 过滤掉MATLAB内部元数据，只保留用户数据
        data_dict = {k: v for k, v in mat_data.items() if not k.startswith('__')}
    except FileNotFoundError:
        log.error(f"文件未找到: {file_path}")
        return
    except Exception as e:
        log.error(f"加载文件 '{file_path}' 时发生错误: {e}")
        return

    # 4. 打印摘要信息
    log.info(f"文件路径: '{file_path}'")
    log.info(f"包含 {len(data_dict)} 个变量 (键): {', '.join(data_dict.keys())}")
    log.info("-" * 25)

    # 5. 逐一打印每个变量的详细信息
    for key, value in data_dict.items():
        log.info(f"  键名 (Key): '{key}'")
        log.info(f"    - Python 类型: {type(value)}")
        try:
            # 根据变量类型提供不同的详细信息
            if sp.issparse(value):
                log.info(f"    - 格式: SciPy Sparse Matrix")
                log.info(f"    - 形状 (shape): {value.shape}")
                log.info(f"    - 非零元素数 (nnz): {value.nnz}")
                log.info(f"    - 数据类型 (dtype): {value.dtype}")
            elif isinstance(value, np.ndarray):
                log.info(f"    - 格式: NumPy Array")
                log.info(f"    - 形状 (shape): {value.shape}")
                log.info(f"    - 数据类型 (dtype): {value.dtype}")
            else:
                # 对于非数组类型（如字符串、数字），直接显示其值
                log.info(f"    - 值 (value): {value}")
        except AttributeError:
            log.warning(f"    - 无法获取键 '{key}' 的详细属性 (可能不是标准数组类型)。")
            log.info(f"    - 值 (value): {value}")


def show_mat_file_info(dataset_name: str):
    log_raw.info(f'mat文件信息')
    log_raw.info('==============================================================')
    print_mat_file_info(dataset_name)
    log_raw.info('==============================================================\n')
