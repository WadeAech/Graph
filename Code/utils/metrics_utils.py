import logging
import torch
from torch_geometric.data import Data
from torch_geometric.utils import homophily, degree, remove_self_loops
from collections import Counter

log = logging.getLogger('app')
log_raw = logging.getLogger('raw')


class GraphMetrics:
    """
    一个用于计算图的各种指标的工具类。(兼容旧版 PyG)
    名字前带 _ 的函数，是这个指标的辅助函数
    """

    def __init__(self, data: Data):
        if not isinstance(data, Data):
            raise TypeError("输入必须是 PyG 的 Data 对象")
        self.data = data
        self.metrics = {}

    # ... (其他方法 calculate_node_homophily, calculate_edge_homophily 等保持不变) ...
    def calculate_node_homophily(self):
        h_node = homophily(self.data.edge_index, self.data.y, method='node')
        self.metrics['node_homophily'] = h_node
        return h_node

    def calculate_edge_homophily(self):
        h_edge = homophily(self.data.edge_index, self.data.y, method='edge')
        self.metrics['edge_homophily'] = h_edge
        return h_edge

    def calculate_label_distribution(self):
        labels = self.data.y.cpu().numpy()
        total_nodes = len(labels)
        if total_nodes == 0:
            distribution = {}
        else:
            label_counts = Counter(labels)
            distribution = {
                k: {
                    'count': v,
                    'percentage': v / total_nodes
                }
                for k, v in sorted(label_counts.items())
            }
        self.metrics['label_distribution'] = distribution
        return distribution

    def _calculate_class_distribution(self):
        """
        内部辅助方法，用于计算 adjusted_homo 所需的度加权类别分布 (p_bar)。
        此版本明确计算“总度数”(入度+出度)，以适应所有类型的图。
        """
        labels = self.data.y.squeeze()
        num_nodes = self.data.num_nodes
        edge_index = self.data.edge_index

        # 1. 分别计算入度和出度
        in_degree = degree(edge_index[1], num_nodes=num_nodes, dtype=torch.float32)
        out_degree = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.float32)

        # 2. 计算总度数
        total_degree_vec = in_degree + out_degree

        # 3. 基于总度数计算 p_bar
        num_classes = labels.max().item() + 1
        p_bar = torch.zeros(num_classes, dtype=torch.float32, device=labels.device)  # <-- 修正行
        p_bar.scatter_add_(0, labels, total_degree_vec)

        total_degree_sum = torch.sum(total_degree_vec)
        if total_degree_sum > 0:
            p_bar = p_bar / total_degree_sum

        p_bar[torch.where(p_bar == 0)] = 1e-8
        return p_bar

    def calculate_adjusted_homo(self):
        """
        计算调整后的同质性，该同质性考虑了随机情况下的期望值。
        """
        # 确保先计算 edge_homophily
        if 'edge_homophily' not in self.metrics:
            self.calculate_edge_homophily()

        edge_homo = self.metrics['edge_homophily']

        # 计算随机期望的同质性
        p_bar = self._calculate_class_distribution()
        expected_homo = torch.sum(p_bar ** 2).item()

        # 调整公式
        denominator = 1 - expected_homo
        if denominator == 0:
            adj_homo = 0.0
        else:
            adj_homo = (edge_homo - expected_homo) / denominator

        self.metrics['adjusted_homo'] = adj_homo
        return adj_homo

    # --- ^^^ NEW METHOD ADDED ^^^ ---

    def _calculate_compatibility_matrix(self):
        """
        内部辅助方法，计算 class_homophily 所需的 c x c 兼容性矩阵 H。
        H[i,j] 是类别 i 的节点的邻居是类别 j 的比例。
        """
        edge_index, labels = self.data.edge_index, self.data.y.squeeze()
        num_classes = labels.max().item() + 1

        # 1. 移除自环
        edge_index_no_loops, _ = remove_self_loops(edge_index)
        src_node, targ_node = edge_index_no_loops[0, :], edge_index_no_loops[1, :]

        # 2. 构建 H 矩阵
        H = torch.zeros((num_classes, num_classes), dtype=torch.float32)
        for k in range(num_classes):
            # 找到所有类别为 k 的源节点
            mask = (labels[src_node] == k)
            # 获取这些源节点指向的目标节点的标签
            neighbor_labels = labels[targ_node[mask]]
            if neighbor_labels.numel() > 0:
                # 统计邻居标签的分布
                counts = torch.bincount(neighbor_labels, minlength=num_classes)
                H[k, :] = counts.float()

        # 3. 按行归一化，得到概率
        row_sums = H.sum(dim=1, keepdim=True)
        # 避免除以零
        H[row_sums.squeeze() > 0] /= row_sums[row_sums > 0]

        return H

    def calculate_class_homophily(self):
        """
        计算 class_homophily, 衡量超出随机性的类内连接程度。
        """
        labels = self.data.y.squeeze()
        num_classes = labels.max().item() + 1

        # 1. 获取兼容性矩阵 H
        H = self._calculate_compatibility_matrix()

        # 2. 获取类别分布
        class_counts = torch.bincount(labels, minlength=num_classes)
        proportions = class_counts.float() / self.data.num_nodes

        # 3. 计算 "超额" 同质性
        val = 0.0
        for k in range(num_classes):
            # 仅累加 H[k,k] (局部同质性) 超出 proportions[k] (全局占比) 的部分
            excess_homophily = torch.clamp(H[k, k] - proportions[k], min=0)
            if not torch.isnan(excess_homophily):
                val += excess_homophily.item()

        # 4. 归一化
        if num_classes > 1:
            class_homo = val / (num_classes - 1)
        else:
            class_homo = 0.0  # 如果只有一个类，则为0

        self.metrics['class_homophily'] = class_homo
        return class_homo

    def calculate_label_informativeness(self):
        """
        计算 Label Informativeness (LI) 指标。
        该指标衡量邻居标签对节点标签的预测能力。
        """
        edge_index, labels = self.data.edge_index, self.data.y.squeeze()
        num_nodes = self.data.num_nodes
        num_edges = edge_index.shape[1]
        num_classes = labels.max().item() + 1

        if num_edges == 0:
            self.metrics['label_informativeness'] = 0.0
            return 0.0

        # 1. 计算联合概率分布 p(c1, c2)
        # 这是一个 c x c 的矩阵，存储了每种标签对在边上出现的次数
        joint_counts = torch.zeros((num_classes, num_classes), dtype=torch.float32)

        # 获取每条边的源节点和目标节点的标签
        src_labels = labels[edge_index[0]]
        tgt_labels = labels[edge_index[1]]

        # 使用 PyTorch 的高效方法计算共现次数
        # 将 (c1, c2) 映射到一个唯一的 ID
        pair_ids = src_labels * num_classes + tgt_labels
        pair_counts = torch.bincount(pair_ids, minlength=num_classes * num_classes)
        joint_counts = pair_counts.view(num_classes, num_classes).float()

        # 归一化得到联合概率
        p_c1_c2 = joint_counts / num_edges

        # 2. 计算度加权的类别分布 p_bar(c)
        # 复用为 adj_homo 编写的稳健的辅助函数
        p_bar = self._calculate_class_distribution()

        # 3. 计算熵
        # 联合熵 H(Y_ξ, Y_η)
        # 注意: p * log(p) 当 p=0 时为 0
        p_c1_c2_nolog = p_c1_c2.clone()
        p_c1_c2_nolog[p_c1_c2 == 0] = 1  # log(1)=0, 避免 log(0)
        h_joint = -torch.sum(p_c1_c2 * torch.log2(p_c1_c2_nolog))

        # 边缘熵 H(Y_ξ)
        p_bar_nolog = p_bar.clone()
        p_bar_nolog[p_bar == 0] = 1
        h_marginal = -torch.sum(p_bar * torch.log2(p_bar_nolog))

        # 4. 应用最终公式
        if h_marginal == 0:
            # 如果边缘熵为0 (例如所有节点都属于同一类)，则信息量也为0
            li_score = 0.0
        else:
            # 根据论文中的公式 LI = 2 - H_joint / H_marginal
            # 这里 H_joint 和 H_marginal 都是正值
            li_score = (2 * h_marginal - h_joint) / h_marginal
            # 也可以是 I(X;Y)/H(X) = (H(X)+H(Y)-H(X,Y))/H(X)
            # H(X)=H(Y)=h_marginal, so (2*h_marginal - h_joint) / h_marginal

        self.metrics['label_informativeness'] = li_score.item()
        return li_score.item()

    # --- ^^^ NEW METHOD ADDED ^^^ ---

    def calculate_average_degree(self):
        if self.data.num_nodes == 0:
            return 0.0
        avg_deg = self.data.num_edges / self.data.num_nodes
        self.metrics['average_degree'] = avg_deg
        return avg_deg

    def calculate_linkx_homophily(self, batch_size=1000000):
        """
        计算特征同质性 (linkx_homophily)。
        此版本使用分批处理，以避免在大型图上出现显存不足的问题。
        """
        if self.data.x is None:
            self.metrics['linkx_homophily'] = 0.0
            return 0.0

        edge_index, x = self.data.edge_index, self.data.x
        num_edges = edge_index.shape[1]

        if num_edges == 0:
            self.metrics['linkx_homophily'] = 0.0
            return 0.0

        # 1. 对所有节点特征只归一化一次
        x_norm = torch.nn.functional.normalize(x, p=2, dim=-1)

        total_similarity_sum = 0.0
        # 2. 按批次遍历所有边
        for i in range(0, num_edges, batch_size):
            # a. 获取当前批次的边
            end = min(i + batch_size, num_edges)
            row_batch, col_batch = edge_index[:, i:end]

            # b. 仅为当前批次计算相似度（点积）
            similarities_in_batch = (x_norm[row_batch] * x_norm[col_batch]).sum(dim=-1)

            # c. 累加当前批次的相似度总和
            total_similarity_sum += similarities_in_batch.sum().item()

        # 3. 计算所有边的平均相似度
        h_linkx = total_similarity_sum / num_edges

        self.metrics['linkx_homophily'] = h_linkx
        return h_linkx

    def calculate_isolated_nodes_ratio(self):
        """
        计算孤立节点的比例。(手动实现，兼容所有 PyG 版本)
        """
        if self.data.num_nodes == 0:
            return 0.0

        # --- 手动实现逻辑开始 ---
        # 1. 获取所有参与了边的节点 (去重)
        connected_nodes = torch.unique(self.data.edge_index)

        # 2. 创建一个包含所有节点的集合
        all_nodes = set(range(self.data.num_nodes))

        # 3. 从所有节点中移除连接的节点，剩下的就是孤立节点
        isolated_nodes = all_nodes - set(connected_nodes.cpu().numpy())

        num_isolated_nodes = len(isolated_nodes)
        # --- 手动实现逻辑结束 ---

        ratio = num_isolated_nodes / self.data.num_nodes
        self.metrics['isolated_nodes_ratio'] = ratio
        return ratio

    def calculate_all(self, verbose=True):
        """
        计算所有支持的指标。
        """
        log.info("开始计算图指标...")
        self.calculate_node_homophily()  # 标签同配性
        self.calculate_edge_homophily()  # 标签同配性
        self.calculate_adjusted_homo()  # 标签同配性
        self.calculate_class_homophily()  # 标签同配性
        self.calculate_label_informativeness()  # 结构同配性
        self.calculate_linkx_homophily(10000)  # 特征同配性
        self.calculate_label_distribution()
        self.calculate_average_degree()
        self.calculate_isolated_nodes_ratio()

        if verbose:
            for key, value in self.metrics.items():
                if key == 'label_distribution':
                    print(f"{key:<25}:")
                    if not value:
                        print("  - (无标签)")
                    else:
                        for label, stats in value.items():
                            # --- vvv 这里是修改的核心 vvv ---
                            # 将格式从 :.2% (两位小数百分比) 改为 :.4f (四位小数浮点数)
                            print(f"  - Class {label}: {stats['count']:<5} nodes ({stats['percentage']:.4f})")
                            # --- ^^^ 这里是修改的核心 ^^^ ---
                elif isinstance(value, float):
                    print(f"{key:<25}: {value:.4f}")
                else:
                    print(f"{key:<25}: {value}")

        return self.metrics
