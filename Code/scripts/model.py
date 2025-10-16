# model.py (更新后)

# import torch
import torch.nn as nn
# import torch.nn.functional as torch_func
from torch_geometric.nn import GCNConv, GATConv
from .layers import ResidualModuleWrapper, MLP  # 假设我们保留了自定义层

GRAPH_CONV_LAYERS = (
    GCNConv,
    GATConv,
    ResidualModuleWrapper,
    MLP,
    # 未来您创建的任何新的图结构层，只需在这里添加即可
)

COMPUTER_LAYERS = (
    'gcn',
    'gat',
    'residual',
    'mlp',
    # 未来您创建的任何新的图结构层，只需在这里添加即可
)


# --- 辅助函数：get_activation (不变) ---
def get_activation(name: str):
    # ... (代码不变)
    activations = {
        'relu': nn.ReLU(),
        'elu': nn.ELU(),
        'leaky_relu': nn.LeakyReLU(),
        'gelu': nn.GELU(),
    }
    if name not in activations:
        raise ValueError(f'未知的激活函数: {name}')
    return activations[name]


# --- 唯一的通用GNN类 (变得更简单、更通用) ---
class GenericGNN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        # 现在的layers列表里包含了卷积、激活、Dropout等所有模块
        self.layers = nn.ModuleList(layers)

        self.reset_parameters()

    def forward(self, x, edge_index):
        # 循环所有层，不再需要特殊逻辑
        for layer in self.layers:
            # 判断当前模块是否是图卷积层，只有图卷积层需要edge_index
            if isinstance(layer, GRAPH_CONV_LAYERS):
                x = layer(x, edge_index)
            else:
                # 其他层（如ReLU, Dropout）正常调用
                x = layer(x)
        return x

    def reset_parameters(self):
        """遍历所有子模块并重置它们的参数。"""
        for layer in self.layers:
            # 检查模块是否有 reset_parameters 方法
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


# --- 智能的工厂函数 (变得更智能) ---
def get_model(config: dict, num_features: int, num_classes: int) -> GenericGNN:
    """
    根据新的、更灵活的配置规则动态构建GNN模型。
    """
    layers_config = config['layers']
    layers = []
    input_dim = num_features

    # --- 步骤 1: 预先扫描，找到最后一个计算层 ---
    # 我们从后往前遍历，找到的第一个计算层就是最后一个
    last_compute_layer_idx = -1
    for i in range(len(layers_config) - 1, -1, -1):
        if layers_config[i]['type'].lower() in COMPUTER_LAYERS:
            last_compute_layer_idx = i
            break

    # 如果配置中一个计算层都没有，则抛出错误
    if last_compute_layer_idx == -1:
        raise ValueError("模型配置中必须至少包含一个计算层 (gcn, gat, linear)。")

    # --- 步骤 2: 遍历配置，逐层构建模型 ---
    for i, layer_conf in enumerate(layers_config):
        layer_type = layer_conf['type'].lower()

        # 判断当前层是否是计算层
        if layer_type in COMPUTER_LAYERS:
            # --- 分支 A: 处理计算层 ---

            # 确定最后一个计算层的输出维度
            if i == last_compute_layer_idx:
                output_dim = num_classes
            else:
                output_dim = layer_conf['hidden_dim']

            # b. 根据类型创建层
            if layer_type == 'gcn':
                layer = GCNConv(input_dim, output_dim)
                input_dim = output_dim  # 更新下一层的输入维度
            elif layer_type == 'gat':
                heads = layer_conf.get('heads', 8)
                # 最后一个计算层（输出层）的头数强制为1
                if i == last_compute_layer_idx:
                    heads = 1
                layer = GATConv(input_dim, output_dim, heads=heads)
                input_dim = output_dim * heads  # 更新下一层的输入维度
            elif layer_type == 'linear':
                layer = nn.Linear(input_dim, output_dim)
                input_dim = output_dim  # 更新下一层的输入维度
            elif layer_type == 'residual':
                if layer_conf.get('model') == 'gcn':
                    layer = ResidualModuleWrapper(
                        module_fn=lambda: GCNConv(input_dim, output_dim),
                    )
                    input_dim = output_dim
                elif layer_conf.get('model') == 'gat':
                    heads = layer_conf.get('heads', 8)
                    if i == last_compute_layer_idx:
                        heads = 1
                    layer = ResidualModuleWrapper(
                        module_fn=lambda: GATConv(input_dim, output_dim, heads)
                    )
                    input_dim = output_dim *  heads  # 继续使用上一层的输出维度
            elif layer_type == 'mlp':
                layer = MLP(
                    dim=input_dim,
                    dropout=layer_conf['dropout_rate'],
                    hidden_dim_multiplier=layer_conf.get('hidden_dim_multiplier', 3),
                    input_dim_multiplier=layer_conf.get('input_dim_multiplier', 1)
                )
            layers.append(layer)

        else:
            # --- 分支 B: 处理非计算层 ---
            # 这些层不改变特征维度，所以 input_dim 不需要更新

            if layer_type == 'relu':
                layers.append(nn.ReLU())

            elif layer_type == 'elu':
                layers.append(nn.ELU())

            elif layer_type == 'dropout':
                dropout_rate = layer_conf['dropout_rate']
                layers.append(nn.Dropout(p=dropout_rate))

            elif layer_type == 'layernorm':
                layers.append(nn.LayerNorm(input_dim))

            else:
                raise ValueError(f"未知的层类型: {layer_type}")

    return GenericGNN(layers)
