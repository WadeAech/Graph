"""
文件名：custom_layers.py
概述：
作者：https://github.com/HomerCode
日期：2025/9/13 下午3:37
"""

# coding:UTF-8

import torch.nn as nn

"""
说明：
 - 

函数：
 - 

类：
 - 
"""


class ResidualModuleWrapper(nn.Module):
    """
    通用残差模块封装器：
    - 支持大多数常见 GNN 层（GCN, GAT, SAGE, Transformer, GIN）
    - 自动加残差连接
    - 可以通过 module_fn 适配特殊模块
    """

    def __init__(self, module_fn, residual=True):
        """
        module_fn: 可调用对象，返回一个 nn.Module
            - 如果是标准 GNN 层，可以直接用 lambda dim: GCNConv(dim, dim, ...)
            - 如果是 GINConv，需要用 lambda dim: GINConv(MLP)
        residual: 是否使用残差连接
        """
        super().__init__()
        self.residual = residual
        self.module = module_fn()  # 返回 nn.Module

    def forward(self, x, edge_index):
        x_res = self.module(x, edge_index)
        if self.residual:
            # 3. 检查输入和输出的特征维度是否一致
            if x.shape[1] != x_res.shape[1]:
                # 4. 如果不一致，检查投影层是否已创建
                if not hasattr(self, 'projection'):
                    # 5. 如果未创建，则创建它并存为类属性。
                    #    使用 x.shape[1] 和 x_res.shape[1] 来动态确定维度
                    self.projection = nn.Linear(x.shape[1], x_res.shape[1]).to(x.device)

                # 6. 使用投影层来变换 x 的维度
                x = self.projection(x)

            # 7. 现在维度保证一致，可以安全地相加
            x_res = x + x_res
        return x_res


class MLP(nn.Module):
    # 经典 “Linear → Dropout → Activation → Linear → Dropout” 结构
    def __init__(self, dim, dropout, hidden_dim_multiplier=3, input_dim_multiplier=1):
        super().__init__()
        input_dim = int(dim * input_dim_multiplier)
        hidden_dim = int(dim * hidden_dim_multiplier)
        self.linear_1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(in_features=hidden_dim, out_features=dim)
        self.dropout_2 = nn.Dropout(p=dropout)

    # edge_index 没有使用，只是为了保持兼容性，参与前向传播的层都定义为 (x, edge_index)
    def forward(self, x, edge_index):
        x = self.linear_1(x)
        x = self.dropout_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        x = self.dropout_2(x)

        return x
