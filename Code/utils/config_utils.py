"""
文件名：config_utils.py
概述：
作者：https://github.com/HomerCode
日期：2025/9/16 下午3:26
"""

# coding:UTF-8

import os
import yaml
import logging

log = logging.getLogger('app')

"""
说明：
 - 

函数：
 - 

类：
 - 
"""


def load_config(config_filename: str, project_root: str) -> dict:
    """
    加载指定路径下的YAML配置文件。

    Args:
        config_filename (str): 存放在 'configs/' 目录下的文件名。
        project_root (str): 项目的根目录路径。

    Returns:
        dict: 解析后的配置字典。
    """
    config_path = os.path.join(project_root, 'configs', config_filename)
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        log.info(f"成功加载配置: {config_path}")
        return config
    except FileNotFoundError:
        log.error(f"错误: 未找到配置文件 {config_path}")
        raise
    except Exception as e:
        log.error(f"加载或解析配置文件时出错: {e}")
        raise
