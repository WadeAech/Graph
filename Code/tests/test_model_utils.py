"""
文件名：test_model_utils.py
概述：
作者：https://github.com/HomerCode
日期：2025/9/14 下午4:36
"""

# coding:UTF-8

from utils.model_utils import print_model_architecture

"""
说明：
 - 

函数：
 - 

类：
 - 
"""

if __name__ == '__main__':
    # 调用函数时，传入你想要测试的配置文件名
    print_model_architecture('config_0001.yaml')

    # 如果有其他配置文件，也可以用同样的方式测试
    # show_model_architecture('gcn_config.yaml')
    # show_model_architecture('gat_config.yaml')
