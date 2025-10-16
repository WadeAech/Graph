"""
文件名：test_data_utils.py
概述：
作者：https://github.com/HomerCode
日期：2025/9/14 下午4:19
"""

# coding:UTF-8

import os
from utils.data_utils import show_dataset_info  # , show_mat_file_info, show_npz_file_info
from utils.data_utils import apply_torch_load_patch
from utils.logger_utils import setup_logging  # <-- 导入我们的配置函数

"""
说明：
 - 

函数：
 - 

类：
 - 
"""

if __name__ == '__main__':
    apply_torch_load_patch()

    # 1. 为测试日志创建一个临时目录
    temp_log_dir = os.path.join('.', 'results', 'temp_test_logs')
    os.makedirs(temp_log_dir, exist_ok=True)

    # 2. 调用我们自己的函数来初始化日志系统
    setup_logging(log_dir=temp_log_dir)

    # 3. 现在可以正常运行并看到输出了
    # show_dataset_info('Cora')  # 1
    # show_dataset_info('CiteSeer')  # 1
    # show_dataset_info('PubMed')  # 1
    #
    # show_dataset_info('Roman-empire') 1
    # show_dataset_info('Amazon-ratings') 1
    # show_dataset_info('Minesweeper') 1
    # show_dataset_info('Tolokers') 1
    # show_dataset_info('Questions') 1

    # show_dataset_info('squirrel_directed')
    # show_dataset_info('squirrel')
    # show_dataset_info('squirrel_filtered_directed')
    # show_dataset_info('squirrel_filtered') 1

    # show_dataset_info('chameleon_directed')
    # show_dataset_info('chameleon')
    # show_dataset_info('chameleon_filtered_directed')
    # show_dataset_info('chameleon_filtered') 1

    # show_dataset_info('cornell') 1
    # show_dataset_info('Cornell')
    # show_dataset_info('texas') 1
    # show_dataset_info('texas_4_classes')
    # show_dataset_info('Texas')
    # show_dataset_info('wisconsin') 1
    # show_dataset_info('Wisconsin')

    # show_dataset_info('actor') 1
    # show_dataset_info('Actor')

    # show_dataset_info('Flickr')
    # show_dataset_info('CoraFull')  1
    # show_dataset_info('CiteSeerFull')  1
    # show_dataset_info('PubMedFull')  1
    # show_dataset_info('Cora_ML')
    # show_dataset_info('DBLP')
    # show_dataset_info('Computers') 1
    # show_dataset_info('Photo') 1
    # show_dataset_info('CS') 1
    # show_dataset_info('Physics') 1
    # show_dataset_info('WikiCS') 1

    # show_dataset_info('BlogCatalog')
    # show_dataset_info('ogbn-arxiv')
    # show_dataset_info('ogbn-products')  考虑

    # show_mat_file_info('genius-v2')
    # show_dataset_info('genius')
    # show_dataset_info('genius-v2')

    # show_npz_file_info('squirrel')
    # show_mat_file_info('film')

    # show_dataset_info('film') 考虑
    # show_dataset_info('deezer-europe')  考虑
    # show_mat_file_info('deezer-europe')
    # show_dataset_info('penn94')

    # show_dataset_info('cora_dgl')
    # show_dataset_info('citeseer_dgl')
    # show_dataset_info('pubmed_dgl')
    # show_dataset_info('flickr_dgl') 1

    print('---------')
    # show_dataset_info('Cora')  # 1
    # show_dataset_info('CiteSeer')  # 2
    # show_dataset_info('PubMed')  # 3

    # show_dataset_info('Roman-empire')  # 4
    # show_dataset_info('Amazon-ratings')  # 5
    # show_dataset_info('Minesweeper')  # 6
    # show_dataset_info('Tolokers')  # 7
    # show_dataset_info('Questions')  # 8

    # show_dataset_info('Computers')  # 9
    # show_dataset_info('Photo')  # 10
    # show_dataset_info('CS')  # 11
    # show_dataset_info('Physics')  # 12

    # show_dataset_info('CoraFull')  # 13
    # show_dataset_info('CiteSeerFull') # 14 缺乏对照组
    # show_dataset_info('PubMedFull')  # 15 缺乏对照组

    # show_dataset_info('genius')  # 22

    # npz 格式
    # show_dataset_info('squirrel_filtered')  # 16
    # show_dataset_info('chameleon_filtered')  # 17
    show_dataset_info('cornell')  # 18 疑似过拟合，暂时不处理
    show_dataset_info('texas')  # 19 疑似过拟合，暂时不处理; 极度不平衡的数据集
    show_dataset_info('wisconsin')  # 20 疑似过拟合，暂时不处理
    show_dataset_info('actor')  # 21 疑似过拟合，暂时不处理

    # mat 格式
    # show_dataset_info('flickr_dgl')  # 暂时不处理
    # show_dataset_info('WikiCS')  # 暂时不处理
    # show_dataset_info('ogbn-arxiv')
    # show_dataset_info('ogbn-products')  # OOM
    # show_dataset_info('film')  # 暂时不处理
