"""
文件名：logger_setup.py
概述：
作者：https://github.com/HomerCode
日期：2025/9/14 下午5:33
"""

# coding:UTF-8

import os
import sys
import logging

"""
说明：
 - 

函数：
 - 

类：
 - 
"""


def setup_logging(log_dir: str):
    """
    配置项目的日志记录器。

    Args:
        log_dir (str): 本次运行的专属结果目录，日志文件将保存在这里。
    """
    # --- 配置带格式的 Logger ('app') ---
    log = logging.getLogger('app')
    log.setLevel(logging.INFO)
    log.propagate = False

    # 清除旧的 handlers，防止重复记录
    if log.hasHandlers():
        log.handlers.clear()

    # 控制台处理器
    handler_stdout_formatted = logging.StreamHandler(sys.stdout)
    formatter_formatted = logging.Formatter('【%(asctime)s】 %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler_stdout_formatted.setFormatter(formatter_formatted)
    log.addHandler(handler_stdout_formatted)

    # 文件处理器
    log_file_path = os.path.join(log_dir, 'run.log')
    handler_file_formatted = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    handler_file_formatted.setFormatter(formatter_formatted)
    log.addHandler(handler_file_formatted)

    # --- 配置裸输出的 Logger ('raw') ---
    log_raw = logging.getLogger('raw')
    log_raw.setLevel(logging.INFO)
    log_raw.propagate = False

    if log_raw.hasHandlers():
        log_raw.handlers.clear()

    # 控制台处理器
    handler_stdout_raw = logging.StreamHandler(sys.stdout)
    formatter_raw = logging.Formatter('%(message)s')
    handler_stdout_raw.setFormatter(formatter_raw)
    log_raw.addHandler(handler_stdout_raw)

    # 文件处理器
    handler_file_raw = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')  # 使用追加模式'a'
    handler_file_raw.setFormatter(formatter_raw)
    log_raw.addHandler(handler_file_raw)

    log_raw.info(f'日志')
    log_raw.info('==============================================================')
    log.info(f"日志系统已启动，日志文件将保存至: {log_file_path}")
    log_raw.info('==============================================================\n')
