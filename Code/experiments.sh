#!/bin/bash

# --- 打印开始信息 ---
echo "================================================="
echo "开始批量执行GNN实验..."
echo "实验开始时间: $(date)"
echo "================================================="

# --- 定义要运行的配置文件列表 ---
CONFIGS=(
#    "config_0001_01.yaml"
#    "config_0001_02.yaml"
#    "config_0001_03.yaml"
#    "config_0001_04.yaml"
#    "config_0001_05.yaml"
#    "config_0001_06.yaml"
#    "config_0001_07.yaml"
#    "config_0001_08.yaml"
#    "config_0001_09.yaml"
#    "config_0001_10.yaml"
#    "config_0001_11.yaml"
#    "config_0001_12.yaml"
#    "config_0001_13.yaml"
#    "config_0001_14.yaml"
#    "config_0001_15.yaml"
#    "config_0001_16.yaml"
#    "config_0001_17.yaml"
#    "config_0001_18.yaml"
#    "config_0001_19.yaml"
#    "config_0001_20.yaml"
#    "config_0001_21.yaml"
#    "config_0001_22.yaml"

    "config_0002_01.yaml"
    "config_0002_02.yaml"
    "config_0002_03.yaml"
    "config_0002_04.yaml"
    "config_0002_05.yaml"
    "config_0002_06.yaml"
    "config_0002_07.yaml"
    "config_0002_08.yaml"
    "config_0002_09.yaml"
    "config_0002_10.yaml"
    "config_0002_11.yaml"
    "config_0002_12.yaml"
    "config_0002_13.yaml"
    "config_0002_14.yaml"
    "config_0002_15.yaml"
    "config_0002_16.yaml"
    "config_0002_17.yaml"
    "config_0002_18.yaml"
#    "config_0002_19.yaml"
    "config_0002_20.yaml"
    "config_0002_21.yaml"
    "config_0002_22.yaml"

)

# --- 循环执行每一个实验 ---
for config_file in "${CONFIGS[@]}"
do
    echo ""
    echo "-------------------------------------------------"
    echo "正在运行配置: $config_file"
    echo "-------------------------------------------------"
    
    # 调用您的主程序，传入对应的配置文件名
    # 假设 main.py 在 scripts/ 目录下
    python -m scripts.main --config "$config_file"
    
    # 检查上一个命令是否成功执行
    if [ $? -ne 0 ]; then
        echo "!!!! 错误: $config_file 实验运行失败。中止批量任务。 !!!!"
        exit 1
    fi
done

echo ""
echo "================================================="
echo "所有实验已成功完成！"
echo "实验结束时间: $(date)"
echo "================================================="