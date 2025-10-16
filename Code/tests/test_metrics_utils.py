import logging
from utils.data_utils import load_dataset, print_dataset_info
from utils.logger_utils import setup_logging
from utils.metrics_utils import GraphMetrics
from utils.save_utils import create_run_results_dir

log = logging.getLogger('app')
log_raw = logging.getLogger('raw')
run_results_dir = create_run_results_dir('config_0001_01.yaml', '/')
setup_logging(log_dir=run_results_dir)
log = logging.getLogger('app')
log_raw = logging.getLogger('raw')

# 加载数据集
dataset = load_dataset('genius')
cora_data = dataset[0]

# 打印基本信息
print_dataset_info('genius')

# 打印度量
metrics_calculator = GraphMetrics(cora_data)

metrics_calculator.calculate_all()
# all_metrics_dict = metrics_calculator.calculate_all()

# print("\n方法返回的字典内容:")
# print(all_metrics_dict)
#
# node_homo = all_metrics_dict.get('node_homophily')
# print(f"\n单独获取的节点同质性是: {node_homo:.4f}")
