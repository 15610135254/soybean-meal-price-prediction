import os
import logging

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 默认数据文件路径
DEFAULT_DATA_FILE_PATH = '../model_data/date1.csv'

# 当前使用的数据文件路径
current_data_file_path = DEFAULT_DATA_FILE_PATH

def reset_data_file_path():
    global current_data_file_path
    current_data_file_path = DEFAULT_DATA_FILE_PATH
    logger.info(f"已重置数据文件路径为默认值: {current_data_file_path}")
    return current_data_file_path

def set_data_file_path(path):
    global current_data_file_path
    current_data_file_path = path
    logger.info(f"已设置数据文件路径为: {current_data_file_path}")
    return current_data_file_path

def get_data_file_path():
    return current_data_file_path

def get_full_data_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, current_data_file_path)
