import os
import h5py

def read_hdf5_file(filepath):
    """
    读取 HDF5 文件并打印其内容。
    """
    try:
        with h5py.File(filepath, 'r') as f:
            print(f"--- 内容在 {filepath} ---")
            f.visititems(print_hdf5_item)
    except Exception as e:
        print(f"无法读取 {filepath}: {e}")

def print_hdf5_item(name, obj):
    """
    打印 HDF5 文件中的项目名称和类型。
    """
    print(f"名称: {name}, 类型: {type(obj)}")
    if isinstance(obj, h5py.Dataset):
        print(f"  形状: {obj.shape}, 数据类型: {obj.dtype}")

if __name__ == "__main__":
    models_dir = "best_models"  # 假设脚本与 best_models 文件夹在同一目录下
    
    # 检查目录是否存在
    if not os.path.isdir(models_dir):
        print(f"错误：找不到目录 '{models_dir}'")
        exit()

    for filename in os.listdir(models_dir):
        if filename.endswith(".h5"):
            filepath = os.path.join(models_dir, filename)
            read_hdf5_file(filepath) 