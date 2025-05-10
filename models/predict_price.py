import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from datetime import datetime
import glob
from .feature_transformers import TechnicalIndicatorTransformer  # 改为相对导入

# 设置随机种子以确保结果可重现
np.random.seed(42)
tf.random.set_seed(42)

class ModelPredictor:
    def __init__(self, model_type=None, model_path=None, pipeline_path=None, scaler_path=None, look_back=30):
        self.model_type_requested = model_type # 保存请求的模型类型
        self.model_path = model_path
        self.pipeline_path = pipeline_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pipelines/preprocessing_pipeline.pkl')
        self.scaler_path = scaler_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scalers/target_scaler.pkl')
        self.look_back = look_back

        # 加载模型和相关组件
        self.load_components()

    def load_components(self):
        """加载模型、预处理管道和缩放器"""
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')

        if self.model_path is None: # 如果没有直接提供模型路径
            if self.model_type_requested:
                # 如果请求了特定类型的模型，例如 'LSTM'
                # 我们需要查找类似 'lstm_*.h5' 或 'LSTM_*.h5' 的文件
                # 同时处理大小写不敏感的模型类型
                search_pattern = os.path.join(model_dir, f'{self.model_type_requested.lower()}_*.h5')
                model_files = glob.glob(search_pattern)
                if not model_files: # 如果小写没找到，尝试大写
                    search_pattern_upper = os.path.join(model_dir, f'{self.model_type_requested.upper()}_*.h5')
                    model_files = glob.glob(search_pattern_upper)

                if not model_files: # 如果特定类型模型未找到，尝试不区分类型的最通用名称，如 'lstm.h5'
                    search_pattern_exact = os.path.join(model_dir, f'{self.model_type_requested.lower()}.h5')
                    model_files = glob.glob(search_pattern_exact)
                    if not model_files:
                        search_pattern_exact_upper = os.path.join(model_dir, f'{self.model_type_requested.upper()}.h5')
                        model_files = glob.glob(search_pattern_exact_upper)

                if not model_files:
                    raise FileNotFoundError(f"未找到类型为 '{self.model_type_requested}' 的模型文件。搜索路径: {search_pattern} 和 {search_pattern_upper} 及精确匹配")
                self.model_path = max(model_files, key=os.path.getmtime)
                print(f"自动选择类型为 '{self.model_type_requested}' 的最新模型: {self.model_path}")
            else:
                # 如果未请求特定类型，则加载所有类型中最新的模型
                all_model_files = glob.glob(os.path.join(model_dir, '*.h5'))
                if not all_model_files:
                    raise FileNotFoundError("在 saved_models 目录中未找到任何模型文件")
                self.model_path = max(all_model_files, key=os.path.getmtime)
                print(f"未指定模型类型，自动选择最新的模型: {self.model_path}")

        # 加载模型
        try:
            self.model = load_model(self.model_path)
            print(f"成功加载模型: {self.model_path}")

            # 提取实际加载的模型类型和时间戳
            model_filename = os.path.basename(self.model_path)
            # self.model_type 应该反映实际加载的模型的类型
            self.model_type_loaded = model_filename.split('_')[0].upper()
            # 如果文件名不包含'_'，例如 'lstm.h5'，则直接使用文件名（去除.h5）作为类型
            if '_' not in model_filename:
                 self.model_type_loaded = model_filename.replace('.h5', '').upper()

            self.timestamp = '_'.join(model_filename.split('_')[-2:]).replace('.h5', '') if '_' in model_filename else 'N/A'
            print(f"实际加载的模型类型: {self.model_type_loaded}, 时间戳: {self.timestamp}")
        except Exception as e:
            raise RuntimeError(f"加载模型时出错: {e}")

        # 加载预处理管道
        try:
            self.preprocessing_pipeline = joblib.load(self.pipeline_path)
            print(f"成功加载预处理管道: {self.pipeline_path}")
        except Exception as e:
            raise RuntimeError(f"加载预处理管道时出错: {e}")

        # 加载目标缩放器
        try:
            self.target_scaler = joblib.load(self.scaler_path)
            print(f"成功加载目标缩放器: {self.scaler_path}")
        except Exception as e:
            raise RuntimeError(f"加载目标缩放器时出错: {e}")

    def prepare_data(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("输入数据必须是pandas DataFrame类型")

        # 检查并添加缺失的列
        required_columns = ['星期', '月份', '季度']
        data_copy = data.copy()

        # 添加日期相关特征
        if '日期' in data_copy.columns:
            # 如果有日期列，从中提取时间特征
            try:
                data_copy['日期'] = pd.to_datetime(data_copy['日期'])
                if '星期' not in data_copy.columns:
                    data_copy['星期'] = data_copy['日期'].dt.dayofweek
                if '月份' not in data_copy.columns:
                    data_copy['月份'] = data_copy['日期'].dt.month
                if '季度' not in data_copy.columns:
                    data_copy['季度'] = data_copy['日期'].dt.quarter
                print("已从日期列生成时间特征")
            except Exception as e:
                print(f"从日期生成特征时出错: {e}")
        else:
            # 如果没有日期列，添加默认值
            for col in required_columns:
                if col not in data_copy.columns:
                    data_copy[col] = 0
                    print(f"添加缺失列 '{col}' 并设置为默认值0")

        # 应用预处理管道
        try:
            processed_data = self.preprocessing_pipeline.transform(data_copy)
            print(f"数据预处理完成，处理后形状: {processed_data.shape}")
        except Exception as e:
            # 如果仍然失败，打印更详细的错误信息
            print(f"预处理失败: {e}")
            print(f"数据列: {data_copy.columns.tolist()}")
            print(f"预处理管道需要的列可能不匹配")
            raise RuntimeError(f"数据预处理失败: {e}")

        # 创建时间序列窗口
        X = []
        for i in range(len(processed_data) - self.look_back + 1):
            X.append(processed_data[i:(i + self.look_back)])

        # 转换为numpy数组
        X = np.array(X)

        # 根据模型类型调整数据形状
        # 使用 self.model_type_loaded 来确保与加载的模型一致
        try:
            # 获取模型的输入形状
            input_shape = self.model.input_shape
            print(f"模型期望的输入形状: {input_shape}")

            # 检查当前数据形状
            print(f"当前数据形状: {X.shape}")

            # 根据模型类型和期望的输入形状调整数据
            if self.model_type_loaded == 'MLP':
                # 检查模型是否期望3D输入
                if len(input_shape) == 3:
                    # 如果模型期望3D输入，但数据已经是3D，则不需要调整
                    if len(X.shape) == 3:
                        print("MLP模型期望3D输入，数据已经是3D形状，无需调整")
                    # 如果数据是2D，需要重塑为3D
                    elif len(X.shape) == 2:
                        # 假设特征数是输入形状的最后一维
                        features_per_step = input_shape[-1]
                        time_steps = input_shape[-2]
                        X = X.reshape(X.shape[0], time_steps, features_per_step)
                        print(f"将2D数据重塑为3D: {X.shape}")
                else:
                    # 如果模型期望2D输入，但数据是3D，则需要展平
                    if len(X.shape) == 3:
                        X = X.reshape(X.shape[0], -1)
                        print(f"将3D数据展平为2D: {X.shape}")
            elif self.model_type_loaded == 'CNN' or self.model_type_loaded == 'LSTM':
                # CNN和LSTM模型通常需要3D数据 [样本数, 时间步, 特征数]
                # 确保数据是3D形状
                if len(X.shape) != 3:
                    # 如果不是3D，尝试重塑为3D
                    if len(input_shape) == 3:
                        features_per_step = input_shape[-1]
                        time_steps = input_shape[-2]
                        X = X.reshape(X.shape[0], time_steps, features_per_step)
                        print(f"将数据重塑为3D: {X.shape}")
        except Exception as e:
            print(f"调整数据形状时出错: {e}")
            print("继续使用原始数据形状")

        print(f"预测数据准备完成，形状: {X.shape}")
        return X

    def predict(self, data):
        # 准备数据
        X = self.prepare_data(data)

        # 进行预测
        try:
            predictions_scaled = self.model.predict(X)
            print(f"预测完成，缩放后的预测结果形状: {predictions_scaled.shape}")
        except Exception as e:
            raise RuntimeError(f"预测失败: {e}")

        # 反向转换预测值
        predictions = self.target_scaler.inverse_transform(predictions_scaled)
        print(f"预测结果反向转换完成，最终形状: {predictions.shape}")

        return predictions

    def predict_next_n_days(self, data, n_days=5):
        if len(data) < self.look_back:
            raise ValueError(f"输入数据长度({len(data)})小于look_back({self.look_back})")

        # 复制原始数据，避免修改原始数据
        working_data = data.copy()

        # 存储预测结果
        future_predictions = []

        for _ in range(n_days):
            # 准备最近的look_back天数据
            recent_data = working_data.iloc[-self.look_back:]

            # 预测下一天
            next_day_pred = self.predict(recent_data)
            next_day_value = next_day_pred[0][0]
            future_predictions.append(next_day_value)

            # 创建新的一行，包含预测值
            new_row = working_data.iloc[-1:].copy()
            target_col = working_data.columns[0]
            new_row[target_col] = next_day_value

            # 将新行添加到工作数据中
            working_data = pd.concat([working_data, new_row], ignore_index=True)

        return np.array(future_predictions)

    def plot_predictions(self, actual, predictions, title=None):
        plt.figure(figsize=(12, 6))

        # 绘制实际值和预测值
        plt.plot(actual, 'b-', label='实际值')
        plt.plot(predictions, 'r--', label='预测值')

        # 设置图表标题和标签
        # 使用 self.model_type_loaded
        plt.title(title or f'{self.model_type_loaded}模型预测结果')
        plt.xlabel('样本')
        plt.ylabel('值')
        plt.legend()

        # 保存图表
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        save_path = os.path.join(results_dir, f'{self.model_type_loaded.lower()}_prediction_{timestamp}.png')
        plt.savefig(save_path)
        plt.close()

        print(f"预测结果图表已保存到: {save_path}")

        return save_path

def predict_next_day(data_file, model_type=None):
    try:
        # 加载数据
        data = pd.read_csv(data_file)
        print(f"成功加载数据: {data_file}, 形状: {data.shape}")

        # 初始化预测器，传入模型类型
        predictor = ModelPredictor(model_type=model_type, look_back=30)

        # 预测下一天
        predictions = predictor.predict_next_n_days(data, n_days=1)
        prediction_value = predictions[0]

        print(f"下一天的预测值: {prediction_value:.4f}")
        return prediction_value

    except Exception as e:
        print(f"预测过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    data_file = "models/model_data/date1.csv"  # 默认数据文件路径
    predict_next_day(data_file, model_type='LSTM') # 测试加载特定LSTM模型
    predict_next_day(data_file, model_type='MLP') # 测试加载特定MLP模型
    predict_next_day(data_file, model_type='CNN') # 测试加载特定CNN模型
