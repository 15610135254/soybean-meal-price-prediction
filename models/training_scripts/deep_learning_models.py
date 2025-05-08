import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
import os
import time
import random
from datetime import datetime
import platform
import json

# 设置GPU内存增长，避免一次性占用所有GPU内存
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 对每个GPU设置内存增长
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"找到 {len(gpus)} 个GPU设备，已设置内存动态增长")

        # 设置可见的GPU设备
        tf.config.set_visible_devices(gpus, 'GPU')

        # 启用混合精度训练 (如果GPU支持)
        from tensorflow.keras import mixed_precision
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print(f"混合精度策略已设置为: {policy.name}")

        # 检查是否在Apple Silicon芯片上运行
        is_apple_silicon = platform.processor() == 'arm'

        # 如果是Apple Silicon，尝试启用Metal支持
        if is_apple_silicon:
            try:
                # 导入Metal支持（前提是已安装tensorflow-metal）
                from tensorflow.python.compiler.mlcompute import mlcompute
                print("ML Compute加速已启用")
                # 使用GPU进行加速
                mlcompute.set_mlc_device(device_name='gpu')
                print(f"ML Compute设备: {mlcompute.get_mlc_device()}")
            except ImportError:
                print("未检测到tensorflow-metal，将使用标准GPU加速")
    except RuntimeError as e:
        print(f"GPU设置错误: {e}")
else:
    print("未检测到GPU设备，将使用CPU运行")

from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from tensorflow.keras.layers import Input, Concatenate, GlobalAveragePooling1D, Attention, Add, TimeDistributed
from tensorflow.keras.layers import Bidirectional, GRU, LeakyReLU, PReLU, Reshape, RepeatVector, Lambda
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from config import TRAINING_CONFIG, MLP_CONFIG, LSTM_CONFIG, CNN_CONFIG

# 设置随机种子，保证结果可复现
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)  # tensorflow 2.9.0使用这种方式设置随机种子

print(f"TensorFlow版本: {tf.__version__}")
# print(f"Keras版本: {tf.keras.__version__}") # Keras版本通常与TensorFlow版本一致，且此属性可能在新版中移除

# 检测可用的GPU设备
print("可用设备:")
for device in tf.config.list_physical_devices():
    print(f" - {device.name} ({device.device_type})")

class DeepLearningModels:
    def __init__(self, data_file, target_col='收盘价', look_back=TRAINING_CONFIG['look_back']):
        """
        初始化深度学习模型类

        Args:
            data_file: 数据文件路径
            target_col: 目标预测列名
            look_back: 使用过去多少天的数据来预测
        """
        self.data_file = data_file
        self.target_col = target_col
        self.look_back = look_back
        self.models = {}
        self.history = {}
        self.predictions = {}
        self.metrics = {}

        # 创建必要的目录
        self.create_directories()

        # 加载和准备数据
        self.load_and_prepare_data()

    def create_directories(self):
        """创建必要的目录"""
        directories = ['saved_models', 'logs', 'checkpoints', 'results', 'scalers']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def load_and_prepare_data(self):
        """加载并准备数据"""
        print(f"正在加载数据: {self.data_file}")
        self.df = pd.read_csv(self.data_file)
        print(f"数据加载后，DataFrame 形状: {self.df.shape}")

        # 检查数据集格式并进行必要的映射
        if 'date' in self.df.columns and '日期' not in self.df.columns:
            print("检测到新数据集格式，进行列名映射...")
            # 列名映射（新数据集到旧数据集的映射）
            self.column_mapping = {
                'date': '日期',
                'open': '开盘价',
                'high': '最高价',
                'low': '最低价',
                'close': '收盘价',
                'volume': '成交量',
                'hold': '持仓量',
                'MA_5': 'MA5',
                'HV_20': '波动率',
                'ATR_14': 'ATR',
                'RSI_14': 'RSI',
                'OBV': 'OBV',
                'MACD': 'MACD'
            }

            # 将日期列转换为datetime
            self.df['date'] = pd.to_datetime(self.df['date'])

            # 更新目标列名
            if self.target_col == '收盘价' and 'close' in self.df.columns:
                self.target_col = 'close'
                print(f"目标列已更新为: {self.target_col}")

            # 不进行列名映射，直接使用新的列名
            self.date_col = 'date'
        else:
            # 原始数据集格式
            self.df['日期'] = pd.to_datetime(self.df['日期'])
            self.date_col = '日期'

        # 按日期升序排序
        self.df = self.df.sort_values(self.date_col)

        # 根据用户要求，不添加新的技术指标特征，将直接从现有列中选择特征。
        # 因此，我们注释掉 self.add_technical_indicators()
        # self.add_technical_indicators()
        # The print statement below is also removed as no indicators are added.
        # print(f"添加技术指标后，DataFrame 形状: {self.df.shape}")
        print("信息: 根据用户要求，不添加新的技术指标。将从原始数据列中选择特征。")

        # 显示数据的基本信息 (在选择特征之前)
        # print(f"数据准备阶段 - DataFrame 形状 (在排序之后，特征选择之前): {self.df.shape}") # 这行可以省略，因为紧接着就有select_features
        print(f"原始数据行数: {len(self.df)}")
        print(f"原始数据列数 (包括日期和目标列): {len(self.df.columns)}")
        print(f"时间范围: {self.df[self.date_col].min()} 至 {self.df[self.date_col].max()}")

        # 选择要使用的特征 (此方法将被修改为仅从现有列中选择)
        self.features = self.select_features() # self.select_features() 将被修改
        # self.features 现在是特征列名的列表.
        # self.df 本身在列数上可能没有改变，但 select_features 内部的 to_numeric 可能已修改了某些列的数据类型。
        print(f"选择的特征列表 ({len(self.features)} 个特征): {self.features}")
        # print(f"选择特征后 ({len(self.features)} 个特征)，DataFrame 形状: {self.df.shape}") # DataFrame的列数本身不因特征列表的创建而改变

        # 处理缺失值
        self.handle_missing_values()

        # 分割数据集
        self.split_data()

        # 数据标准化
        self.scale_data()

        # 准备时间序列数据
        self.prepare_time_series()

    def add_technical_indicators(self):
        """添加技术指标"""
        # 确定使用的列名
        if 'close' in self.df.columns:
            # 新数据集格式
            close_col = 'close'
            open_col = 'open'
            high_col = 'high'
            low_col = 'low'
            volume_col = 'volume'
            date_col = 'date'

            # 检查MA列的格式
            ma5_col = 'MA_5' if 'MA_5' in self.df.columns else 'MA5'
            ma10_col = 'MA_10' if 'MA_10' in self.df.columns else 'MA10'
            ma20_col = 'MA_20' if 'MA_20' in self.df.columns else 'MA20'
            ma30_col = 'MA_30' if 'MA_30' in self.df.columns else 'MA30'

            # 检查RSI列的格式
            rsi_col = 'RSI_14' if 'RSI_14' in self.df.columns else 'RSI'

            # 检查MACD列的格式
            macd_col = 'MACD'
        else:
            # 旧数据集格式
            close_col = '收盘价'
            open_col = '开盘价'
            high_col = '最高价'
            low_col = '最低价'
            volume_col = '成交量'
            date_col = '日期'

            ma5_col = 'MA5'
            ma10_col = 'MA10'
            ma20_col = 'MA20'
            ma30_col = 'MA30'

            rsi_col = 'RSI'
            macd_col = 'MACD'

        # 基本价格特征
        self.df['价格变化'] = self.df[close_col].diff()
        self.df['价格变化率'] = self.df[close_col].pct_change()
        self.df['日内波动率'] = (self.df[high_col] - self.df[low_col]) / self.df[open_col]

        # 移动平均线特征
        ma_cols = {5: ma5_col, 10: ma10_col, 20: ma20_col, 30: ma30_col}
        for ma, col in ma_cols.items():
            if col in self.df.columns:
                self.df[f'MA{ma}_diff'] = self.df[close_col] - self.df[col]
                self.df[f'MA{ma}_slope'] = self.df[col].diff()
                self.df[f'MA{ma}_std'] = self.df[close_col].rolling(window=ma).std()

        # 波动率指标
        self.df['20日波动率'] = self.df[close_col].rolling(window=20).std() / self.df[close_col].rolling(window=20).mean()
        self.df['10日波动率'] = self.df[close_col].rolling(window=10).std() / self.df[close_col].rolling(window=10).mean()

        # MACD相关指标
        if macd_col in self.df.columns:
            self.df['MACD_diff'] = self.df[macd_col].diff()
            self.df['MACD_slope'] = self.df[macd_col].diff(3)

        # RSI相关指标
        if rsi_col in self.df.columns:
            self.df['RSI_diff'] = self.df[rsi_col].diff()
            self.df['RSI_slope'] = self.df[rsi_col].diff(3)
            self.df['RSI_MA5'] = self.df[rsi_col].rolling(window=5).mean()

        # KDJ相关指标
        if all(x in self.df.columns for x in ['K', 'D', 'J']):
            self.df['KD_diff'] = self.df['K'] - self.df['D']
            self.df['KD_cross'] = (self.df['K'] > self.df['D']).astype(int)

        # 成交量特征
        self.df['成交量变化率'] = self.df[volume_col].pct_change()
        self.df['成交量MA5'] = self.df[volume_col].rolling(window=5).mean()
        self.df['量价背离'] = (self.df['价格变化率'] * self.df['成交量变化率'] < 0).astype(int)

        # 趋势特征
        if ma5_col in self.df.columns:
            self.df['上升趋势'] = (self.df[close_col] > self.df[ma5_col]).astype(int)

            if ma10_col in self.df.columns and ma20_col in self.df.columns:
                self.df['强势上升'] = ((self.df[ma5_col] > self.df[ma10_col]) &
                                  (self.df[ma10_col] > self.df[ma20_col])).astype(int)

        # 时间特征
        self.df['星期'] = self.df[date_col].dt.dayofweek
        self.df['月份'] = self.df[date_col].dt.month
        self.df['季度'] = self.df[date_col].dt.quarter

        # 新数据集特有的特征
        if 'a_close' in self.df.columns and 'c_close' in self.df.columns:
            self.df['豆粕_大豆价差'] = self.df[close_col] - self.df['a_close']
            self.df['豆粕_玉米价差'] = self.df[close_col] - self.df['c_close']
            self.df['豆粕_大豆比率'] = self.df[close_col] / self.df['a_close']
            self.df['豆粕_玉米比率'] = self.df[close_col] / self.df['c_close']

        # 删除包含无穷值的行
        self.df = self.df.replace([np.inf, -np.inf], np.nan)

    def handle_missing_values(self):
        """处理缺失值"""
        print(f"开始处理缺失值，DataFrame 形状: {self.df.shape}")
        # print(f"处理前，每列的NaN数量:\n{self.df[self.features].isnull().sum().sort_values(ascending=False)}")

        # 1. 前向填充 (ffill)
        self.df[self.features] = self.df[self.features].fillna(method='ffill')
        print(f"执行 ffill 后，DataFrame 形状: {self.df.shape}")
        # print(f"执行 ffill 后，每列的NaN数量:\n{self.df[self.features].isnull().sum().sort_values(ascending=False)}")

        # 2. 后向填充 (bfill) - 对 ffill 后可能在开头的NaN有帮助
        self.df[self.features] = self.df[self.features].fillna(method='bfill')
        print(f"执行 bfill 后，DataFrame 形状: {self.df.shape}")
        # print(f"执行 bfill 后，每列的NaN数量:\n{self.df[self.features].isnull().sum().sort_values(ascending=False)}")

        # 3. 对于仍然是 NaN 的列 (通常是由于原始数据不足以计算指标导致整列 NaN), 用 0 填充
        nan_counts_after_fills = self.df[self.features].isnull().sum()
        for feature in self.features:
            if nan_counts_after_fills[feature] == len(self.df):
                print(f"警告: 特征 '{feature}' 在ffill和bfill后仍然全是NaN (共{len(self.df)}行)。将用0填充此列。")
                self.df[feature] = self.df[feature].fillna(0)
            elif nan_counts_after_fills[feature] > 0:
                 # 对于那些不是整列NaN但仍有NaN的列，先尝试用该列的均值填充
                # 如果均值本身是NaN（例如，如果列中有效值很少），则后续用0填充
                mean_val = self.df[feature].mean()
                if pd.isna(mean_val):
                    print(f"警告: 特征 '{feature}' 的均值为NaN，将用0填充剩余NaN。")
                    self.df[feature] = self.df[feature].fillna(0)
                else:
                    self.df[feature] = self.df[feature].fillna(mean_val)

        print(f"对特定NaN列和均值填充后，DataFrame 形状: {self.df.shape}")
        # print(f"对特定NaN列和均值填充后，每列的NaN数量:\n{self.df[self.features].isnull().sum().sort_values(ascending=False)}")

        # 4. 最后，如果还有零星NaN（理论上此时应该很少了），删除包含这些NaN的行。
        #    或者更保守，对最后剩余的NaN也用0填充，确保不丢失行。
        #    例如: self.df[self.features] = self.df[self.features].fillna(0)
        self.df = self.df.dropna(subset=self.features) 
        print(f"执行 dropna 后，DataFrame 形状: {self.df.shape}")

        if self.df.empty:
            print("-----------------------------------------------------------------")
            print("严重警告: DataFrame 在 handle_missing_values 方法执行完毕后为空！")
            print("这通常意味着：")
            print("  1. 原始数据文件中的数据行数过少，无法满足技术指标计算（特别是回顾窗口较大的指标）和 look_back 周期的需求。")
            print("  2. 特征工程步骤产生了过多的无法填充的NaN值。")
            print("建议措施：")
            print("  - 检查您的原始数据文件 (e.g., date2.csv, date1.csv) 是否有足够的数据。")
            print("  - 尝试减少 DeepLearningModels 初始化时的 look_back 参数。")
            print("  - 检查 add_technical_indicators 方法，暂时移除需要较长回顾期的指标。")
            print("  - 查看上面打印的各阶段NaN数量，找出哪些特征是主要问题源。")
            print("-----------------------------------------------------------------")
            # 可以在这里抛出异常，或者允许流程继续但 split_data 中会捕获

    def select_features(self):
        """
        选择模型使用的特征。
        根据用户要求，此版本仅从加载的原始数据中选择特征，
        排除日期列和目标列，并确保它们是数值类型。
        不添加任何新的计算指标。
        """
        print("信息: 正在从原始加载数据中选择特征，不添加新计算指标。")
        if self.df is None or self.df.empty:
            raise ValueError("错误: DataFrame未加载或为空，无法选择特征。")

        all_loaded_columns = self.df.columns.tolist()

        # 确定要排除的列: 日期列 (self.date_col) 和目标列 (self.target_col)
        # 这些属性应该在调用此方法之前已经被正确设置
        columns_to_exclude = []
        if hasattr(self, 'date_col') and self.date_col:
            columns_to_exclude.append(self.date_col)
        else:
            print("警告: 'self.date_col' 未定义或为空，可能导致日期列未被正确排除。")

        if hasattr(self, 'target_col') and self.target_col:
            columns_to_exclude.append(self.target_col)
        else:
            print("警告: 'self.target_col' 未定义或为空，可能导致目标列未被正确排除。")
        
        # 确保排除列表中的列名确实存在于DataFrame中，避免KeyError
        columns_to_exclude = [col for col in columns_to_exclude if col in all_loaded_columns]

        potential_feature_names = [col for col in all_loaded_columns if col not in columns_to_exclude]
        
        final_selected_features = []
        for feature_name in potential_feature_names:
            if feature_name in self.df.columns: # 再次确认列存在
                if not pd.api.types.is_numeric_dtype(self.df[feature_name]):
                    print(f"信息: 特征 '{feature_name}' 非数值类型，尝试转换为数值类型...")
                    try:
                        # 原地转换列类型，并将无法转换的值设为NaN
                        self.df[feature_name] = pd.to_numeric(self.df[feature_name], errors='coerce')
                        # 检查转换后是否所有值都变为NaN (例如，如果列是纯文本)
                        if self.df[feature_name].isnull().all():
                            print(f"警告: 特征 '{feature_name}' 转换为数值后全为NaN。此特征将不被使用。")
                        else:
                            final_selected_features.append(feature_name)
                            print(f"信息: 特征 '{feature_name}' 已成功转换为数值类型。")
                    except Exception as e:
                        print(f"警告: 转换特征 '{feature_name}' 为数值类型失败: {e}。此特征将不被使用。")
                else:
                    final_selected_features.append(feature_name) # 本身就是数值类型
            else:
                # 这个情况理论上不应该发生，因为是从all_loaded_columns开始的
                print(f"警告: 在特征选择过程中，预期中的列 '{feature_name}' 未在DataFrame中找到。")

        print(f"信息: 最终选择用于模型的特征共 {len(final_selected_features)} 个: {final_selected_features}")
        
        # 根据用户请求检查特征数量，现在将强制要求16个特征 (根据date1.csv的实际情况调整)
        expected_feature_count = 16
        if len(final_selected_features) != expected_feature_count:
            error_message = (
                f"错误: 模型固定需要 {expected_feature_count} 个特征，但实际从数据文件 '{self.data_file}' 中选择了 {len(final_selected_features)} 个。\n"
                f"       检查要点：\n"
                f"         1. 数据文件 '{self.data_file}' 是否包含正确的列数？\n"
                f"         2. 日期列 ('{self.date_col if hasattr(self, 'date_col') else '未定'}') 和目标列 ('{self.target_col if hasattr(self, 'target_col') else '未定'}') 是否被正确识别并排除？\n"
                f"         3. 排除上述两列后，其余的 {expected_feature_count} 个预期特征列是否都存在且为数值类型 (或可成功转换为数值类型)？\n"
                f"       当前所有加载的列: {all_loaded_columns}\n"
                f"       被定义为日期/目标而排除的列: {columns_to_exclude}\n"
                f"       最终选择的特征: {final_selected_features}"
            )
            raise ValueError(error_message)

        return final_selected_features

    def split_data(self, test_size=0.1, val_size=0.1):
        """分割数据为训练集、验证集和测试集"""
        # 确保数据按时间排序
        data = self.df.sort_values(self.date_col)

        # 获取特征和目标变量
        X = data[self.features].values
        y = data[self.target_col].values

        # 总数据量
        n = len(data)

        # 计算测试集和验证集的大小
        test_samples = int(n * test_size)
        val_samples = int(n * val_size)

        # 分割数据
        X_train = X[:n - test_samples - val_samples]
        y_train = y[:n - test_samples - val_samples]

        X_val = X[n - test_samples - val_samples:n - test_samples]
        y_val = y[n - test_samples - val_samples:n - test_samples]

        X_test = X[n - test_samples:]
        y_test = y[n - test_samples:]

        # 保存数据
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

        # 保存日期信息以便可视化
        self.train_dates = data[self.date_col].iloc[:n - test_samples - val_samples]
        self.val_dates = data[self.date_col].iloc[n - test_samples - val_samples:n - test_samples]
        self.test_dates = data[self.date_col].iloc[n - test_samples:]

        print(f"训练集大小: {len(X_train)}")
        print(f"验证集大小: {len(X_val)}")
        print(f"测试集大小: {len(X_test)}")

        if len(X_train) == 0:
            print("错误：训练集为空！请检查数据预处理步骤和原始数据量。")
            raise ValueError("训练集在split_data后为空，无法继续。")

    def scale_data(self):
        """标准化数据"""
        # 创建特征缩放器，限制范围在[-1, 1]之间
        self.feature_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.target_scaler = MinMaxScaler(feature_range=(-1, 1))

        # 使用训练集拟合缩放器
        self.feature_scaler.fit(self.X_train)
        self.target_scaler.fit(self.y_train.reshape(-1, 1))

        # 转换训练集、验证集和测试集
        self.X_train_scaled = self.feature_scaler.transform(self.X_train)
        self.y_train_scaled = self.target_scaler.transform(self.y_train.reshape(-1, 1)).ravel()

        self.X_val_scaled = self.feature_scaler.transform(self.X_val)
        self.y_val_scaled = self.target_scaler.transform(self.y_val.reshape(-1, 1)).ravel()

        self.X_test_scaled = self.feature_scaler.transform(self.X_test)
        self.y_test_scaled = self.target_scaler.transform(self.y_test.reshape(-1, 1)).ravel()

        # 保存标准化器
        scaler_path = 'scalers'
        feature_scaler_file = os.path.join(scaler_path, 'feature_scaler.pkl')
        target_scaler_file = os.path.join(scaler_path, 'target_scaler.pkl')
        joblib.dump(self.feature_scaler, feature_scaler_file)
        joblib.dump(self.target_scaler, target_scaler_file)
        print(f"特征标准化器已保存到: {feature_scaler_file}")
        print(f"目标标准化器已保存到: {target_scaler_file}")

    def prepare_time_series(self):
        """准备时间序列数据"""
        # 创建时间序列样本
        def create_time_series(X, y, time_steps=1):
            Xs, ys = [], []
            for i in range(len(X) - time_steps):
                Xs.append(X[i:(i + time_steps)])
                ys.append(y[i + time_steps])
            return np.array(Xs), np.array(ys)

        # 创建训练集时间序列
        self.X_train_ts, self.y_train_ts = create_time_series(
            self.X_train_scaled, self.y_train_scaled, self.look_back
        )

        # 创建验证集时间序列
        self.X_val_ts, self.y_val_ts = create_time_series(
            self.X_val_scaled, self.y_val_scaled, self.look_back
        )

        # 创建测试集时间序列
        self.X_test_ts, self.y_test_ts = create_time_series(
            self.X_test_scaled, self.y_test_scaled, self.look_back
        )

        print(f"时间序列训练集形状: {self.X_train_ts.shape}")
        print(f"时间序列验证集形状: {self.X_val_ts.shape}")
        print(f"时间序列测试集形状: {self.X_test_ts.shape}")

    def _create_dataset(self, X, y, batch_size, shuffle=False):
        """辅助函数，用于从NumPy数组创建tf.data.Dataset"""
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(X))
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def create_mlp_model(self, params=None):
        """创建改进的多层感知机模型，增强时序特征提取能力，适用于GPU训练"""
        if params is None:
            params = {
                'units1': 256,  # 增加网络规模，利用GPU计算能力
                'units2': 128,
                'units3': 64,
                'units4': 32,
                'dropout': 0.2,  # 保持适度的dropout
                'learning_rate': 0.0005,  # 适中的学习率
                'l2_reg': 0.0001  # 轻微的正则化
            }

        # 使用函数式API来构建更复杂的模型结构
        inputs = Input(shape=(self.look_back, len(self.features)))

        # 批归一化输入
        normalized = BatchNormalization()(inputs)

        # 创建多个不同尺度的时序特征提取分支
        # 短期特征
        time_features1 = Conv1D(filters=64, kernel_size=2, padding='same')(normalized)
        time_features1 = BatchNormalization()(time_features1)
        time_features1 = PReLU()(time_features1)
        time_features1 = GlobalAveragePooling1D()(time_features1)

        # 中期特征
        time_features2 = Conv1D(filters=64, kernel_size=3, padding='same')(normalized)
        time_features2 = BatchNormalization()(time_features2)
        time_features2 = PReLU()(time_features2)
        time_features2 = GlobalAveragePooling1D()(time_features2)

        # 长期特征
        time_features3 = Conv1D(filters=64, kernel_size=5, padding='same')(normalized)
        time_features3 = BatchNormalization()(time_features3)
        time_features3 = PReLU()(time_features3)
        time_features3 = GlobalAveragePooling1D()(time_features3)

        # 创建全局特征提取分支 - 直接扁平化处理
        global_features = Flatten()(normalized)

        # 合并所有特征
        merged_features = Concatenate()([time_features1, time_features2, time_features3, global_features])

        # 添加残差连接的深度前馈网络
        x = Dense(params['units1'], kernel_regularizer=l2(params['l2_reg']))(merged_features)
        x = BatchNormalization()(x)
        x = PReLU()(x)  # 使用PReLU代替LeakyReLU
        x_res1 = x  # 保存用于残差连接

        x = Dense(params['units2'], kernel_regularizer=l2(params['l2_reg']))(x)
        x = BatchNormalization()(x)
        x = PReLU()(x)
        x = Dropout(params['dropout'])(x)

        # 第一个残差连接
        x_res2 = Dense(params['units2'])(x_res1)
        x = Add()([x, x_res2])

        x = Dense(params['units3'], kernel_regularizer=l2(params['l2_reg']))(x)
        x = BatchNormalization()(x)
        x = PReLU()(x)
        x = Dropout(params['dropout'])(x)

        x = Dense(params['units4'], kernel_regularizer=l2(params['l2_reg']))(x)
        x = BatchNormalization()(x)
        x = PReLU()(x)

        # 输出层前的额外层，确保输出稳定性
        x = Dense(8, activation='linear')(x)

        # 输出层
        outputs = Dense(1)(x)

        # 创建模型
        model = Model(inputs=inputs, outputs=outputs, name="MLP_Enhanced")

        # 编译模型
        optimizer = Adam(
            learning_rate=params['learning_rate'],
            clipnorm=0.7,
            clipvalue=0.3
        )
        model.compile(
            optimizer=optimizer,
            loss='mse',  # 使用MSE损失函数
            metrics=[RootMeanSquaredError(name='rmse'), MeanAbsoluteError(name='mae')]
        )

        return model

    def create_lstm_model(self, params=None):
        """创建改进的LSTM模型，解决过拟合和长期依赖问题，适用于GPU训练"""
        if params is None:
            params = {
                'lstm_units1': 128,  # 增加单元数量，利用GPU计算能力
                'lstm_units2': 64,
                'dense_units1': 64,
                'dense_units2': 32,
                'dropout': 0.2,  # 保持适度的dropout
                'learning_rate': 0.0008,
                'l2_reg': 0.0001  # 轻微的正则化
            }

        # 使用函数式API构建模型
        inputs = Input(shape=(self.look_back, len(self.features)))

        # 批归一化输入
        normalized = BatchNormalization()(inputs)

        # 使用LSTM，GPU上性能更好
        rnn1 = Bidirectional(LSTM(
            params['lstm_units1'],
            return_sequences=True,
            kernel_regularizer=l2(params['l2_reg']/2),
            recurrent_dropout=0.0,  # GPU上不支持循环dropout，设为0
            activation='tanh',
            unroll=True  # 在GPU上展开循环以提高性能
        ))(normalized)
        rnn1 = BatchNormalization()(rnn1)

        # 添加注意力机制
        attention_layer = Attention()([rnn1, rnn1])

        # 捕获全局信息
        rnn2 = Bidirectional(LSTM(
            params['lstm_units2'],
            kernel_regularizer=l2(params['l2_reg']/4)
        ))(attention_layer)
        rnn2 = BatchNormalization()(rnn2)

        # 增强后端网络
        x = Dense(params['dense_units1'])(rnn2)
        x = BatchNormalization()(x)
        x = PReLU()(x)
        x = Dropout(params['dropout'])(x)

        x = Dense(params['dense_units2'])(x)
        x = BatchNormalization()(x)
        x = PReLU()(x)

        # 输出层
        outputs = Dense(1)(x)

        # 创建模型
        model = Model(inputs=inputs, outputs=outputs, name="LSTM_Enhanced")

        # 编译模型
        optimizer = Adam(
            learning_rate=params['learning_rate'],
            clipnorm=0.5,
            clipvalue=0.3,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=[RootMeanSquaredError(name='rmse'), MeanAbsoluteError(name='mae')]
        )

        return model

    def create_cnn_model(self, params=None):
        """创建改进的CNN模型，解决局部特征提取问题和特征丢失问题，适用于GPU训练"""
        if params is None:
            params = {
                'filters1': 128,  # 增加滤波器数量，利用GPU计算能力
                'filters2': 64,
                'filters3': 32,
                'kernel_size': 3,
                'pool_size': 2,
                'dense_units1': 64,
                'dense_units2': 32,
                'dropout': 0.15,  # 保持适度的dropout
                'learning_rate': 0.0008,
                'l2_reg': 0.0001  # 轻微的正则化
            }

        # 使用函数式API构建模型
        inputs = Input(shape=(self.look_back, len(self.features)))

        # 批归一化输入
        x = BatchNormalization()(inputs)

        # 多分支卷积网络
        # 分支1: 较小卷积核
        conv1 = Conv1D(
            filters=params['filters1'],
            kernel_size=2,
            padding='same',
            kernel_regularizer=l2(params['l2_reg']/4),
            activation='relu'  # 使用ReLU激活函数，在GPU上更高效
        )(x)
        conv1 = BatchNormalization()(conv1)

        # 分支2: 中等卷积核
        conv2 = Conv1D(
            filters=params['filters1'],
            kernel_size=3,
            padding='same',
            kernel_regularizer=l2(params['l2_reg']/4),
            activation='relu'
        )(x)
        conv2 = BatchNormalization()(conv2)

        # 分支3: 较大卷积核
        conv3 = Conv1D(
            filters=params['filters1'],
            kernel_size=5,
            padding='same',
            kernel_regularizer=l2(params['l2_reg']/4),
            activation='relu'
        )(x)
        conv3 = BatchNormalization()(conv3)

        # 合并分支
        merged = Concatenate()([conv1, conv2, conv3])

        # 添加额外卷积层精炼特征
        refined = Conv1D(
            filters=params['filters2'],
            kernel_size=3,
            padding='same',
            kernel_regularizer=l2(params['l2_reg']/4),
            activation='relu'
        )(merged)
        refined = BatchNormalization()(refined)
        refined = MaxPooling1D(pool_size=params['pool_size'])(refined)
        refined = Dropout(params['dropout'])(refined)

        # 再添加一层卷积
        refined = Conv1D(
            filters=params['filters3'],
            kernel_size=3,
            padding='same',
            kernel_regularizer=l2(params['l2_reg']/4),
            activation='relu'
        )(refined)
        refined = BatchNormalization()(refined)

        # 全局特征
        global_feature = GlobalAveragePooling1D()(refined)

        # 增强后端网络
        x = Dense(params['dense_units1'], activation='relu')(global_feature)
        x = BatchNormalization()(x)
        x = Dropout(params['dropout'])(x)

        x = Dense(params['dense_units2'], activation='relu')(x)
        x = BatchNormalization()(x)

        # 输出层
        outputs = Dense(1)(x)

        # 创建模型
        model = Model(inputs=inputs, outputs=outputs, name="CNN_Enhanced")

        # 编译模型
        optimizer = Adam(
            learning_rate=params['learning_rate'],
            clipnorm=0.5,
            clipvalue=0.3
        )
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=[RootMeanSquaredError(name='rmse'), MeanAbsoluteError(name='mae')]
        )

        return model

    def test_models(self, epochs=5, batch_size=16):
        """小规模测试所有模型"""
        print("\n开始小规模测试...")

        models_to_test = ['MLP', 'LSTM', 'CNN']
        results = {}

        # 创建小规模数据
        test_size = min(100, len(self.X_train_ts))
        X_test_small_np = self.X_train_ts[:test_size]
        y_test_small_np = self.y_train_ts[:test_size]

        # 创建 tf.data.Dataset
        test_dataset_small = self._create_dataset(X_test_small_np, y_test_small_np, batch_size)

        print(f"使用{test_size}个样本进行小规模测试")

        for model_type in models_to_test:
            print(f"\n测试 {model_type} 模型...")

            # 创建模型
            if model_type == 'MLP':
                model = self.create_mlp_model()
            elif model_type == 'LSTM':
                model = self.create_lstm_model()
            elif model_type == 'CNN':
                model = self.create_cnn_model()

            print(f"{model_type} 模型摘要:")
            model.summary()

            # 简单训练几个epoch
            model.fit(
                test_dataset_small, # 使用tf.data.Dataset
                epochs=epochs,
                # batch_size 参数在Dataset中已定义，这里不需要
                verbose=1
            )

            # 评估模型
            loss, rmse, mae = model.evaluate(test_dataset_small, verbose=0) # 使用tf.data.Dataset
            results[model_type] = {'loss': loss, 'rmse': rmse, 'mae': mae}

            # 试一下预测
            preds = model.predict(test_dataset_small.take(1)) # 预测也使用Dataset，取一个批次
            actuals_batch = list(test_dataset_small.take(1).as_numpy_iterator())[0][1]
            print(f"样本预测结果 vs 实际值 (来自一个小批次):")
            for i in range(min(5, len(preds))):
                print(f"预测: {preds[i][0]:.4f}, 实际: {actuals_batch[i]:.4f}")

        print("\n小规模测试结果摘要:")
        for model_type, metrics in results.items():
            print(f"{model_type}: 损失={metrics['loss']:.4f}, RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")

        return results

    def train_model(self, model_type, params=None, epochs=TRAINING_CONFIG['epochs'],
                   batch_size=TRAINING_CONFIG['batch_size'], resume_training=False,
                   run_small_test=False):
        """训练模型，可选是否先进行小规模测试"""

        # 如果需要先进行小规模测试
        if run_small_test:
            self.test_models(epochs=5, batch_size=batch_size//2)

        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 设置模型保存路径
        model_path = f'saved_models/{model_type.lower()}_{timestamp}'
        checkpoint_path = f'checkpoints/{model_type.lower()}_{timestamp}'
        log_path = f'logs/{model_type.lower()}_{timestamp}'

        # 确定模型类型和创建/加载模型
        if resume_training and os.path.exists(f'{model_path}.h5'):
            print(f"正在加载已有模型: {model_path}.h5")
            model = load_model(f'{model_path}.h5')
        else:
            if model_type.lower() == 'mlp':
                params = params or MLP_CONFIG
                model = self.create_mlp_model(params)
            elif model_type.lower() == 'lstm':
                params = params or LSTM_CONFIG
                model = self.create_lstm_model(params)
            elif model_type.lower() == 'cnn':
                params = params or CNN_CONFIG
                model = self.create_cnn_model(params)
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")

        # 创建回调函数
        callbacks = [
            # 早停策略
            EarlyStopping(
                monitor='val_loss',
                patience=TRAINING_CONFIG['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            # 学习率调度器
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=TRAINING_CONFIG['reduce_lr_patience'],
                min_lr=TRAINING_CONFIG['min_lr'],
                verbose=1
            ),
            # 模型检查点
            ModelCheckpoint(
                filepath=f'{checkpoint_path}.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            # 训练日志记录
            CSVLogger(f'{log_path}.csv', append=resume_training)
        ]

        # 创建 tf.data.Dataset
        train_dataset = self._create_dataset(self.X_train_ts, self.y_train_ts, batch_size, shuffle=True)
        val_dataset = self._create_dataset(self.X_val_ts, self.y_val_ts, batch_size)

        # 训练开始时间
        start_time = time.time()

        # 训练模型
        history = model.fit(
            train_dataset, # 使用tf.data.Dataset
            epochs=epochs,
            validation_data=val_dataset, # 使用tf.data.Dataset
            callbacks=callbacks,
            verbose=1
        )

        # 训练结束时间
        end_time = time.time()
        training_time = end_time - start_time

        # 保存最终模型
        model.save(f'{model_path}.h5')

        # 输出训练时间和GPU使用情况
        print(f"\n{model_type} 模型训练完成，耗时 {training_time:.2f} 秒")
        print("GPU使用情况:")
        try:
            gpu_devices = tf.config.list_physical_devices('GPU')
            for device in gpu_devices:
                print(f" - {device.name}")
        except:
            print("未检测到GPU设备")

        # 保存模型和训练历史
        self.models[model_type] = model
        self.history[model_type] = history.history

        # 在测试集上进行预测
        self.evaluate_model(model_type, timestamp)

        # 绘制训练历史
        self.plot_training_history(model_type, history, timestamp)

        # --- BEGIN: Added code for saving to JSON ---
        model_info_to_save = {
            'model_type': model_type,
            'timestamp': timestamp,
            'data_file_used': self.data_file,
            'look_back': self.look_back,
            'model_parameters': params, # Actual parameters used for model creation
            'training_time_seconds': round(training_time, 2),
            'evaluation_metrics': self.metrics.get(model_type, {}),
            'saved_model_path': f'{model_path}.h5',
            'checkpoint_path': f'{checkpoint_path}.h5',
            'log_csv_path': f'{log_path}.csv',
            'results_metrics_path': f'results/{model_type.lower()}_{timestamp}_metrics.txt',
            'training_history_plot_path': f'results/{model_type.lower()}_{timestamp}_training_history.png',
            'prediction_details_plot_path': f'results/{model_type.lower()}_{timestamp}_prediction_details.png'
        }

        # Ensure results directory exists for the JSON file (though create_directories should handle it)
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True) # Harmless if already exists
        json_output_path = os.path.join(results_dir, 'all_models_training_summary.json')

        all_models_data = {}
        if os.path.exists(json_output_path):
            try:
                with open(json_output_path, 'r') as f:
                    content = f.read()
                    if content.strip(): # Check if file is not empty
                        all_models_data = json.loads(content)
                    else:
                        print(f"信息: JSON文件 '{json_output_path}' 为空，将创建新内容。")
            except json.JSONDecodeError:
                print(f"警告: JSON文件 '{json_output_path}' 损坏或格式不正确，将创建新文件。")
            except Exception as e:
                print(f"警告: 读取JSON文件 '{json_output_path}' 时发生错误: {e}，将创建新文件。")
        
        # Store runs as a list under each model_type key
        if model_type not in all_models_data:
            all_models_data[model_type] = []
        all_models_data[model_type].append(model_info_to_save)

        try:
            with open(json_output_path, 'w') as f:
                json.dump(all_models_data, f, indent=4, ensure_ascii=False) # ensure_ascii=False for Chinese characters if any in paths
            print(f"模型信息已更新到: {json_output_path}")
        except TypeError as te:
            print(f"错误: 无法序列化模型信息到JSON (可能是参数包含不支持的类型): {te}")
            # Attempt to save with problematic parts converted to string
            try:
                # A more robust serialization would involve custom encoders or careful type checking
                # For now, a simple fallback: convert entire params to str if it's the cause
                if 'model_parameters' in str(te).lower(): # Crude check if params is the issue
                    model_info_to_save['model_parameters'] = str(params)
                # Re-attempt saving if model_info_to_save was modified
                if model_type in all_models_data and all_models_data[model_type]:
                    all_models_data[model_type][-1] = model_info_to_save # Update the last appended item

                with open(json_output_path, 'w') as f:
                    json.dump(all_models_data, f, indent=4, ensure_ascii=False)
                print(f"模型信息已更新到 (部分参数可能已转换为字符串): {json_output_path}")

            except Exception as e_fallback:
                 print(f"错误: 尝试回退保存到JSON失败: {e_fallback}")
        except Exception as e:
            print(f"错误: 无法将模型信息保存到JSON: {e}")
        # --- END: Added code for saving to JSON ---

        return model, history

    def evaluate_model(self, model_type, timestamp):
        """评估模型，包括准确率计算"""
        model = self.models[model_type]

        # 创建测试集的 tf.data.Dataset
        # 注意：评估时 batch_size 可以设置得大一些，并且不需要 shuffle
        test_dataset = self._create_dataset(self.X_test_ts, self.y_test_ts, batch_size=TRAINING_CONFIG['batch_size'] * 2) 

        # 在测试集上进行预测
        predictions_scaled = model.predict(test_dataset)

        # 反向转换预测值
        predictions = self.target_scaler.inverse_transform(predictions_scaled)
        actual = self.target_scaler.inverse_transform(self.y_test_ts.reshape(-1, 1))

        # 保存预测结果
        self.predictions[model_type] = predictions

        # 计算评估指标
        mse = mean_squared_error(actual, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predictions)
        r2 = r2_score(actual, predictions)
        mape = np.mean(np.abs((actual - predictions) / actual)) * 100

        # 计算准确率 - 定义为预测值与实际值的相对误差在5%以内的比例
        tolerance = 0.05  # 5%的容忍度
        correct_predictions = np.sum(np.abs((predictions - actual) / actual) <= tolerance)
        accuracy = (correct_predictions / len(actual)) * 100

        # 保存评估指标
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'Accuracy': accuracy  # 添加准确率指标
        }
        self.metrics[model_type] = metrics

        # 保存评估结果到文件
        results_file = f'results/{model_type.lower()}_{timestamp}_metrics.txt'
        with open(results_file, 'w') as f:
            f.write(f"{model_type} 模型评估结果:\n")
            f.write(f"均方误差 (MSE): {mse:.4f}\n")
            f.write(f"均方根误差 (RMSE): {rmse:.4f}\n")
            f.write(f"平均绝对误差 (MAE): {mae:.4f}\n")
            f.write(f"决定系数 (R2): {r2:.4f}\n")
            f.write(f"平均绝对百分比误差 (MAPE): {mape:.4f}%\n")
            f.write(f"准确率 (Accuracy): {accuracy:.2f}%\n")

        print(f"\n评估结果已保存到: {results_file}")

        # 输出评估指标
        print(f"\n{model_type} 模型评估结果:")
        print(f"均方误差 (MSE): {mse:.4f}")
        print(f"均方根误差 (RMSE): {rmse:.4f}")
        print(f"平均绝对误差 (MAE): {mae:.4f}")
        print(f"决定系数 (R2): {r2:.4f}")
        print(f"平均绝对百分比误差 (MAPE): {mape:.4f}%")
        print(f"准确率 (Accuracy): {accuracy:.2f}%")

    def plot_training_history(self, model_type, history, timestamp):
        """绘制训练历史"""
        plt.figure(figsize=(15, 10))

        # 绘制损失曲线
        plt.subplot(2, 2, 1)
        plt.plot(history.history['loss'], label='训练损失')
        plt.plot(history.history['val_loss'], label='验证损失')
        plt.title(f'{model_type} 模型损失曲线')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.legend()

        # 绘制MAE曲线
        plt.subplot(2, 2, 2)
        plt.plot(history.history['mae'], label='训练MAE')
        plt.plot(history.history['val_mae'], label='验证MAE')
        plt.title(f'{model_type} 模型MAE曲线')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()

        # 绘制学习率曲线（如果存在）
        if 'lr' in history.history:
            plt.subplot(2, 2, 3)
            plt.plot(history.history['lr'])
            plt.title(f'{model_type} 学习率调整')
            plt.xlabel('Epoch')
            plt.ylabel('学习率')
            plt.yscale('log')

        # 绘制预测与实际值对比（使用测试集的一部分）
        plt.subplot(2, 2, 4)

        # 获取预测结果
        if model_type in self.predictions:
            predictions = self.predictions[model_type]
            # 反向转换测试集目标值
            actual = self.target_scaler.inverse_transform(self.y_test_ts.reshape(-1, 1))

            # 只显示前30个点以避免图表过于拥挤
            display_points = min(30, len(predictions))

            plt.plot(range(display_points), actual[:display_points], 'b-', label='实际值')
            plt.plot(range(display_points), predictions[:display_points], 'r--', label='预测值')
            plt.title(f'{model_type} 预测 vs 实际')
            plt.xlabel('样本')
            plt.ylabel('价格')
            plt.legend()

        plt.tight_layout()
        plt.savefig(f'results/{model_type.lower()}_{timestamp}_training_history.png')
        plt.close()

        print(f"\n训练历史图表已保存到: results/{model_type.lower()}_{timestamp}_training_history.png")

        # 额外保存一个预测结果的详细图表
        if model_type in self.predictions:
            plt.figure(figsize=(12, 6))

            predictions = self.predictions[model_type]
            actual = self.target_scaler.inverse_transform(self.y_test_ts.reshape(-1, 1))

            # 计算预测误差
            errors = predictions - actual

            # 绘制预测与实际值
            plt.subplot(2, 1, 1)
            plt.plot(actual, 'b-', label='实际值')
            plt.plot(predictions, 'r--', label='预测值')
            plt.title(f'{model_type} 预测结果详细图')
            plt.ylabel('价格')
            plt.legend()

            # 绘制误差
            plt.subplot(2, 1, 2)
            plt.bar(range(len(errors)), errors.flatten())
            plt.axhline(y=0, color='r', linestyle='-')
            plt.title('预测误差')
            plt.xlabel('样本')
            plt.ylabel('误差')

            plt.tight_layout()
            plt.savefig(f'results/{model_type.lower()}_{timestamp}_prediction_details.png')
            plt.close()

            print(f"预测详细图表已保存到: results/{model_type.lower()}_{timestamp}_prediction_details.png")

    def compare_models(self):
        """比较不同模型的性能，包括准确率指标"""
        if not self.metrics:
            print("没有可比较的模型，请先训练模型")
            return

        # 创建比较结果表格
        comparison = pd.DataFrame({
            '模型': list(self.metrics.keys()),
            'MSE': [m['MSE'] for m in self.metrics.values()],
            'RMSE': [m['RMSE'] for m in self.metrics.values()],
            'MAE': [m['MAE'] for m in self.metrics.values()],
            'R2': [m['R2'] for m in self.metrics.values()],
            'MAPE (%)': [m['MAPE'] for m in self.metrics.values()],
            '准确率 (%)': [m['Accuracy'] for m in self.metrics.values()]
        })

        # 按RMSE排序
        comparison = comparison.sort_values('RMSE')

        print("\n模型比较结果:")
        print(comparison)

        # 找出最佳模型
        best_model_rmse = comparison.iloc[0]['模型']
        print(f"\n根据RMSE指标，最佳模型是: {best_model_rmse}")

        # 按准确率排序找出最佳模型
        comparison_by_acc = comparison.sort_values('准确率 (%)', ascending=False)
        best_model_acc = comparison_by_acc.iloc[0]['模型']
        print(f"根据准确率指标，最佳模型是: {best_model_acc}")

        return comparison

if __name__ == "__main__":
    try:
        # 创建保存模型的目录
        os.makedirs('saved_models', exist_ok=True)

        # 定义数据文件路径
        data_file_old_format = "model_data/date1.csv"  # 用户指定使用此文件
        data_file_new_format = "model_data/date2.csv"
        # 工作区路径: /Users/a/project/models
        # 因此, date1.csv 的绝对路径应为 /Users/a/project/models/model_data/date1.csv
        absolute_path_date1 = os.path.join("/Users/a/project/models", "model_data/date1.csv")

        chosen_data_file = None
        chosen_target_col = None

        # 根据用户请求，优先尝试加载 date1.csv
        if os.path.exists(data_file_old_format):
            chosen_data_file = data_file_old_format
            chosen_target_col = '收盘价'
            print(f"使用用户指定的数据文件: {chosen_data_file}，目标列: {chosen_target_col}")
        elif os.path.exists(absolute_path_date1): # 尝试绝对路径的 date1.csv
            chosen_data_file = absolute_path_date1
            chosen_target_col = '收盘价'
            print(f"使用绝对路径的用户指定数据文件: {chosen_data_file}，目标列: {chosen_target_col}")
        elif os.path.exists(data_file_new_format): # 如果 date1.csv 未找到，尝试 date2.csv
            chosen_data_file = data_file_new_format
            chosen_target_col = 'close'
            print(f"警告: {data_file_old_format} (包括绝对路径) 未找到。尝试使用备选数据文件: {chosen_data_file}，目标列: {chosen_target_col}")
        else:
            error_message = f"错误: 主要数据文件 {data_file_old_format} (包括相对路径和绝对路径 {absolute_path_date1}) 及备选文件 {data_file_new_format}均未找到。请检查文件路径。"
            print(error_message)
            raise FileNotFoundError(error_message)

        # 初始化模型类
        # look_back=15 保持不变，如需调整可修改
        dl_models = DeepLearningModels(data_file=chosen_data_file, target_col=chosen_target_col, look_back=30)

        # 先进行小规模测试
        print("\n进行小规模测试...")
        # 适当增大测试时的batch_size以更好地利用GPU，减少epochs以便快速测试
        dl_models.test_models(epochs=3, batch_size=32) # 小规模测试 epochs 可以少一些

        # 训练MLP模型
        print("\n训练MLP模型...")
        # 移除 epochs=1 的覆盖, 使用TRAINING_CONFIG中的默认值或方法内默认值
        # 增大batch_size以优化GPU训练，可根据实际GPU内存调整
        dl_models.train_model('MLP', batch_size=64)

        # 训练LSTM模型
        # print("\n训练LSTM模型...") # 原本的打印语句可以注释掉或移除
        # dl_models.train_model('LSTM', batch_size=64) # 原本的调用语句

        print("\n训练LSTM模型 (使用config中的LSTM_CONFIG和新的look_back)...")
        dl_models.train_model('LSTM', batch_size=64) # 恢复使用默认配置

        # 训练CNN模型
        print("\n训练CNN模型...")
        dl_models.train_model('CNN', batch_size=64)

        # 比较三个模型的性能
        print("\n比较三个模型的性能:")
        comparison = dl_models.compare_models()

        print("\n测试完成！")

    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()