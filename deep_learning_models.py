import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 使用tensorflow-macos 2.9.0
import tensorflow as tf

# 检查是否在Apple Silicon芯片上运行
import platform
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
        print("未检测到tensorflow-metal，将使用CPU运行")
        print("安装提示: pip install tensorflow-metal")

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.optimizers import Adam, RMSprop
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import time
import random
from datetime import datetime
from config import TRAINING_CONFIG, MLP_CONFIG, LSTM_CONFIG, CNN_CONFIG

# 设置随机种子，保证结果可复现
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)  # tensorflow 2.9.0使用这种方式设置随机种子

print(f"TensorFlow版本: {tf.__version__}")
print(f"Keras版本: {tf.keras.__version__}")

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
        directories = ['saved_models', 'logs', 'checkpoints', 'results']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def load_and_prepare_data(self):
        """加载并准备数据"""
        print(f"正在加载数据: {self.data_file}")
        self.df = pd.read_csv(self.data_file)
        
        # 确保日期列为日期时间类型
        self.df['日期'] = pd.to_datetime(self.df['日期'])
        
        # 按日期升序排序
        self.df = self.df.sort_values('日期')
        
        # 添加更多的技术指标特征
        self.add_technical_indicators()
        
        # 显示数据的基本信息
        print(f"数据行数: {len(self.df)}")
        print(f"数据列数: {len(self.df.columns)}")
        print(f"时间范围: {self.df['日期'].min()} 至 {self.df['日期'].max()}")
        
        # 选择要使用的特征
        self.features = self.select_features()
        
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
        # 基本价格特征
        self.df['价格变化'] = self.df['收盘价'].diff()
        self.df['价格变化率'] = self.df['收盘价'].pct_change()
        self.df['日内波动率'] = (self.df['最高价'] - self.df['最低价']) / self.df['开盘价']
        
        # 移动平均线特征
        for ma in [5, 10, 20, 30]:
            self.df[f'MA{ma}_diff'] = self.df['收盘价'] - self.df[f'MA{ma}']
            self.df[f'MA{ma}_slope'] = self.df[f'MA{ma}'].diff()
            self.df[f'MA{ma}_std'] = self.df['收盘价'].rolling(window=ma).std()
        
        # 波动率指标
        self.df['20日波动率'] = self.df['收盘价'].rolling(window=20).std() / self.df['收盘价'].rolling(window=20).mean()
        self.df['10日波动率'] = self.df['收盘价'].rolling(window=10).std() / self.df['收盘价'].rolling(window=10).mean()
        
        # MACD相关指标
        if 'MACD' in self.df.columns:
            self.df['MACD_diff'] = self.df['MACD'].diff()
            self.df['MACD_slope'] = self.df['MACD'].diff(3)
        
        # RSI相关指标
        if 'RSI' in self.df.columns:
            self.df['RSI_diff'] = self.df['RSI'].diff()
            self.df['RSI_slope'] = self.df['RSI'].diff(3)
            self.df['RSI_MA5'] = self.df['RSI'].rolling(window=5).mean()
        
        # KDJ相关指标
        if all(x in self.df.columns for x in ['K', 'D', 'J']):
            self.df['KD_diff'] = self.df['K'] - self.df['D']
            self.df['KD_cross'] = (self.df['K'] > self.df['D']).astype(int)
        
        # 成交量特征
        self.df['成交量变化率'] = self.df['成交量'].pct_change()
        self.df['成交量MA5'] = self.df['成交量'].rolling(window=5).mean()
        self.df['量价背离'] = (self.df['价格变化率'] * self.df['成交量变化率'] < 0).astype(int)
        
        # 趋势特征
        self.df['上升趋势'] = (self.df['收盘价'] > self.df['MA5']).astype(int)
        self.df['强势上升'] = ((self.df['MA5'] > self.df['MA10']) & (self.df['MA10'] > self.df['MA20'])).astype(int)
        
        # 时间特征
        self.df['星期'] = self.df['日期'].dt.dayofweek
        self.df['月份'] = self.df['日期'].dt.month
        self.df['季度'] = self.df['日期'].dt.quarter
        
        # 删除包含无穷值的行
        self.df = self.df.replace([np.inf, -np.inf], np.nan)
    
    def handle_missing_values(self):
        """处理缺失值"""
        # 对于缺失值较少的列，使用前向填充
        self.df[self.features] = self.df[self.features].fillna(method='ffill')
        
        # 仍然存在的缺失值使用均值填充
        self.df[self.features] = self.df[self.features].fillna(self.df[self.features].mean())
        
        # 删除仍然包含缺失值的行
        self.df = self.df.dropna(subset=self.features)
    
    def select_features(self):
        """选择模型使用的特征"""
        # 基本价格和成交量特征
        basic_features = ['收盘价', '开盘价', '最高价', '最低价', '成交量']
        
        # 技术指标特征
        tech_features = []
        potential_features = ['涨跌幅', 'MA5', 'MA10', 'MA20', 'MA30', 'EMA12', 'EMA26', 'RSI', 'MACD', 'K', 'D', 'J']
        
        for feature in potential_features:
            if feature in self.df.columns:
                tech_features.append(feature)
        
        # 将所有特征合并
        all_features = basic_features + tech_features
        
        # 确保所有特征都是数值类型
        for feature in all_features[:]:  # 使用副本进行迭代
            if feature in self.df.columns and not pd.api.types.is_numeric_dtype(self.df[feature]):
                try:
                    self.df[feature] = pd.to_numeric(self.df[feature], errors='coerce')
                except:
                    all_features.remove(feature)
                    print(f"移除非数值特征: {feature}")
        
        print(f"选择的特征: {all_features}")
        return all_features
    
    def split_data(self, test_size=0.1, val_size=0.1):
        """分割数据为训练集、验证集和测试集"""
        # 确保数据按时间排序
        data = self.df.sort_values('日期')
        
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
        self.train_dates = data['日期'].iloc[:n - test_samples - val_samples]
        self.val_dates = data['日期'].iloc[n - test_samples - val_samples:n - test_samples]
        self.test_dates = data['日期'].iloc[n - test_samples:]
        
        print(f"训练集大小: {len(X_train)}")
        print(f"验证集大小: {len(X_val)}")
        print(f"测试集大小: {len(X_test)}")
    
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
    
    def create_mlp_model(self, params=None):
        """创建多层感知机模型"""
        if params is None:
            params = {
                'units1': 256,
                'units2': 128,
                'units3': 64,
                'units4': 32,
                'dropout': 0.3,
                'learning_rate': 0.001,
                'l2_reg': 0.001
            }
        
        from tensorflow.keras.regularizers import l2
        
        model = Sequential(name="MLP")
        
        # 输入层
        model.add(BatchNormalization(input_shape=(self.look_back, len(self.features))))
        model.add(Flatten())
        
        # 第一个隐藏层
        model.add(Dense(params['units1'], activation='relu',
                       kernel_regularizer=l2(params['l2_reg'])))
        model.add(BatchNormalization())
        model.add(Dropout(params['dropout']))
        
        # 第二个隐藏层
        model.add(Dense(params['units2'], activation='relu',
                       kernel_regularizer=l2(params['l2_reg'])))
        model.add(BatchNormalization())
        model.add(Dropout(params['dropout']))
        
        # 第三个隐藏层
        model.add(Dense(params['units3'], activation='relu',
                       kernel_regularizer=l2(params['l2_reg'])))
        model.add(BatchNormalization())
        model.add(Dropout(params['dropout']))
        
        # 第四个隐藏层
        model.add(Dense(params['units4'], activation='relu',
                       kernel_regularizer=l2(params['l2_reg'])))
        model.add(BatchNormalization())
        model.add(Dropout(params['dropout']))
        
        # 输出层
        model.add(Dense(1))
        
        # 编译模型
        optimizer = Adam(learning_rate=params['learning_rate'],
                        clipnorm=1.0,
                        clipvalue=0.5)
        model.compile(optimizer=optimizer,
                     loss='huber',
                     metrics=['mae'])
        
        return model
    
    def create_lstm_model(self, params=None):
        """创建LSTM模型"""
        if params is None:
            params = {
                'lstm_units1': 128,
                'lstm_units2': 64,
                'lstm_units3': 32,
                'dense_units1': 64,
                'dense_units2': 32,
                'dropout': 0.3,
                'learning_rate': 0.001,
                'l2_reg': 0.001
            }
        
        from tensorflow.keras.regularizers import l2
        
        model = Sequential(name="LSTM")
        
        # 输入层标准化
        model.add(BatchNormalization(input_shape=(self.look_back, len(self.features))))
        
        # 第一个LSTM层
        model.add(LSTM(params['lstm_units1'],
                      return_sequences=True,
                      kernel_regularizer=l2(params['l2_reg']),
                      recurrent_regularizer=l2(params['l2_reg'])))
        model.add(BatchNormalization())
        model.add(Dropout(params['dropout']))
        
        # 第二个LSTM层
        model.add(LSTM(params['lstm_units2'],
                      return_sequences=True,
                      kernel_regularizer=l2(params['l2_reg']),
                      recurrent_regularizer=l2(params['l2_reg'])))
        model.add(BatchNormalization())
        model.add(Dropout(params['dropout']))
        
        # 第三个LSTM层
        model.add(LSTM(params['lstm_units3'],
                      kernel_regularizer=l2(params['l2_reg']),
                      recurrent_regularizer=l2(params['l2_reg'])))
        model.add(BatchNormalization())
        model.add(Dropout(params['dropout']))
        
        # 全连接层
        model.add(Dense(params['dense_units1'], activation='relu',
                       kernel_regularizer=l2(params['l2_reg'])))
        model.add(BatchNormalization())
        model.add(Dropout(params['dropout']))
        
        model.add(Dense(params['dense_units2'], activation='relu',
                       kernel_regularizer=l2(params['l2_reg'])))
        model.add(BatchNormalization())
        model.add(Dropout(params['dropout']))
        
        # 输出层
        model.add(Dense(1))
        
        # 编译模型
        optimizer = Adam(learning_rate=params['learning_rate'],
                        clipnorm=1.0,
                        clipvalue=0.5)
        model.compile(optimizer=optimizer,
                     loss='huber',  # 使用Huber损失函数
                     metrics=['mae'])
        
        return model
    
    def create_cnn_model(self, params=None):
        """创建一维CNN模型"""
        if params is None:
            params = {
                'filters1': 256,
                'filters2': 128,
                'filters3': 64,
                'filters4': 32,
                'kernel_size': 5,
                'pool_size': 2,
                'dense_units1': 128,
                'dense_units2': 64,
                'dropout': 0.2,
                'learning_rate': 0.0005,
                'l2_reg': 0.0005
            }
        
        from tensorflow.keras.regularizers import l2
        from tensorflow.keras.layers import Add
        
        model = Sequential(name="CNN")
        
        # 第一个卷积层
        model.add(BatchNormalization(input_shape=(self.look_back, len(self.features))))
        model.add(Conv1D(filters=params['filters1'],
                        kernel_size=params['kernel_size'],
                        activation='relu',
                        padding='same',
                        kernel_regularizer=l2(params['l2_reg'])))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=params['pool_size']))
        model.add(Dropout(params['dropout']))
        
        # 第二个卷积层
        model.add(Conv1D(filters=params['filters2'],
                        kernel_size=params['kernel_size'],
                        activation='relu',
                        padding='same',
                        kernel_regularizer=l2(params['l2_reg'])))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=params['pool_size']))
        model.add(Dropout(params['dropout']))
        
        # 第三个卷积层
        model.add(Conv1D(filters=params['filters3'],
                        kernel_size=params['kernel_size'],
                        activation='relu',
                        padding='same',
                        kernel_regularizer=l2(params['l2_reg'])))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=params['pool_size']))
        model.add(Dropout(params['dropout']))
        
        # 第四个卷积层
        model.add(Conv1D(filters=params['filters4'],
                        kernel_size=params['kernel_size'],
                        activation='relu',
                        padding='same',
                        kernel_regularizer=l2(params['l2_reg'])))
        model.add(BatchNormalization())
        model.add(Dropout(params['dropout']))
        
        # 展平层
        model.add(Flatten())
        
        # 全连接层
        model.add(Dense(params['dense_units1'],
                       activation='relu',
                       kernel_regularizer=l2(params['l2_reg'])))
        model.add(BatchNormalization())
        model.add(Dropout(params['dropout']))
        
        model.add(Dense(params['dense_units2'],
                       activation='relu',
                       kernel_regularizer=l2(params['l2_reg'])))
        model.add(BatchNormalization())
        model.add(Dropout(params['dropout']))
        
        # 输出层
        model.add(Dense(1))
        
        # 编译模型
        optimizer = Adam(learning_rate=params['learning_rate'],
                        clipnorm=1.0,
                        clipvalue=0.5)
        model.compile(optimizer=optimizer,
                     loss='huber',
                     metrics=['mae'])
        
        return model
    
    def train_model(self, model_type, params=None, epochs=TRAINING_CONFIG['epochs'], 
                   batch_size=TRAINING_CONFIG['batch_size'], resume_training=False):
        """训练模型"""
        
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
        
        # 训练开始时间
        start_time = time.time()
        
        # 训练模型
        history = model.fit(
            self.X_train_ts, self.y_train_ts,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_val_ts, self.y_val_ts),
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
        
        return model, history
    
    def evaluate_model(self, model_type, timestamp):
        """评估模型"""
        model = self.models[model_type]
        
        # 在测试集上进行预测
        predictions_scaled = model.predict(self.X_test_ts)
        
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
        
        # 保存评估指标
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
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
        
        print(f"\n评估结果已保存到: {results_file}")
        
        # 输出评估指标
        print(f"\n{model_type} 模型评估结果:")
        print(f"均方误差 (MSE): {mse:.4f}")
        print(f"均方根误差 (RMSE): {rmse:.4f}")
        print(f"平均绝对误差 (MAE): {mae:.4f}")
        print(f"决定系数 (R2): {r2:.4f}")
        print(f"平均绝对百分比误差 (MAPE): {mape:.4f}%")
    
    def plot_training_history(self, model_type, history, timestamp):
        """绘制训练历史"""
        plt.figure(figsize=(12, 4))
        
        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='训练损失')
        plt.plot(history.history['val_loss'], label='验证损失')
        plt.title(f'{model_type} 模型损失曲线')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.legend()
        
        # 绘制MAE曲线
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='训练MAE')
        plt.plot(history.history['val_mae'], label='验证MAE')
        plt.title(f'{model_type} 模型MAE曲线')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'results/{model_type.lower()}_{timestamp}_training_history.png')
        plt.close()
        
        print(f"\n训练历史图表已保存到: results/{model_type.lower()}_{timestamp}_training_history.png")
    
    def compare_models(self):
        """比较不同模型的性能"""
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
            'MAPE (%)': [m['MAPE'] for m in self.metrics.values()]
        })
        
        # 按RMSE排序
        comparison = comparison.sort_values('RMSE')
        
        print("\n模型比较结果:")
        print(comparison)
        
        # 找出最佳模型
        best_model = comparison.iloc[0]['模型']
        print(f"\n根据RMSE指标，最佳模型是: {best_model}")
        
        return comparison

if __name__ == "__main__":
    try:
        # 创建保存模型的目录
        os.makedirs('saved_models', exist_ok=True)
        
        # 初始化模型类
        dl_models = DeepLearningModels("data.csv", look_back=15)
        
        # 训练MLP模型
        print("\n训练MLP模型...")
        dl_models.train_model('MLP', epochs=50, batch_size=32)
        
        # 训练LSTM模型
        print("\n训练LSTM模型...")
        dl_models.train_model('LSTM', epochs=50, batch_size=32)
        
        # 训练CNN模型
        print("\n训练CNN模型...")
        dl_models.train_model('CNN', epochs=50, batch_size=32)
        
        # 比较三个模型的性能
        print("\n比较三个模型的性能:")
        comparison = dl_models.compare_models()
        
        print("\n测试完成！")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc() 