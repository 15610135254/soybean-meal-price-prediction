import os
import json
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# 配置
MODEL_DIR = "best_models"
SCALER_DIR = "scalers"
METRICS_FILE = "model_metrics.json"

# 模型文件
MODEL_FILES = {
    "MLP": "mlp_20250509_011024.h5",
    "LSTM": "lstm_20250509_011051.h5",
    "CNN": "cnn_20250509_011353.h5"
}

# 列名映射（旧数据集到新数据集的映射）
COLUMN_MAPPING = {
    '日期': 'date',
    '开盘价': 'open',
    '最高价': 'high',
    '最低价': 'low',
    '收盘价': 'close',
    '成交量': 'volume',
    '持仓量': 'hold',
    'MA5': 'MA_5',
    '波动率': 'HV_20',
    'ATR': 'ATR_14',
    'RSI': 'RSI_14',
    'OBV': 'OBV',
    'MACD': 'MACD'
}

def load_test_data():
    """加载测试数据"""
    try:
        # 使用相对路径加载测试数据
        df = pd.read_csv("model_data/date1.csv")
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        # 使用最后20%的数据作为测试集
        test_size = int(len(df) * 0.2)
        test_df = df.iloc[-test_size:]

        return test_df
    except Exception as e:
        print(f"加载测试数据时出错: {e}")
        return None

def prepare_test_data(test_df, look_back=30):
    """准备测试数据"""
    try:
        # 加载标准化器
        feature_scaler_path = os.path.join(SCALER_DIR, 'feature_scaler.pkl')
        target_scaler_path = os.path.join(SCALER_DIR, 'target_scaler.pkl')

        if not os.path.exists(feature_scaler_path) or not os.path.exists(target_scaler_path):
            print(f"错误: 标准化器文件 feature_scaler.pkl 或 target_scaler.pkl 未在 '{SCALER_DIR}' 目录中找到。")
            return None, None, None, None

        feature_scaler = joblib.load(feature_scaler_path)
        target_scaler = joblib.load(target_scaler_path)

        # 定义模型训练时使用的16个特征 (英文列名，不包括目标列'close')
        # 这个列表的顺序应与训练时feature_scaler拟合的顺序一致
        model_features = [
            'open', 'high', 'low', 'volume', 'hold', 'MA_5', 'HV_20', 'ATR_14',
            'RSI_14', 'OBV', 'MACD', 'a_close', 'c_close', 'LPR1Y',
            '大豆产量(万吨)', 'GDP'
        ]

        # 检查所有必要的特征列是否存在于test_df中
        missing_cols = [col for col in model_features if col not in test_df.columns]
        if missing_cols:
            print(f"错误: 测试数据中缺少以下必要的特征列: {missing_cols}")
            return None, None, None, None

        # 确保目标列存在
        if 'close' not in test_df.columns:
            print("错误: 测试数据中缺少目标列 'close'。")
            return None, None, None, None

        # 准备特征和目标
        X = test_df[model_features].values
        y = test_df['close'].values # 目标列

        # 标准化数据
        # 检查feature_scaler期望的特征数量是否与我们提供的匹配
        if feature_scaler.n_features_in_ != X.shape[1]:
            print(f"错误: 特征数量不匹配。feature_scaler期望 {feature_scaler.n_features_in_} 个特征，但测试数据提供了 {X.shape[1]} 个特征。")
            print(f"请确保 '{SCALER_DIR}/feature_scaler.pkl' 是为这 {len(model_features)} 个特定特征正确生成的。")
            return None, None, None, None

        X_scaled = feature_scaler.transform(X)
        y_scaled = target_scaler.transform(y.reshape(-1, 1))

        # 创建时间序列数据
        X_ts = []
        y_ts = []

        for i in range(look_back, len(X_scaled)):
            X_ts.append(X_scaled[i-look_back:i])
            y_ts.append(y_scaled[i])

        X_ts = np.array(X_ts)
        y_ts = np.array(y_ts)

        return X_ts, y_ts, target_scaler, model_features
    except Exception as e:
        print(f"准备测试数据时出错: {e}")
        return None, None, None, None

def evaluate_models():
    """评估所有模型并保存指标"""
    # 加载测试数据
    test_df = load_test_data()
    if test_df is None:
        print("无法加载测试数据，无法评估模型")
        return

    # 准备测试数据
    X_test, y_test, target_scaler, features = prepare_test_data(test_df)
    if X_test is None:
        print("无法准备测试数据，无法评估模型")
        return

    print(f"测试数据形状: X={X_test.shape}, y={y_test.shape}")
    print(f"使用的特征: {features}")

    # 评估结果
    metrics = {}

    # 评估每个模型
    for model_name, model_filename in MODEL_FILES.items():
        model_path = os.path.join(MODEL_DIR, model_filename)
        print(f"\n评估模型: {model_path}")

        if not os.path.exists(model_path):
            print(f"错误：模型文件 {model_path} 不存在。")
            continue

        try:
            # 加载模型
            model = load_model(model_path)

            # 预测
            predictions_scaled = model.predict(X_test)

            # 反标准化
            predictions = target_scaler.inverse_transform(predictions_scaled)
            actual = target_scaler.inverse_transform(y_test.reshape(-1, 1))

            # 计算指标
            mse = mean_squared_error(actual, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actual, predictions)
            r2 = r2_score(actual, predictions)

            # 计算准确率（基于MAPE）
            mape = np.mean(np.abs((actual - predictions) / actual)) * 100
            accuracy = max(0, 100 - mape)  # 将MAPE转换为准确率

            # 保存指标
            metrics[model_name.lower()] = {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2),
                'mape': float(mape),
                'accuracy': float(accuracy)
            }

            print(f"{model_name} 模型评估结果:")
            print(f"均方误差 (MSE): {mse:.4f}")
            print(f"均方根误差 (RMSE): {rmse:.4f}")
            print(f"平均绝对误差 (MAE): {mae:.4f}")
            print(f"决定系数 (R2): {r2:.4f}")
            print(f"平均绝对百分比误差 (MAPE): {mape:.4f}%")
            print(f"准确率: {accuracy:.2f}%")

        except Exception as e:
            print(f"评估模型 {model_name} 时出错: {e}")
            import traceback
            traceback.print_exc()

    # 保存指标到JSON文件
    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"\n模型评估指标已保存到: {METRICS_FILE}")
    return metrics

if __name__ == "__main__":
    # 确保目录存在
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(SCALER_DIR, exist_ok=True)

    # 评估模型
    metrics = evaluate_models()

    if metrics:
        print("\n模型评估摘要:")
        for model_name, model_metrics in metrics.items():
            print(f"{model_name}: 准确率={model_metrics['accuracy']:.2f}%, RMSE={model_metrics['rmse']:.4f}")
