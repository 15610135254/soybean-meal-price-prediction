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
    "MLP": "mlp_20250508_223913.h5",
    "LSTM": "lstm_20250508_224629.h5",
    "CNN": "cnn_20250508_224004.h5"
}

def load_test_data():
    """加载测试数据"""
    try:
        # 尝试加载测试数据
        # 注意：在实际应用中，您需要确保有一个专门的测试数据集
        df = pd.read_csv("/Users/a/project/models/model_data/date1.csv")
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values('日期')
        
        # 使用最后20%的数据作为测试集
        test_size = int(len(df) * 0.2)
        test_df = df.iloc[-test_size:]
        
        return test_df
    except Exception as e:
        print(f"加载测试数据时出错: {e}")
        return None

def prepare_test_data(test_df, look_back=20):
    """准备测试数据"""
    try:
        # 加载标准化器
        feature_scaler = joblib.load(os.path.join(SCALER_DIR, 'feature_scaler.pkl'))
        target_scaler = joblib.load(os.path.join(SCALER_DIR, 'target_scaler.pkl'))
        
        # 选择特征（与训练时相同）
        basic_features = ['收盘价', '开盘价', '最高价', '最低价', '成交量']
        tech_features = []
        potential_features = ['涨跌幅', 'MA5', 'MA10', 'MA20', 'MA30', 'EMA12', 'EMA26', 'RSI', 'MACD', 'K', 'D', 'J']
        
        for feature in potential_features:
            if feature in test_df.columns:
                tech_features.append(feature)
        
        features = basic_features + tech_features
        
        # 过滤掉不存在或非数值的列
        final_features = []
        for feature in features:
            if feature in test_df.columns and pd.api.types.is_numeric_dtype(test_df[feature]):
                final_features.append(feature)
        
        # 准备特征和目标
        X = test_df[final_features].values
        y = test_df['收盘价'].values
        
        # 标准化数据
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
        
        return X_ts, y_ts, target_scaler, final_features
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
