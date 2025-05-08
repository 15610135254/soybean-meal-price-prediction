import pandas as pd
import numpy as np
import os
import joblib
import glob
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# --- 配置 ---
DEFAULT_DATA_FILE = "model_data/date1.csv"  # 默认数据文件路径
LOOK_BACK = 20  # 设置默认值为20，这是时间序列预测中常用的窗口大小
MODEL_DIR = "saved_models"
SCALER_DIR = "scalers"
TARGET_COL = 'close' # 确认目标列

# 列名映射（新数据集到旧数据集的映射）
COLUMN_MAPPING = {
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

def get_latest_data_file():
    """获取数据文件路径，与趋势图保持一致"""
    # 首先检查默认数据文件是否存在
    if os.path.exists(DEFAULT_DATA_FILE):
        print(f"使用默认数据文件: {DEFAULT_DATA_FILE}")
        return DEFAULT_DATA_FILE

    # 如果默认文件不存在，查找model_data和data目录下的所有CSV文件
    data_files = glob.glob("model_data/*.csv") + glob.glob("data/*.csv")

    if not data_files:
        # 如果没有找到任何CSV文件，返回默认路径（即使它不存在）
        print(f"未找到任何CSV文件，使用默认文件路径: {DEFAULT_DATA_FILE}")
        return DEFAULT_DATA_FILE

    # 按文件修改时间排序，返回最新的文件
    latest_file = max(data_files, key=os.path.getmtime)
    print(f"默认数据文件不存在，使用最新的数据文件: {latest_file}")
    return latest_file

# 指定要使用的模型文件
MODEL_FILES = {
    "MLP": "mlp_20250502_225158.h5",
    "LSTM": "lstm_20250502_225234.h5",
    "CNN": "cnn_20250502_225554.h5"
}

# --- 辅助函数 (需要与 deep_learning_models.py 中的逻辑保持一致) ---

def add_technical_indicators(df):
    """添加技术指标 (简化版，确保与训练时使用的特征一致)"""
    # 确保使用正确的列名
    close_col = '收盘价' if '收盘价' in df.columns else 'close'
    open_col = '开盘价' if '开盘价' in df.columns else 'open'
    high_col = '最高价' if '最高价' in df.columns else 'high'
    low_col = '最低价' if '最低价' in df.columns else 'low'

    # 基本价格特征
    df['价格变化'] = df[close_col].diff()
    df['价格变化率'] = df[close_col].pct_change()
    df['日内波动率'] = (df[high_col] - df[low_col]) / df[open_col]

    # 移动平均线特征 (仅添加训练时实际使用的)
    # 检查是否有MA5或MA_5列
    ma_columns = {}
    for ma in [5, 10, 20, 30]:
        if f'MA{ma}' in df.columns:
            ma_columns[ma] = f'MA{ma}'
        elif f'MA_{ma}' in df.columns:
            ma_columns[ma] = f'MA_{ma}'

    for ma, col_name in ma_columns.items():
        df[f'MA{ma}_diff'] = df[close_col] - df[col_name]
        df[f'MA{ma}_slope'] = df[col_name].diff()
        # df[f'MA{ma}_std'] = df[close_col].rolling(window=ma).std() # 训练时可能未使用

    # 其他指标 (确保这些列在 data.csv 或基础计算中存在)
    # ... (根据 deep_learning_models.py 中 select_features 实际选择的添加)
    # 例如:
    # if 'MACD' in df.columns:
    #     df['MACD_diff'] = df['MACD'].diff()
    # if 'RSI' in df.columns:
    #     df['RSI_diff'] = df['RSI'].diff()

    # 时间特征 (如果训练时使用了)
    # df['星期'] = df['日期'].dt.dayofweek
    # df['月份'] = df['日期'].dt.month

    # 替换无穷值
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

def select_features(df):
    """选择模型使用的特征 (必须与训练时完全一致)"""
    # 注意：这里的列表需要和 deep_learning_models.py 中 select_features 函数最终返回的列表完全一致
    # 根据错误信息，训练时使用了17个特征。尝试只使用基础特征和技术指标特征。

    # 检查数据集的列名格式
    if 'close' in df.columns and '收盘价' not in df.columns:
        # 新数据集格式
        basic_features = ['close', 'open', 'high', 'low', 'volume']
    else:
        # 旧数据集格式
        basic_features = ['收盘价', '开盘价', '最高价', '最低价', '成交量']

    tech_features = []
    # 检查可能的技术指标列名
    potential_features = [
        # 旧格式
        '涨跌幅', 'MA5', 'MA10', 'MA20', 'MA30', 'EMA12', 'EMA26', 'RSI', 'MACD', 'K', 'D', 'J',
        # 新格式
        'MA_5', 'HV_20', 'ATR_14', 'RSI_14', 'OBV', 'MACD'
    ]

    for feature in potential_features:
        if feature in df.columns:
            tech_features.append(feature)

    # 合并基础和技术特征
    all_features = basic_features + tech_features

    # 过滤掉不存在或非数值的列
    final_features = []
    for feature in all_features:
        if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]):
            final_features.append(feature)

    # 确保特征列表有效

    print(f"预测时使用的特征: {final_features}")
    return final_features

def handle_missing_values(df, features):
    """处理缺失值 (与训练时一致)"""
    df_subset = df[features].copy()
    df_subset = df_subset.fillna(method='ffill') # 先前向填充
    df_subset = df_subset.fillna(df_subset.mean()) # 再均值填充
    # 确保没有NaN剩下，否则模型会出错
    if df_subset.isnull().values.any():
        print("警告：处理后仍有缺失值，可能导致预测错误。")
        # 可以选择填充0或其他策略
        df_subset = df_subset.fillna(0)
    df[features] = df_subset
    return df

# --- 主预测逻辑 ---
def main():
    # 1. 加载数据
    data_file = get_latest_data_file()
    print(f"正在加载数据: {data_file}")
    try:
        df = pd.read_csv(data_file)

        # 检查数据集的列名，并进行必要的映射
        if 'date' in df.columns and '日期' not in df.columns:
            print("检测到新数据集格式，进行列名映射...")
            # 将日期列转换为datetime
            df['date'] = pd.to_datetime(df['date'])
            # 重命名列以兼容旧代码
            df.rename(columns={'date': '日期',
                              'open': '开盘价',
                              'high': '最高价',
                              'low': '最低价',
                              'close': '收盘价',
                              'volume': '成交量'}, inplace=True)
        else:
            # 原始数据集格式
            df['日期'] = pd.to_datetime(df['日期'])

        df = df.sort_values('日期')
    except FileNotFoundError:
        print(f"错误：数据文件 {data_file} 未找到。")
        return

    # 2. 加载标准化器
    try:
        feature_scaler = joblib.load(os.path.join(SCALER_DIR, 'feature_scaler.pkl'))
        target_scaler = joblib.load(os.path.join(SCALER_DIR, 'target_scaler.pkl'))
        print("标准化器加载成功。")
    except FileNotFoundError:
        print(f"错误：标准化器文件未在 {SCALER_DIR} 找到。请先运行训练脚本生成标准化器。")
        return

    # 3. 准备最新数据进行预测
    print("正在准备最新数据...")
    # a. 添加技术指标
    df_processed = add_technical_indicators(df.copy())

    # b. 选择特征 (确保与训练时一致)
    features = select_features(df_processed)
    if not features:
        print("错误：未能选择任何有效特征进行预测。")
        return

    # c. 处理缺失值 (只处理选定的特征)
    df_processed = handle_missing_values(df_processed, features)

    # d. 提取最后 look_back 条数据
    if len(df_processed) < LOOK_BACK:
        print(f"错误：数据不足 {LOOK_BACK} 条，无法进行预测。需要 {LOOK_BACK} 条，现有 {len(df_processed)} 条。")
        return

    latest_data = df_processed[features].iloc[-LOOK_BACK:].values

    # e. 标准化数据
    try:
        latest_data_scaled = feature_scaler.transform(latest_data)
    except ValueError as e:
         print(f"错误：标准化数据时出错。特征数量可能不匹配。错误信息：{e}")
         print(f"期望特征数量: {feature_scaler.n_features_in_}")
         print(f"实际特征数量: {latest_data.shape[1]}")
         print(f"使用的特征: {features}")
         return

    # f. 重塑数据以符合模型输入要求 (1, look_back, num_features)
    X_predict = np.reshape(latest_data_scaled, (1, LOOK_BACK, len(features)))

    print(f"准备好的预测输入形状: {X_predict.shape}")

    # 4. 加载模型并预测
    print("\n--- 开始预测下一交易日收盘价 ---")
    predictions = {}

    for model_name, model_filename in MODEL_FILES.items():
        model_path = os.path.join(MODEL_DIR, model_filename)
        print(f"\n加载模型: {model_path}")

        if not os.path.exists(model_path):
            print(f"错误：模型文件 {model_path} 不存在。")
            continue

        try:
            model = load_model(model_path)

            # 执行预测
            prediction_scaled = model.predict(X_predict)

            # 反标准化预测结果
            prediction = target_scaler.inverse_transform(prediction_scaled)

            predictions[model_name] = prediction[0][0]
            print(f"{model_name} 模型预测收盘价: {prediction[0][0]:.2f}")

        except Exception as e:
            print(f"错误：加载或预测模型 {model_name} 时出错: {e}")
            import traceback
            traceback.print_exc()

    # 5. 输出汇总预测结果
    print("\n--- 预测汇总 ---")
    if predictions:
        for name, pred_price in predictions.items():
            print(f"{name}: {pred_price:.2f}")
    else:
        print("未能成功获取任何模型的预测结果。")

def predict_with_model(model_type, days=7):
    """
    使用指定的模型进行预测

    Args:
        model_type: 模型类型 ('mlp', 'lstm', 或 'cnn')
        days: 预测天数

    Returns:
        dict: 包含预测结果的字典，格式为 {'dates': [...], 'prices': [...]}
    """
    print(f"使用 {model_type.upper()} 模型预测未来 {days} 天的价格")

    # 1. 加载数据
    data_file = get_latest_data_file()
    try:
        df = pd.read_csv(data_file)

        # 检查数据集的列名，并进行必要的映射
        if 'date' in df.columns and '日期' not in df.columns:
            print("检测到新数据集格式，进行列名映射...")
            # 将日期列转换为datetime
            df['date'] = pd.to_datetime(df['date'])
            # 重命名列以兼容旧代码
            df.rename(columns={'date': '日期',
                              'open': '开盘价',
                              'high': '最高价',
                              'low': '最低价',
                              'close': '收盘价',
                              'volume': '成交量'}, inplace=True)
        else:
            # 原始数据集格式
            df['日期'] = pd.to_datetime(df['日期'])

        df = df.sort_values('日期')
    except FileNotFoundError:
        print(f"错误：数据文件 {data_file} 未找到。")
        return {'error': f'数据文件 {data_file} 未找到'}

    # 2. 加载标准化器
    try:
        feature_scaler = joblib.load(os.path.join(SCALER_DIR, 'feature_scaler.pkl'))
        target_scaler = joblib.load(os.path.join(SCALER_DIR, 'target_scaler.pkl'))
    except FileNotFoundError:
        print(f"错误：标准化器文件未在 {SCALER_DIR} 找到。")
        return {'error': f'标准化器文件未找到'}

    # 3. 准备最新数据进行预测
    df_processed = add_technical_indicators(df.copy())
    features = select_features(df_processed)
    if not features:
        print("错误：未能选择任何有效特征进行预测。")
        return {'error': '未能选择有效特征'}

    df_processed = handle_missing_values(df_processed, features)

    if len(df_processed) < LOOK_BACK:
        print(f"错误：数据不足 {LOOK_BACK} 条，无法进行预测。")
        return {'error': f'数据不足 {LOOK_BACK} 条'}

    latest_data = df_processed[features].iloc[-LOOK_BACK:].values

    try:
        latest_data_scaled = feature_scaler.transform(latest_data)
    except ValueError as e:
        print(f"错误：标准化数据时出错。{e}")
        return {'error': f'标准化数据时出错: {str(e)}'}

    X_predict = np.reshape(latest_data_scaled, (1, LOOK_BACK, len(features)))

    # 4. 加载指定的模型
    model_type = model_type.upper()
    if model_type not in MODEL_FILES:
        print(f"错误：不支持的模型类型 {model_type}")
        return {'error': f'不支持的模型类型: {model_type}'}

    model_filename = MODEL_FILES[model_type]
    model_path = os.path.join(MODEL_DIR, model_filename)

    if not os.path.exists(model_path):
        print(f"错误：模型文件 {model_path} 不存在。")
        return {'error': f'模型文件不存在: {model_filename}'}

    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"错误：加载模型 {model_type} 时出错: {e}")
        return {'error': f'加载模型时出错: {str(e)}'}

    # 5. 进行预测
    predictions = []
    prediction_dates = []

    # 获取最后一天的日期
    last_date = df['日期'].iloc[-1]

    # 使用当前数据进行第一次预测
    current_input = X_predict

    for i in range(days):
        # 预测下一天
        prediction_scaled = model.predict(current_input)
        prediction = target_scaler.inverse_transform(prediction_scaled)[0][0]

        # 计算下一天的日期
        next_date = last_date + pd.Timedelta(days=i+1)

        # 保存预测结果
        predictions.append(float(prediction))
        prediction_dates.append(next_date.strftime('%Y-%m-%d'))

        # 准备下一次预测的输入
        # 移除最早的一天数据，添加新预测的一天
        if i < days - 1:  # 不需要为最后一天准备下一次预测
            # 获取当前输入的最后一天（除去第一天）
            next_input = current_input[0, 1:, :]

            # 创建新的一天数据（复制最后一天的特征，但更新目标值）
            new_day = np.copy(next_input[-1:, :])

            # 更新目标值（假设目标值是第一个特征）
            # 注意：这里简化处理，实际应用中可能需要更复杂的逻辑
            new_day_scaled = np.copy(new_day)
            new_day_scaled[0, 0] = prediction_scaled[0][0]

            # 合并数据
            next_input = np.vstack([next_input, new_day_scaled])

            # 重塑为模型输入格式
            current_input = np.reshape(next_input, (1, LOOK_BACK, len(features)))

    print(f"预测完成，共 {len(predictions)} 天")

    return {
        'dates': prediction_dates,
        'prices': predictions
    }

if __name__ == "__main__":
    main()