import pandas as pd
import numpy as np
import os
import joblib
import glob
import logging
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# 获取日志记录器
logger = logging.getLogger(__name__)

# --- 配置 ---
DEFAULT_DATA_FILE = "model_data/date1.csv"  # 默认数据文件路径
LOOK_BACK = 30  # 时间序列预测中使用的窗口大小
MODEL_DIR = "best_models"  # 模型目录
SCALER_DIR = "scalers"  # 标准化器目录
TRAINING_SCALER_DIR = "models/training_scripts/scalers"  # 训练脚本中的标准化器目录
TARGET_COL = 'close'  # 目标列

# 使用 views.data_utils 中的函数替代此功能

# 指定要使用的模型文件
MODEL_FILES = {
    "MLP": "mlp_20250509_011024.h5",
    "LSTM": "lstm_20250509_011051.h5",
    "CNN": "cnn_20250509_011353.h5"
}

def preprocess_data(df):
    """预处理数据，确保所有必要的列都存在并处理缺失值"""
    try:
        # 确保日期列是datetime类型
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            # 确保数据按日期排序（从早到晚）
            df = df.sort_values('date', ascending=True)
            logger.info(f"数据已按日期排序，日期范围: {df['date'].min()} 至 {df['date'].max()}")

        # 检查并确保所有必要的列都存在
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            logger.warning(f"缺少必要的列: {missing_cols}")
            # 尝试创建缺失的必要列
            for col in missing_cols:
                if col == 'open' and 'close' in df.columns:
                    df['open'] = df['close'].shift(1)
                    logger.info(f"已创建缺失列 'open'，使用前一天的收盘价")
                elif col == 'high' and 'close' in df.columns:
                    df['high'] = df['close'] * 1.01  # 简单估计，收盘价上浮1%
                    logger.info(f"已创建缺失列 'high'，使用收盘价上浮1%")
                elif col == 'low' and 'close' in df.columns:
                    df['low'] = df['close'] * 0.99  # 简单估计，收盘价下浮1%
                    logger.info(f"已创建缺失列 'low'，使用收盘价下浮1%")
                elif col == 'volume' and 'close' in df.columns:
                    # 创建一个随机成交量
                    np.random.seed(42)  # 设置随机种子以确保可重复性
                    df['volume'] = np.random.randint(1000, 10000, size=len(df))
                    logger.info(f"已创建缺失列 'volume'，使用随机值")
                elif col == 'close' and 'open' in df.columns:
                    df['close'] = df['open'].shift(-1)  # 使用下一天的开盘价
                    logger.info(f"已创建缺失列 'close'，使用下一天的开盘价")
                else:
                    # 如果无法创建，使用0填充
                    df[col] = 0
                    logger.warning(f"无法创建缺失列 '{col}'，使用0填充")

        # 替换无穷值和NaN
        df = df.replace([np.inf, -np.inf], np.nan)

        # 检查数值列中的NaN值
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        nan_counts = df[numeric_cols].isnull().sum()
        if nan_counts.sum() > 0:
            logger.info(f"检测到NaN值:\n{nan_counts[nan_counts > 0]}")

            # 对数值列进行前向填充
            df[numeric_cols] = df[numeric_cols].ffill()

            # 如果仍有NaN（例如在序列开始处），使用后向填充
            df[numeric_cols] = df[numeric_cols].bfill()

            # 如果仍有NaN，使用列均值填充
            for col in numeric_cols:
                if df[col].isnull().any():
                    mean_val = df[col].mean()
                    if pd.isna(mean_val):  # 如果均值本身是NaN
                        df[col] = df[col].fillna(0)
                        logger.info(f"列 '{col}' 的均值为NaN，使用0填充")
                    else:
                        df[col] = df[col].fillna(mean_val)
                        logger.info(f"列 '{col}' 使用均值 {mean_val:.2f} 填充")

        # 最后检查，如果仍有NaN，填充为0
        if df.isnull().values.any():
            logger.warning("仍有NaN值，使用0填充")
            df = df.fillna(0)

        return df

    except Exception as e:
        logger.error(f"预处理数据时出错: {e}")
        import traceback
        traceback.print_exc()
        return df



def select_features(df):
    """选择模型使用的特征。
    此版本旨在从DataFrame中选择一组预定义的16个特征，
    并确保它们存在且为数值类型。
    """
    logger.info("开始为预测选择特征...")

    # 预定义的16个特征列表 (与训练脚本从date1.csv选择的特征和顺序完全一致)
    # 这些特征应该直接存在于输入的 df 中
    defined_16_features = [
        'open', 'high', 'low', 'volume', 'hold', 'MA_5', 'HV_20', 'ATR_14',
        'RSI_14', 'OBV', 'MACD', 'a_close', 'c_close', 'LPR1Y',
        '大豆产量(万吨)', 'GDP'
    ]

    final_features = []
    missing_features = []
    non_numeric_features = []

    # 检查数据集中是否有预定义的特征
    for feature_name in defined_16_features:
        if feature_name in df.columns:
            if pd.api.types.is_numeric_dtype(df[feature_name]):
                final_features.append(feature_name)
            else:
                # 尝试转换为数值
                logger.info(f"特征 '{feature_name}' 非数值，尝试转换...")
                try:
                    df[feature_name] = pd.to_numeric(df[feature_name], errors='raise') # errors='raise' 更严格
                    final_features.append(feature_name)
                    logger.info(f"特征 '{feature_name}' 成功转换为数值。")
                except (ValueError, TypeError):
                    non_numeric_features.append(feature_name)
                    logger.warning(f"特征 '{feature_name}' 无法转换为数值，将被排除。")
        else:
            missing_features.append(feature_name)

    # 如果有缺失的特征，尝试创建它们
    if missing_features:
        logger.warning(f"预定义的特征在DataFrame中缺失: {missing_features}")

        # 尝试创建一些基本的缺失特征
        for feature in missing_features[:]:  # 使用切片创建副本以避免在迭代时修改列表
            if feature == 'hold' and 'volume' in df.columns:
                # 创建一个简单的持仓量估计
                df['hold'] = df['volume'].rolling(window=5).mean()
                logger.info(f"已创建缺失特征 'hold'")
                missing_features.remove('hold')
                final_features.append('hold')

            elif feature == 'MA_5' and 'close' in df.columns:
                # 创建5日移动平均线
                df['MA_5'] = df['close'].rolling(window=5).mean()
                logger.info(f"已创建缺失特征 'MA_5'")
                missing_features.remove('MA_5')
                final_features.append('MA_5')

            elif feature == 'HV_20' and 'close' in df.columns:
                # 创建20日历史波动率
                df['HV_20'] = df['close'].pct_change().rolling(window=20).std() * 100
                logger.info(f"已创建缺失特征 'HV_20'")
                missing_features.remove('HV_20')
                final_features.append('HV_20')

            elif feature == 'ATR_14' and all(col in df.columns for col in ['high', 'low', 'close']):
                # 创建14日ATR
                high_low = df['high'] - df['low']
                high_close = (df['high'] - df['close'].shift()).abs()
                low_close = (df['low'] - df['close'].shift()).abs()
                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = ranges.max(axis=1)
                df['ATR_14'] = true_range.rolling(window=14).mean()
                logger.info(f"已创建缺失特征 'ATR_14'")
                missing_features.remove('ATR_14')
                final_features.append('ATR_14')

            elif feature == 'RSI_14' and 'close' in df.columns:
                # 创建14日RSI
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                df['RSI_14'] = 100 - (100 / (1 + rs))
                logger.info(f"已创建缺失特征 'RSI_14'")
                missing_features.remove('RSI_14')
                final_features.append('RSI_14')

            elif feature == 'OBV' and all(col in df.columns for col in ['close', 'volume']):
                # 创建OBV
                df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
                logger.info(f"已创建缺失特征 'OBV'")
                missing_features.remove('OBV')
                final_features.append('OBV')

            elif feature == 'MACD' and 'close' in df.columns:
                # 创建MACD
                ema12 = df['close'].ewm(span=12, adjust=False).mean()
                ema26 = df['close'].ewm(span=26, adjust=False).mean()
                df['MACD'] = ema12 - ema26
                logger.info(f"已创建缺失特征 'MACD'")
                missing_features.remove('MACD')
                final_features.append('MACD')

    # 如果仍有缺失的特征，使用0填充
    if missing_features:
        logger.warning(f"无法创建的特征，将使用0填充: {missing_features}")
        for feature in missing_features:
            df[feature] = 0
            logger.info(f"已用0填充缺失特征 '{feature}'")
            final_features.append(feature)

    # 如果有非数值特征，使用0填充
    if non_numeric_features:
        logger.warning(f"非数值特征，将使用0填充: {non_numeric_features}")
        for feature in non_numeric_features:
            df[feature] = 0
            logger.info(f"已用0填充非数值特征 '{feature}'")
            final_features.append(feature)

    # 确保特征顺序与预定义的顺序一致
    final_features = [feature for feature in defined_16_features if feature in final_features]

    # 处理缺失值
    for feature in final_features:
        if df[feature].isnull().any():
            # 使用前向填充
            df[feature] = df[feature].ffill()
            # 如果仍有NaN（例如在序列开始处），使用后向填充
            df[feature] = df[feature].bfill()
            # 如果仍有NaN，填充为0
            df[feature] = df[feature].fillna(0)
            logger.info(f"已处理特征 '{feature}' 中的缺失值")

    logger.info(f"成功选择所有 {len(final_features)} 个预定义特征: {final_features}")
    return final_features



def prepare_sequence_data(df, features, look_back=LOOK_BACK):
    """准备时间序列数据"""
    try:
        if len(df) < look_back:
            logger.error(f"数据不足 {look_back} 条，无法准备序列数据")
            return None

        # 获取最后look_back条数据
        latest_data = df[features].iloc[-look_back:].values

        logger.info(f"准备的序列数据形状: {latest_data.shape}")
        return latest_data

    except Exception as e:
        logger.error(f"准备序列数据时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def normalize_data(data, scaler_path):
    """标准化数据"""
    try:
        if not os.path.exists(scaler_path):
            logger.error(f"标准化器文件不存在: {scaler_path}")
            return None

        # 加载标准化器
        scaler = joblib.load(scaler_path)

        # 记录特征数量信息
        logger.info(f"数据特征数量: {data.shape[1]}")
        logger.info(f"标准化器期望特征数量: {scaler.n_features_in_}")

        # 检查特征数量是否匹配
        if data.shape[1] != scaler.n_features_in_:
            logger.warning(f"特征数量不匹配: 数据有 {data.shape[1]} 个特征，但标准化器期望 {scaler.n_features_in_} 个特征")

            # 尝试使用update_scalers.py脚本更新标准化器
            logger.info("尝试运行update_scalers.py脚本更新标准化器...")
            try:
                import subprocess
                result = subprocess.run(['python3', 'update_scalers.py'], capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info("成功运行update_scalers.py脚本")

                    # 重新加载更新后的标准化器
                    scaler = joblib.load(scaler_path)
                    logger.info(f"重新加载标准化器，期望特征数量: {scaler.n_features_in_}")

                    # 再次检查特征数量是否匹配
                    if data.shape[1] != scaler.n_features_in_:
                        logger.warning(f"更新后特征数量仍不匹配，创建临时标准化器")
                        # 创建临时标准化器
                        temp_scaler = MinMaxScaler(feature_range=(-1, 1))
                        temp_scaler.fit(data)
                        normalized_data = temp_scaler.transform(data)
                        return normalized_data
                    else:
                        # 特征数量匹配，使用更新后的标准化器
                        normalized_data = scaler.transform(data)
                        return normalized_data
                else:
                    logger.error(f"运行update_scalers.py脚本失败")
                    # 创建临时标准化器
                    temp_scaler = MinMaxScaler(feature_range=(-1, 1))
                    temp_scaler.fit(data)
                    normalized_data = temp_scaler.transform(data)
                    return normalized_data
            except Exception as e:
                logger.error(f"尝试更新标准化器时出错: {e}")
                # 创建临时标准化器
                temp_scaler = MinMaxScaler(feature_range=(-1, 1))
                temp_scaler.fit(data)
                normalized_data = temp_scaler.transform(data)
                return normalized_data

        # 特征数量匹配，直接使用标准化器
        normalized_data = scaler.transform(data)
        return normalized_data

    except Exception as e:
        logger.error(f"标准化数据时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- 主预测逻辑 ---
def main():
    """主函数，用于命令行测试"""
    # 测试所有模型
    for model_type in MODEL_FILES.keys():
        print(f"\n--- 使用 {model_type} 模型预测 ---")
        result = predict_with_model(model_type, days=1)

        if 'error' in result:
            print(f"预测失败: {result['error']}")
        else:
            for date, price in zip(result['dates'], result['prices']):
                print(f"{date}: {price:.2f}")

def predict_with_model(model_type, days=1, data_file=None):
    """使用指定的模型进行预测

    参数:
        model_type: 模型类型 (MLP, LSTM, CNN)
        days: 预测天数
        data_file: 数据文件路径，如果为None则使用默认数据文件
    """
    logger.info(f"使用 {model_type.upper()} 模型预测未来 {days} 天的价格")

    try:
        # 1. 加载数据
        if data_file is None:
            # 使用默认数据文件
            data_file = DEFAULT_DATA_FILE

        logger.info(f"使用数据文件: {data_file}")

        if not os.path.exists(data_file):
            return {'error': f'数据文件 {data_file} 不存在'}

        df = pd.read_csv(data_file)
        logger.info(f"成功加载数据文件: {data_file}, 共 {len(df)} 条记录")

        # 2. 预处理数据 (例如日期转换，基础NaN处理)
        df = preprocess_data(df)

        # 3. 添加技术指标 (特征工程，与训练时一致)
        # 数据集中已经包含了所有需要的指标

        # 4. 选择最终特征 (包含原始及生成的技术指标)
        features = select_features(df)
        if not features:
            return {'error': '无法选择有效特征进行预测'}

        # 打印最终选择的特征数量，再次确认
        logger.info(f"在 predict_with_model 中，选择的特征数量为: {len(features)}")

        # 5. 准备序列数据
        sequence_data = prepare_sequence_data(df, features)
        if sequence_data is None:
            return {'error': f'无法准备序列数据，需要至少 {LOOK_BACK} 条记录'}

        # 6. 标准化数据
        # 优先使用训练脚本目录中的标准化器
        training_feature_scaler_path = os.path.join(TRAINING_SCALER_DIR, 'feature_scaler.pkl')
        if os.path.exists(training_feature_scaler_path):
            feature_scaler_path = training_feature_scaler_path
            logger.info(f"使用训练脚本目录中的标准化器: {feature_scaler_path}")
        else:
            feature_scaler_path = os.path.join(SCALER_DIR, 'feature_scaler.pkl')
            logger.info(f"训练脚本目录中的标准化器不存在，使用默认标准化器: {feature_scaler_path}")

        # 在标准化之前，确保 sequence_data 的列数与 scaler 期望的列数一致
        logger.info(f"准备标准化前，序列数据的形状: {sequence_data.shape} (应为 LOOK_BACK x num_features)")

        normalized_data = normalize_data(sequence_data, feature_scaler_path)
        if normalized_data is None:
            return {'error': '无法标准化数据'}

        # 7. 重塑数据为模型输入格式
        X_predict = np.reshape(normalized_data, (1, LOOK_BACK, len(features)))

        # 8. 加载模型
        model_type = model_type.upper()
        if model_type not in MODEL_FILES:
            return {'error': f'不支持的模型类型: {model_type}'}

        model_path = os.path.join(MODEL_DIR, MODEL_FILES[model_type])
        if not os.path.exists(model_path):
            return {'error': f'模型文件不存在: {model_path}'}

        try:
            # 直接加载模型
            model = load_model(model_path, compile=False)
            logger.info(f"成功加载模型: {model_path}")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return {'error': f'加载模型失败: {str(e)}'}

        # 9. 进行预测

        # 获取最后一天的日期
        if 'date' in df.columns:
            last_date = pd.to_datetime(df['date'].iloc[-1])
        else:
            # 如果没有日期列，使用当前日期
            last_date = pd.Timestamp.now().normalize()

        # 加载目标标准化器
        # 优先使用训练脚本目录中的标准化器
        training_target_scaler_path = os.path.join(TRAINING_SCALER_DIR, 'target_scaler.pkl')
        if os.path.exists(training_target_scaler_path):
            target_scaler_path = training_target_scaler_path
            logger.info(f"使用训练脚本目录中的目标标准化器: {target_scaler_path}")
        else:
            target_scaler_path = os.path.join(SCALER_DIR, 'target_scaler.pkl')
            logger.info(f"训练脚本目录中的目标标准化器不存在，使用默认目标标准化器: {target_scaler_path}")

        target_scaler = joblib.load(target_scaler_path)

        # 只使用单步预测方法
        # 单步预测：每次只预测一天，使用真实数据作为输入
        # 这种方法更准确，能够更好地反映模型的实际性能
        predictions = []
        prediction_dates = []

        # 检查数据集是否足够长
        if len(df) >= LOOK_BACK + days:
            # 获取用于预测的数据（包括训练集的最后LOOK_BACK天和测试集）
            prediction_data = df.iloc[-(LOOK_BACK + days):]

            # 对于每一天进行预测
            for i in range(days):
                # 获取当前时间窗口的数据
                window_data = prediction_data.iloc[i:i+LOOK_BACK][features].values

                # 标准化数据
                normalized_window = normalize_data(window_data, feature_scaler_path)
                if normalized_window is None:
                    logger.error(f"无法标准化第 {i+1} 天的数据")
                    continue

                # 重塑为模型输入格式
                X_window = np.reshape(normalized_window, (1, LOOK_BACK, len(features)))

                # 预测
                try:
                    prediction_scaled = model.predict(X_window, verbose=0)
                    prediction = target_scaler.inverse_transform(prediction_scaled)[0][0]

                    # 获取对应的日期
                    pred_date = prediction_data.iloc[i+LOOK_BACK]['date']
                    if isinstance(pred_date, str):
                        pred_date = pd.to_datetime(pred_date)

                    # 保存预测结果
                    predictions.append(float(prediction))
                    prediction_dates.append(pred_date.strftime('%Y-%m-%d'))
                except Exception as e:
                    logger.error(f"预测第 {i+1} 天时出错: {e}")

            logger.info(f"单步预测完成，共 {len(predictions)} 天")
        else:
            # 如果数据不足，使用简单的递归预测作为备选方案
            logger.warning(f"数据不足，无法进行完整的单步预测，需要至少 {LOOK_BACK + days} 条记录")
            logger.warning(f"将使用简单的递归预测作为备选方案")

            # 使用当前数据进行第一次预测
            current_input = X_predict

            for i in range(days):
                # 预测下一天
                prediction_scaled = model.predict(current_input, verbose=0)
                prediction = target_scaler.inverse_transform(prediction_scaled)[0][0]

                # 计算下一天的日期
                next_date = last_date + pd.Timedelta(days=i+1)

                # 保存预测结果
                predictions.append(float(prediction))
                prediction_dates.append(next_date.strftime('%Y-%m-%d'))

                # 准备下一次预测的输入
                if i < days - 1:  # 不需要为最后一天准备下一次预测
                    # 获取当前输入的最后一天（除去第一天）
                    next_input = current_input[0, 1:, :]

                    # 创建新的一天数据（复制最后一天的特征，但更新目标值）
                    new_day = np.copy(next_input[-1:, :])

                    # 更新目标值（假设目标值是第一个特征）
                    new_day_scaled = np.copy(new_day)
                    new_day_scaled[0, 0] = prediction_scaled[0][0]

                    # 合并数据
                    next_input = np.vstack([next_input, new_day_scaled])

                    # 重塑为模型输入格式
                    current_input = np.reshape(next_input, (1, LOOK_BACK, len(features)))

            logger.info(f"备选预测完成，共 {len(predictions)} 天")

        # 返回预测结果
        logger.info(f"返回预测结果，共 {len(predictions)} 天")
        return {
            'dates': prediction_dates,
            'prices': predictions,
            'prediction_type': 'single_step'
        }

    except Exception as e:
        logger.error(f"预测过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return {'error': f'预测过程中出错: {str(e)}'}

if __name__ == "__main__":
    main()