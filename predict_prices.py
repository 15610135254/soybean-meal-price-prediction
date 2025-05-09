import pandas as pd
import numpy as np
import os
import joblib
import glob
import logging
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 配置 ---
DEFAULT_DATA_FILE = "model_data/date1.csv"  # 默认数据文件路径
LOOK_BACK = 30  # 时间序列预测中使用的窗口大小
MODEL_DIR = "best_models"  # 模型目录
SCALER_DIR = "scalers"  # 标准化器目录
TARGET_COL = 'close'  # 目标列

def get_latest_data_file():
    """获取数据文件路径，优先使用默认数据文件"""
    if os.path.exists(DEFAULT_DATA_FILE):
        logger.info(f"使用默认数据文件: {DEFAULT_DATA_FILE}")
        return DEFAULT_DATA_FILE

    # 如果默认文件不存在，查找model_data和data目录下的所有CSV文件
    data_files = glob.glob("model_data/*.csv") + glob.glob("data/*.csv")

    if not data_files:
        logger.warning(f"未找到任何CSV文件，使用默认文件路径: {DEFAULT_DATA_FILE}")
        return DEFAULT_DATA_FILE

    # 按文件修改时间排序，返回最新的文件
    latest_file = max(data_files, key=os.path.getmtime)
    logger.info(f"默认数据文件不存在，使用最新的数据文件: {latest_file}")
    return latest_file

# 指定要使用的模型文件
MODEL_FILES = {
    "MLP": "mlp_20250509_011024.h5",
    "LSTM": "lstm_20250509_011051.h5",
    "CNN": "cnn_20250509_011353.h5"
}

def preprocess_data(df):
    """预处理数据，确保所有必要的列都存在"""
    try:
        # 确保日期列是datetime类型
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        # 确保数据按日期排序
        if 'date' in df.columns:
            df = df.sort_values('date')

        # 检查并确保所有必要的列都存在
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            logger.warning(f"缺少必要的列: {missing_cols}")

        # 替换无穷值和NaN
        df = df.replace([np.inf, -np.inf], np.nan)

        # 对数值列进行简单的前向填充
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill')

        # 如果仍有NaN，使用列均值填充
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

        # 如果仍有NaN，填充为0
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

    if missing_features:
        logger.error(f"错误: 预定义的特征在DataFrame中缺失: {missing_features}")
    if non_numeric_features:
        logger.error(f"错误: 预定义的特征非数值且无法转换: {non_numeric_features}")

    # 只有当所有16个预定义特征都成功找到并且是数值类型时，才认为是成功的
    if len(final_features) == len(defined_16_features):
        logger.info(f"成功选择所有 {len(final_features)} 个预定义特征: {final_features}")
        # 返回的final_features将保持defined_16_features中的顺序
        return final_features
    else:
        logger.error(f"未能选择全部16个预定义特征。实际有效特征数量: {len(final_features)}。请检查数据源和上述错误。")
        return [] # 返回空列表表示失败，以便上游处理



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
                    logger.info(result.stdout)

                    # 重新加载更新后的标准化器
                    scaler = joblib.load(scaler_path)
                    logger.info(f"重新加载标准化器，期望特征数量: {scaler.n_features_in_}")

                    # 再次检查特征数量是否匹配
                    if data.shape[1] != scaler.n_features_in_:
                        logger.warning(f"更新后特征数量仍不匹配，创建临时标准化器")
                        # 创建临时标准化器
                        temp_scaler = MinMaxScaler()
                        temp_scaler.fit(data)
                        normalized_data = temp_scaler.transform(data)
                        return normalized_data
                    else:
                        # 特征数量匹配，使用更新后的标准化器
                        normalized_data = scaler.transform(data)
                        return normalized_data
                else:
                    logger.error(f"运行update_scalers.py脚本失败: {result.stderr}")
                    # 创建临时标准化器
                    temp_scaler = MinMaxScaler()
                    temp_scaler.fit(data)
                    normalized_data = temp_scaler.transform(data)
                    return normalized_data
            except Exception as e:
                logger.error(f"尝试更新标准化器时出错: {e}")
                # 创建临时标准化器
                temp_scaler = MinMaxScaler()
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

def predict_with_model(model_type, days=1):
    """使用指定的模型进行预测"""
    logger.info(f"使用 {model_type.upper()} 模型预测未来 {days} 天的价格")

    try:
        # 1. 加载数据
        data_file = get_latest_data_file()
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
        feature_scaler_path = os.path.join(SCALER_DIR, 'feature_scaler.pkl')
        # 在标准化之前，确保 sequence_data 的列数与 scaler 期望的列数一致
        # scaler 是基于训练数据拟合的，训练数据经过了特征工程，特征数量约为41
        # sequence_data 此时应该也是约41列
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
        predictions = []
        prediction_dates = []

        # 获取最后一天的日期
        if 'date' in df.columns:
            last_date = pd.to_datetime(df['date'].iloc[-1])
        else:
            # 如果没有日期列，使用当前日期
            last_date = pd.Timestamp.now().normalize()

        # 加载目标标准化器
        target_scaler_path = os.path.join(SCALER_DIR, 'target_scaler.pkl')
        target_scaler = joblib.load(target_scaler_path)

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

        logger.info(f"预测完成，共 {len(predictions)} 天")

        return {
            'dates': prediction_dates,
            'prices': predictions
        }

    except Exception as e:
        logger.error(f"预测过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return {'error': f'预测过程中出错: {str(e)}'}

if __name__ == "__main__":
    main()