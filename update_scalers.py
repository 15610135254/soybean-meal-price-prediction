#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标准化器更新脚本
此脚本用于更新特征标准化器和目标标准化器，以适应新的数据格式和特征数量
"""

import os
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.preprocessing import MinMaxScaler

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置
DEFAULT_DATA_FILE = "model_data/date1.csv"  # 默认数据文件路径
SCALER_DIR = "models/training_scripts/scalers"  # 训练脚本中的标准化器目录
TARGET_COL = 'close'  # 目标列
LOOK_BACK = 30  # 时间序列预测中使用的窗口大小

# 确保标准化器目录存在
os.makedirs(SCALER_DIR, exist_ok=True)

def load_data(data_file=DEFAULT_DATA_FILE):
    """加载数据文件"""
    try:
        logger.info(f"正在加载数据文件: {data_file}")
        if not os.path.exists(data_file):
            logger.error(f"数据文件不存在: {data_file}")
            return None

        # 读取CSV文件
        df = pd.read_csv(data_file)
        logger.info(f"成功加载数据文件，形状: {df.shape}")
        logger.info(f"数据列名: {df.columns.tolist()}")

        return df
    except Exception as e:
        logger.error(f"加载数据文件时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def select_features(df):
    """选择特征列"""
    try:
        # 排除不用于特征的列
        exclude_cols = ['date']

        # 如果目标列是特征之一，也将其排除
        if TARGET_COL in df.columns and TARGET_COL != 'date':
            exclude_cols.append(TARGET_COL)

        # 选择所有数值列作为特征
        features = []
        for col in df.columns:
            if col not in exclude_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    features.append(col)
                else:
                    # 尝试转换为数值
                    try:
                        df[col] = pd.to_numeric(df[col], errors='raise')
                        features.append(col)
                        logger.info(f"特征 '{col}' 成功转换为数值。")
                    except (ValueError, TypeError):
                        logger.warning(f"特征 '{col}' 无法转换为数值，将被排除。")

        logger.info(f"选择的特征列表 ({len(features)} 个特征): {features}")
        return features, df
    except Exception as e:
        logger.error(f"选择特征时出错: {e}")
        import traceback
        traceback.print_exc()
        return [], df

def handle_missing_values(df, features):
    """处理缺失值"""
    try:
        # 检查缺失值
        missing_values = df[features].isnull().sum()
        if missing_values.sum() > 0:
            logger.info(f"检测到缺失值:\n{missing_values[missing_values > 0]}")

            # 1. 首先使用前向填充 (ffill) 处理缺失值
            df[features] = df[features].fillna(method='ffill')

            # 2. 然后使用后向填充 (bfill) 处理仍然存在的缺失值
            df[features] = df[features].fillna(method='bfill')

            # 3. 对于仍然是 NaN 的列，用 0 填充
            nan_counts_after_fills = df[features].isnull().sum()
            for feature in features:
                if nan_counts_after_fills[feature] > 0:
                    if nan_counts_after_fills[feature] == len(df):
                        logger.warning(f"特征 '{feature}' 在ffill和bfill后仍然全是NaN，将用0填充此列。")
                    df[feature] = df[feature].fillna(0)

        return df
    except Exception as e:
        logger.error(f"处理缺失值时出错: {e}")
        import traceback
        traceback.print_exc()
        return df

def create_and_save_scalers(df, features):
    """创建并保存标准化器"""
    try:
        # 创建特征缩放器，限制范围在[-1, 1]之间
        feature_scaler = MinMaxScaler(feature_range=(-1, 1))
        target_scaler = MinMaxScaler(feature_range=(-1, 1))

        # 使用所有数据拟合缩放器
        feature_scaler.fit(df[features].values)

        # 拟合目标标准化器
        if TARGET_COL in df.columns:
            target_values = df[TARGET_COL].values.reshape(-1, 1)
            target_scaler.fit(target_values)
            logger.info(f"目标列 '{TARGET_COL}' 的标准化器已创建")
        else:
            logger.warning(f"目标列 '{TARGET_COL}' 不在数据集中，使用默认值创建目标标准化器")
            # 创建一个默认的目标标准化器
            dummy_values = np.array([0, 1]).reshape(-1, 1)
            target_scaler.fit(dummy_values)

        # 定义标准化器文件路径
        feature_scaler_file = os.path.join(SCALER_DIR, 'feature_scaler.pkl')
        target_scaler_file = os.path.join(SCALER_DIR, 'target_scaler.pkl')

        # 备份旧的标准化器
        if os.path.exists(feature_scaler_file):
            backup_feature_scaler_file = os.path.join(SCALER_DIR, 'feature_scaler_backup.pkl')
            os.rename(feature_scaler_file, backup_feature_scaler_file)
            logger.info(f"已备份旧的特征标准化器到: {backup_feature_scaler_file}")

        if os.path.exists(target_scaler_file):
            backup_target_scaler_file = os.path.join(SCALER_DIR, 'target_scaler_backup.pkl')
            os.rename(target_scaler_file, backup_target_scaler_file)
            logger.info(f"已备份旧的目标标准化器到: {backup_target_scaler_file}")

        # 保存新的标准化器
        joblib.dump(feature_scaler, feature_scaler_file)
        joblib.dump(target_scaler, target_scaler_file)

        logger.info(f"特征标准化器已保存到: {feature_scaler_file}")
        logger.info(f"目标标准化器已保存到: {target_scaler_file}")
        logger.info(f"特征标准化器期望特征数量: {feature_scaler.n_features_in_}")

        return True
    except Exception as e:
        logger.error(f"创建和保存标准化器时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    try:
        logger.info("开始更新标准化器...")

        # 加载数据
        df = load_data()
        if df is None:
            logger.error("无法加载数据，更新标准化器失败")
            return False

        # 选择特征
        features, df = select_features(df)
        if not features:
            logger.error("无法选择特征，更新标准化器失败")
            return False

        # 处理缺失值
        df = handle_missing_values(df, features)

        # 创建并保存标准化器
        success = create_and_save_scalers(df, features)

        if success:
            logger.info("标准化器更新成功")
            return True
        else:
            logger.error("标准化器更新失败")
            return False
    except Exception as e:
        logger.error(f"更新标准化器时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()
