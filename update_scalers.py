import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置
DATA_FILE = "model_data/date1.csv"  # 数据文件路径
SCALER_DIR = "scalers"  # 标准化器目录
LOOK_BACK = 20  # 时间序列预测中使用的窗口大小

def load_data():
    """加载数据集"""
    try:
        if not os.path.exists(DATA_FILE):
            logger.error(f"数据文件不存在: {DATA_FILE}")
            return None
            
        df = pd.read_csv(DATA_FILE)
        logger.info(f"成功加载数据文件: {DATA_FILE}, 共 {len(df)} 条记录")
        
        # 确保日期列是datetime类型
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
        # 确保数据按日期排序
        if 'date' in df.columns:
            df = df.sort_values('date')
            
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
        logger.error(f"加载数据时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def select_features(df):
    """选择模型使用的特征"""
    try:
        # 基本特征 - 必须包含
        basic_features = ['close', 'open', 'high', 'low', 'volume']
        
        # 确保所有基本特征都存在
        missing_basic = [f for f in basic_features if f not in df.columns]
        if missing_basic:
            logger.warning(f"缺少基本特征: {missing_basic}")
            # 使用可用的基本特征
            basic_features = [f for f in basic_features if f in df.columns]
        
        # 技术指标特征 - 可选
        tech_features = []
        potential_features = [
            'hold', 'MA_5', 'HV_20', 'ATR_14', 'RSI_14', 'OBV', 'MACD',
            'a_close', 'c_close', 'LPR1Y', '大豆产量(万吨)', 'GDP'
        ]
        
        for feature in potential_features:
            if feature in df.columns:
                tech_features.append(feature)
        
        # 合并特征
        all_features = basic_features + tech_features
        
        # 过滤掉非数值列
        final_features = []
        for feature in all_features:
            if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]):
                final_features.append(feature)
        
        if not final_features:
            logger.error("没有找到任何有效特征")
            return []
        
        logger.info(f"使用的特征: {final_features}")
        return final_features
    
    except Exception as e:
        logger.error(f"选择特征时出错: {e}")
        import traceback
        traceback.print_exc()
        return []

def create_scalers(df, features):
    """创建并保存标准化器"""
    try:
        # 确保标准化器目录存在
        if not os.path.exists(SCALER_DIR):
            os.makedirs(SCALER_DIR)
            logger.info(f"创建标准化器目录: {SCALER_DIR}")
        
        # 提取特征数据
        X = df[features].values
        y = df['close'].values.reshape(-1, 1)
        
        # 创建特征标准化器
        feature_scaler = MinMaxScaler()
        feature_scaler.fit(X)
        
        # 创建目标标准化器
        target_scaler = MinMaxScaler()
        target_scaler.fit(y)
        
        # 保存标准化器
        feature_scaler_path = os.path.join(SCALER_DIR, 'feature_scaler.pkl')
        target_scaler_path = os.path.join(SCALER_DIR, 'target_scaler.pkl')
        
        joblib.dump(feature_scaler, feature_scaler_path)
        joblib.dump(target_scaler, target_scaler_path)
        
        logger.info(f"特征标准化器已保存: {feature_scaler_path}")
        logger.info(f"目标标准化器已保存: {target_scaler_path}")
        
        # 记录特征数量
        logger.info(f"特征标准化器特征数量: {feature_scaler.n_features_in_}")
        
        return feature_scaler, target_scaler
        
    except Exception as e:
        logger.error(f"创建标准化器时出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    """主函数"""
    logger.info("开始更新标准化器...")
    
    # 1. 加载数据
    df = load_data()
    if df is None:
        logger.error("加载数据失败，无法更新标准化器")
        return
    
    # 2. 选择特征
    features = select_features(df)
    if not features:
        logger.error("选择特征失败，无法更新标准化器")
        return
    
    # 3. 创建并保存标准化器
    feature_scaler, target_scaler = create_scalers(df, features)
    if feature_scaler is None or target_scaler is None:
        logger.error("创建标准化器失败")
        return
    
    logger.info("标准化器更新完成")

if __name__ == "__main__":
    main()
