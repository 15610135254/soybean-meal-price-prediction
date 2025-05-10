import pandas as pd
import numpy as np

def preprocess_new_data(input_file, output_file=None):
    """
    对新格式的期货历史数据进行预处理
    
    Args:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径，默认与输入文件相同
        
    Returns:
        处理后的DataFrame
    """
    print(f"正在读取新格式数据文件: {input_file}")
    # 读取CSV文件
    df = pd.read_csv(input_file)
    
    # 显示基本信息
    print("原始数据基本信息:")
    print(f"行数: {len(df)}")
    print(f"列数: {len(df.columns)}")
    print(f"列名: {df.columns.tolist()}")
    
    # 1. 检查缺失值
    missing_values = df.isnull().sum()
    print("\n缺失值统计:")
    print(missing_values)
    
    # 2. 转换日期列为日期时间类型
    try:
        df['date'] = pd.to_datetime(df['date'])
        print("\n已将'date'列转换为日期时间类型")
    except Exception as e:
        print(f"转换日期列时出错: {str(e)}")
    
    # 3. 转换数值列并处理异常值
    numeric_cols = ['close', 'open', 'high', 'low', 'volume']
    for col in numeric_cols:
        if col in df.columns:
            # 检查是否已经是数值类型
            if not pd.api.types.is_numeric_dtype(df[col]):
                # 清理非数字字符
                df[col] = df[col].astype(str).str.replace(',', '').str.strip()
                # 转换为数值类型
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 处理异常值 (例如替换3倍标准差以外的值为NaN)
            mean = df[col].mean()
            std = df[col].std()
            def check_outlier(x, mean, std):
                return x < mean - 3 * std or x > mean + 3 * std
            def handle_nan_values(x):
                return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            outliers = df[col].apply(check_outlier, args=(mean, std))
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                print(f"\n'{col}'列检测到 {outlier_count} 个异常值")
                # 可以选择保留这些记录但标记出来，此处不进行替换
            df[col] = df[col].apply(handle_nan_values)
    
    # 4. 填充缺失值 (使用前一个值填充)
    df = df.fillna(method='ffill')
    
    # 4.1 处理剩余的NaN值（如果前向填充后仍有NaN）
    # 对于数值列，用列的平均值填充
    for col in numeric_cols:
        if col in df.columns and df[col].isna().any():
            col_mean = df[col].mean()
            df[col] = df[col].fillna(col_mean)
            print(f"'{col}'列中的NaN值已用平均值 {col_mean:.2f} 填充")
    
    # 5. 添加新特征（如果需要）
    # 注意：新数据集可能已经包含了一些技术指标，我们只添加缺失的指标
    
    # 检查是否已有MA_5，如果没有则添加
    if 'MA_5' not in df.columns:
        df['MA_5'] = df['close'].rolling(window=5).mean()
        print("已添加MA_5指标")
    
    # 检查是否已有MA_10，如果没有则添加
    if 'MA_10' not in df.columns:
        df['MA_10'] = df['close'].rolling(window=10).mean()
        print("已添加MA_10指标")
    
    # 检查是否已有MA_20，如果没有则添加
    if 'MA_20' not in df.columns:
        df['MA_20'] = df['close'].rolling(window=20).mean()
        print("已添加MA_20指标")
    
    # 检查是否已有RSI_14，如果没有则添加
    if 'RSI_14' not in df.columns:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        print("已添加RSI_14指标")
    
    # 检查是否已有MACD，如果没有则添加
    if 'MACD' not in df.columns:
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        print("已添加MACD指标")
    
    # 添加价格变动和日内波幅
    df['price_change'] = df['close'] - df['open']
    df['daily_range'] = (df['high'] - df['low']) / df['open']
    
    # 添加涨跌幅
    df['price_change_pct'] = df['close'].pct_change() * 100
    
    # 添加成交量变化率
    df['volume_change_pct'] = df['volume'].pct_change() * 100
    
    # 6. 按日期排序（升序，旧日期在前）
    df = df.sort_values(by='date', ascending=True)
    
    # 统计处理后的信息
    print("\n处理后的数据信息:")
    print(f"行数: {len(df)}")
    print(f"列数: {len(df.columns)}")
    print(f"列名: {df.columns.tolist()}")
    
    # 保存处理后的数据
    if output_file is None:
        output_file = input_file
    
    # 最终检查：确保没有NaN值
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                # 数值列用0填充
                df[col] = df[col].fillna(0)
                print(f"'{col}'列中的NaN值已用0填充")
            else:
                # 非数值列用空字符串填充
                df[col] = df[col].fillna('')
                print(f"'{col}'列中的NaN值已用空字符串填充")
    
    # 使用numpy的nan_to_num函数处理inf和NaN
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].apply(lambda x: np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))
    
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n预处理后的数据已保存到: {output_file}")
    
    return df

if __name__ == "__main__":
    input_file = "model_data/date1.csv"
    # 将处理后的数据保存回原文件
    df_processed = preprocess_new_data(input_file)
    
    # 显示处理后的数据样例
    print("\n处理后的数据预览:")
    print(df_processed.head(10))
