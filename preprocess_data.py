import pandas as pd
import numpy as np

def preprocess_data(input_file, output_file=None):
    """
    对期货历史数据进行预处理
    
    Args:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径，默认与输入文件相同
    
    Returns:
        处理后的DataFrame
    """
    print(f"正在读取数据文件: {input_file}")
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
        df['日期'] = pd.to_datetime(df['日期'])
        print("\n已将'日期'列转换为日期时间类型")
    except Exception as e:
        print(f"转换日期列时出错: {str(e)}")
    
    # 3. 转换数值列并处理异常值
    numeric_cols = ['收盘价', '开盘价', '最高价', '最低价', '成交量']
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
            outliers = df[col].apply(lambda x: x < mean - 3 * std or x > mean + 3 * std)
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                print(f"\n'{col}'列检测到 {outlier_count} 个异常值")
                # 可以选择保留这些记录但标记出来，此处不进行替换
    
    # 4. 填充缺失值 (使用前一个值填充)
    df = df.fillna(method='ffill')
    
    # 5. 添加新特征
    # 5.1 计算涨跌幅
    df['涨跌幅'] = df['收盘价'].pct_change() * 100
    # 5.2 计算5日移动平均线
    df['MA5'] = df['收盘价'].rolling(window=5).mean()
    # 5.3 计算10日移动平均线
    df['MA10'] = df['收盘价'].rolling(window=10).mean()
    # 5.4 计算交易量变化率
    df['成交量变化率'] = df['成交量'].pct_change() * 100
    # 5.5 添加一周中的天数
    df['星期'] = df['日期'].dt.dayofweek + 1  # 1-7，1代表星期一
    
    # 6. 按日期排序（降序，最新日期在前）
    df = df.sort_values(by='日期', ascending=False)
    
    # 统计处理后的信息
    print("\n处理后的数据信息:")
    print(f"行数: {len(df)}")
    print(f"列数: {len(df.columns)}")
    print(f"列名: {df.columns.tolist()}")
    
    # 保存处理后的数据
    if output_file is None:
        output_file = input_file
    
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n预处理后的数据已保存到: {output_file}")
    
    return df

if __name__ == "__main__":
    input_file = "data.csv"
    # 将处理后的数据保存回原文件
    df_processed = preprocess_data(input_file)
    
    # 显示处理后的数据样例
    print("\n处理后的数据预览:")
    print(df_processed.head(10)) 