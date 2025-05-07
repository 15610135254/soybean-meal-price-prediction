import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
import time

def get_data(url):
    """
    从新浪财经网站提取期货历史数据
    
    Args:
        url: 新浪财经期货历史数据URL
        
    Returns:
        DataFrame: 包含期货历史数据的DataFrame
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'
    }
    
    try:
        print(f"正在请求URL: {url}")
        response = requests.get(url, headers=headers)
        response.encoding = 'gbk'  # 新浪财经网站使用GBK编码
        
        print(f"状态码: {response.status_code}")
        
        # 解析HTML
        soup = BeautifulSoup(response.text, 'lxml')
        
        # 找到所有表格
        tables = soup.find_all('table')
        
        if len(tables) < 4:
            print("未找到足够的表格，请检查网页结构")
            return None
        
        # 根据检查结果，我们需要的数据在第4个表格中
        data_table = tables[3]  # 索引从0开始，所以第4个表格是索引3
        
        # 提取表格中的所有行
        rows = data_table.find_all('tr')
        
        if len(rows) < 2:
            print("表格中没有足够的行")
            return None
        
        # 提取表头
        header_row = rows[1]  # 第二行包含实际的表头（日期，收盘价等）
        headers = [th.get_text().strip() for th in header_row.find_all('td')]
        
        if not headers:
            print("未找到表头")
            return None
        
        print(f"表头: {headers}")
        
        # 提取数据行
        data_rows = rows[2:]  # 从第三行开始是数据
        
        data_list = []
        for row in data_rows:
            cells = row.find_all('td')
            if cells:
                row_data = [cell.get_text().strip() for cell in cells]
                if row_data and len(row_data) == len(headers):
                    data_list.append(row_data)
        
        # 创建DataFrame
        df = pd.DataFrame(data_list, columns=headers)
        
        # 转换数据类型
        if not df.empty:
            # 转换日期列
            date_col = headers[0]  # 第一列是日期
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            
            # 转换数值列
            numeric_cols = headers[1:]
            for col in numeric_cols:
                # 清理数值（去除逗号等非数字字符）
                df[col] = df[col].apply(lambda x: re.sub(r'[^\d.]', '', str(x)) if pd.notna(x) else x)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"成功提取了 {len(df)} 行数据")
        return df
        
    except Exception as e:
        print(f"提取数据时出错: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

def save_to_csv(df, filename="futures_data.csv"):
    """保存数据到CSV文件"""
    if df is not None and not df.empty:
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"数据已保存到 {filename}")
    else:
        print("没有数据可保存")

def get_pages_range(base_url, start_page=1, end_page=120):
    """
    提取指定页码范围的数据
    
    Args:
        base_url: 基础URL
        start_page: 起始页码
        end_page: 结束页码
        
    Returns:
        DataFrame: 合并所有页面数据的DataFrame
    """
    all_data = []
    
    for page in range(start_page, end_page + 1):
        # 构建分页URL
        page_url = f"{base_url}&page={page}"
        print(f"\n正在提取第 {page} 页数据...")
        
        # 获取当前页数据
        df = get_data(page_url)
        
        if df is not None and not df.empty:
            all_data.append(df)
            # 添加延迟，避免请求过快
            time.sleep(1)
        else:
            print(f"第 {page} 页数据提取失败或为空，停止爬取")
            break
    
    # 合并所有数据
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\n总共提取了 {len(combined_df)} 行数据")
        return combined_df
    else:
        print("未提取到任何数据")
        return None

if __name__ == "__main__":
    base_url = "https://vip.stock.finance.sina.com.cn/q/view/vFutures_History.php?jys=dce&pz=M&hy=M0&breed=M0&type=inner&start=1990-03-01&end=2025-05-02"
    
    print("请选择爬取模式:")
    print("1. 爬取指定页数")
    print("2. 爬取指定页码范围")
    
    choice = input("请输入选择 (1/2): ").strip()
    
    if choice == '1':
        # 爬取前n页
        max_pages = int(input("请输入要爬取的最大页数: ") or 10)
        df = get_pages_range(base_url, 1, max_pages)
    elif choice == '2':
        # 爬取指定页码范围
        start_page = int(input("请输入起始页码: ") or 1)
        end_page = int(input("请输入结束页码: ") or 120)
        df = get_pages_range(base_url, start_page, end_page)
    else:
        # 默认只爬取第一页
        print("选择无效，将默认爬取第一页数据")
        df = get_data(base_url)
    
    if df is not None:
        # 显示前10行数据
        print("\n数据预览:")
        print(df.head(10))
        
        # 显示数据类型
        print("\n数据类型:")
        print(df.dtypes)
        
        # 保存数据
        filename = input("请输入保存文件名 (默认为futures_data.csv): ").strip() or "futures_data.csv"
        save_to_csv(df, filename)