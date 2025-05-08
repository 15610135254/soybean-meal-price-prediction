from flask import Blueprint, render_template, current_app, jsonify, request
from werkzeug.utils import secure_filename
import pandas as pd
import json
import os
import logging
import sys
import numpy as np
from werkzeug.utils import secure_filename
from views.auth import login_required, admin_required

# 自定义JSON编码器，处理NaN值
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_)):
            return bool(obj)
        if pd.isna(obj) or np.isnan(obj):
            return None
        return super(NpEncoder, self).default(obj)

# 创建一个名为 'visualization' 的 Blueprint
bp = Blueprint('visualization', __name__)

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 数据文件路径 (相对于当前文件)
DATA_FILE_PATH = '../model_data/date1.csv'

# 备用数据文件路径，以防主路径不可用
BACKUP_DATA_PATHS = [
    'model_data/date1.csv',
    '../model_data/date1.csv',
    'date1.csv',
    'data.csv',
    'data/data.csv',
    '../data/data.csv'
]

# 模型指标文件路径
METRICS_FILE_PATH = '../model_metrics.json'
BACKUP_METRICS_PATHS = [
    'model_metrics.json',
    '../model_metrics.json'
]

# 文件上传配置
ALLOWED_EXTENSIONS = {'csv'}
UPLOAD_FOLDER = '../model_data'  # 相对于当前文件的路径

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_data_folder_path():
    """获取数据文件夹的绝对路径"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, UPLOAD_FOLDER))

@bp.route('/')
@login_required
def view_data():
    """数据可视化页面路由（需要登录）"""
    chart_data_json = None

    # 使用相对于当前文件的路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    full_data_path = os.path.join(current_dir, DATA_FILE_PATH)

    # 如果主路径不存在，尝试备用路径
    if not os.path.exists(full_data_path):
        logger.warning(f"主数据文件 {full_data_path} 未找到，尝试备用路径")

        # 尝试相对于当前文件的备用路径
        for backup_path in BACKUP_DATA_PATHS:
            temp_path = os.path.join(current_dir, backup_path)
            if os.path.exists(temp_path):
                full_data_path = temp_path
                logger.info(f"使用备用数据文件: {full_data_path}")
                break

        # 如果仍然找不到，尝试相对于项目根目录的路径
        if not os.path.exists(full_data_path):
            base_dir = os.path.dirname(current_app.root_path)
            for backup_path in BACKUP_DATA_PATHS:
                temp_path = os.path.join(base_dir, backup_path)
                if os.path.exists(temp_path):
                    full_data_path = temp_path
                    logger.info(f"使用项目根目录下的备用数据文件: {full_data_path}")
                    break

    try:
        # 尝试加载数据文件
        if os.path.exists(full_data_path):
            logger.info(f"正在加载数据文件: {full_data_path}")
            df = pd.read_csv(full_data_path)

            # 检查数据集的列名格式
            if 'date' in df.columns and '日期' not in df.columns:
                logger.info("检测到新数据集格式，进行列名映射...")
                # 确保日期列是 datetime 类型
                df['date'] = pd.to_datetime(df['date'])
                # 按日期从早到晚排序（升序）
                df = df.sort_values('date', ascending=True)
                logger.info("已对数据按日期升序排序（从早到晚）")
                logger.info(f"数据日期范围: {df['date'].min()} 至 {df['date'].max()}")
                logger.info(f"排序后前10个日期: {', '.join(df['date'].dt.strftime('%Y-%m-%d').head(10).tolist())}")
                logger.info(f"排序后后10个日期: {', '.join(df['date'].dt.strftime('%Y-%m-%d').tail(10).tolist())}")

                # 准备图表数据 (选择日期和收盘价)
                chart_data = {
                    'labels': df['date'].dt.strftime('%Y-%m-%d').tolist(),  # 日期格式化为字符串
                    'closing_prices': df['close'].tolist(),
                    'opening_prices': df['open'].tolist() if 'open' in df.columns else [],
                    'high_prices': df['high'].tolist() if 'high' in df.columns else [],
                    'low_prices': df['low'].tolist() if 'low' in df.columns else [],
                    'volumes': df['volume'].tolist() if 'volume' in df.columns else []
                }
            else:
                # 旧数据集格式
                # 确保日期列是 datetime 类型并按日期从早到晚排序（升序）
                df['日期'] = pd.to_datetime(df['日期'])
                df = df.sort_values('日期', ascending=True)
                logger.info("已对数据按日期升序排序（从早到晚）")
                logger.info(f"数据日期范围: {df['日期'].min()} 至 {df['日期'].max()}")
                logger.info(f"排序后前10个日期: {', '.join(df['日期'].dt.strftime('%Y-%m-%d').head(10).tolist())}")
                logger.info(f"排序后后10个日期: {', '.join(df['日期'].dt.strftime('%Y-%m-%d').tail(10).tolist())}")

                # 准备图表数据 (选择日期和收盘价)
                chart_data = {
                    'labels': df['日期'].dt.strftime('%Y-%m-%d').tolist(),  # 日期格式化为字符串
                    'closing_prices': df['收盘价'].tolist(),
                    'opening_prices': df['开盘价'].tolist() if '开盘价' in df.columns else [],
                    'high_prices': df['最高价'].tolist() if '最高价' in df.columns else [],
                    'low_prices': df['最低价'].tolist() if '最低价' in df.columns else [],
                    'volumes': df['成交量'].tolist() if '成交量' in df.columns else []
                }

            # 添加所有技术指标数据（如果存在）
            # 检查是否是新数据集格式
            is_new_format = 'date' in df.columns and '日期' not in df.columns

            # 移动平均线
            if is_new_format:
                # 新数据集格式
                for column in ['MA_5', 'MA_10', 'MA_20', 'MA_30', 'MA_60']:
                    if column in df.columns:
                        chart_data[column.lower().replace('_', '')] = df[column].tolist()
            else:
                # 旧数据集格式
                for column in ['MA5', 'MA10', 'MA20', 'MA30', 'MA60', 'EMA12', 'EMA26']:
                    if column in df.columns:
                        chart_data[column.lower()] = df[column].tolist()

            # RSI指标
            if is_new_format:
                if 'RSI_14' in df.columns:
                    chart_data['rsi'] = df['RSI_14'].tolist()
            else:
                if 'RSI' in df.columns:
                    chart_data['rsi'] = df['RSI'].tolist()

            # MACD指标
            if 'MACD' in df.columns:
                chart_data['MACD'] = df['MACD'].tolist()

            if not is_new_format:
                for column in ['MACD_Signal', 'MACD_Hist']:
                    if column in df.columns:
                        chart_data[column] = df[column].tolist()

            # KDJ指标
            for column in ['RSV', 'K', 'D', 'J']:
                if column in df.columns:
                    chart_data[column] = df[column].tolist()

            # 布林带指标
            for column in ['中轨线', '标准差', '上轨线', '下轨线']:
                if column in df.columns:
                    chart_data[column] = df[column].tolist()

            # 成交量指标
            for column in ['成交量变化率', '相对成交量', '成交量MA5', '成交量MA10']:
                if column in df.columns:
                    chart_data[column] = df[column].tolist()

            # 其他技术指标和特征
            for column in ['涨跌幅', '日内波幅', '价格变动', '突破MA5', '突破MA10', '突破MA20', '金叉', '死叉']:
                if column in df.columns:
                    chart_data[column] = df[column].tolist()

            # 新数据集特有的指标
            if is_new_format:
                if 'HV_20' in df.columns:
                    chart_data['hv20'] = df['HV_20'].tolist()
                if 'ATR_14' in df.columns:
                    chart_data['atr14'] = df['ATR_14'].tolist()
                if 'OBV' in df.columns:
                    chart_data['obv'] = df['OBV'].tolist()

            # 将数据转换为 JSON 格式传递给模板
            chart_data_json = json.dumps(chart_data, ensure_ascii=False)
            logger.info(f"成功加载数据，共 {len(df)} 条记录")

        else:
            logger.error(f"无法找到任何有效的数据文件")
            chart_data_json = json.dumps({
                'labels': [],
                'closing_prices': [],
                'error': '无法找到数据文件'
            }, ensure_ascii=False)

    except FileNotFoundError:
        logger.error(f"错误：数据文件 {full_data_path} 未找到")
        chart_data_json = json.dumps({
            'labels': [],
            'closing_prices': [],
            'error': f'数据文件未找到: {os.path.basename(full_data_path)}'
        }, ensure_ascii=False)

    except Exception as e:
        logger.error(f"加载或处理数据时出错: {e}")
        chart_data_json = json.dumps({
            'labels': [],
            'closing_prices': [],
            'error': f'数据处理错误: {str(e)}'
        }, ensure_ascii=False)

    # 加载模型指标
    model_metrics = load_model_metrics()

    # 调试输出模型指标
    logger.info(f"模型指标: {model_metrics}")

    # 返回模板，传递数据
    return render_template(
        'visualization/view_data.html',
        chart_data=chart_data_json,
        model_metrics=json.dumps(model_metrics),
        data_path=os.path.basename(full_data_path),
        data_exists=os.path.exists(full_data_path)
    )

def load_model_metrics():
    """加载模型评估指标"""
    # 默认指标数据，如果无法加载文件时使用
    default_metrics = {
        'mlp': {'accuracy': 97.71, 'rmse': 104.08},
        'lstm': {'accuracy': 97.34, 'rmse': 144.14},
        'cnn': {'accuracy': 97.25, 'rmse': 127.27}
    }

    # 尝试从 all_models_training_summary.json 加载模型指标
    try:
        # 获取项目根目录
        base_dir = os.path.dirname(current_app.root_path)
        summary_path = os.path.join(base_dir, 'results', 'all_models_training_summary.json')

        # 检查文件是否存在
        if os.path.exists(summary_path):
            logger.info(f"正在从 all_models_training_summary.json 加载模型指标")
            with open(summary_path, 'r') as f:
                summary_data = json.load(f)

            # 从 summary_data 中提取每个模型的最新评估指标
            metrics = {}

            # 处理 MLP 模型
            if 'MLP' in summary_data and summary_data['MLP']:
                latest_mlp = summary_data['MLP'][-1]  # 获取最新的 MLP 模型数据
                if 'evaluation_metrics' in latest_mlp:
                    eval_metrics = latest_mlp['evaluation_metrics']
                    metrics['mlp'] = {
                        'accuracy': eval_metrics.get('Accuracy', default_metrics['mlp']['accuracy']),
                        'rmse': eval_metrics.get('RMSE', default_metrics['mlp']['rmse']),
                        'mae': eval_metrics.get('MAE', 0),
                        'r2': eval_metrics.get('R2', 0),
                        'mape': eval_metrics.get('MAPE', 0)
                    }

            # 处理 LSTM 模型
            if 'LSTM' in summary_data and summary_data['LSTM']:
                latest_lstm = summary_data['LSTM'][-1]  # 获取最新的 LSTM 模型数据
                if 'evaluation_metrics' in latest_lstm:
                    eval_metrics = latest_lstm['evaluation_metrics']
                    metrics['lstm'] = {
                        'accuracy': eval_metrics.get('Accuracy', default_metrics['lstm']['accuracy']),
                        'rmse': eval_metrics.get('RMSE', default_metrics['lstm']['rmse']),
                        'mae': eval_metrics.get('MAE', 0),
                        'r2': eval_metrics.get('R2', 0),
                        'mape': eval_metrics.get('MAPE', 0)
                    }

            # 处理 CNN 模型
            if 'CNN' in summary_data and summary_data['CNN']:
                latest_cnn = summary_data['CNN'][-1]  # 获取最新的 CNN 模型数据
                if 'evaluation_metrics' in latest_cnn:
                    eval_metrics = latest_cnn['evaluation_metrics']
                    metrics['cnn'] = {
                        'accuracy': eval_metrics.get('Accuracy', default_metrics['cnn']['accuracy']),
                        'rmse': eval_metrics.get('RMSE', default_metrics['cnn']['rmse']),
                        'mae': eval_metrics.get('MAE', 0),
                        'r2': eval_metrics.get('R2', 0),
                        'mape': eval_metrics.get('MAPE', 0)
                    }

            # 确保所有必要的模型都存在
            if all(model in metrics for model in ['mlp', 'lstm', 'cnn']):
                logger.info(f"成功从 all_models_training_summary.json 加载模型指标: {metrics}")
                return metrics
            else:
                # 如果缺少某些模型，使用默认值填充
                logger.warning(f"从 all_models_training_summary.json 加载的模型指标不完整，使用默认值填充缺失的模型")
                for model in ['mlp', 'lstm', 'cnn']:
                    if model not in metrics:
                        metrics[model] = default_metrics[model]
                return metrics

        # 如果 all_models_training_summary.json 不存在，尝试使用旧的指标文件
        logger.warning(f"all_models_training_summary.json 不存在，尝试使用旧的指标文件")

        # 使用相对于当前文件的路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        full_metrics_path = os.path.join(current_dir, METRICS_FILE_PATH)

        # 如果主路径不存在，尝试备用路径
        if not os.path.exists(full_metrics_path):
            logger.warning(f"主指标文件 {full_metrics_path} 未找到，尝试备用路径")

            # 尝试相对于当前文件的备用路径
            for backup_path in BACKUP_METRICS_PATHS:
                temp_path = os.path.join(current_dir, backup_path)
                if os.path.exists(temp_path):
                    full_metrics_path = temp_path
                    logger.info(f"使用备用指标文件: {full_metrics_path}")
                    break

            # 如果仍然找不到，尝试相对于项目根目录的路径
            if not os.path.exists(full_metrics_path):
                for backup_path in BACKUP_METRICS_PATHS:
                    temp_path = os.path.join(base_dir, backup_path)
                    if os.path.exists(temp_path):
                        full_metrics_path = temp_path
                        logger.info(f"使用项目根目录下的备用指标文件: {full_metrics_path}")
                        break

        # 尝试加载旧的指标文件
        if os.path.exists(full_metrics_path):
            logger.info(f"正在加载旧的模型指标文件: {full_metrics_path}")
            with open(full_metrics_path, 'r') as f:
                metrics = json.load(f)

            # 确保指标文件包含所有必要的模型
            if all(model in metrics for model in ['mlp', 'lstm', 'cnn']):
                return metrics

        # 如果所有尝试都失败，返回默认指标数据
        logger.warning(f"无法加载任何模型指标文件，使用默认指标数据")
        return default_metrics

    except Exception as e:
        logger.error(f"加载模型指标时出错: {e}")
        import traceback
        traceback.print_exc()
        # 返回默认指标数据
        return default_metrics

@bp.route('/api/model-metrics')
def get_model_metrics():
    """API端点：获取模型评估指标"""
    metrics = load_model_metrics()
    return jsonify(metrics)

@bp.route('/api/predict')
def predict():
    """API端点：使用选定的模型进行预测"""
    # 获取请求参数
    model_type = request.args.get('model', 'mlp')
    days = int(request.args.get('days', 7))

    # 记录请求信息
    logger.info(f"收到预测请求: 模型={model_type}, 天数={days}")

    try:
        # 导入预测模块
        import sys
        import os

        # 获取当前文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # 获取项目根目录
        base_dir = os.path.dirname(current_app.root_path)

        # 将项目根目录添加到系统路径
        if base_dir not in sys.path:
            sys.path.append(base_dir)

        # 导入预测函数
        from predict_prices import predict_with_model

        # 调用预测函数
        result = predict_with_model(model_type, days)

        # 返回预测结果
        return jsonify(result)

    except Exception as e:
        logger.error(f"预测时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@bp.route('/api/edit', methods=['POST'])
@admin_required
def edit_data():
    """API端点：编辑数据（仅管理员）"""
    try:
        # 获取请求数据
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "没有提供数据"}), 400

        # 获取编辑的数据
        date = data.get('date')
        open_price = data.get('open')
        high_price = data.get('high')
        low_price = data.get('low')
        close_price = data.get('close')
        volume = data.get('volume')

        # 验证必要的字段
        if not date or not close_price:
            return jsonify({"success": False, "error": "日期和收盘价是必填字段"}), 400

        # 获取数据文件路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        full_data_path = os.path.join(current_dir, DATA_FILE_PATH)

        # 加载数据文件
        if not os.path.exists(full_data_path):
            return jsonify({"success": False, "error": "数据文件不存在"}), 404

        # 读取CSV文件
        df = pd.read_csv(full_data_path)

        # 检查数据集的列名格式
        if 'date' in df.columns and '日期' not in df.columns:
            # 新数据集格式
            # 确保日期列是datetime类型
            df['date'] = pd.to_datetime(df['date'])

            # 查找要编辑的行
            date_obj = pd.to_datetime(date)
            mask = df['date'] == date_obj

            if not mask.any():
                return jsonify({"success": False, "error": f"未找到日期为 {date} 的数据"}), 404

            # 更新数据
            if open_price is not None:
                df.loc[mask, 'open'] = float(open_price)
            if high_price is not None:
                df.loc[mask, 'high'] = float(high_price)
            if low_price is not None:
                df.loc[mask, 'low'] = float(low_price)
            if close_price is not None:
                df.loc[mask, 'close'] = float(close_price)
            if volume is not None:
                df.loc[mask, 'volume'] = int(volume)

            # 确保数据按日期从早到晚排序（升序）
            df = df.sort_values('date', ascending=True)
            logger.info("已对数据按日期升序排序（从早到晚）")
        else:
            # 旧数据集格式
            # 确保日期列是datetime类型
            df['日期'] = pd.to_datetime(df['日期'])

            # 查找要编辑的行
            date_obj = pd.to_datetime(date)
            mask = df['日期'] == date_obj

            if not mask.any():
                return jsonify({"success": False, "error": f"未找到日期为 {date} 的数据"}), 404

            # 更新数据
            if open_price is not None:
                df.loc[mask, '开盘价'] = float(open_price)
            if high_price is not None:
                df.loc[mask, '最高价'] = float(high_price)
            if low_price is not None:
                df.loc[mask, '最低价'] = float(low_price)
            if close_price is not None:
                df.loc[mask, '收盘价'] = float(close_price)
            if volume is not None:
                df.loc[mask, '成交量'] = int(volume)

            # 确保数据按日期从早到晚排序（升序）
            df = df.sort_values('日期', ascending=True)
            logger.info("已对数据按日期升序排序（从早到晚）")

        # 保存更新后的数据到原文件，不更换数据源
        df.to_csv(full_data_path, index=False)

        # 预处理数据，但不更改DATA_FILE_PATH
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # 检查数据集的列名格式
        if 'date' in df.columns and '日期' not in df.columns:
            # 新数据集格式
            from data.preprocess_new_data import preprocess_new_data
            df = preprocess_new_data(full_data_path, full_data_path)

            # 准备返回数据
            chart_data = {
                'labels': df['date'].dt.strftime('%Y-%m-%d').tolist(),
                'closing_prices': df['close'].tolist(),
                'opening_prices': df['open'].tolist() if 'open' in df.columns else [],
                'high_prices': df['high'].tolist() if 'high' in df.columns else [],
                'low_prices': df['low'].tolist() if 'low' in df.columns else [],
                'volumes': df['volume'].tolist() if 'volume' in df.columns else []
            }

            # 添加所有技术指标数据（如果存在）
            # 移动平均线
            for column in ['MA_5', 'MA_10', 'MA_20', 'MA_30', 'MA_60']:
                if column in df.columns:
                    chart_data[column.lower().replace('_', '')] = df[column].tolist()

            # RSI指标
            if 'RSI_14' in df.columns:
                chart_data['rsi'] = df['RSI_14'].tolist()

            # MACD指标
            if 'MACD' in df.columns:
                chart_data['MACD'] = df['MACD'].tolist()

            # 波动率指标
            if 'HV_20' in df.columns:
                chart_data['hv20'] = df['HV_20'].tolist()

            # ATR指标
            if 'ATR_14' in df.columns:
                chart_data['atr14'] = df['ATR_14'].tolist()

            # OBV指标
            if 'OBV' in df.columns:
                chart_data['obv'] = df['OBV'].tolist()

            # 价格变动和日内波幅
            if 'price_change' in df.columns:
                chart_data['price_change'] = df['price_change'].tolist()
            if 'daily_range' in df.columns:
                chart_data['daily_range'] = df['daily_range'].tolist()

            # 涨跌幅
            if 'price_change_pct' in df.columns:
                chart_data['price_change_pct'] = df['price_change_pct'].tolist()

            # 成交量变化率
            if 'volume_change_pct' in df.columns:
                chart_data['volume_change_pct'] = df['volume_change_pct'].tolist()
        else:
            # 旧数据集格式
            from data.preprocess_data import preprocess_data
            df = preprocess_data(full_data_path, full_data_path)

            # 准备返回数据
            chart_data = {
                'labels': df['日期'].dt.strftime('%Y-%m-%d').tolist(),
                'closing_prices': df['收盘价'].tolist(),
                'opening_prices': df['开盘价'].tolist() if '开盘价' in df.columns else [],
                'high_prices': df['最高价'].tolist() if '最高价' in df.columns else [],
                'low_prices': df['最低价'].tolist() if '最低价' in df.columns else [],
                'volumes': df['成交量'].tolist() if '成交量' in df.columns else []
            }

            # 添加所有技术指标数据（如果存在）
            # 移动平均线
            for column in ['MA5', 'MA10', 'MA20', 'MA30', 'MA60', 'EMA12', 'EMA26']:
                if column in df.columns:
                    chart_data[column.lower()] = df[column].tolist()

            # RSI指标
            if 'RSI' in df.columns:
                chart_data['rsi'] = df['RSI'].tolist()

            # MACD指标
            for column in ['MACD', 'MACD_Signal', 'MACD_Hist']:
                if column in df.columns:
                    chart_data[column] = df[column].tolist()

            # KDJ指标
            for column in ['RSV', 'K', 'D', 'J']:
                if column in df.columns:
                    chart_data[column] = df[column].tolist()

            # 布林带指标
            for column in ['中轨线', '标准差', '上轨线', '下轨线']:
                if column in df.columns:
                    chart_data[column] = df[column].tolist()

            # 成交量指标
            for column in ['成交量变化率', '相对成交量', '成交量MA5', '成交量MA10']:
                if column in df.columns:
                    chart_data[column] = df[column].tolist()

            # 其他技术指标和特征
            for column in ['涨跌幅', '日内波幅', '价格变动', '突破MA5', '突破MA10', '突破MA20', '金叉', '死叉']:
                if column in df.columns:
                    chart_data[column] = df[column].tolist()

        # 返回成功响应
        response_data = {
            "success": True,
            "message": "数据编辑成功",
            "chart_data": chart_data,
            "rows": len(df)
        }

        return current_app.response_class(
            json.dumps(response_data, cls=NpEncoder),
            mimetype='application/json'
        )

    except Exception as e:
        logger.error(f"编辑数据时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@bp.route('/api/delete', methods=['POST'])
@admin_required
def delete_data():
    """API端点：删除数据（仅管理员）"""
    try:
        # 获取请求数据
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "没有提供数据"}), 400

        # 获取删除日期
        delete_date = data.get('date')

        # 验证必要的字段
        if not delete_date:
            return jsonify({"success": False, "error": "删除日期是必填字段"}), 400

        # 获取数据文件路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        full_data_path = os.path.join(current_dir, DATA_FILE_PATH)

        # 加载数据文件
        if not os.path.exists(full_data_path):
            return jsonify({"success": False, "error": "数据文件不存在"}), 404

        # 读取CSV文件
        df = pd.read_csv(full_data_path)

        # 检查数据集的列名格式
        if 'date' in df.columns and '日期' not in df.columns:
            # 新数据集格式
            # 确保日期列是datetime类型
            df['date'] = pd.to_datetime(df['date'])

            # 记录删除前的数据信息
            logger.info(f"删除前数据行数: {len(df)}")
            logger.info(f"要删除的日期: {delete_date}")

            # 转换删除日期为datetime对象
            delete_date_obj = pd.to_datetime(delete_date)

            # 记录删除前的行数
            original_rows = len(df)

            # 只删除指定日期的数据
            df = df[df['date'] != delete_date_obj]

            # 确保数据按日期从早到晚排序（升序）
            df = df.sort_values('date', ascending=True)
        else:
            # 旧数据集格式
            # 确保日期列是datetime类型
            df['日期'] = pd.to_datetime(df['日期'])

            # 记录删除前的数据信息
            logger.info(f"删除前数据行数: {len(df)}")
            logger.info(f"要删除的日期: {delete_date}")

            # 转换删除日期为datetime对象
            delete_date_obj = pd.to_datetime(delete_date)

            # 记录删除前的行数
            original_rows = len(df)

            # 只删除指定日期的数据
            df = df[df['日期'] != delete_date_obj]

            # 确保数据按日期从早到晚排序（升序）
            df = df.sort_values('日期', ascending=True)

        # 计算删除的行数
        deleted_rows = original_rows - len(df)
        logger.info(f"删除的行数: {deleted_rows}")

        if deleted_rows <= 0:
            return jsonify({"success": False, "error": "未找到指定日期的数据"}), 404

        # 保存更新后的数据
        df.to_csv(full_data_path, index=False)

        # 预处理数据
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # 检查数据集的列名格式
        if 'date' in df.columns and '日期' not in df.columns:
            # 新数据集格式
            from data.preprocess_new_data import preprocess_new_data
            df = preprocess_new_data(full_data_path, full_data_path)

            # 准备返回数据
            chart_data = {
                'labels': df['date'].dt.strftime('%Y-%m-%d').tolist(),
                'closing_prices': df['close'].tolist(),
                'opening_prices': df['open'].tolist() if 'open' in df.columns else [],
                'high_prices': df['high'].tolist() if 'high' in df.columns else [],
                'low_prices': df['low'].tolist() if 'low' in df.columns else [],
                'volumes': df['volume'].tolist() if 'volume' in df.columns else []
            }

            # 添加所有技术指标数据（如果存在）
            # 移动平均线
            for column in ['MA_5', 'MA_10', 'MA_20', 'MA_30', 'MA_60']:
                if column in df.columns:
                    chart_data[column.lower().replace('_', '')] = df[column].tolist()

            # RSI指标
            if 'RSI_14' in df.columns:
                chart_data['rsi'] = df['RSI_14'].tolist()

            # MACD指标
            if 'MACD' in df.columns:
                chart_data['MACD'] = df['MACD'].tolist()

            # 波动率指标
            if 'HV_20' in df.columns:
                chart_data['hv20'] = df['HV_20'].tolist()

            # ATR指标
            if 'ATR_14' in df.columns:
                chart_data['atr14'] = df['ATR_14'].tolist()

            # OBV指标
            if 'OBV' in df.columns:
                chart_data['obv'] = df['OBV'].tolist()

            # 价格变动和日内波幅
            if 'price_change' in df.columns:
                chart_data['price_change'] = df['price_change'].tolist()
            if 'daily_range' in df.columns:
                chart_data['daily_range'] = df['daily_range'].tolist()

            # 涨跌幅
            if 'price_change_pct' in df.columns:
                chart_data['price_change_pct'] = df['price_change_pct'].tolist()

            # 成交量变化率
            if 'volume_change_pct' in df.columns:
                chart_data['volume_change_pct'] = df['volume_change_pct'].tolist()
        else:
            # 旧数据集格式
            from data.preprocess_data import preprocess_data
            df = preprocess_data(full_data_path, full_data_path)

            # 准备返回数据
            chart_data = {
                'labels': df['日期'].dt.strftime('%Y-%m-%d').tolist(),
                'closing_prices': df['收盘价'].tolist(),
                'opening_prices': df['开盘价'].tolist() if '开盘价' in df.columns else [],
                'high_prices': df['最高价'].tolist() if '最高价' in df.columns else [],
                'low_prices': df['最低价'].tolist() if '最低价' in df.columns else [],
                'volumes': df['成交量'].tolist() if '成交量' in df.columns else []
            }

            # 添加所有技术指标数据（如果存在）
            # 移动平均线
            for column in ['MA5', 'MA10', 'MA20', 'MA30', 'MA60', 'EMA12', 'EMA26']:
                if column in df.columns:
                    chart_data[column.lower()] = df[column].tolist()

            # RSI指标
            if 'RSI' in df.columns:
                chart_data['rsi'] = df['RSI'].tolist()

            # MACD指标
            for column in ['MACD', 'MACD_Signal', 'MACD_Hist']:
                if column in df.columns:
                    chart_data[column] = df[column].tolist()

            # KDJ指标
            for column in ['RSV', 'K', 'D', 'J']:
                if column in df.columns:
                    chart_data[column] = df[column].tolist()

            # 布林带指标
            for column in ['中轨线', '标准差', '上轨线', '下轨线']:
                if column in df.columns:
                    chart_data[column] = df[column].tolist()

            # 成交量指标
            for column in ['成交量变化率', '相对成交量', '成交量MA5', '成交量MA10']:
                if column in df.columns:
                    chart_data[column] = df[column].tolist()

            # 其他技术指标和特征
            for column in ['涨跌幅', '日内波幅', '价格变动', '突破MA5', '突破MA10', '突破MA20', '金叉', '死叉']:
                if column in df.columns:
                    chart_data[column] = df[column].tolist()

        # 返回成功响应
        response_data = {
            "success": True,
            "message": f"成功删除 {deleted_rows} 条数据",
            "chart_data": chart_data,
            "rows": len(df)
        }

        return current_app.response_class(
            json.dumps(response_data, cls=NpEncoder),
            mimetype='application/json'
        )

    except Exception as e:
        logger.error(f"删除数据时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@bp.route('/api/upload', methods=['POST'])
@admin_required
def upload_file():
    """API端点：上传数据文件（仅管理员）"""
    # 检查是否有文件
    if 'file' not in request.files:
        logger.error("没有文件部分")
        return jsonify({"success": False, "error": "没有选择文件"}), 400

    file = request.files['file']

    # 检查文件名是否为空
    if file.filename == '':
        logger.error("没有选择文件")
        return jsonify({"success": False, "error": "没有选择文件"}), 400

    # 检查文件类型
    if not allowed_file(file.filename):
        logger.error(f"不允许的文件类型: {file.filename}")
        return jsonify({"success": False, "error": "只允许上传CSV文件"}), 400

    try:
        # 获取数据文件夹路径
        data_folder = get_data_folder_path()

        # 确保目录存在
        os.makedirs(data_folder, exist_ok=True)

        # 安全地获取文件名
        filename = secure_filename(file.filename)

        # 始终使用默认文件名date1.csv，替换现有数据
        save_path = os.path.join(data_folder, 'date1.csv')
        logger.info(f"上传的文件将保存为: {save_path}")

        # 保存文件
        file.save(save_path)
        logger.info(f"文件已保存到: {save_path}")

        # 预处理数据文件
        try:
            # 导入预处理模块
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from data.preprocess_new_data import preprocess_new_data

            # 预处理数据
            df = preprocess_new_data(save_path, save_path)

            # 始终更新DATA_FILE_PATH为默认路径
            global DATA_FILE_PATH
            DATA_FILE_PATH = '../model_data/date1.csv'
            logger.info("已更新DATA_FILE_PATH为默认路径")

            # 准备返回数据
            chart_data = {
                'labels': df['date'].dt.strftime('%Y-%m-%d').tolist(),
                'closing_prices': df['close'].tolist(),
                'opening_prices': df['open'].tolist() if 'open' in df.columns else [],
                'high_prices': df['high'].tolist() if 'high' in df.columns else [],
                'low_prices': df['low'].tolist() if 'low' in df.columns else [],
                'volumes': df['volume'].tolist() if 'volume' in df.columns else []
            }

            # 添加所有技术指标数据（如果存在）
            # 移动平均线
            for column in ['MA_5', 'MA_10', 'MA_20', 'MA_30', 'MA_60']:
                if column in df.columns:
                    chart_data[column.lower().replace('_', '')] = df[column].tolist()

            # RSI指标
            if 'RSI_14' in df.columns:
                chart_data['rsi'] = df['RSI_14'].tolist()

            # MACD指标
            if 'MACD' in df.columns:
                chart_data['MACD'] = df['MACD'].tolist()

            # 波动率指标
            if 'HV_20' in df.columns:
                chart_data['hv20'] = df['HV_20'].tolist()

            # ATR指标
            if 'ATR_14' in df.columns:
                chart_data['atr14'] = df['ATR_14'].tolist()

            # OBV指标
            if 'OBV' in df.columns:
                chart_data['obv'] = df['OBV'].tolist()

            # 价格变动和日内波幅
            if 'price_change' in df.columns:
                chart_data['price_change'] = df['price_change'].tolist()
            if 'daily_range' in df.columns:
                chart_data['daily_range'] = df['daily_range'].tolist()

            # 涨跌幅
            if 'price_change_pct' in df.columns:
                chart_data['price_change_pct'] = df['price_change_pct'].tolist()

            # 成交量变化率
            if 'volume_change_pct' in df.columns:
                chart_data['volume_change_pct'] = df['volume_change_pct'].tolist()

            # 处理NaN值
            response_data = {
                "success": True,
                "message": "文件上传成功",
                "filename": os.path.basename(save_path),
                "chart_data": chart_data,
                "rows": len(df)
            }

            # 使用自定义JSON编码器处理NaN值
            return current_app.response_class(
                json.dumps(response_data, cls=NpEncoder),
                mimetype='application/json'
            )

        except Exception as e:
            logger.error(f"预处理数据时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            response_data = {
                "success": True,
                "message": "文件已上传，但预处理失败",
                "filename": os.path.basename(save_path),
                "error": str(e)
            }
            return current_app.response_class(
                json.dumps(response_data, cls=NpEncoder),
                mimetype='application/json'
            )

    except Exception as e:
        logger.error(f"上传文件时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        response_data = {"success": False, "error": str(e)}
        return current_app.response_class(
            json.dumps(response_data, cls=NpEncoder),
            mimetype='application/json',
            status=500
        )

@bp.route('/model-evaluation')
@admin_required
def model_evaluation():
    """模型评估页面（仅管理员）"""
    # 加载模型指标
    model_metrics = load_model_metrics()

    # 获取模型文件信息
    model_files = []

    # 初始化模型详情，使用默认值
    model_details = {
        'mlp': {'params': 6688896, 'layers': 25, 'input_shape': '(None, 10, 15)', 'output_shape': '(None, 1)'},
        'lstm': {'params': 11655784, 'layers': 18, 'input_shape': '(None, 10, 15)', 'output_shape': '(None, 1)'},
        'cnn': {'params': 11674952, 'layers': 22, 'input_shape': '(None, 10, 15)', 'output_shape': '(None, 1)'}
    }

    # 记录初始模型详情
    logger.info(f"初始模型详情: {model_details}")

    try:
        # 获取项目根目录
        base_dir = os.path.dirname(current_app.root_path)
        model_dir = os.path.join(base_dir, 'best_models')

        # 记录模型目录路径
        logger.info(f"模型目录路径: {model_dir}")

        # 如果模型目录不存在，尝试其他可能的路径
        if not os.path.exists(model_dir):
            # 尝试相对于当前文件的路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            alt_model_dir = os.path.join(current_dir, '../best_models')
            if os.path.exists(alt_model_dir):
                model_dir = alt_model_dir
                logger.info(f"使用备用模型目录路径: {model_dir}")
            else:
                # 尝试直接使用best_models路径
                if os.path.exists('best_models'):
                    model_dir = 'best_models'
                    logger.info(f"使用相对模型目录路径: {model_dir}")
                else:
                    logger.warning(f"无法找到模型目录: {model_dir}, {alt_model_dir}, best_models")

        if os.path.exists(model_dir):
            # 尝试导入TensorFlow以获取模型详细信息
            try:
                import tensorflow as tf
                tf_available = True
            except ImportError:
                tf_available = False
                logger.warning("TensorFlow未安装，无法加载模型详细信息")

            for file in os.listdir(model_dir):
                if file.endswith('.h5'):
                    file_path = os.path.join(model_dir, file)
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # 转换为MB
                    file_time = os.path.getmtime(file_path)

                    # 提取模型类型
                    model_type = 'unknown'
                    if 'mlp' in file.lower():
                        model_type = 'mlp'
                    elif 'lstm' in file.lower():
                        model_type = 'lstm'
                    elif 'cnn' in file.lower():
                        model_type = 'cnn'

                    # 创建模型文件信息
                    model_info = {
                        'name': file,
                        'type': model_type,
                        'size': f"{file_size:.2f} MB",
                        'modified': pd.to_datetime(file_time, unit='s').strftime('%Y-%m-%d %H:%M:%S'),
                        'path': file_path
                    }

                    # 如果TensorFlow可用，尝试加载模型获取更多信息
                    if tf_available:
                        try:
                            model = tf.keras.models.load_model(file_path)

                            # 获取模型参数
                            total_params = model.count_params()
                            trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
                            non_trainable_params = total_params - trainable_params

                            # 获取模型结构信息
                            input_shape = str(model.input_shape)
                            output_shape = str(model.output_shape)

                            # 添加到模型信息
                            model_info['total_params'] = f"{total_params:,}"
                            model_info['trainable_params'] = f"{trainable_params:,}"
                            model_info['non_trainable_params'] = f"{non_trainable_params:,}"
                            model_info['layers'] = len(model.layers)
                            model_info['input_shape'] = input_shape
                            model_info['output_shape'] = output_shape

                            # 更新模型类型详情
                            if model_type in model_details:
                                model_details[model_type]['params'] = total_params
                                model_details[model_type]['layers'] = len(model.layers)
                                model_details[model_type]['input_shape'] = input_shape
                                model_details[model_type]['output_shape'] = output_shape

                        except Exception as e:
                            logger.warning(f"加载模型 {file} 时出错: {str(e)}")

                    model_files.append(model_info)
    except Exception as e:
        logger.error(f"获取模型文件信息时出错: {str(e)}")

    # 按模型类型排序
    model_files.sort(key=lambda x: x['type'])

    return render_template(
        'visualization/model_evaluation.html',
        model_metrics=model_metrics,
        model_files=model_files,
        model_details=model_details
    )

@bp.route('/api/model-details/<model_type>', methods=['GET'])
@admin_required
def get_model_details(model_type):
    """API端点：获取模型详细信息（仅管理员）"""
    try:
        # 获取项目根目录
        base_dir = os.path.dirname(current_app.root_path)
        model_dir = os.path.join(base_dir, 'best_models')

        # 记录模型目录路径
        logger.info(f"API - 模型目录路径: {model_dir}")

        # 如果模型目录不存在，尝试其他可能的路径
        if not os.path.exists(model_dir):
            # 尝试相对于当前文件的路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            alt_model_dir = os.path.join(current_dir, '../best_models')
            if os.path.exists(alt_model_dir):
                model_dir = alt_model_dir
                logger.info(f"API - 使用备用模型目录路径: {model_dir}")
            else:
                # 尝试直接使用best_models路径
                if os.path.exists('best_models'):
                    model_dir = 'best_models'
                    logger.info(f"API - 使用相对模型目录路径: {model_dir}")
                else:
                    logger.warning(f"API - 无法找到模型目录: {model_dir}, {alt_model_dir}, best_models")

        # 查找对应类型的模型文件
        model_file = None

        # 检查模型目录是否存在
        if os.path.exists(model_dir):
            # 列出模型目录中的所有文件
            model_files = os.listdir(model_dir)
            logger.info(f"API - 模型目录中的文件: {model_files}")

            # 查找对应类型的模型文件
            for file in model_files:
                if file.endswith('.h5') and model_type.lower() in file.lower():
                    model_file = os.path.join(model_dir, file)
                    logger.info(f"API - 找到模型文件: {model_file}")
                    break

        if not model_file:
            return jsonify({
                'success': False,
                'message': f'未找到{model_type.upper()}模型文件'
            }), 404

        # 尝试导入TensorFlow以获取模型详细信息
        try:
            import tensorflow as tf

            # 加载模型
            model = tf.keras.models.load_model(model_file)

            # 获取模型结构
            layers_info = []
            for i, layer in enumerate(model.layers):
                layer_info = {
                    'name': layer.name,
                    'type': layer.__class__.__name__,
                    'params': layer.count_params(),
                    'input_shape': str(layer.input_shape),
                    'output_shape': str(layer.output_shape)
                }

                # 获取层的配置
                config = layer.get_config()
                if 'units' in config:
                    layer_info['units'] = config['units']
                if 'activation' in config:
                    layer_info['activation'] = config['activation']
                if 'filters' in config:
                    layer_info['filters'] = config['filters']
                if 'kernel_size' in config:
                    layer_info['kernel_size'] = config['kernel_size']
                if 'pool_size' in config:
                    layer_info['pool_size'] = config['pool_size']
                if 'rate' in config:
                    layer_info['dropout_rate'] = config['rate']
                if 'strides' in config:
                    layer_info['strides'] = config['strides']
                if 'padding' in config:
                    layer_info['padding'] = config['padding']
                if 'kernel_regularizer' in config and config['kernel_regularizer']:
                    layer_info['regularizer'] = 'L2' if 'l2' in str(config['kernel_regularizer']).lower() else 'L1'
                if 'recurrent_dropout' in config:
                    layer_info['recurrent_dropout'] = config['recurrent_dropout']

                # 添加更多层配置信息
                if layer.__class__.__name__ == 'Dense':
                    layer_info['description'] = f"全连接层，{config['units']}个神经元"
                elif layer.__class__.__name__ == 'Conv1D':
                    layer_info['description'] = f"一维卷积层，{config['filters']}个过滤器，核大小{config['kernel_size']}"
                elif layer.__class__.__name__ == 'LSTM' or layer.__class__.__name__ == 'GRU':
                    layer_info['description'] = f"循环神经网络层，{config['units']}个单元"
                elif layer.__class__.__name__ == 'Dropout':
                    layer_info['description'] = f"丢弃层，丢弃率{config['rate']}"
                elif layer.__class__.__name__ == 'BatchNormalization':
                    layer_info['description'] = "批归一化层，用于加速训练和提高模型稳定性"
                elif layer.__class__.__name__ == 'MaxPooling1D':
                    layer_info['description'] = f"最大池化层，池化大小{config['pool_size']}"
                elif layer.__class__.__name__ == 'GlobalAveragePooling1D':
                    layer_info['description'] = "全局平均池化层，用于降维"
                elif layer.__class__.__name__ == 'Flatten':
                    layer_info['description'] = "展平层，将多维输入转换为一维"
                elif layer.__class__.__name__ == 'Concatenate':
                    layer_info['description'] = "连接层，合并多个输入特征"
                elif layer.__class__.__name__ == 'Add':
                    layer_info['description'] = "加法层，实现残差连接"
                elif layer.__class__.__name__ == 'Input':
                    layer_info['description'] = f"输入层，形状{layer.input_shape}"
                elif 'Activation' in layer.__class__.__name__ or layer.__class__.__name__ == 'PReLU' or layer.__class__.__name__ == 'LeakyReLU':
                    layer_info['description'] = f"激活层，增加网络非线性能力"
                else:
                    layer_info['description'] = f"{layer.__class__.__name__}层"

                layers_info.append(layer_info)

            # 获取模型总体信息
            total_params = model.count_params()
            trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
            non_trainable_params = total_params - trainable_params

            # 获取模型指标
            model_metrics = load_model_metrics()
            metrics = model_metrics.get(model_type.lower(), {})

            # 构建模型架构描述
            model_architecture_description = get_model_architecture_description(model_type.lower())

            # 构建响应
            response = {
                'success': True,
                'model_name': os.path.basename(model_file),
                'model_type': model_type.upper(),
                'total_params': total_params,
                'trainable_params': trainable_params,
                'non_trainable_params': non_trainable_params,
                'layers_count': len(model.layers),
                'input_shape': str(model.input_shape),
                'output_shape': str(model.output_shape),
                'layers': layers_info,
                'metrics': metrics,
                'architecture_description': model_architecture_description
            }

            return jsonify(response)

        except ImportError:
            return jsonify({
                'success': False,
                'message': 'TensorFlow未安装，无法获取模型详细信息'
            }), 500
        except Exception as e:
            logger.error(f"获取模型详细信息时出错: {str(e)}")
            return jsonify({
                'success': False,
                'message': f'获取模型详细信息时出错: {str(e)}'
            }), 500

    except Exception as e:
        logger.error(f"获取模型详细信息时出错: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'获取模型详细信息时出错: {str(e)}'
        }), 500

def get_model_architecture_description(model_type):
    """获取模型架构描述"""
    if model_type == 'mlp':
        return """
        多层感知机 (MLP) 模型架构采用了增强型设计，包含多个全连接层和批归一化层。

        主要特点：
        1. 使用批归一化加速训练并提高稳定性
        2. 采用PReLU激活函数增强非线性表达能力
        3. 引入残差连接避免梯度消失问题
        4. 使用Dropout层减少过拟合
        5. 采用L2正则化提高泛化能力

        该模型特别适合处理豆粕期货价格的非线性关系，能够有效捕捉价格变动模式。
        """
    elif model_type == 'lstm':
        return """
        长短期记忆网络 (LSTM) 模型架构采用了双向GRU设计，专门用于捕捉时间序列数据中的长期依赖关系。

        主要特点：
        1. 使用双向GRU层代替传统LSTM，减轻计算负担
        2. 批归一化层用于稳定训练过程
        3. 采用小的循环dropout防止过拟合
        4. 多层设计增强模型表达能力
        5. 使用tanh激活函数捕捉非线性关系

        该模型特别适合处理豆粕期货价格的时间序列特性，能够记忆长期价格趋势并预测未来走势。
        """
    elif model_type == 'cnn':
        return """
        卷积神经网络 (CNN) 模型架构采用了多分支设计，专门用于提取时间序列数据中的局部特征和模式。

        主要特点：
        1. 多分支卷积结构，使用不同大小的卷积核捕捉不同尺度的模式
        2. 批归一化层加速训练并提高稳定性
        3. 全局平均池化层提取全局特征
        4. 使用PReLU激活函数增强非线性表达能力
        5. 采用L2正则化减少过拟合

        该模型特别适合识别豆粕期货价格中的局部模式和趋势，计算效率高且预测准确度好。
        """
    else:
        return "未找到模型架构描述"

@bp.route('/api/run-evaluation', methods=['POST'])
@admin_required
def run_evaluation():
    """API端点：运行模型评估（仅管理员）"""
    try:
        # 导入评估模块
        import sys
        import os

        # 获取项目根目录
        base_dir = os.path.dirname(current_app.root_path)

        # 将项目根目录添加到系统路径
        if base_dir not in sys.path:
            sys.path.append(base_dir)

        # 导入评估函数
        from model_metrics import evaluate_models

        # 运行评估
        metrics = evaluate_models()

        if metrics:
            return jsonify({
                'success': True,
                'message': '模型评估完成',
                'metrics': metrics
            })
        else:
            return jsonify({
                'success': False,
                'message': '模型评估失败，请检查日志'
            }), 500

    except Exception as e:
        logger.error(f"运行模型评估时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'评估过程中出错: {str(e)}'
        }), 500