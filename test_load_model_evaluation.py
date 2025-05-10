import os
import json
import glob

def test_load_model_evaluation():
    """测试从JSON文件中加载模型评估结果"""
    try:
        # 使用相对路径 models/results
        results_dir = 'models/results'
        print(f"模型评估结果目录: {results_dir}")

        # 使用glob模块查找所有JSON文件
        try:
            all_files = glob.glob(os.path.join(results_dir, '*.json'))
            print(f"找到的所有JSON文件: {all_files}")
        except Exception as e:
            print(f"查找JSON文件时出错: {str(e)}")
            all_files = []

        # 根据文件名确定模型类型
        model_files = {}
        for file_path in all_files:
            if not file_path:  # 跳过空行
                continue
            file_name = os.path.basename(file_path)
            if file_name.startswith('mlp_'):
                model_files['mlp'] = file_path
            elif file_name.startswith('lstm_'):
                model_files['lstm'] = file_path
            elif file_name.startswith('cnn_'):
                model_files['cnn'] = file_path

        print(f"模型文件映射: {model_files}")

        # 如果没有找到文件，使用硬编码的文件名
        if not model_files:
            print("未找到任何模型评估结果文件，使用硬编码的文件名")
            model_files = {
                'mlp': os.path.join(results_dir, 'mlp_corr_pearson_20250510_203310_metrics.json'),
                'lstm': os.path.join(results_dir, 'lstm_corr_pearson_20250510_203354_metrics.json'),
                'cnn': os.path.join(results_dir, 'cnn_corr_pearson_20250510_203909_metrics.json')
            }

        # 检查文件是否存在
        for model_type, file_path in model_files.items():
            print(f"检查 {model_type.upper()} 模型的评估结果文件: {file_path}")
            if os.path.exists(file_path):
                print(f"{model_type.upper()} 模型的评估结果文件存在")
            else:
                print(f"{model_type.upper()} 模型的评估结果文件不存在: {file_path}")
                model_files[model_type] = None

        # 记录找到的文件
        found_files = [f for f in model_files.values() if f is not None]
        print(f"找到 {len(found_files)} 个模型评估结果文件")

        # 初始化模型指标
        metrics = {
            'mlp': {'rmse': 0, 'mae': 0, 'mape': 0, 'accuracy': 0, 'r2': 0},
            'lstm': {'rmse': 0, 'mae': 0, 'mape': 0, 'accuracy': 0, 'r2': 0},
            'cnn': {'rmse': 0, 'mae': 0, 'mape': 0, 'accuracy': 0, 'r2': 0}
        }

        # 对每个模型进行评估
        for model_type in ['mlp', 'lstm', 'cnn']:
            try:
                model_file = model_files.get(model_type)

                if model_file and os.path.exists(model_file):
                    print(f"使用 {model_type.upper()} 模型的评估结果文件: {os.path.basename(model_file)}")

                    # 读取JSON文件
                    try:
                        # 直接使用Python的文件读取功能
                        with open(model_file, 'r', encoding='utf-8') as f:
                            file_content = f.read()

                        if not file_content:
                            print(f"文件内容为空: {model_file}")
                            continue

                        print(f"文件内容: {file_content[:100]}...")  
                        model_data = json.loads(file_content)
                        print(f"成功解析JSON数据: {model_type}")
                    except Exception as e:
                        print(f"读取或解析JSON文件时出错: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        continue

                    # 提取指标
                    if 'metrics' in model_data:
                        metrics[model_type]['rmse'] = float(model_data['metrics'].get('rmse', 0))
                        metrics[model_type]['mae'] = float(model_data['metrics'].get('mae', 0))
                        metrics[model_type]['mape'] = float(model_data['metrics'].get('mape', 0))
                        metrics[model_type]['accuracy'] = float(model_data['metrics'].get('accuracy', 0))
                        metrics[model_type]['r2'] = float(model_data['metrics'].get('r2', 0))

                        print(f"{model_type.upper()} 模型指标: RMSE={metrics[model_type]['rmse']:.2f}, MAE={metrics[model_type]['mae']:.2f}, MAPE={metrics[model_type]['mape']:.2f}%, 准确率={metrics[model_type]['accuracy']:.2f}%, R2={metrics[model_type]['r2']:.4f}")
                    else:
                        print(f"{model_type.upper()} 模型评估结果文件中没有metrics字段")
                else:
                    print(f"未找到 {model_type.upper()} 模型的评估结果文件")
            except Exception as e:
                print(f"{model_type.upper()} 模型评估结果处理时出错: {str(e)}")
                import traceback
                traceback.print_exc()

        # 记录最终的指标数据
        print(f"最终的指标数据: {metrics}")

        return metrics
    except Exception as e:
        print(f"加载模型评估结果时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    metrics = test_load_model_evaluation()
    print("\n最终结果:")
    print(json.dumps(metrics, indent=4))
