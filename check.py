import joblib
import os

scaler_dir = "/Users/a/project/models/scalers" # 请确保这是正确的路径
scaler_files = ["feature_scaler.pkl", "target_scaler.pkl", "new_feature_scaler.pkl"]
results = {}

print(f"Inspecting scalers in: {scaler_dir}")

for fname in scaler_files:
    fpath = os.path.join(scaler_dir, fname)
    if os.path.exists(fpath):
        try:
            scaler = joblib.load(fpath)
            # MinMaxScaler 对象通常有 n_features_in_ 属性
            if hasattr(scaler, 'n_features_in_'):
                num_features = scaler.n_features_in_
            # 有些旧版本或不同类型的 scaler 可能用 len(scaler.scale_)
            elif hasattr(scaler, 'scale_') and scaler.scale_ is not None:
                num_features = len(scaler.scale_)
            else:
                num_features = "无法确定特征数量"
            results[fname] = num_features
            print(f"  {fname}: {num_features} features")
        except Exception as e:
            results[fname] = f"加载时出错: {e}"
            print(f"  {fname}: 加载时出错 - {e}")
    else:
        results[fname] = "文件未找到"
        print(f"  {fname}: 文件未找到")

print("\nSummary:")
for fname, count in results.items():
    print(f"  {fname}: {count}")
