#!/bin/bash

# 检查是否为Apple Silicon芯片
PROCESSOR=$(uname -p)
if [[ "$PROCESSOR" != "arm" ]]; then
    echo "当前脚本针对Apple Silicon芯片Mac，您的处理器为: $PROCESSOR"
    echo "脚本仍将继续，但请注意部分配置可能不适用"
fi

echo "===== 期货数据分析与预测环境配置工具 ====="
echo "此脚本将安装TensorFlow和相关依赖以支持Apple Silicon芯片"
echo ""

# 检查conda是否已安装
if ! command -v conda &> /dev/null; then
    echo "未检测到conda。请先安装Miniconda或Anaconda。"
    echo "访问 https://docs.conda.io/en/latest/miniconda.html 下载安装"
    exit 1
fi

echo "请输入新环境名称 (默认: finance_ml):"
read ENV_NAME
ENV_NAME=${ENV_NAME:-finance_ml}

echo "正在创建新的conda环境: $ENV_NAME..."
conda create -y -n $ENV_NAME python=3.9

echo "激活环境..."
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

echo "安装TensorFlow依赖..."
conda install -y -c apple tensorflow-deps

echo "安装TensorFlow (macOS优化版)..."
pip install tensorflow-macos

echo "安装TensorFlow Metal插件 (GPU加速)..."
pip install tensorflow-metal

echo "安装项目所需的其他依赖..."
pip install pandas numpy matplotlib scikit-learn

echo "===== 安装完成 ====="
echo "使用以下命令激活环境:"
echo "  conda activate $ENV_NAME"
echo ""
echo "建议测试TensorFlow安装:"
echo 'python -c "import tensorflow as tf; print(tf.__version__); print(\"GPU可用:\", bool(tf.config.list_physical_devices(\"GPU\")))"'
echo ""
echo "若需要利用Apple Silicon芯片的ML Compute加速，确保在代码中添加:"
echo 'from tensorflow.python.compiler.mlcompute import mlcompute'
echo 'mlcompute.set_mlc_device(device_name=\"gpu\")' 