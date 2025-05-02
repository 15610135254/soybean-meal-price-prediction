"""
训练配置参数
"""

# 通用训练参数
TRAINING_CONFIG = {
    'epochs': 200,  # 增加训练轮数
    'batch_size': 64,
    'validation_split': 0.1,
    'test_size': 0.1,
    'look_back': 20,  # 增加历史数据窗口大小
    'early_stopping_patience': 20,  # 增加早停耐心值
    'reduce_lr_patience': 10,  # 增加学习率调整耐心值
    'min_lr': 1e-6
}

# MLP模型参数
MLP_CONFIG = {
    'units1': 512,  # 增加神经元数量
    'units2': 256,
    'units3': 128,
    'units4': 64,
    'dropout': 0.3,
    'learning_rate': 0.001,
    'l2_reg': 0.001
}

# LSTM模型参数
LSTM_CONFIG = {
    'lstm_units1': 256,  # 增加LSTM单元数量
    'lstm_units2': 128,
    'lstm_units3': 64,
    'dense_units1': 128,
    'dense_units2': 64,
    'dropout': 0.3,
    'learning_rate': 0.001,
    'l2_reg': 0.001
}

# CNN模型参数
CNN_CONFIG = {
    'filters1': 512,  # 增加卷积核数量
    'filters2': 256,
    'filters3': 128,
    'filters4': 64,
    'kernel_size': 5,
    'pool_size': 2,
    'dense_units1': 256,
    'dense_units2': 128,
    'dropout': 0.2,
    'learning_rate': 0.0005,
    'l2_reg': 0.0005
} 