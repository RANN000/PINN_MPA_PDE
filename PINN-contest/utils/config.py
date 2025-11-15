"""
配置管理
"""

import json
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """从JSON文件加载配置"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """保存配置到JSON文件"""
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)


# 默认配置
DEFAULT_CONFIG = {
    'task1_helmholtz': {
        'layers': [2, 50, 50, 50, 1],
        'activation': 'tanh',
        'learning_rate': 0.001,
        'weight_decay': 0.0,
        'wavenumber': 4,
        'lambda_pde': 1.0,
        'lambda_bc': 1.0,
        'n_epochs': 2000,
        'batch_size': 256,
        'log_every': 100
    },
    'task2_high_wavenumber': {
        'layers': [2, 100, 100, 100, 1],
        'activation': 'tanh',
        'learning_rate': 0.0001,
        'weight_decay': 0.0,
        'wavenumber': 100,
        'lambda_pde': 1.0,
        'lambda_bc': 1.0,
        'n_epochs': 5000,
        'batch_size': 512,
        'log_every': 500
    },
    'task3_poisson_inverse': {
        'layers_u': [2, 50, 50, 50, 1],
        'layers_lambda': [2, 20, 20, 1],
        'activation': 'tanh',
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        'lambda_pde': 1.0,
        'lambda_data': 10.0,
        'n_epochs': 3000,
        'batch_size': 128,
        'log_every': 100
    }
}

