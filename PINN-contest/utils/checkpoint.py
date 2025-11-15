"""
Checkpoint管理
"""

import torch
import os
from pathlib import Path


def save_checkpoint(state_dict, path, **kwargs):
    """
    保存checkpoint
    
    Args:
        state_dict: 模型state_dict
        path: 保存路径
        **kwargs: 其他需要保存的信息
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'state_dict': state_dict,
        **kwargs
    }
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(path, device='cpu'):
    """
    加载checkpoint
    
    Args:
        path: checkpoint路径
        device: 加载设备
        
    Returns:
        dict: checkpoint内容
    """
    checkpoint = torch.load(path, map_location=device)
    print(f"Checkpoint loaded: {path}")
    return checkpoint

