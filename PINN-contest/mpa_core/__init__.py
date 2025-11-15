"""
MPA Core Module - Method-Problem Alignment Theory
"""

from .alignment_metrics import compute_gdc, compute_ts, compute_lsm, compute_all_metrics
from .problem_profiler import ProblemProfiler, profile_task1, profile_task2, profile_task3
from .optimizer_selector import OptimizerSelector, select_optimizer_by_mpa

__all__ = [
    'compute_gdc', 'compute_ts', 'compute_lsm', 'compute_all_metrics',
    'ProblemProfiler', 'profile_task1', 'profile_task2', 'profile_task3',
    'OptimizerSelector', 'select_optimizer_by_mpa'
]

