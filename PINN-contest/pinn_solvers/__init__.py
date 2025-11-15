"""
PINN Solvers Module
"""

from .base_pinn import BasePINN, MLP
from .helmholtz_solver import HelmholtzPINN, train_helmholtz_pinn
from .network_architectures import (
    FourierFeatureMapping, 
    AdaptiveSinActivation,
    ResidualUnit,
    MFFM_PINN,
    R_gPINN,
    DualNetworkPINN,
    create_network
)

__all__ = [
    'BasePINN', 'MLP', 
    'HelmholtzPINN', 'train_helmholtz_pinn',
    'FourierFeatureMapping', 'AdaptiveSinActivation', 'ResidualUnit',
    'MFFM_PINN', 'R_gPINN', 'DualNetworkPINN', 'create_network'
]

