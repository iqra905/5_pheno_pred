# multiscale/utils/__init__.py
"""
Utility modules for metrics, plotting, and I/O operations

This module provides essential utilities including:
- Comprehensive metric calculations
- Visualization and plotting functions
- Checkpoint management and I/O operations
- Result serialization and reporting
"""

from .metrics import (
    compute_final_metrics,
    print_final_results
)
from .plotting import plot_multilabel_metrics
from .checkpointing import (
    save_checkpoint,
    find_latest_checkpoint,
    cleanup_old_checkpoints,
    write_results,
    get_hyperparameters_dict
)

__all__ = [
    # Metrics and evaluation
    'compute_final_metrics',
    'print_final_results',
    
    # Visualization
    'plot_multilabel_metrics',
    
    # Checkpoint management
    'save_checkpoint',
    'find_latest_checkpoint', 
    'cleanup_old_checkpoints',
    
    # Results and I/O
    'write_results',
    'get_hyperparameters_dict'
]

# Version info
__version__ = "1.0.0"
__description__ = "Utilities for metrics, plotting, and I/O operations"

# Supported metrics
AVAILABLE_METRICS = [
    'accuracy',
    'sensitivity', 
    'specificity',
    'auc',
    'confusion_matrix',
    'precision',
    'recall',
    'f1_score'
]

# Supported plot types
AVAILABLE_PLOTS = [
    'training_curves',
    'loss_progression',
    'learning_rate',
    'disease_specific_metrics',
    'combined_dashboard'
]

# File formats for results
SUPPORTED_FORMATS = [
    'txt',  # Human-readable text
    'csv',  # Machine-readable CSV
    'json', # JSON format
    'pkl'   # Pickle format
]