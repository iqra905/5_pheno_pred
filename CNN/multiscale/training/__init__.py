# multiscale/training/__init__.py
"""
Training modules including schedulers, trainers, and early stopping

This module provides comprehensive training infrastructure including:
- Main trainer class with training loops
- Custom learning rate schedulers
- Early stopping mechanisms
- Training utilities and helpers
"""

from .trainer import Trainer
from .schedulers import (
    WarmupExponential,
    ExponentialDecay,
    get_scheduler,
    print_lr
)
from .early_stopping import EarlyStopping

__all__ = [
    # Main trainer
    'Trainer',
    
    # Custom schedulers
    'WarmupExponential',
    'ExponentialDecay',
    
    # Scheduler utilities
    'get_scheduler',
    'print_lr',
    
    # Training control
    'EarlyStopping'
]

# Version info
__version__ = "1.0.0"
__description__ = "Training infrastructure for genomic disease prediction models"

# Scheduler registry
SCHEDULER_REGISTRY = {
    'warmup_exponential': WarmupExponential,
    'exponential_decay': ExponentialDecay,
}

# Available PyTorch schedulers (for reference)
PYTORCH_SCHEDULERS = [
    'plateau',
    'cosine', 
    'step',
    'multistep',
    'exponential'
]