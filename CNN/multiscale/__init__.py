# src/__init__.py
"""
Enhanced Genomic Disease Prediction Framework

A modular framework for multilabel disease prediction using genomic data
with advanced deep learning architectures including multi-scale convolutions,
transformers, and disease-specific attention mechanisms.
"""
__description__ = "Enhanced Genomic Disease Prediction Framework"

# Import main components for easy access
from .main import main
from .config import parse_args, validate_hardcoded_parameters
from .models import MultilabelGenotypeModel
from .training import Trainer, EarlyStopping
from .data import MultilabelGenotypeDataset
from .utils import compute_final_metrics, plot_multilabel_metrics

# Package-level exports
__all__ = [
    # Main entry point
    'main',
    
    # Configuration
    'parse_args',
    'validate_hardcoded_parameters',
    
    # Core model
    'MultilabelGenotypeModel',
    
    # Training components
    'Trainer',
    'EarlyStopping',
    
    # Data handling
    'MultilabelGenotypeDataset',
    
    # Utilities
    'compute_final_metrics',
    'plot_multilabel_metrics',
]