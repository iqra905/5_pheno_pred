# multiscale/data/__init__.py
"""
Data processing and dataset modules

This module handles all data-related operations including:
- Dataset classes for genomic data
- Data preprocessing and normalization
- Quality control and validation
- Sample statistics and visualization
"""

from .dataset import MultilabelGenotypeDataset
from .preprocessing import (
    CovariateNormalizer,
    get_input_size,
    clean_phenotype_data,
    print_sample_examples,
    print_disease_statistics
)

__all__ = [
    # Main dataset class
    'MultilabelGenotypeDataset',
    
    # Preprocessing utilities
    'CovariateNormalizer',
    'get_input_size',
    'clean_phenotype_data',
    
    # Data analysis functions
    'print_sample_examples',
    'print_disease_statistics'
]

# Version info
__version__ = "1.0.0"
__description__ = "Data processing and dataset handling for genomic disease prediction"