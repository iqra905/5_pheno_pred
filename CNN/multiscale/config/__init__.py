# multiscale/config/__init__.py
"""
Configuration and argument parsing modules

This module handles all configuration management including:
- Command-line argument parsing
- Parameter validation
- Configuration file handling
- Hyperparameter management
"""

from .args import (
    parse_args,
    validate_hardcoded_parameters,
    parse_int_list,
    parse_str_list,
    parse_nested_int_list
)

__all__ = [
    # Main configuration functions
    'parse_args',
    'validate_hardcoded_parameters',
    
    # Helper parsing functions
    'parse_int_list',
    'parse_str_list', 
    'parse_nested_int_list'
]

# Version info
__version__ = "1.0.0"
__description__ = "Configuration and argument parsing for genomic disease prediction"