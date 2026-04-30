# multiscale/models/__init__.py
"""
Neural network model modules

This module contains all neural network architectures and components including:
- Main multilabel genotype prediction model
- Attention mechanisms for disease-specific processing
- Multi-scale convolution blocks
- Transformer components with pretrained initialization
"""

from .base_model import MultilabelGenotypeModel
from .attention import (
    DiseaseSpecificAttention,
    SeparateDiseaseHead
)
from .convolution import (
    MultiScaleConvBlock,
    ParallelMultiScaleConvBlock
)
from .transformer import (
    GenomicTransformerBlock,
    PositionalEncoding
)

__all__ = [
    # Main model
    'MultilabelGenotypeModel',
    
    # Attention mechanisms
    'DiseaseSpecificAttention',
    'SeparateDiseaseHead',
    
    # Convolution components
    'MultiScaleConvBlock',
    'ParallelMultiScaleConvBlock',
    
    # Transformer components
    'GenomicTransformerBlock',
    'PositionalEncoding'
]

# Model registry for easy access
MODEL_REGISTRY = {
    'multilabel_genotype': MultilabelGenotypeModel,
}

# Component registry
COMPONENT_REGISTRY = {
    'attention': {
        'disease_specific': DiseaseSpecificAttention,
        'separate_heads': SeparateDiseaseHead,
    },
    'convolution': {
        'multi_scale': MultiScaleConvBlock,
        'parallel_multi_scale': ParallelMultiScaleConvBlock,
    },
    'transformer': {
        'genomic_transformer': GenomicTransformerBlock,
        'positional_encoding': PositionalEncoding,
    }
}