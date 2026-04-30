import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import copy
import glob
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer
from torch.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
import argparse
import csv
import time
import shutil
import torch.nn.functional as F

# Try to import mamba-ssm and transformers
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
    print("Using official mamba-ssm implementation.")
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba-ssm not found. Using custom implementation.")

try:
    from transformers import ViTConfig, ViTModel, BertModel, BertConfig
    TRANSFORMERS_AVAILABLE = True
    print("transformers library available for pretrained initialization.")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not found. Pretrained initialization disabled.")

def parse_int_list(s):
    return [int(x) for x in s.split(',')]

def parse_str_list(s):
    return [x.strip() for x in s.split(',')]

def parse_nested_int_list(s):
    """Parse nested lists like '15,63,255;7,31,127;3,15,63' into [[15,63,255], [7,31,127], [3,15,63]]"""
    if not s or s.lower() == 'none':
        return None
    layers = s.split(';')
    return [[int(x) for x in layer.split(',')] for layer in layers]

def parse_layer_config(s):
    """Parse layer configuration like 'T,M,T,M' into ['transformer', 'mamba', 'transformer', 'mamba']"""
    if not s:
        return []
    mapping = {'T': 'transformer', 'M': 'mamba'}
    return [mapping.get(x.strip().upper(), x.strip().lower()) for x in s.split(',')]

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Transformer-Mamba Genomic Disease Prediction")
    parser.add_argument("-ID", type=str, default="Unified_Exp_01", help="ID of the experiment")
    parser.add_argument("-exp_dir", type=str, default='/mnt/fast/nobackup/users/if00208/5_disease_experiments/CNN/results/5d_multilabel/multiscale/mamba', help="Directory to save experiment results")
    parser.add_argument("-genotype_dir", type=str, default='/mnt/fast/datasets/ucdatasets/gwas/gwas_mono_rm/sampled_data_5M_unq_npy_dup/sampled_data_5M_unq_npy', help="Directory containing genotype files")
    parser.add_argument("-phenotype_file", type=str, default='/mnt/fast/datasets/ucdatasets/gwas/data_files/merged_v8_pcs_chip_added_Iqra_1_cleaned.xlsx', help="Path to phenotype file")

    # Model and training parameters
    parser.add_argument("-bs", type=int, default=5, help="Batch size for training")
    parser.add_argument("-dropout", type=float, default=0.5, help="Dropout rate for the model")
    parser.add_argument("-epochs", type=int, default=3, help="Number of epochs for training")
    parser.add_argument("-lr", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("-peak_lr", type=float, default=1e-2, help="Peak learning rate for WarmupExponential scheduler")
    parser.add_argument("-final_lr", type=float, default=1e-5, help="Final learning rate for custom schedulers")
    parser.add_argument("-act", type=str, default="gelu", choices=["tanh","relu","gelu"], help="Activation function for the model")
    parser.add_argument("-sch", type=str, default="exponential_decay", choices=["none","plateau", "cosine", "step","multistep","explr","warmup_exponential", "exponential_decay"], help="Learning rate scheduler")
    parser.add_argument("-df", type=float, default=0.2, help="Decay factor for custom schedulers")
    parser.add_argument("-opt", type=str, default="adamw", choices=["adam", "adamw", "sgd"], help="Optimizer to use")
    parser.add_argument("-wd", type=float, default=0.5, help="Weight decay for optimizer")

    # Model architecture - CNN layers
    parser.add_argument("-kernel_sizes", type=parse_int_list, default=[128,64,32], help="Convolution Kernel Size")
    parser.add_argument("-stride", type=parse_int_list, default=[16,16,16], help="Convolution Stride")
    parser.add_argument("-conv_channels", type=parse_int_list, default=[4,8,16], help="Convolution channels")
    parser.add_argument("-fc_layers", type=parse_int_list, default=[128,64], help="Fully connected layers")

    # Enhanced architecture parameters
    parser.add_argument("-use_multi_scale", type=int, default=1, choices=[0, 1], help="Whether to use multi-scale convolutions (0: no, 1: yes)")
    parser.add_argument("-use_disease_attention", type=int, default=0, choices=[0, 1], help="Whether to use disease-specific attention (0: no, 1: yes)")
    parser.add_argument("-use_separate_heads", type=int, default=0, choices=[0, 1], help="Whether to use separate disease heads (0: no, 1: yes)")
    parser.add_argument("-attention_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("-attention_dim", type=int, default=256, help="Attention dimension")
    
    # Multi-scale configuration
    parser.add_argument("-multi_scale_kernels", type=parse_int_list, default=[15,127], help="Multi-scale kernel sizes for first layer")
    parser.add_argument("-multi_scale_strides", type=parse_int_list, default=[8,16], help="Multi-scale strides for first layer")
    parser.add_argument("-multi_scale_fusion", type=str, default="parallel", choices=["cross_scale", "parallel"], help="Multi-scale fusion strategy")
    parser.add_argument("-multi_scale_mode", type=str, default="hardcoded", choices=["progressive", "hardcoded"], help="Multi-scale mode")
    parser.add_argument("-hardcoded_kernels", type=parse_nested_int_list, default='16,128,1024;16,64,512;16,32,256', help="Hardcoded kernel sizes for all layers")
    parser.add_argument("-hardcoded_strides", type=parse_nested_int_list, default='16,16,16;16,16,16;16,16,16', help="Hardcoded stride values for all layers")
    parser.add_argument("-use_pointwise_conv", type=int, default=0, choices=[0, 1], help="Whether to use pointwise convolution")
    parser.add_argument("-pointwise_channels", type=int, default=4, help="Number of output channels for pointwise convolution")

    parser.add_argument("-use_pooling", type=int, default=0, choices=[0, 1], help="Whether to use Pooling after convolution layers")
    parser.add_argument("-pool_size", type=int, default=256, help="Size of the adaptive pooling output")
    parser.add_argument("-pool_type", type=str, default="max", choices=["max", "avg"], help="Type of adaptive pooling")
    
    # Early stopping parameters
    parser.add_argument("-patience", type=int, default=15, help="Patience for early stopping")
    parser.add_argument("-min_delta", type=float, default=1e-4, help="Minimum change for early stopping")
    
    # Data-specific parameters
    parser.add_argument("-cov", type=int, default=1, choices=[0, 1], help="Whether to include PC's as covariates")
    parser.add_argument("-use_age", type=int, default=1, choices=[0, 1], help="Whether to include age in covariates")
    parser.add_argument("-use_gender", type=int, default=1, choices=[0, 1], help="Whether to include gender in covariates")
    parser.add_argument("-use_bmi", type=int, default=1, choices=[0, 1], help="Whether to include BMI in covariates")

    # Normalization parameters
    parser.add_argument("-norm_age", type=str, default="none", choices=["none", "standard", "minmax", "robust", "quantile", "power"], help="Normalization method for age")
    parser.add_argument("-norm_pcs", type=str, default="none", choices=["none", "standard", "minmax", "robust", "quantile", "power"], help="Normalization method for PCs")
    parser.add_argument("-norm_gender", type=str, default="none", choices=["none", "minmax"], help="Normalization method for gender")
    parser.add_argument("-norm_bmi", type=str, default="none", choices=["none", "standard", "minmax", "robust", "quantile", "power"], help="Normalization method for BMI")
    parser.add_argument("-disease_labels", type=parse_str_list, default="pros01,panca,crc,breacancer,t2dm", help="Comma-separated list of disease column names")

    # UNIFIED SEQUENCE MODELING ARGUMENTS
    
    # Architecture mode selection
    parser.add_argument("-sequence_mode", type=str, default="transformer", choices=["transformer", "mamba", "hybrid"], help="Sequence modeling mode: 'transformer', 'mamba', or 'hybrid' (interleaved)")
    
    # Hybrid mode configuration
    parser.add_argument("-layer_config", type=str, default="M,T", help="Layer configuration for hybrid mode. Use T for transformer, M for mamba. Example: 'T,M,T,M'")
    parser.add_argument("-hybrid_mode", type=str, default="parallel", choices=["sequential", "parallel"], help="Hybrid architecture mode: 'sequential' (layers in series) or 'parallel' (branches in parallel)")
    
    # Intermediate pooling arguments
    parser.add_argument("-use_intermediate_pooling", type=int, default=0, choices=[0, 1], help="Whether to use pooling between hybrid layers (0: no, 1: yes)")
    parser.add_argument("-intermediate_pooling_strategy", type=str, default="mean", choices=["mean", "max", "attention", "chunked", "multiscale", "multihead", "conv", "hierarchical"], help="Pooling strategy to use between hybrid layers")
    parser.add_argument("-intermediate_pooling_layers", type=str, default="0", help="Comma-separated layer indices after which to apply pooling (e.g., '0,2' for after 1st and 3rd layer)")
    parser.add_argument("-intermediate_pool_factor", type=int, default=2, help="Reduction factor for intermediate pooling (e.g., 2 means reduce by half)")
    parser.add_argument("-intermediate_chunked_num_chunks", type=int, default=10, help="Number of chunks for intermediate chunked pooling")
    parser.add_argument("-intermediate_chunked_pool_type", type=str, default="mean", choices=["max", "mean"], help="Pooling type for intermediate chunked pooling")
    parser.add_argument("-intermediate_conv_target_length", type=int, default=512, help="Target sequence length after intermediate convolutional pooling")

    # Shared sequence modeling parameters
    parser.add_argument("-d_model", type=int, default=384, help="Model dimension for sequence layers")
    parser.add_argument("-num_layers", type=int, default=2, help="Total number of sequence layers")
    parser.add_argument("-sequence_dropout", type=float, default=0.1, help="Dropout rate for sequence layers")
    parser.add_argument("-use_layer_norm", type=int, default=1, choices=[0, 1], help="Whether to use layer normalization")
    
    # Transformer-specific parameters
    parser.add_argument("-transformer_heads", type=int, default=8, help="Number of attention heads in transformer")
    parser.add_argument("-transformer_ff_dim", type=int, default=1024, help="Transformer feedforward dimension")
    parser.add_argument("-use_positional_encoding", type=int, default=1, choices=[0, 1], help="Whether to use positional encoding")
    parser.add_argument("-max_seq_len", type=int, default=10000, help="Maximum sequence length for positional encoding")
    
    # Transformer pretrained initialization
    parser.add_argument("-init_from_pretrained", type=int, default=0, choices=[0, 1], help="Whether to initialize from pretrained model")
    parser.add_argument("-pretrained_model_type", type=str, default="bert", choices=["auto", "vit", "bert"], help="Type of pretrained model")
    parser.add_argument("-pretrained_model_name", type=str, default="bert-base-uncased", help="Name of pretrained model")
    parser.add_argument("-init_layers_fraction", type=float, default=1.0, help="Fraction of layers to initialize from pretrained")
    parser.add_argument("-layer_init_strategy", type=str, default="middle", choices=["first", "middle", "last", "random", "custom"], help="Strategy for selecting pretrained layers")
    parser.add_argument("-custom_layer_indices", type=str, default="4,6", help="Custom layer indices for pretrained initialization")
    
    # Mamba-specific parameters
    parser.add_argument("-mamba_d_state", type=int, default=16, help="Mamba state dimension")
    parser.add_argument("-mamba_d_conv", type=int, default=4, help="Mamba local convolution width")
    parser.add_argument("-mamba_expand", type=int, default=2, help="Mamba expansion factor")
    
    # Token and pooling strategies
    parser.add_argument("-use_cls_token", type=int, default=0, choices=[0, 1], help="Whether to use class token (transformer mode)")
    parser.add_argument("-use_covariate_tokens", type=int, default=1, choices=[0, 1], help="Whether to process covariates as tokens")
    parser.add_argument("-covariate_token_strategy", type=str, default="combined", choices=["separate", "combined"], help="Covariate tokenization strategy")
    parser.add_argument("-covariate_embed_dim", type=int, default=64, help="Embedding dimension for covariate tokens")

    # Update pooling strategy choices
    parser.add_argument("-pooling_strategy", type=str, default="mean", 
                       choices=["mean", "max", "attention", "concat", "chunked", "multiscale", "multihead", "conv", "hierarchical"], 
                       help="Pooling strategy for final sequence representation")
    
    # Chunked pooling parameters
    parser.add_argument("-chunked_num_chunks", type=int, default=512, help="Number of chunks for chunked pooling strategy")
    parser.add_argument("-chunked_pool_type", type=str, default="mean", choices=["mean", "max"], help="Pooling type within each chunk")
    
    # Multi-scale pooling parameters
    parser.add_argument("-multiscale_window_sizes", type=parse_int_list, default=[8,32,128,512], help="Window sizes for multi-scale pooling (comma-separated)")
    
    # Multi-head pooling parameters  
    parser.add_argument("-multihead_num_heads", type=int, default=8, help="Number of attention heads for multi-head pooling")
    parser.add_argument("-multihead_head_dim", type=int, default=256, help="Dimension of each attention head")
    
    # Convolutional pooling parameters
    parser.add_argument("-conv_target_length", type=int, default=512, help="Target sequence length after convolutional pooling")
    parser.add_argument("-conv_num_layers", type=int, default=1, help="Number of convolutional layers for downsampling")
    
    # Hierarchical pooling parameters
    parser.add_argument("-hierarchical_levels", type=parse_int_list, default=[512,128,32,8], help="Hierarchical pooling levels (comma-separated)")
    
    # Checkpoint parameters
    parser.add_argument("-resume", type=int, default=1, choices=[0, 1], help="Whether to resume from checkpoint")
    parser.add_argument("-keep_checkpoints", type=int, default=1, help="Number of recent checkpoints to keep")
    
    return parser.parse_args()

def get_pooling_kwargs(args):
    """Extract pooling-specific kwargs from arguments"""
    pooling_kwargs = {}
    
    if args.pooling_strategy == "chunked":
        pooling_kwargs = {
            'num_chunks': args.chunked_num_chunks,
            'pool_type': args.chunked_pool_type
        }
    elif args.pooling_strategy == "multiscale":
        pooling_kwargs = {
            'window_sizes': args.multiscale_window_sizes
        }
    elif args.pooling_strategy == "multihead":
        pooling_kwargs = {
            'num_heads': args.multihead_num_heads,
            'head_dim': args.multihead_head_dim
        }
    elif args.pooling_strategy == "conv":
        pooling_kwargs = {
            'target_length': args.conv_target_length,
            'conv_layers': args.conv_num_layers
        }
    elif args.pooling_strategy == "hierarchical":
        pooling_kwargs = {
            'levels': args.hierarchical_levels
        }
    
    return pooling_kwargs

def calculate_pooling_output_dim(pooling_strategy, d_model, pooling_kwargs):
    """Calculate output dimension for different pooling strategies"""
    if pooling_strategy == "chunked":
        return pooling_kwargs['num_chunks'] * d_model
    elif pooling_strategy == "multiscale":
        return len(pooling_kwargs['window_sizes']) * d_model
    elif pooling_strategy == "multihead":
        return pooling_kwargs['num_heads'] * pooling_kwargs['head_dim']
    elif pooling_strategy == "conv":
        return d_model * pooling_kwargs['target_length']
    elif pooling_strategy == "hierarchical":
        return len(pooling_kwargs['levels']) * d_model
    elif pooling_strategy in ["mean", "max", "attention"]:
        return d_model
    else:  # concat
        return None  # Will be calculated later

# class ChunkedPooling(nn.Module):
#     """Divide sequence into chunks and pool within each chunk"""
#     def __init__(self, num_chunks=32, pool_type='mean'):
#         super(ChunkedPooling, self).__init__()
#         self.num_chunks = num_chunks
#         self.pool_type = pool_type
        
#     def forward(self, x):
#         # x: [batch_size, seq_len, d_model]
#         batch_size, seq_len, d_model = x.shape
        
#         # Calculate chunk size
#         chunk_size = seq_len // self.num_chunks
#         trimmed_len = chunk_size * self.num_chunks
        
#         # Trim sequence to fit exact chunks
#         x_trimmed = x[:, :trimmed_len, :]  # [batch, trimmed_len, d_model]
        
#         # Reshape to chunks
#         x_chunks = x_trimmed.view(batch_size, self.num_chunks, chunk_size, d_model)
        
#         # Pool within each chunk
#         if self.pool_type == 'mean':
#             pooled_chunks = torch.mean(x_chunks, dim=2)  # [batch, num_chunks, d_model]
#         elif self.pool_type == 'max':
#             pooled_chunks, _ = torch.max(x_chunks, dim=2)
        
#         # Flatten chunks
#         output = pooled_chunks.view(batch_size, -1)  # [batch, num_chunks * d_model]
#         return output

class ChunkedPooling(nn.Module):
    """Efficient non-overlapping chunked pooling with vectorized operations"""
    def __init__(self, num_chunks=512, pool_type='mean'):
        super(ChunkedPooling, self).__init__()
        self.num_chunks = num_chunks
        self.pool_type = pool_type
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape
        
        # Handle edge cases
        if seq_len < self.num_chunks:
            padding = self.num_chunks - seq_len
            x = F.pad(x, (0, 0, 0, padding))
            return x.view(batch_size, -1)
        
        if seq_len == self.num_chunks:
            return x.view(batch_size, -1)
        
        # Calculate chunk sizes
        base_chunk_size = seq_len // self.num_chunks
        num_larger_chunks = seq_len % self.num_chunks
        
        # Split into two groups: larger chunks and base chunks
        larger_chunk_size = base_chunk_size + 1
        larger_total_len = num_larger_chunks * larger_chunk_size
        base_total_len = (self.num_chunks - num_larger_chunks) * base_chunk_size
        
        pooled_chunks = []
        
        # Process larger chunks (if any)
        if num_larger_chunks > 0:
            x_larger = x[:, :larger_total_len, :]  # [batch, larger_total_len, d_model]
            
            # Reshape to [batch, num_larger_chunks, larger_chunk_size, d_model]
            x_larger = x_larger.view(batch_size, num_larger_chunks, larger_chunk_size, d_model)
            
            # Pool along chunk dimension
            if self.pool_type == 'mean':
                pooled_larger = torch.mean(x_larger, dim=2)  # [batch, num_larger_chunks, d_model]
            elif self.pool_type == 'max':
                pooled_larger, _ = torch.max(x_larger, dim=2)
            
            pooled_chunks.append(pooled_larger)
        
        # Process base chunks (if any)
        if base_chunk_size > 0 and (self.num_chunks - num_larger_chunks) > 0:
            x_base = x[:, larger_total_len:larger_total_len + base_total_len, :]
            
            # Reshape to [batch, num_base_chunks, base_chunk_size, d_model]
            num_base_chunks = self.num_chunks - num_larger_chunks
            x_base = x_base.view(batch_size, num_base_chunks, base_chunk_size, d_model)
            
            # Pool along chunk dimension
            if self.pool_type == 'mean':
                pooled_base = torch.mean(x_base, dim=2)  # [batch, num_base_chunks, d_model]
            elif self.pool_type == 'max':
                pooled_base, _ = torch.max(x_base, dim=2)
            
            pooled_chunks.append(pooled_base)
        
        # Concatenate all pooled chunks
        output = torch.cat(pooled_chunks, dim=1)  # [batch, num_chunks, d_model]
        
        # Flatten output
        output = output.view(batch_size, -1)  # [batch, num_chunks * d_model]
        
        return output

class MultiScalePooling(nn.Module):
    """Apply different pooling window sizes to capture multi-scale patterns"""
    def __init__(self, d_model, window_sizes=[8, 32, 128, 512]):
        super(MultiScalePooling, self).__init__()
        self.window_sizes = window_sizes
        self.d_model = d_model
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape
        x_transposed = x.transpose(1, 2)  # [batch, d_model, seq_len]
        
        scale_features = []
        
        for window_size in self.window_sizes:
            # Apply average pooling with specific window size
            if window_size >= seq_len:
                # Global pooling if window larger than sequence
                pooled = F.adaptive_avg_pool1d(x_transposed, 1)
            else:
                # Strided pooling with specific window size
                pooled = F.avg_pool1d(x_transposed, 
                                    kernel_size=min(window_size, seq_len), 
                                    stride=max(1, window_size//4),  # 25% stride for overlap
                                    padding=window_size//2)
                # Then adaptive pool to fixed size for concatenation
                target_size = max(1, seq_len//window_size)
                pooled = F.adaptive_avg_pool1d(pooled, target_size)
            
            # Global pool each scale to get fixed size output
            scale_feature = pooled.mean(dim=2)  # [batch, d_model]
            scale_features.append(scale_feature)
        
        return torch.cat(scale_features, dim=1)  # [batch, len(window_sizes) * d_model]

class HierarchicalPooling(nn.Module):
    """Progressive abstraction levels from fine to coarse"""
    def __init__(self, d_model, levels=[256, 64, 16, 4]):
        super(HierarchicalPooling, self).__init__()
        self.levels = sorted(levels, reverse=True)  # Coarse to fine
        self.d_model = d_model
        
        # Learnable projections for each level
        self.level_projections = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in levels
        ])
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape
        x_transposed = x.transpose(1, 2)  # [batch, d_model, seq_len]
        
        hierarchical_features = []
        current_representation = x_transposed
        
        for i, level in enumerate(self.levels):
            # Ensure level doesn't exceed current sequence length
            actual_level = min(level, current_representation.size(2))
            
            # Pool current representation to this level
            level_repr = F.adaptive_avg_pool1d(current_representation, actual_level)
            
            # Apply level-specific projection
            level_repr = level_repr.transpose(1, 2)  # [batch, actual_level, d_model]
            level_repr = self.level_projections[i](level_repr)
            
            # Global pool this level and store
            level_feature = level_repr.mean(dim=1)  # [batch, d_model]
            hierarchical_features.append(level_feature)
            
            # Update current representation for next level
            if i < len(self.levels) - 1:
                current_representation = level_repr.transpose(1, 2)
        
        return torch.cat(hierarchical_features, dim=1)  # [batch, len(levels) * d_model]

class AttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        #self.projection = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        # Compute attention weights
        attention_weights = F.softmax(self.attention(x), dim=1)
        # Apply attention and project
        weighted = torch.sum(x * attention_weights, dim=1)
        #return self.projection(weighted)
        return weighted
        
class MultiHeadPooling(nn.Module):
    """Multiple attention-based pooling heads"""
    def __init__(self, d_model, num_heads=8, head_dim=64):
        super(MultiHeadPooling, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.d_model = d_model
        
        # Multiple attention pooling heads
        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, head_dim),
                nn.Tanh(),
                nn.Linear(head_dim, 1)
            ) for _ in range(num_heads)
        ])
        
        # Projection layers for each head
        self.projections = nn.ModuleList([
            nn.Linear(d_model, head_dim) for _ in range(num_heads)
        ])
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        head_outputs = []
        
        for i in range(self.num_heads):
            # Compute attention weights
            attention_weights = F.softmax(self.attention_heads[i](x), dim=1)
            
            # Apply attention and project
            weighted = torch.sum(x * attention_weights, dim=1)  # [batch_size, d_model]
            projected = self.projections[i](weighted)  # [batch_size, head_dim]
            head_outputs.append(projected)
            
        return torch.cat(head_outputs, dim=1)  # [batch_size, num_heads * head_dim]

class ConvolutionalPooling(nn.Module):
    """Use 1D convolutions to intelligently downsample"""
    def __init__(self, d_model, target_length=128, conv_layers=3):
        super(ConvolutionalPooling, self).__init__()
        self.target_length = target_length
        
        # Progressive downsampling layers
        layers = []
        current_dim = d_model
        
        for i in range(conv_layers):
            layers.extend([
                nn.Conv1d(current_dim, current_dim, kernel_size=5, stride=2, padding=1),
                nn.BatchNorm1d(current_dim),
                nn.GELU(),
                nn.Conv1d(current_dim, current_dim, kernel_size=1),  # Pointwise
                nn.BatchNorm1d(current_dim),
                nn.GELU(),
            ])
            
        self.conv_layers = nn.Sequential(*layers)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(target_length)
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model] -> [batch_size, d_model, seq_len]
        x = x.transpose(1, 2)
        
        # Apply convolutions
        x = self.conv_layers(x)
        
        # Final adaptive pooling
        x = self.adaptive_pool(x)
        
        # Flatten: [batch_size, d_model * target_length]
        return x.view(x.size(0), -1)

def validate_hardcoded_parameters(args):
    """Validate hardcoded multi-scale parameters"""
    if args.multi_scale_mode == "hardcoded":
        if args.hardcoded_kernels is None or args.hardcoded_strides is None:
            raise ValueError("When using hardcoded multi-scale mode, both --hardcoded_kernels and --hardcoded_strides must be provided")
        
        if len(args.hardcoded_kernels) != len(args.conv_channels):
            raise ValueError(f"Number of hardcoded kernel layers ({len(args.hardcoded_kernels)}) must match number of conv_channels ({len(args.conv_channels)})")
        
        if len(args.hardcoded_strides) != len(args.conv_channels):
            raise ValueError(f"Number of hardcoded stride layers ({len(args.hardcoded_strides)}) must match number of conv_channels ({len(args.conv_channels)})")
        
        # Check that all layers have the same number of branches
        num_branches = len(args.hardcoded_kernels[0])
        for i, layer_kernels in enumerate(args.hardcoded_kernels):
            if len(layer_kernels) != num_branches:
                raise ValueError(f"All layers must have the same number of branches. Layer {i} has {len(layer_kernels)} branches, expected {num_branches}")
        
        for i, layer_strides in enumerate(args.hardcoded_strides):
            if len(layer_strides) != num_branches:
                raise ValueError(f"All layers must have the same number of branches. Layer {i} has {len(layer_strides)} stride values, expected {num_branches}")
        
        args.multi_scale_kernels = args.hardcoded_kernels[0]
        args.multi_scale_strides = args.hardcoded_strides[0]

def validate_sequence_config(args):
    """Validate sequence modeling configuration"""
    if args.sequence_mode == "hybrid":
        if not args.layer_config:
            raise ValueError("layer_config must be specified for hybrid mode")
        
        layer_config = parse_layer_config(args.layer_config)
        if len(layer_config) != args.num_layers:
            raise ValueError(f"layer_config length ({len(layer_config)}) must match num_layers ({args.num_layers})")
        
        # Validate layer types
        valid_types = {'transformer', 'mamba'}
        for i, layer_type in enumerate(layer_config):
            if layer_type not in valid_types:
                raise ValueError(f"Invalid layer type '{layer_type}' at position {i}. Must be 'transformer' or 'mamba'")
        
        if args.hybrid_mode == "parallel":
            # Check that we have equal numbers of Mamba and Transformer layers
            mamba_count = layer_config.count('mamba')
            transformer_count = layer_config.count('transformer')
            
            if mamba_count != transformer_count:
                raise ValueError(f"Parallel hybrid mode requires equal numbers of Mamba ({mamba_count}) and Transformer ({transformer_count}) layers")
            
            if args.num_layers % 2 != 0:
                raise ValueError(f"Parallel hybrid mode requires even number of layers, got {args.num_layers}")
            
            print(f"Parallel hybrid mode validated: {mamba_count} Mamba + {transformer_count} Transformer branches")
        else:
            print(f"Sequential hybrid mode validated with configuration: {layer_config}")
    
    elif args.sequence_mode == "mamba" and not MAMBA_AVAILABLE:
        print("Warning: Mamba mode selected but mamba-ssm not available. Using custom implementation.")
    
    elif args.sequence_mode == "transformer" and args.init_from_pretrained and not TRANSFORMERS_AVAILABLE:
        print("Warning: Pretrained initialization requested but transformers library not available. Using random initialization.")
        args.init_from_pretrained = 0
    
class ParallelHybridProcessor(nn.Module):
    """Parallel processing of Mamba and Transformer branches"""
    
    def __init__(self, d_model, layer_config, transformer_heads, transformer_ff_dim,
                 mamba_d_state, mamba_d_conv, mamba_expand, dropout, use_layer_norm):
        super(ParallelHybridProcessor, self).__init__()
        
        self.d_model = d_model
        self.layer_config = layer_config
        self.num_branches = len(layer_config)
        self.use_layer_norm = use_layer_norm
        
        # Calculate channels per branch
        self.channels_per_branch = d_model // self.num_branches
        self.remaining_channels = d_model % self.num_branches
        
        print(f"  Parallel Hybrid Processor:")
        print(f"    - Total branches: {self.num_branches}")
        print(f"    - Channels per branch: {self.channels_per_branch}")
        print(f"    - Remaining channels: {self.remaining_channels}")
        
        # Create parallel branches
        self.branches = nn.ModuleList()
        self.branch_norms = nn.ModuleList() if use_layer_norm else None
        
        for i, layer_type in enumerate(layer_config):
            # Calculate actual channels for this branch (handle remainder)
            branch_channels = self.channels_per_branch
            if i < self.remaining_channels:
                branch_channels += 1
            
            if layer_type == "transformer":
                # Adjust transformer parameters for smaller dimension
                branch_heads = max(1, transformer_heads * branch_channels // d_model)
                branch_ff_dim = transformer_ff_dim * branch_channels // d_model
                
                branch = nn.TransformerEncoderLayer(
                    d_model=branch_channels,
                    nhead=branch_heads,
                    dim_feedforward=branch_ff_dim,
                    dropout=dropout,
                    activation='gelu',
                    batch_first=False
                )
                print(f"    - Branch {i}: Transformer ({branch_channels}D, {branch_heads}H, {branch_ff_dim}FF)")
                
            elif layer_type == "mamba":
                if MAMBA_AVAILABLE:
                    branch = Mamba(d_model=branch_channels, d_state=mamba_d_state, 
                                 d_conv=mamba_d_conv, expand=mamba_expand)
                else:
                    branch = CustomMambaLayer(branch_channels, mamba_d_state, 
                                            mamba_d_conv, mamba_expand, dropout)
                print(f"    - Branch {i}: Mamba ({branch_channels}D, state={mamba_d_state})")
            
            self.branches.append(branch)
            
            if use_layer_norm:
                self.branch_norms.append(nn.LayerNorm(branch_channels))
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Split input across branches
        branch_outputs = []
        start_idx = 0
        
        for i, (branch, layer_type) in enumerate(zip(self.branches, self.layer_config)):
            # Calculate channels for this branch
            branch_channels = self.channels_per_branch
            if i < self.remaining_channels:
                branch_channels += 1
            
            # Extract branch input
            end_idx = start_idx + branch_channels
            branch_input = x[:, :, start_idx:end_idx]  # [batch_size, seq_len, branch_channels]
            
            # Apply layer normalization if enabled
            if self.use_layer_norm:
                branch_input_norm = self.branch_norms[i](branch_input)
            else:
                branch_input_norm = branch_input
            
            # Process through branch
            if layer_type == "transformer":
                # Transformer expects [seq_len, batch_size, d_model]
                branch_input_t = branch_input_norm.transpose(0, 1)
                branch_output_t = branch(branch_input_t)
                branch_output = branch_output_t.transpose(0, 1)
            else:  # mamba
                # Mamba expects [batch_size, seq_len, d_model]
                branch_output = branch(branch_input_norm)
            
            # Residual connection
            branch_output = branch_output + branch_input
            
            branch_outputs.append(branch_output)
            start_idx = end_idx
        
        # Concatenate branch outputs
        output = torch.cat(branch_outputs, dim=-1)  # [batch_size, seq_len, d_model]
        
        return output

def get_input_size(genotype_file):
    if genotype_file.endswith('.npy'):
        genotype_data = np.load(genotype_file)
        return genotype_data.shape[0]  # Number of SNPs

class WarmupExponential:
    def __init__(self, optimizer, start_lr, peak_lr, final_lr, warmup_steps, t_total, decay_factor, last_epoch=-1):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.peak_lr = peak_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.decay_factor = decay_factor
        self.current_step = last_epoch + 1 if last_epoch > -1 else 0

    def get_lr(self):
        if self.current_step >= self.t_total:
            return self.final_lr
        elif self.current_step < self.warmup_steps:
            return self.start_lr * math.exp(math.log(self.peak_lr / self.start_lr) * (self.current_step / self.warmup_steps))
        else:
            progress = (self.current_step - self.warmup_steps) / (self.t_total - self.warmup_steps)
            decay = progress ** self.decay_factor
            return self.peak_lr * (self.final_lr / self.peak_lr) ** decay

    def step(self):
        if self.current_step < self.t_total:
            self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

class ExponentialDecay:
    def __init__(self, optimizer, start_lr, final_lr, total_steps, decay_factor, last_epoch=-1):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.final_lr = final_lr
        self.total_steps = total_steps
        self.decay_factor = decay_factor
        self.current_step = last_epoch + 1 if last_epoch > -1 else 0
    
    def get_lr(self):
        if self.current_step >= self.total_steps:
            return self.final_lr
        
        progress = self.current_step / self.total_steps
        decay = progress ** self.decay_factor
        return self.start_lr * (self.final_lr / self.start_lr) ** decay
    
    def step(self):
        if self.current_step < self.total_steps:
            self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
    
    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.min_delta = min_delta
        self.val_loss_min = float('inf')
        self.initial_run = True  # Flag to track if this is the first call after initialization

    def __call__(self, val_loss, global_best_loss=None):
        # Special case for first call after loading from checkpoint
        if self.initial_run and global_best_loss is not None:
            # Synchronize with global best loss on first run
            if self.best_loss is None or abs(self.best_loss - global_best_loss) > 1e-6:
                print(f'EarlyStopping: Initializing with global best loss: {global_best_loss:.6f}')
                self.best_loss = global_best_loss
            self.initial_run = False
        
        # If best_loss is still None after initialization
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.verbose:
                print(f'EarlyStopping: Initial validation loss: {val_loss:.6f}')
                
        elif val_loss > self.best_loss - self.min_delta:
            # No improvement
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience} (no improvement)')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # There was an improvement according to EarlyStopping criteria
            prev_best = self.best_loss
            self.best_loss = val_loss
            self.counter = 0
            
            if self.verbose:
                if global_best_loss is not None and val_loss > global_best_loss:
                    print(f'EarlyStopping: Reset patience counter (current: {val_loss:.6f}, previous: {prev_best:.6f})')
                else:
                    print(f'EarlyStopping: Validation loss improved from {prev_best:.6f} to {val_loss:.6f}')

    def state_dict(self):
        """Return state as a dictionary for checkpoint saving"""
        return {
            'counter': self.counter,
            'best_loss': self.best_loss,
            'early_stop': self.early_stop,
            'val_loss_min': self.val_loss_min,
            'initial_run': False  # Always set to False when saving
        }
    
    def load_state_dict(self, state_dict):
        """Load state from checkpoint"""
        self.counter = state_dict['counter']
        self.best_loss = state_dict['best_loss']
        self.early_stop = state_dict['early_stop']
        self.val_loss_min = state_dict['val_loss_min']
        self.initial_run = True  # Always reset to True when loading, to force synchronization

class CovariateNormalizer:
    def __init__(self, method="standard"):
        self.method = method
        self.scaler = None
        
        if method == "standard":
            self.scaler = StandardScaler()
        elif method == "minmax":
            self.scaler = MinMaxScaler()
        elif method == "robust":
            self.scaler = RobustScaler()
        elif method == "quantile":
            self.scaler = QuantileTransformer(output_distribution='normal')
        elif method == "power":
            self.scaler = PowerTransformer(method='yeo-johnson')

    def fit(self, data):
        if self.method != "none" and data is not None and self.scaler is not None:
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            self.scaler.fit(data)
    
    def transform(self, data):
        if self.method != "none" and data is not None and self.scaler is not None:
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            return self.scaler.transform(data)
        return data

class MultilabelGenotypeDataset(Dataset):
    def __init__(self, file_list, phenotype_data, disease_labels, use_covariates=True, use_age=True, 
                 use_gender=True, use_bmi=True, norm_age="standard", norm_pcs="standard", norm_gender="none", norm_bmi="standard",
                 fit_normalizers=True, normalizers=None):
        self.file_list = file_list
        self.phenotype_data = phenotype_data
        self.disease_labels = disease_labels
        self.use_covariates = use_covariates
        self.use_age = use_age
        self.use_gender = use_gender
        self.use_bmi = use_bmi

        # Verify that all disease label columns exist in the phenotype data
        for label in self.disease_labels:
            if label not in self.phenotype_data.columns:
                raise ValueError(f"Disease label column '{label}' not found in phenotype data. "
                               f"Available columns are: {', '.join(self.phenotype_data.columns)}")

        if normalizers is None:
            self.age_normalizer = CovariateNormalizer(norm_age)
            self.pcs_normalizer = CovariateNormalizer(norm_pcs)
            self.gender_normalizer = CovariateNormalizer(norm_gender)
            self.bmi_normalizer = CovariateNormalizer(norm_bmi)

            # Fit normalizers if this is training set
            if fit_normalizers:
                self._fit_normalizers()
        else:
            # Use the provided normalizers: for Test data
            self.age_normalizer = normalizers['age']
            self.pcs_normalizer = normalizers['pcs']
            self.gender_normalizer = normalizers['gender']
            self.bmi_normalizer = normalizers['bmi']
        
        
        # Log initialization
        print(f"\nDataset Initialization:")
        print(f"- Number of files: {len(file_list)}")
        print(f"- Disease labels: {', '.join(disease_labels)}")
        print(f"- Using PCs: {use_covariates} (normalization: {norm_pcs})")
        print(f"- Using age: {use_age} (normalization: {norm_age})")
        print(f"- Using gender: {use_gender} (normalization: {norm_gender})")
        print(f"- Using BMI: {use_bmi} (normalization: {norm_bmi})")

        # Print disease prevalence information
        self._print_disease_statistics()

        # Print sample information for the first few samples
        self._print_sample_examples(3)

    def _print_sample_examples(self, num_samples=3):
        """Print detailed information about the first few samples in the dataset"""
        print(f"\nSample Examples (first {num_samples}):")
        for i in range(min(num_samples, len(self.file_list))):
            file_path = self.file_list[i]
            sample_id = int(file_path.split('sample_')[1].split('.npy')[0])
            
            # Get labels for all diseases for this sample
            disease_status = {}
            sample_row = self.phenotype_data[self.phenotype_data['new_order'] == sample_id]
            
            if not sample_row.empty:
                for disease in self.disease_labels:
                    status = sample_row[disease].values[0]
                    disease_status[disease] = int(status)
                
                # Include demographic info if available
                demographics = []
                if 'Sex' in self.phenotype_data.columns and self.use_gender:
                    gender = sample_row['Sex'].values[0]
                    demographics.append(f"Gender: {gender}")
                if 'Agexit' in self.phenotype_data.columns and self.use_age:
                    age = sample_row['Agexit'].values[0]
                    demographics.append(f"Age: {age}")
                if 'Bmi_C' in self.phenotype_data.columns and self.use_bmi:
                    bmi = sample_row['Bmi_C'].values[0]
                    demographics.append(f"BMI: {bmi}")
                
                demo_str = ", ".join(demographics)
                if demo_str:
                    demo_str = f" ({demo_str})"
                
                print(f"Sample ID: {sample_id}{demo_str}")
                print(f"  Disease status: {disease_status}")
            else:
                print(f"Sample ID: {sample_id} - No matching phenotype data found")

    def _print_disease_statistics(self):
        """Print statistics about disease prevalence in the dataset"""
        # Get the sample IDs in this split
        sample_ids = []
        for file_path in self.file_list:

            sample_id = int(file_path.split('sample_')[1].split('.npy')[0])
            sample_ids.append(sample_id)
        
        # Filter phenotype data to only include samples in this split
        split_phenotype = self.phenotype_data[self.phenotype_data['new_order'].isin(sample_ids)]
        
        print("\nDisease Prevalence Statistics:")
        for disease in self.disease_labels:
            if disease in self.phenotype_data.columns:
                count = split_phenotype[disease].sum()
                total = len(split_phenotype)
                percentage = (count / total) * 100
                print(f"- {disease}: {count}/{total} ({percentage:.2f}%)")

    def _fit_normalizers(self):
        """Fit all normalizers on training data"""
        if self.use_covariates:
            # Get all PC data as a matrix (n_samples, n_pcs)
            pc_data = np.array([self.phenotype_data[f'PC{i}'].values for i in range(1, 11)]).T
            self.pcs_normalizer.fit(pc_data)
        
        if self.use_age:
            age_data = self.phenotype_data['Agexit'].values
            self.age_normalizer.fit(age_data)
        
        if self.use_gender:
            gender_data = self.phenotype_data['Sex'].values
            self.gender_normalizer.fit(gender_data)
        
        if self.use_bmi:
            bmi_data = self.phenotype_data['Bmi_C'].values
            self.bmi_normalizer.fit(bmi_data)
    
    def get_normalizers(self):
        """Return the fitted normalizers"""
        return {
            'age': self.age_normalizer,
            'pcs': self.pcs_normalizer,
            'gender': self.gender_normalizer,
            'bmi': self.bmi_normalizer,
        }
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        genotype_file = self.file_list[idx]

        sample_id_str = os.path.basename(genotype_file).replace("sample_", "").replace(".npy", "")
        
        sample_id = int(sample_id_str)
        
        # Get labels for all diseases
        labels = self.phenotype_data.loc[self.phenotype_data['new_order'] == sample_id, self.disease_labels].values[0]
        
        # Process covariates with normalization if needed
        covariates_list = []
        # PCs
        if self.use_covariates:
            # Get PC values as matrix (1, n_pcs)
            pc_data = np.array([
                self.phenotype_data.loc[self.phenotype_data['new_order'] == sample_id, f'PC{i}'].values[0] 
                for i in range(1, 11)]).reshape(1, -1)
            
            # Transform PCs
            normalized_pcs = self.pcs_normalizer.transform(pc_data).flatten()
            covariates_list.append(normalized_pcs)
                
        # Age
        if self.use_age:
            # Get and normalize age
            age = self.phenotype_data.loc[self.phenotype_data['new_order'] == sample_id, 'Agexit'].values[0]
            normalized_age = self.age_normalizer.transform(np.array([[age]])).flatten()
            covariates_list.append(normalized_age)
        
        # Gender
        if self.use_gender:
            # Get and normalize gender
            gender = self.phenotype_data.loc[self.phenotype_data['new_order'] == sample_id, 'Sex'].values[0]
            normalized_gender = self.gender_normalizer.transform(np.array([[gender]])).flatten()
            covariates_list.append(normalized_gender)

        # BMI
        if self.use_bmi:
            # Get and normalize BMI
            bmi = self.phenotype_data.loc[self.phenotype_data['new_order'] == sample_id, 'Bmi_C'].values[0]
            normalized_bmi = self.bmi_normalizer.transform(np.array([[bmi]])).flatten()
            covariates_list.append(normalized_bmi)
        
        # Combine all covariates
        covariates = np.concatenate(covariates_list) if covariates_list else np.array([])
        covariates_tensor = torch.tensor(covariates, dtype=torch.float32)

        if '.npy' in genotype_file:
            genotype_data = np.load(genotype_file)  # Shape: (5M, 3)
            genotype_tensor = torch.from_numpy(genotype_data).float()
        
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        return genotype_tensor, covariates_tensor, labels_tensor

# Custom Mamba implementation if mamba-ssm is not available
class CustomMambaLayer(nn.Module):
    """A simplified Mamba-like layer for genomic sequences"""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        
        d_inner = int(self.expand * d_model)
        
        # Input projection
        self.in_proj = nn.Linear(d_model, d_inner * 2)
        
        # Convolution for local dependencies
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_inner  # Depthwise convolution
        )
        
        # State space parameters (simplified)
        self.x_proj = nn.Linear(d_inner, d_state)
        self.dt_proj = nn.Linear(d_inner, d_inner)
        
        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()
        
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        
        # Input projection and split
        xz = self.in_proj(x)  # (batch, seq_len, 2 * d_inner)
        x_input, z = xz.chunk(2, dim=-1)  # Each: (batch, seq_len, d_inner)
        
        # Apply convolution (needs channel-first format)
        x_conv = self.conv1d(x_input.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        x_conv = self.act(x_conv)
        
        # Simplified state space modeling (this is a very basic approximation)
        # In real Mamba, this would involve selective state space computation
        x_state = self.x_proj(x_conv)  # Project to state dimension
        dt = F.softplus(self.dt_proj(x_conv))  # Time step
        
        # Simple recurrent-like processing (not true SSM)
        y = x_conv * torch.sigmoid(dt)
        
        # Gate with z
        y = y * self.act(z)
        
        # Output projection
        output = self.out_proj(y)
        return self.dropout(output)

# class AttentionPooling(nn.Module):
#     """Learnable attention-based pooling for sequence outputs"""
#     def __init__(self, input_dim, hidden_dim=256):
#         super(AttentionPooling, self).__init__()
#         self.attention = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.Tanh(),
#             nn.Linear(hidden_dim, 1)
#         )
#         self.softmax = nn.Softmax(dim=1)
        
#     def forward(self, x):
#         """
#         Args:
#             x: [batch_size, seq_len, input_dim]
#         Returns:
#             pooled: [batch_size, input_dim]
#         """
#         # Compute attention weights
#         attention_weights = self.attention(x)  # [batch_size, seq_len, 1]
#         attention_weights = self.softmax(attention_weights)  # [batch_size, seq_len, 1]
        
#         # Apply attention weights
#         pooled = torch.sum(x * attention_weights, dim=1)  # [batch_size, input_dim]
        
#         return pooled

class CovariateTokenEmbedder(nn.Module):
    """Convert covariates into tokens for sequence processing"""
    def __init__(self, d_model, embed_dim=64, strategy="combined", use_age=True, use_gender=True, use_bmi=True, use_pcs=True):
        super(CovariateTokenEmbedder, self).__init__()
        
        self.use_age = use_age
        self.use_gender = use_gender  
        self.use_bmi = use_bmi
        self.use_pcs = use_pcs
        self.d_model = d_model
        self.strategy = strategy

        # Calculate total covariate dimensions
        total_cov_dim = 0
        if use_pcs:
            total_cov_dim += 10  # 10 PCs
        if use_age:
            total_cov_dim += 1   # Age
        if use_gender:
            total_cov_dim += 1   # Gender  
        if use_bmi:
            total_cov_dim += 1   # BMI
            
        self.total_cov_dim = total_cov_dim
        
        if strategy == "separate":
            # Separate embedders for each covariate type
            self.embedders = nn.ModuleDict()
            
            if use_age:
                self.embedders['age'] = nn.Sequential(
                    nn.Linear(1, embed_dim),
                    nn.GELU(),
                    nn.Linear(embed_dim, d_model)
                )
                
            if use_gender:
                self.embedders['gender'] = nn.Sequential(
                    nn.Linear(1, embed_dim),
                    nn.GELU(), 
                    nn.Linear(embed_dim, d_model)
                )
                
            if use_bmi:
                self.embedders['bmi'] = nn.Sequential(
                    nn.Linear(1, embed_dim),
                    nn.GELU(),
                    nn.Linear(embed_dim, d_model)
                )
                
            if use_pcs:
                self.embedders['pcs'] = nn.Sequential(
                    nn.Linear(10, embed_dim),
                    nn.GELU(),
                    nn.Linear(embed_dim, d_model)
                )
            
            # Token type embeddings for separate tokens (optional - helps transformer distinguish token types)
            self.token_type_embeddings = nn.Embedding(5, d_model)  # genomic, age, gender, bmi, pcs
        
        elif strategy == "combined":
            # Single embedder for all covariates combined
            if total_cov_dim > 0:
                self.combined_embedder = nn.Sequential(
                    nn.Linear(total_cov_dim, embed_dim),
                    nn.GELU(),
                    nn.Linear(embed_dim, d_model)
                )

            # Single token type embedding for combined token
            self.token_type_embeddings = nn.Embedding(2, d_model)  # genomic, combined_covariates
        
        print(f"  CovariateTokenEmbedder created:")
        print(f"    - Strategy: {strategy}")
        print(f"    - Total covariate dimensions: {total_cov_dim}")
        print(f"    - Embed dim: {embed_dim} -> Model dim: {d_model}")
    
    def forward(self, covariates_dict):
        if not covariates_dict:  # Empty dictionary
            print("WARNING: NO COVARIATES USED - returning none for embeddings")
            return None, None

        batch_size = list(covariates_dict.values())[0].size(0)

        if self.strategy == "separate":
            return self._forward_separate(covariates_dict, batch_size)
        elif self.strategy == "combined":
            return self._forward_combined(covariates_dict, batch_size)
        else:
            # Fallback case
            print(f"Warning: Unknown covariate token strategy '{self.strategy}', returning None")
            return None, None

    def _forward_separate(self, covariates_dict, batch_size):
        tokens = []
        token_types = []
        token_idx = 1  # 0 reserved for genomic tokens       
        
        if self.use_age and 'age' in covariates_dict:
            age_token = self.embedders['age'](covariates_dict['age'].unsqueeze(-1))
            age_token = age_token.unsqueeze(1)
            tokens.append(age_token)
            token_types.append(torch.full((batch_size, 1), token_idx, device=age_token.device))
            token_idx += 1
            
        if self.use_gender and 'gender' in covariates_dict:
            gender_token = self.embedders['gender'](covariates_dict['gender'].unsqueeze(-1))
            gender_token = gender_token.unsqueeze(1)
            tokens.append(gender_token)
            token_types.append(torch.full((batch_size, 1), token_idx, device=gender_token.device))
            token_idx += 1
            
        if self.use_bmi and 'bmi' in covariates_dict:
            bmi_token = self.embedders['bmi'](covariates_dict['bmi'].unsqueeze(-1))
            bmi_token = bmi_token.unsqueeze(1)
            tokens.append(bmi_token)
            token_types.append(torch.full((batch_size, 1), token_idx, device=bmi_token.device))
            token_idx += 1
            
        if self.use_pcs and 'pcs' in covariates_dict:
            pcs_token = self.embedders['pcs'](covariates_dict['pcs'])
            pcs_token = pcs_token.unsqueeze(1)
            tokens.append(pcs_token)
            token_types.append(torch.full((batch_size, 1), token_idx, device=pcs_token.device))
            token_idx += 1
        
        if tokens:
            covariate_tokens = torch.cat(tokens, dim=1)
            token_type_ids = torch.cat(token_types, dim=1)
            # Add token type embeddings
            type_embeddings = self.token_type_embeddings(token_type_ids)
            covariate_tokens = covariate_tokens + type_embeddings

            return covariate_tokens, token_type_ids
        else:
            return None, None
    
    def _forward_combined(self, covariates_dict, batch_size):
        """Create a single combined token for all covariates"""
        # Collect all covariates in a consistent order
        cov_features = []
        
        # Always follow the same order: PCs, Age, Gender, BMI
        if self.use_pcs and 'pcs' in covariates_dict:
            cov_features.append(covariates_dict['pcs'])  # [batch_size, 10]
        
        if self.use_age and 'age' in covariates_dict:
            cov_features.append(covariates_dict['age'].unsqueeze(-1))  # [batch_size, 1]
            
        if self.use_gender and 'gender' in covariates_dict:
            cov_features.append(covariates_dict['gender'].unsqueeze(-1))  # [batch_size, 1]
            
        if self.use_bmi and 'bmi' in covariates_dict:
            cov_features.append(covariates_dict['bmi'].unsqueeze(-1))  # [batch_size, 1]
        
        if cov_features:
            # Concatenate all covariates
            combined_covariates = torch.cat(cov_features, dim=1)  # [batch_size, total_cov_dim]
            
            # Create single combined token
            combined_token = self.combined_embedder(combined_covariates)
            combined_token = combined_token.unsqueeze(1)

            # Token type for combined covariate token
            token_type_ids = torch.full((batch_size, 1), 1, device=combined_token.device)  # 1 for combined covariates
            
            # Add token type embedding
            type_embeddings = self.token_type_embeddings(token_type_ids)
            combined_token = combined_token + type_embeddings

            return combined_token, token_type_ids
        else:
            return None, None

# Disease-specific components
class DiseaseSpecificAttention(nn.Module):
    """Disease-specific attention mechanism"""
    def __init__(self, feature_dim, num_diseases, num_heads=8):
        super(DiseaseSpecificAttention, self).__init__()
        self.num_diseases = num_diseases
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.head_dim = feature_dim // num_heads
        
        # Disease-specific query generators
        self.disease_queries = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim) for _ in range(num_diseases)
        ])
        
        # Shared key and value projections
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)

        # Output projection
        self.output_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, feature_dim)
        batch_size, seq_len, feature_dim = x.size()
        
        # Generate keys and values (shared across diseases)
        keys = self.key_proj(x)  # (batch_size, seq_len, feature_dim)
        values = self.value_proj(x)  # (batch_size, seq_len, feature_dim)
        
        # Reshape for multi-head attention
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        disease_outputs = []
        
        for disease_idx in range(self.num_diseases):
            # Generate disease-specific queries
            # Use global average pooling to create a single query vector per disease
            global_context = torch.mean(x, dim=1)  # (batch_size, feature_dim)
            queries = self.disease_queries[disease_idx](global_context)  # (batch_size, feature_dim)
            queries = queries.unsqueeze(1)  # (batch_size, 1, feature_dim)
            
            # Reshape queries for multi-head attention
            queries = queries.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
            # queries shape: (batch_size, num_heads, 1, head_dim)
            
            # Compute attention scores
            scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # scores shape: (batch_size, num_heads, 1, seq_len)
            
            # Apply softmax
            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            # Apply attention to values
            attended = torch.matmul(attention_weights, values)
            # attended shape: (batch_size, num_heads, 1, head_dim)
            
            # Reshape and project
            attended = attended.transpose(1, 2).contiguous().view(batch_size, 1, feature_dim)
            attended = self.output_proj(attended)
            
            # Squeeze to get (batch_size, feature_dim)
            disease_outputs.append(attended.squeeze(1))
        
        # Stack disease-specific features
        return torch.stack(disease_outputs, dim=1)  # (batch_size, num_diseases, feature_dim)

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer"""
    def __init__(self, d_model, max_len=10000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register as buffer (not a parameter, but part of the model state)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MultiScaleConvBlock(nn.Module):
    """Multi-scale convolution block with different kernel sizes and strides"""
    def __init__(self, in_channels, out_channels, kernel_sizes, strides, act, dropout_rate, pool_type="max", 
                use_pointwise_conv=False, pointwise_channels=16, is_final_layer=False):
        super(MultiScaleConvBlock, self).__init__()
        
        self.branches = nn.ModuleList()
        self.pool_type = pool_type
        self.num_branches = len(kernel_sizes)
        self.use_pointwise_conv = use_pointwise_conv
        self.pointwise_channels = pointwise_channels
        self.is_final_layer = is_final_layer
        
        # Create multiple parallel convolution branches with different strides
        for i, (kernel_size, stride) in enumerate(zip(kernel_sizes, strides)):
            # Each branch gets the full out_channels
            branch_channels = out_channels
            
            # Ensure proper padding to maintain reasonable output sizes
            padding = kernel_size // 2
            
            # Main convolution layers (no pointwise here)
            branch = nn.Sequential(
                nn.Conv1d(in_channels, branch_channels, 
                         kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm1d(branch_channels),
                self.get_activation(act)
            )
            self.branches.append(branch)
        
        # Add pointwise convolution layers for final layer only
        if self.use_pointwise_conv and self.is_final_layer:
            self.pointwise_branches = nn.ModuleList()
            for i in range(self.num_branches):
                pointwise_conv = nn.Sequential(
                    nn.Conv1d(out_channels, pointwise_channels, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm1d(pointwise_channels),
                    self.get_activation(act)
                )
                self.pointwise_branches.append(pointwise_conv)
        
        # Initialize empty ModuleList for pooling layers (will be populated dynamically)
        self.branch_pools = nn.ModuleList()
        self.pooling_needed = [False] * self.num_branches  # Track which branches need pooling
        self._pools_initialized = False
    
    def get_activation(self, name):
        activations = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'gelu': nn.GELU()
        }
        return activations.get(name, nn.ReLU())
    
    def _initialize_pooling_layers(self, target_length, device):
        """Initialize pooling layers only for branches that need them"""
        if self._pools_initialized:
            return
            
        # Clear any existing pooling layers
        self.branch_pools = nn.ModuleList()
        
        # Add pooling layers only where needed
        for i in range(self.num_branches):
            if self.pooling_needed[i]:
                if self.pool_type.lower() == "max":
                    pool_layer = nn.AdaptiveMaxPool1d(target_length).to(device)
                elif self.pool_type.lower() == "avg":
                    pool_layer = nn.AdaptiveAvgPool1d(target_length).to(device)
                else:
                    pool_layer = nn.AdaptiveMaxPool1d(target_length).to(device)
                self.branch_pools.append(pool_layer)
            else:
                # Add Identity layer as placeholder (won't show pooling in architecture)
                self.branch_pools.append(nn.Identity())
        
        self._pools_initialized = True
    
    def forward(self, x):
        branch_outputs = []
        output_lengths = []
        
        # Process all branches through main convolutions first
        for i, branch in enumerate(self.branches):
            output = branch(x)
            branch_outputs.append(output)
            output_lengths.append(output.size(2))
        
        # Determine target length (use the smallest to avoid upsampling)
        target_length = min(output_lengths)
        
        # Determine which branches need pooling
        for i, length in enumerate(output_lengths):
            self.pooling_needed[i] = (length != target_length)
        
        # Initialize pooling layers if not done already
        if not self._pools_initialized:
            self._initialize_pooling_layers(target_length, x.device)
        
        # Apply pooling only where needed
        standardized_outputs = []
        for i, output in enumerate(branch_outputs):
            if self.pooling_needed[i]:
                # Update pooling layer size if target length changed
                if hasattr(self.branch_pools[i], 'output_size'):
                    if self.branch_pools[i].output_size != target_length:
                        if self.pool_type.lower() == "max":
                            self.branch_pools[i] = nn.AdaptiveMaxPool1d(target_length).to(output.device)
                        else:
                            self.branch_pools[i] = nn.AdaptiveAvgPool1d(target_length).to(output.device)
                output = self.branch_pools[i](output)
            standardized_outputs.append(output)
        
        # Apply pointwise convolution if this is the final layer and pointwise is enabled
        if self.use_pointwise_conv and self.is_final_layer:
            pointwise_outputs = []
            for i, output in enumerate(standardized_outputs):
                pointwise_output = self.pointwise_branches[i](output)
                pointwise_outputs.append(pointwise_output)
            standardized_outputs = pointwise_outputs
        
        # Concatenate along channel dimension
        result = torch.cat(standardized_outputs, dim=1)

        return result


class SeparateDiseaseHead(nn.Module):
    """Separate prediction head for each disease"""
    def __init__(self, input_dim, hidden_dims, dropout_rate, act):
        super(SeparateDiseaseHead, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self.get_activation(act),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(prev_dim, 1))
        self.head = nn.Sequential(*layers)
    
    def get_activation(self, name):
        activations = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'gelu': nn.GELU()
        }
        return activations.get(name, nn.ReLU())
    
    def forward(self, x):
        return self.head(x)

# Main unified sequence modeling block
class UnifiedSequenceBlock(nn.Module):
    """Unified block that can handle transformer, mamba, or hybrid sequence modeling"""
    
    def __init__(self, input_dim, d_model, sequence_mode="transformer", layer_config=None,
                 num_layers=4, dropout=0.1, use_layer_norm=True,
                 hybrid_mode="sequential",
                 # Transformer parameters
                 transformer_heads=8, transformer_ff_dim=1024, use_positional_encoding=True, max_seq_len=10000,
                 init_from_pretrained=False, pretrained_model_name="bert-base-uncased", pretrained_model_type="auto",
                 init_layers_fraction=1.0, layer_init_strategy="middle", custom_layer_indices="",
                 # Mamba parameters
                 mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
                 # Token and pooling parameters
                 use_cls_token=False, use_covariate_tokens=True, covariate_embed_dim=64, 
                 covariate_token_strategy="combined", pooling_strategy="mean",
                 use_age=True, use_gender=True, use_bmi=True, use_pcs=True,
                 # Intermediate pooling parameters
                 use_intermediate_pooling=False,
                 intermediate_pooling_strategy="mean",
                 intermediate_pooling_layers=None,
                 intermediate_pooling_kwargs=None,
                 # Advanced pooling parameters
                 **pooling_kwargs):
        
        super(UnifiedSequenceBlock, self).__init__()

        self.hybrid_mode = hybrid_mode
        self.input_dim = input_dim
        self.d_model = d_model
        self.sequence_mode = sequence_mode
        self.num_layers = num_layers
        self.use_layer_norm = use_layer_norm
        self.use_cls_token = use_cls_token
        self.use_covariate_tokens = use_covariate_tokens
        self.pooling_strategy = pooling_strategy
        
        # Intermediate pooling configuration
        self.use_intermediate_pooling = use_intermediate_pooling
        self.intermediate_pooling_strategy = intermediate_pooling_strategy
        self.intermediate_pooling_layers = intermediate_pooling_layers or []
        self.intermediate_pooling_kwargs = intermediate_pooling_kwargs or {}
        
        if pooling_strategy == "chunked":
            self.custom_pooling = ChunkedPooling(**pooling_kwargs)
        elif pooling_strategy == "multiscale":
            self.custom_pooling = MultiScalePooling(d_model=d_model, **pooling_kwargs)
        elif pooling_strategy == "multihead":
            self.custom_pooling = MultiHeadPooling(d_model=d_model, **pooling_kwargs)
        elif pooling_strategy == "conv":
            self.custom_pooling = ConvolutionalPooling(d_model=d_model, **pooling_kwargs)
        elif pooling_strategy == "hierarchical":
            self.custom_pooling = HierarchicalPooling(d_model=d_model, **pooling_kwargs)
        elif pooling_strategy == "attention":
            self.attention_pooling = AttentionPooling(d_model)
        
        # Calculate final output dimension
        self.final_output_dim = calculate_pooling_output_dim(pooling_strategy, d_model, pooling_kwargs)
        if self.final_output_dim is None:  # concat case
            self.final_output_dim = d_model  # Will be calculated dynamically        
        
        # Layer normalization
        if use_layer_norm:
            self.final_norm = nn.LayerNorm(d_model)
        
        # Create intermediate pooling layers if needed
        self.intermediate_poolers = nn.ModuleDict()
        if use_intermediate_pooling and intermediate_pooling_layers:
            print(f"    - Creating intermediate pooling after layers: {intermediate_pooling_layers}")
            
            # Validate and warn about pooling strategy choices
            aggressive_strategies = ["attention", "multiscale", "multihead", "conv", "hierarchical"]
            moderate_strategies = ["mean", "max", "chunked"]
            
            if intermediate_pooling_strategy in aggressive_strategies:
                print(f"    ⚠ WARNING: '{intermediate_pooling_strategy}' pooling reduces sequence to very few tokens (often just 1).")
                print(f"              This may be too aggressive for intermediate pooling between layers.")
                print(f"              Consider using 'mean', 'max', or 'chunked' for better sequence preservation.")

            for layer_idx in intermediate_pooling_layers:
                if intermediate_pooling_strategy in ["mean", "max"]:
                    # Simple pooling doesn't need a module
                    self.intermediate_poolers[str(layer_idx)] = None
                    pool_factor = intermediate_pooling_kwargs.get('pool_factor', 2)
                    print(f"      - Layer {layer_idx}: {intermediate_pooling_strategy} pooling (factor={pool_factor})")

                elif intermediate_pooling_strategy == "attention":
                    self.intermediate_poolers[str(layer_idx)] = AttentionPooling(d_model)
                    print(f"      - Layer {layer_idx}: Attention pooling (output: 1 token)")
                    
                elif intermediate_pooling_strategy == "chunked":
                    num_chunks = intermediate_pooling_kwargs.get('num_chunks', 10)
                    pool_type = intermediate_pooling_kwargs.get('pool_type', 'max')
                    self.intermediate_poolers[str(layer_idx)] = ChunkedPooling(
                        num_chunks=num_chunks,
                        pool_type=pool_type
                    )
                    print(f"      - Layer {layer_idx}: Chunked pooling (num_chunks={num_chunks}, type={pool_type})")
                
                elif intermediate_pooling_strategy == "conv":
                    target_length = intermediate_pooling_kwargs.get('target_length', 1024)
                    self.intermediate_poolers[str(layer_idx)] = ConvolutionalPooling(
                        d_model=d_model,
                        target_length=target_length
                    )
                    print(f"      - Layer {layer_idx}: Convolutional pooling (target_length={target_length})")

                elif intermediate_pooling_strategy == "multiscale":
                    self.intermediate_poolers[str(layer_idx)] = MultiScalePooling(d_model=d_model, **self.intermediate_pooling_kwargs)
                    print(f"      - Layer {layer_idx}: Multi-scale pooling (output: 1 token)")
                    
                elif intermediate_pooling_strategy == "multihead":
                    self.intermediate_poolers[str(layer_idx)] = MultiHeadPooling(d_model=d_model, **self.intermediate_pooling_kwargs)
                    print(f"      - Layer {layer_idx}: Multi-head pooling (output: 1 token)")
                    
                elif intermediate_pooling_strategy == "hierarchical":
                    self.intermediate_poolers[str(layer_idx)] = HierarchicalPooling(d_model=d_model, **self.intermediate_pooling_kwargs)
                    print(f"      - Layer {layer_idx}: Hierarchical pooling (output: 1 token)")
        
        print(f"  Creating Unified Sequence Block:")
        print(f"    - Mode: {sequence_mode}")
        print(f"    - Number of layers: {num_layers}")
        print(f"    - Model dimension: {d_model}")
        print(f"    - Class token: {'Enabled' if use_cls_token else 'Disabled'}")
        print(f"    - Covariate tokens: {'Enabled' if use_covariate_tokens else 'Disabled'}")
        print(f"    - Pooling strategy: {pooling_strategy}")
        if use_intermediate_pooling and intermediate_pooling_layers:
            print(f"    - Intermediate pooling: {intermediate_pooling_strategy} after layers {intermediate_pooling_layers}")
        
        # Input projection
        self.input_projection = None
        if input_dim != d_model:
            self.input_projection = nn.Linear(input_dim, d_model)
            print(f"    - Input projection: {input_dim} → {d_model}")
        
        # Class token (for transformer mode)
        if use_cls_token and sequence_mode in ["transformer", "hybrid"]:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
            print(f"    - Class token initialized")
        
        # Positional encoding (for transformer layers)
        if use_positional_encoding and sequence_mode in ["transformer", "hybrid"]:
            self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
            print(f"    - Positional encoding enabled")
        
        # Covariate token embedder
        if use_covariate_tokens:
            self.covariate_embedder = CovariateTokenEmbedder(
                d_model, covariate_embed_dim, covariate_token_strategy, use_age, use_gender, use_bmi, use_pcs
            )
        
        # Sequence layers
        if sequence_mode == "transformer":
            self._create_transformer_layers(transformer_heads, transformer_ff_dim, dropout)
            if init_from_pretrained:
                self._initialize_from_pretrained(pretrained_model_name, pretrained_model_type, 
                                                init_layers_fraction, layer_init_strategy, custom_layer_indices)
        
        elif sequence_mode == "mamba":
            self._create_mamba_layers(mamba_d_state, mamba_d_conv, mamba_expand, dropout)
        
        elif sequence_mode == "hybrid":
            if layer_config is None:
                layer_config = parse_layer_config("T,M,T,M")  # Default hybrid config
            self.layer_config = layer_config
            print(f"    - Hybrid configuration: {layer_config}")
            print(f"    - Hybrid mode: {hybrid_mode}")
            
            if hybrid_mode == "sequential":
                self._create_hybrid_layers(layer_config, transformer_heads, transformer_ff_dim,
                                         mamba_d_state, mamba_d_conv, mamba_expand, dropout)
            elif hybrid_mode == "parallel":
                self._create_parallel_hybrid_layers(layer_config, transformer_heads, transformer_ff_dim,
                                                  mamba_d_state, mamba_d_conv, mamba_expand, dropout)
        
        print(f"    - Final output dimension: {self.final_output_dim}")
    
    def _create_transformer_layers(self, num_heads, ff_dim, dropout):
        """Create transformer encoder layers"""
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        print(f"    - Transformer layers: {self.num_layers} layers, {num_heads} heads")
    
    def _create_mamba_layers(self, d_state, d_conv, expand, dropout):
        """Create Mamba layers"""
        self.mamba_layers = nn.ModuleList()
        
        for i in range(self.num_layers):
            if MAMBA_AVAILABLE:
                mamba_layer = Mamba(d_model=self.d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            else:
                mamba_layer = CustomMambaLayer(self.d_model, d_state, d_conv, expand, dropout)
            self.mamba_layers.append(mamba_layer)
        
        if self.use_layer_norm:
            self.layer_norms = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(self.num_layers)])
        
        print(f"    - Mamba layers: {self.num_layers} layers, d_state={d_state}, d_conv={d_conv}")
    
    def _create_hybrid_layers(self, layer_config, transformer_heads, transformer_ff_dim,
                            mamba_d_state, mamba_d_conv, mamba_expand, dropout):
        """Create hybrid sequence with interleaved transformer and mamba layers"""
        self.hybrid_layers = nn.ModuleList()
        self.hybrid_layer_norms = nn.ModuleList() if self.use_layer_norm else None
        
        for i, layer_type in enumerate(layer_config):
            if layer_type == "transformer":
                layer = nn.TransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=transformer_heads,
                    dim_feedforward=transformer_ff_dim,
                    dropout=dropout,
                    activation='gelu',
                    batch_first=False
                )
            elif layer_type == "mamba":
                if MAMBA_AVAILABLE:
                    layer = Mamba(d_model=self.d_model, d_state=mamba_d_state, d_conv=mamba_d_conv, expand=mamba_expand)
                else:
                    layer = CustomMambaLayer(self.d_model, mamba_d_state, mamba_d_conv, mamba_expand, dropout)
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
            
            self.hybrid_layers.append(layer)
            
            if self.use_layer_norm:
                self.hybrid_layer_norms.append(nn.LayerNorm(self.d_model))
        
        print(f"    - Hybrid layers created with configuration: {layer_config}")
    
    def _create_parallel_hybrid_layers(self, layer_config, transformer_heads, transformer_ff_dim,
                                     mamba_d_state, mamba_d_conv, mamba_expand, dropout):
        """Create parallel hybrid processing"""
        self.parallel_processor = ParallelHybridProcessor(
            d_model=self.d_model,
            layer_config=layer_config,
            transformer_heads=transformer_heads,
            transformer_ff_dim=transformer_ff_dim,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            dropout=dropout,
            use_layer_norm=self.use_layer_norm
        )
        
        print(f"    - Parallel hybrid processor created")

    def _initialize_from_pretrained(self, model_name, model_type, init_fraction, init_strategy, custom_indices):
        """Initialize transformer weights from pretrained models"""
        if not TRANSFORMERS_AVAILABLE:
            print("    - Warning: transformers library not available, skipping pretrained initialization")
            return
        
        print(f"    - Initializing from pretrained model: {model_name}")
        try:
            if model_type == "vit":
                self._initialize_from_vit()
            elif model_type == "bert":
                self._initialize_from_bert()
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
        except Exception as e:
            print(f"  - Warning: Could not initialize from pretrained model: {e}")
            print(f"  - Continuing with random initialization")

    def _initialize_from_vit(self):
        """Initialize from Vision Transformer"""
        from transformers import ViTModel
        
        # Load pretrained ViT model
        pretrained_vit = ViTModel.from_pretrained(self.pretrained_model_name)
        pretrained_layers = pretrained_vit.encoder.layer
        
        print(f"  - Loaded pretrained ViT model with {len(pretrained_layers)} layers")
        print(f"  - Pretrained hidden size: {pretrained_vit.config.hidden_size}")
        print(f"  - Pretrained num heads: {pretrained_vit.config.num_attention_heads}")
        print(f"  - Pretrained intermediate size: {pretrained_vit.config.intermediate_size}")
        
        # Calculate how many layers to initialize
        num_layers_to_init = int(self.num_layers * self.init_layers_fraction)
        print(f"  - Initializing {num_layers_to_init} out of {self.num_layers} custom layers")
        
        # Determine which pretrained layers to use
        pretrained_layer_indices = self._select_pretrained_layers(
            len(pretrained_layers), 
            num_layers_to_init
        )
        
        print(f"  - Using pretrained layers: {pretrained_layer_indices}")
        print(f"  - Layer selection strategy: {self.layer_init_strategy}")
        
        self._copy_vit_weights(pretrained_layers, pretrained_layer_indices)
        
        print(f"  - ViT weight initialization completed successfully")
    
    def _initialize_from_bert(self):
        """Initialize from BERT model"""
        from transformers import BertModel
        
        # Load pretrained BERT model
        pretrained_bert = BertModel.from_pretrained(self.pretrained_model_name)
        pretrained_layers = pretrained_bert.encoder.layer
        
        print(f"  - Loaded pretrained BERT model with {len(pretrained_layers)} layers")
        print(f"  - Pretrained hidden size: {pretrained_bert.config.hidden_size}")
        print(f"  - Pretrained num heads: {pretrained_bert.config.num_attention_heads}")
        print(f"  - Pretrained intermediate size: {pretrained_bert.config.intermediate_size}")
        
        # Calculate how many layers to initialize
        num_layers_to_init = int(self.num_layers * self.init_layers_fraction)
        print(f"  - Initializing {num_layers_to_init} out of {self.num_layers} custom layers")
        
        # Determine which pretrained layers to use
        pretrained_layer_indices = self._select_pretrained_layers(
            len(pretrained_layers), 
            num_layers_to_init
        )
        
        print(f"  - Using pretrained layers: {pretrained_layer_indices}")
        print(f"  - Layer selection strategy: {self.layer_init_strategy}")
        
        self._copy_bert_weights(pretrained_layers, pretrained_layer_indices)
        
        print(f"  - BERT weight initialization completed successfully")
    
    def _select_pretrained_layers(self, total_pretrained_layers, num_layers_needed):
        """Select which pretrained layers to use based on strategy"""
        
        if self.layer_init_strategy == "first":
            # Use first N layers: [0, 1, 2, ...]
            selected = list(range(min(num_layers_needed, total_pretrained_layers)))
            print(f"    Selected first layers: {selected}")
            return selected
        
        elif self.layer_init_strategy == "middle":
            # Use middle layers
            if total_pretrained_layers < num_layers_needed:
                return list(range(total_pretrained_layers))
            
            # Calculate middle range
            start_idx = (total_pretrained_layers - num_layers_needed) // 2
            end_idx = start_idx + num_layers_needed
            selected = list(range(start_idx, end_idx))
            
            print(f"    Selected middle layers from index {start_idx} to {end_idx-1}: {selected}")
            return selected
        
        elif self.layer_init_strategy == "last":
            # Use last N layers: [..., -3, -2, -1]
            start_idx = max(0, total_pretrained_layers - num_layers_needed)
            selected = list(range(start_idx, total_pretrained_layers))
            
            print(f"    Selected last layers from index {start_idx} to {total_pretrained_layers-1}: {selected}")
            return selected
        
        elif self.layer_init_strategy == "random":
            # Randomly select N layers
            import random
            available_indices = list(range(total_pretrained_layers))
            selected = sorted(random.sample(available_indices, min(num_layers_needed, total_pretrained_layers)))
            
            print(f"    Randomly selected layers: {selected}")
            return selected
        
        elif self.layer_init_strategy == "custom":
            # Use user-specified layers
            if not self.custom_layer_indices:
                print(f"    Warning: custom strategy selected but no indices provided, falling back to first layers")
                return list(range(min(num_layers_needed, total_pretrained_layers)))
            
            try:
                custom_indices = [int(x.strip()) for x in self.custom_layer_indices.split(',')]
                # Validate indices
                valid_indices = [idx for idx in custom_indices if 0 <= idx < total_pretrained_layers]
                
                if len(valid_indices) != len(custom_indices):
                    print(f"    Warning: Some indices were invalid. Using valid indices: {valid_indices}")
                
                # Take only what we need
                selected = valid_indices[:num_layers_needed]
                print(f"    Using custom layer indices: {selected}")
                return selected
                
            except ValueError as e:
                print(f"    Error parsing custom indices: {e}, falling back to first layers")
                return list(range(min(num_layers_needed, total_pretrained_layers)))
        
        else:
            # Fallback to first layers
            print(f"    Unknown strategy '{self.layer_init_strategy}', using first layers")
            return list(range(min(num_layers_needed, total_pretrained_layers)))
    
    def _copy_vit_weights(self, pretrained_layers, pretrained_layer_indices):
        """Copy weights from ViT layers"""
        for i, pretrained_idx in enumerate(pretrained_layer_indices):
            if i < len(self.transformer_encoder.layers):
                print(f"    Initializing custom layer {i} from ViT layer {pretrained_idx}")
                self._copy_vit_layer_weights(
                    pretrained_layers[pretrained_idx], 
                    self.transformer_encoder.layers[i],
                    layer_idx=i
                )
    
    def _copy_bert_weights(self, pretrained_layers, pretrained_layer_indices):
        """Copy weights from BERT layers"""
        for i, pretrained_idx in enumerate(pretrained_layer_indices):
            if i < len(self.transformer_encoder.layers):
                print(f"    Initializing custom layer {i} from BERT layer {pretrained_idx}")
                self._copy_bert_layer_weights(
                    pretrained_layers[pretrained_idx], 
                    self.transformer_encoder.layers[i],
                    layer_idx=i
                )

    def _adapt_weight_tensor(self, pretrained_weight, target_shape, weight_name):
        """Intelligently adapt pretrained weights to target shape"""
        pretrained_shape = pretrained_weight.shape
        
        if pretrained_shape == target_shape:
            return pretrained_weight.clone()
        
        #print(f"       Adapting {weight_name}: {pretrained_shape} → {target_shape}")
        
        # For 2D weights (Linear layer weights)
        if len(target_shape) == 2 and len(pretrained_shape) == 2:
            target_out, target_in = target_shape
            pretrained_out, pretrained_in = pretrained_shape
            
            # Create new tensor with target shape
            adapted_weight = torch.zeros(target_shape, dtype=pretrained_weight.dtype, device=pretrained_weight.device)
            
            # Copy overlapping region
            copy_out = min(target_out, pretrained_out)
            copy_in = min(target_in, pretrained_in)
            
            adapted_weight[:copy_out, :copy_in] = pretrained_weight[:copy_out, :copy_in]
            
            # Initialize remaining weights
            if target_out > pretrained_out or target_in > pretrained_in:
                # Use Xavier/Glorot initialization for new weights
                with torch.no_grad():
                    fan_in, fan_out = target_in, target_out
                    std = math.sqrt(2.0 / (fan_in + fan_out))
                    
                    # Initialize new rows (if target_out > pretrained_out)
                    if target_out > pretrained_out:
                        adapted_weight[copy_out:, :copy_in].normal_(0, std)
                    
                    # Initialize new columns (if target_in > pretrained_in)  
                    if target_in > pretrained_in:
                        adapted_weight[:copy_out, copy_in:].normal_(0, std)
                        
                    # Initialize new intersection (if both dimensions expanded)
                    if target_out > pretrained_out and target_in > pretrained_in:
                        adapted_weight[copy_out:, copy_in:].normal_(0, std)
            
            # adaptation_info = f"copied {copy_out}×{copy_in}, initialized {target_out-copy_out}×{target_in-copy_in}"
            # print(f"        ✓ {adaptation_info}")
            return adapted_weight
            
        # For 1D weights (bias vectors, layer norm weights)
        elif len(target_shape) == 1 and len(pretrained_shape) == 1:
            target_dim = target_shape[0]
            pretrained_dim = pretrained_shape[0]
            
            adapted_weight = torch.zeros(target_shape, dtype=pretrained_weight.dtype, device=pretrained_weight.device)
            
            copy_dim = min(target_dim, pretrained_dim)
            adapted_weight[:copy_dim] = pretrained_weight[:copy_dim]
            
            # Initialize remaining elements (bias usually to 0, others to small random)
            if target_dim > pretrained_dim:
                if 'bias' in weight_name.lower():
                    adapted_weight[copy_dim:] = 0.0
                else:
                    adapted_weight[copy_dim:].normal_(0, 0.02)
            
            # adaptation_info = f"copied {copy_dim}, initialized {target_dim-copy_dim}"
            # print(f"        ✓ {adaptation_info}")
            return adapted_weight
        
        # For unsupported shapes, fall back to Xavier initialization
        else:
            print(f"         Unsupported shape adaptation, using Xavier initialization")
            adapted_weight = torch.zeros(target_shape, dtype=pretrained_weight.dtype, device=pretrained_weight.device)
            torch.nn.init.xavier_uniform_(adapted_weight)
            return adapted_weight
    
    def _adapt_bias_tensor(self, pretrained_bias, target_shape, bias_name):
        """Adapt bias tensors with appropriate initialization"""
        if pretrained_bias.shape == target_shape:
            return pretrained_bias.clone()
            
        #print(f"      Adapting {bias_name}: {pretrained_bias.shape} → {target_shape}")

        target_dim = target_shape[0]
        pretrained_dim = pretrained_bias.shape[0]
        
        adapted_bias = torch.zeros(target_shape, dtype=pretrained_bias.dtype, device=pretrained_bias.device)
        
        copy_dim = min(target_dim, pretrained_dim)
        adapted_bias[:copy_dim] = pretrained_bias[:copy_dim]
        
        # Bias initialization for new elements is usually 0
        if target_dim > pretrained_dim:
            adapted_bias[copy_dim:] = 0.0

        #print(f"        ✓ copied {copy_dim} bias elements, initialized {target_dim-copy_dim} to zero")
        return adapted_bias
    
    def _copy_vit_layer_weights(self, pretrained_layer, custom_layer, layer_idx):
        """Copy weights from pretrained ViT layer to custom layer"""
        print(f"    Initializing layer {layer_idx} from ViT")
        
        try:
            # Get pretrained attention weights (ViT structure)
            pretrained_attn = pretrained_layer.attention.attention
            custom_attn = custom_layer.self_attn
            
            # Adapt and copy Q, K, V weights
            embed_dim = custom_attn.embed_dim
            
            with torch.no_grad():
                # Adapt Q, K, V weights
                q_adapted = self._adapt_weight_tensor(pretrained_attn.query.weight, (embed_dim, embed_dim), "Query weight")
                k_adapted = self._adapt_weight_tensor(pretrained_attn.key.weight, (embed_dim, embed_dim), "Key weight")
                v_adapted = self._adapt_weight_tensor(pretrained_attn.value.weight, (embed_dim, embed_dim), "Value weight")
                
                # Copy adapted weights to the combined in_proj_weight tensor
                custom_attn.in_proj_weight[:embed_dim].copy_(q_adapted)
                custom_attn.in_proj_weight[embed_dim:2*embed_dim].copy_(k_adapted)
                custom_attn.in_proj_weight[2*embed_dim:].copy_(v_adapted)
                
                # Adapt and copy Q, K, V biases
                q_bias_adapted = self._adapt_bias_tensor(pretrained_attn.query.bias, (embed_dim,), "Query bias")
                k_bias_adapted = self._adapt_bias_tensor(pretrained_attn.key.bias, (embed_dim,), "Key bias")
                v_bias_adapted = self._adapt_bias_tensor(pretrained_attn.value.bias, (embed_dim,), "Value bias")
                
                custom_attn.in_proj_bias[:embed_dim].copy_(q_bias_adapted)
                custom_attn.in_proj_bias[embed_dim:2*embed_dim].copy_(k_bias_adapted)
                custom_attn.in_proj_bias[2*embed_dim:].copy_(v_bias_adapted)
                
                # Adapt and copy output projection (ViT structure)
                out_weight_adapted = self._adapt_weight_tensor(
                    pretrained_layer.attention.output.dense.weight, 
                    custom_attn.out_proj.weight.shape, 
                    "Output projection weight"
                )
                out_bias_adapted = self._adapt_bias_tensor(
                    pretrained_layer.attention.output.dense.bias,
                    custom_attn.out_proj.bias.shape,
                    "Output projection bias"
                )
                
                custom_attn.out_proj.weight.copy_(out_weight_adapted)
                custom_attn.out_proj.bias.copy_(out_bias_adapted)
            
            #print(f"      Successfully adapted and copied attention weights")

            # Adapt and copy layer norm weights (ViT structure)
            with torch.no_grad():
                # First layer norm (before attention)
                ln1_weight_adapted = self._adapt_weight_tensor(
                    pretrained_layer.layernorm_before.weight,
                    custom_layer.norm1.weight.shape,
                    "LayerNorm1 weight"
                )
                ln1_bias_adapted = self._adapt_bias_tensor(
                    pretrained_layer.layernorm_before.bias,
                    custom_layer.norm1.bias.shape,
                    "LayerNorm1 bias"
                )
                
                custom_layer.norm1.weight.copy_(ln1_weight_adapted)
                custom_layer.norm1.bias.copy_(ln1_bias_adapted)
                
                # Second layer norm (after attention)
                ln2_weight_adapted = self._adapt_weight_tensor(
                    pretrained_layer.layernorm_after.weight,
                    custom_layer.norm2.weight.shape,
                    "LayerNorm2 weight"
                )
                ln2_bias_adapted = self._adapt_bias_tensor(
                    pretrained_layer.layernorm_after.bias,
                    custom_layer.norm2.bias.shape,
                    "LayerNorm2 bias"
                )
                
                custom_layer.norm2.weight.copy_(ln2_weight_adapted)
                custom_layer.norm2.bias.copy_(ln2_bias_adapted)
            
            # Copy feedforward weights (same structure as BERT)
            self._copy_feedforward_weights(pretrained_layer, custom_layer)
            
            print(f"    ViT layer {layer_idx} initialization completed successfully")
            
        except Exception as e:
            print(f"      Error adapting ViT layer {layer_idx}: {e}")
            self._fallback_xavier_init(custom_layer)
    
    def _copy_bert_layer_weights(self, pretrained_layer, custom_layer, layer_idx):
        """Copy weights from pretrained BERT layer to custom layer"""
        print(f"    Initializing layer {layer_idx} from BERT")
        
        try:
            # Get pretrained attention weights (BERT structure)
            pretrained_attn = pretrained_layer.attention.self  # BERT uses .self
            custom_attn = custom_layer.self_attn
            
            # Adapt and copy Q, K, V weights
            embed_dim = custom_attn.embed_dim
            
            with torch.no_grad():
                # Adapt Q, K, V weights
                q_adapted = self._adapt_weight_tensor(pretrained_attn.query.weight, (embed_dim, embed_dim), "Query weight")
                k_adapted = self._adapt_weight_tensor(pretrained_attn.key.weight, (embed_dim, embed_dim), "Key weight")
                v_adapted = self._adapt_weight_tensor(pretrained_attn.value.weight, (embed_dim, embed_dim), "Value weight")
                
                # Copy adapted weights to the combined in_proj_weight tensor
                custom_attn.in_proj_weight[:embed_dim].copy_(q_adapted)
                custom_attn.in_proj_weight[embed_dim:2*embed_dim].copy_(k_adapted)
                custom_attn.in_proj_weight[2*embed_dim:].copy_(v_adapted)
                
                # Adapt and copy Q, K, V biases
                q_bias_adapted = self._adapt_bias_tensor(pretrained_attn.query.bias, (embed_dim,), "Query bias")
                k_bias_adapted = self._adapt_bias_tensor(pretrained_attn.key.bias, (embed_dim,), "Key bias")
                v_bias_adapted = self._adapt_bias_tensor(pretrained_attn.value.bias, (embed_dim,), "Value bias")
                
                custom_attn.in_proj_bias[:embed_dim].copy_(q_bias_adapted)
                custom_attn.in_proj_bias[embed_dim:2*embed_dim].copy_(k_bias_adapted)
                custom_attn.in_proj_bias[2*embed_dim:].copy_(v_bias_adapted)
                
                # Adapt and copy output projection (BERT structure)
                out_weight_adapted = self._adapt_weight_tensor(
                    pretrained_layer.attention.output.dense.weight,  # BERT path
                    custom_attn.out_proj.weight.shape, 
                    "Output projection weight"
                )
                out_bias_adapted = self._adapt_bias_tensor(
                    pretrained_layer.attention.output.dense.bias,   # BERT path
                    custom_attn.out_proj.bias.shape,
                    "Output projection bias"
                )
                
                custom_attn.out_proj.weight.copy_(out_weight_adapted)
                custom_attn.out_proj.bias.copy_(out_bias_adapted)
            
            # Adapt and copy layer norm weights (BERT structure)
            with torch.no_grad():
                # BERT has attention.output.LayerNorm and output.LayerNorm
                ln1_weight_adapted = self._adapt_weight_tensor(
                    pretrained_layer.attention.output.LayerNorm.weight,  # BERT path
                    custom_layer.norm1.weight.shape,
                    "LayerNorm1 weight"
                )
                ln1_bias_adapted = self._adapt_bias_tensor(
                    pretrained_layer.attention.output.LayerNorm.bias,   # BERT path
                    custom_layer.norm1.bias.shape,
                    "LayerNorm1 bias"
                )
                
                custom_layer.norm1.weight.copy_(ln1_weight_adapted)
                custom_layer.norm1.bias.copy_(ln1_bias_adapted)
                
                # Second layer norm
                ln2_weight_adapted = self._adapt_weight_tensor(
                    pretrained_layer.output.LayerNorm.weight,           # BERT path
                    custom_layer.norm2.weight.shape,
                    "LayerNorm2 weight"
                )
                ln2_bias_adapted = self._adapt_bias_tensor(
                    pretrained_layer.output.LayerNorm.bias,            # BERT path
                    custom_layer.norm2.bias.shape,
                    "LayerNorm2 bias"
                )
                
                custom_layer.norm2.weight.copy_(ln2_weight_adapted)
                custom_layer.norm2.bias.copy_(ln2_bias_adapted)
            
            # Copy feedforward weights (same structure as ViT)
            self._copy_feedforward_weights(pretrained_layer, custom_layer)
            
            print(f"    BERT layer {layer_idx} initialization completed successfully")
            
        except Exception as e:
            print(f"      Error adapting BERT layer {layer_idx}: {e}")
            self._fallback_xavier_init(custom_layer)
    
    def _copy_feedforward_weights(self, pretrained_layer, custom_layer):
        """Copy feedforward weights (same structure for both ViT and BERT)"""
        pretrained_ff = pretrained_layer.intermediate
        pretrained_output = pretrained_layer.output
        custom_ff = custom_layer.linear1
        custom_output = custom_layer.linear2
        
        with torch.no_grad():
            # First feedforward layer
            ff1_weight_adapted = self._adapt_weight_tensor(
                pretrained_ff.dense.weight,
                custom_ff.weight.shape,
                "Feedforward1 weight"
            )
            ff1_bias_adapted = self._adapt_bias_tensor(
                pretrained_ff.dense.bias,
                custom_ff.bias.shape,
                "Feedforward1 bias"
            )
            
            custom_ff.weight.copy_(ff1_weight_adapted)
            custom_ff.bias.copy_(ff1_bias_adapted)
            
            # Second feedforward layer (output projection)
            ff2_weight_adapted = self._adapt_weight_tensor(
                pretrained_output.dense.weight,
                custom_output.weight.shape,
                "Feedforward2 weight"
            )
            ff2_bias_adapted = self._adapt_bias_tensor(
                pretrained_output.dense.bias,
                custom_output.bias.shape,
                "Feedforward2 bias"
            )
            
            custom_output.weight.copy_(ff2_weight_adapted)
            custom_output.bias.copy_(ff2_bias_adapted)
    
    
    def _fallback_xavier_init(self, custom_layer):
        """Fallback initialization using Xavier uniform"""
        print(f"      Applying Xavier initialization fallback")
        try:
            with torch.no_grad():
                for param in custom_layer.parameters():
                    if len(param.shape) >= 2:
                        torch.nn.init.xavier_uniform_(param)
                    else:
                        torch.nn.init.zeros_(param)
            print(f"      Xavier initialization completed for layer {layer_idx}")
        except Exception as fallback_error:
            print(f"      Even fallback initialization failed: {fallback_error}")
    
    def _parse_covariates_tensor(self, covariates_tensor):
        """Parse concatenated covariates tensor into individual components"""
        if covariates_tensor is None or covariates_tensor.numel() == 0:
            return {}
        
        covariates_dict = {}
        start_idx = 0
        
        if hasattr(self.covariate_embedder, 'use_pcs') and self.covariate_embedder.use_pcs:
            if start_idx + 10 <= covariates_tensor.size(1):
                covariates_dict['pcs'] = covariates_tensor[:, start_idx:start_idx+10]
                start_idx += 10
        
        if hasattr(self.covariate_embedder, 'use_age') and self.covariate_embedder.use_age:
            if start_idx < covariates_tensor.size(1):
                covariates_dict['age'] = covariates_tensor[:, start_idx]
                start_idx += 1
                
        if hasattr(self.covariate_embedder, 'use_gender') and self.covariate_embedder.use_gender:
            if start_idx < covariates_tensor.size(1):
                covariates_dict['gender'] = covariates_tensor[:, start_idx]
                start_idx += 1
                
        if hasattr(self.covariate_embedder, 'use_bmi') and self.covariate_embedder.use_bmi:
            if start_idx < covariates_tensor.size(1):
                covariates_dict['bmi'] = covariates_tensor[:, start_idx]
                start_idx += 1
        
        return covariates_dict
    
    def forward(self, x, covariates=None):
        """
        Args:
            x: (batch_size, channels, seq_len) from conv layers
            covariates: (batch_size, num_covariates) or None
        Returns:
            Features for classification
        """
        batch_size, channels, seq_len = x.shape
        
        # Reshape to (batch_size, seq_len, channels)
        x = x.transpose(1, 2)
        
        # Project to model dimension
        if self.input_projection is not None:
            x = self.input_projection(x)
        
        # Add class token if enabled
        if self.use_cls_token and hasattr(self, 'cls_token'):
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            seq_len += 1
        
        # Add covariate tokens if enabled
        covariate_seq_len = 0
        if self.use_covariate_tokens and covariates is not None:
            covariates_dict = self._parse_covariates_tensor(covariates)
            covariate_result = self.covariate_embedder(covariates_dict)
            
            if covariate_result is not None:
                covariate_tokens, _ = covariate_result
                if covariate_tokens is not None:
                    x = torch.cat([x, covariate_tokens], dim=1)
                    covariate_seq_len = covariate_tokens.size(1)
        
        # Process through sequence layers
        if self.sequence_mode == "transformer":
            x = self._forward_transformer(x)
        elif self.sequence_mode == "mamba":
            x = self._forward_mamba(x)
        elif self.sequence_mode == "hybrid":
            if self.hybrid_mode == "sequential":
                x = self._forward_hybrid(x)
            elif self.hybrid_mode == "parallel":
                x = self._forward_parallel_hybrid(x)
        
        # Apply final normalization
        if self.use_layer_norm and hasattr(self, 'final_norm'):
            x = self.final_norm(x)
        
        # Apply pooling strategy
        return self._apply_pooling(x)
    
    def _forward_transformer(self, x):
        """Forward pass for transformer mode"""
        # Reshape for transformer: (batch_size, seq_len, d_model) -> (seq_len, batch_size, d_model)
        x = x.transpose(0, 1)
        
        # Add positional encoding
        if hasattr(self, 'pos_encoding'):
            x = self.pos_encoding(x)
        
        # Apply transformer layers
        x = self.transformer_encoder(x)
        
        # Reshape back: (seq_len, batch_size, d_model) -> (batch_size, seq_len, d_model)
        x = x.transpose(0, 1)
        
        return x
    
    def _forward_mamba(self, x):
        """Forward pass for mamba mode"""
        # Mamba expects (batch_size, seq_len, d_model)
        for i, mamba_layer in enumerate(self.mamba_layers):
            residual = x
            
            if self.use_layer_norm:
                x = self.layer_norms[i](x)
            
            x = mamba_layer(x)
            
            # Residual connection
            x = x + residual
        
        return x
    
    def _forward_hybrid(self, x):
        """Forward pass for hybrid mode with interleaved layers and optional intermediate pooling"""
        # Need to handle different input formats for transformer vs mamba layers
        for i, (layer, layer_type) in enumerate(zip(self.hybrid_layers, self.layer_config)):
            residual = x
            
            if self.use_layer_norm:
                x = self.hybrid_layer_norms[i](x)
            
            if layer_type == "transformer":
                # Transformer layer expects (seq_len, batch_size, d_model)
                x_transposed = x.transpose(0, 1)
                x_transposed = layer(x_transposed)
                x = x_transposed.transpose(0, 1)
            
            elif layer_type == "mamba":
                # Mamba layer expects (batch_size, seq_len, d_model)
                x = layer(x)
            
            # Check if we will apply intermediate pooling after this layer
            will_pool = self.use_intermediate_pooling and i in self.intermediate_pooling_layers
            
            # Apply residual connection BEFORE pooling
            x = x + residual
            
            # Apply intermediate pooling if configured for this layer
            if will_pool:
                x_before = x
                x = self._apply_intermediate_pooling(x, i)
                
                # Validate output shape
                if len(x.shape) != 3:
                    raise ValueError(
                        f"Intermediate pooling output has incorrect number of dimensions: {x.shape}. "
                        f"Expected 3D tensor (batch_size, seq_len, d_model)"
                    )
                
                if x.shape[0] != x_before.shape[0]:
                    raise ValueError(
                        f"Intermediate pooling changed batch size: {x_before.shape[0]} -> {x.shape[0]}"
                    )
                
                if x.shape[2] != x_before.shape[2]:
                    raise ValueError(
                        f"Intermediate pooling changed model dimension: {x_before.shape[2]} -> {x.shape[2]}"
                    )
                
                #print(f"    ✓ Validation passed: {x_before.shape} -> {x.shape}")
        
        return x
    
    def _apply_intermediate_pooling(self, x, layer_idx):
        """Apply intermediate pooling to reduce sequence length between hybrid layers"""
        batch_size, seq_len, d_model = x.shape

        #print(f"  DEBUG: Applying intermediate pooling after layer {layer_idx}")
        #print(f"    Input shape: {x.shape}")
        
        if str(layer_idx) in self.intermediate_poolers and self.intermediate_poolers[str(layer_idx)] is not None:
            # Use module-based pooling
            pooler = self.intermediate_poolers[str(layer_idx)]
            
            # Handle different pooling types based on their output shapes
            if isinstance(pooler, AttentionPooling):
                # AttentionPooling: (batch_size, seq_len, d_model) -> (batch_size, d_model)
                pooled = pooler(x)  # (batch_size, d_model)
                pooled = pooled.unsqueeze(1)  # (batch_size, 1, d_model)
                #print(f"    AttentionPooling output: {pooled.shape}")
                
            elif isinstance(pooler, ChunkedPooling):
                # ChunkedPooling: (batch_size, seq_len, d_model) -> (batch_size, num_chunks * d_model)
                pooled_flat = pooler(x)  # (batch_size, num_chunks * d_model)
                num_chunks = pooled_flat.size(1) // d_model
                pooled = pooled_flat.reshape(batch_size, num_chunks, d_model)
                #print(f"    ChunkedPooling output: {pooled.shape} (num_chunks={num_chunks})")
                
            elif isinstance(pooler, MultiScalePooling):
                # MultiScalePooling: (batch_size, seq_len, d_model) -> (batch_size, d_model)
                pooled = pooler(x)  # (batch_size, d_model)
                pooled = pooled.unsqueeze(1)  # (batch_size, 1, d_model)
                #print(f"    MultiScalePooling output: {pooled.shape}")
                
            elif isinstance(pooler, MultiHeadPooling):
                # MultiHeadPooling: (batch_size, seq_len, d_model) -> (batch_size, d_model)
                pooled = pooler(x)  # (batch_size, d_model)
                pooled = pooled.unsqueeze(1)  # (batch_size, 1, d_model)
                #print(f"    MultiHeadPooling output: {pooled.shape}")
                
            # elif isinstance(pooler, ConvolutionalPooling):
            #     # ConvolutionalPooling: (batch_size, seq_len, d_model) -> (batch_size, d_model)
            #     pooled = pooler(x)  # (batch_size, d_model)
            #     pooled = pooled.unsqueeze(1)  # (batch_size, 1, d_model)
            #     print(f"    ConvolutionalPooling output: {pooled.shape}")

            elif isinstance(pooler, ConvolutionalPooling):
                # ConvolutionalPooling: (batch_size, seq_len, d_model) -> (batch_size, d_model * target_length)
                pooled_flat = pooler(x)  # (batch_size, d_model * target_length)
                target_length = pooled_flat.size(1) // d_model
                pooled = pooled_flat.reshape(batch_size, target_length, d_model)
                #print(f"    ConvolutionalPooling output: {pooled.shape} (target_length={target_length})")
                
            elif isinstance(pooler, HierarchicalPooling):
                # HierarchicalPooling: (batch_size, seq_len, d_model) -> (batch_size, d_model)
                pooled = pooler(x)  # (batch_size, d_model)
                pooled = pooled.unsqueeze(1)  # (batch_size, 1, d_model)
                #print(f"    HierarchicalPooling output: {pooled.shape}")
                
            else:
                # Unknown pooler type - try to handle gracefully
                pooled = pooler(x)
                if len(pooled.shape) == 2:  # (batch_size, d_model)
                    pooled = pooled.unsqueeze(1)  # (batch_size, 1, d_model)
                #print(f"    Generic pooling output: {pooled.shape}")
            
            return pooled
        
        # Simple pooling strategies (mean/max)
        elif self.intermediate_pooling_strategy == "mean":
            # Get pool factor from kwargs or use default
            pool_factor = self.intermediate_pooling_kwargs.get('pool_factor', 2)
            target_len = max(1, seq_len // pool_factor)
            
            # Adaptive average pooling
            # Input: (batch, seq_len, d_model)
            # Need to transpose to (batch, d_model, seq_len) for pooling
            x_transposed = x.transpose(1, 2)  # (batch, d_model, seq_len)
            pooled = F.adaptive_avg_pool1d(x_transposed, target_len)  # (batch, d_model, target_len)
            pooled = pooled.transpose(1, 2)  # (batch, target_len, d_model)
            
            print(f"    Mean pooling: {x.shape} -> {pooled.shape} (factor={pool_factor})")
            return pooled
            
        elif self.intermediate_pooling_strategy == "max":
            # Get pool factor from kwargs or use default
            pool_factor = self.intermediate_pooling_kwargs.get('pool_factor', 2)
            target_len = max(1, seq_len // pool_factor)
            
            # Adaptive max pooling
            x_transposed = x.transpose(1, 2)  # (batch, d_model, seq_len)
            pooled = F.adaptive_max_pool1d(x_transposed, target_len)  # (batch, d_model, target_len)
            pooled = pooled.transpose(1, 2)  # (batch, target_len, d_model)
            
            print(f"    Max pooling: {x.shape} -> {pooled.shape} (factor={pool_factor})")
            return pooled
        
        # Fallback: no pooling
        print(f"    No pooling applied, returning original shape: {x.shape}")
        return x
    
    def _forward_parallel_hybrid(self, x):
        """Forward pass for parallel hybrid mode"""
        # Process through parallel branches
        x = self.parallel_processor(x)
        return x
    
    def _apply_pooling(self, x):
        """Apply the specified pooling strategy"""
        if self.use_cls_token and hasattr(self, 'cls_token'):
            # Use first token (class token)
            result = x[:, 0]
            print(f"DEBUG: CLS token override - output: {result.shape}")
            return result
        
        elif self.pooling_strategy in ["chunked", "multiscale", "multihead", "conv", "hierarchical"]:
            return self.custom_pooling(x)
        
        elif self.pooling_strategy == "mean":
            return torch.mean(x, dim=1)
        
        elif self.pooling_strategy == "max":
            return torch.max(x, dim=1)[0]
        
        elif self.pooling_strategy == "attention":
            return self.attention_pooling(x)
        
        elif self.pooling_strategy == "concat":
            #print(f"DEBUG: Executing concat pooling")
            #print(f"DEBUG: Input details - batch: {x.size(0)}, seq: {x.size(1)}, dim: {x.size(2)}")
            result = x.reshape(x.size(0), -1)
            #print(f"DEBUG: Concat output: {result.shape}")
            expected = x.size(1) * x.size(2)
            #print(f"DEBUG: Expected features: {expected}, Actual: {result.size(1)}")
            return result

        
        else:
            # Fallback to mean pooling
            return torch.mean(x, dim=1)

# Main unified model
class UnifiedGenomicModel(nn.Module):
    """Unified genomic disease prediction model with flexible sequence modeling"""
    
    def __init__(self, input_size, num_diseases, kernel_sizes, stride, conv_channels, fc_layers, act, dropout_rate, 
                 use_covariates=True, use_age=True, use_gender=True, use_bmi=True, num_covariates=10, 
                 use_pooling=True, pool_size=16, pool_type="max",
                 use_multi_scale=True, use_disease_attention=True, use_separate_heads=True, 
                 attention_heads=8, attention_dim=256, multi_scale_kernels=None, multi_scale_strides=None,
                 multi_scale_fusion="cross_scale", multi_scale_mode="progressive", hardcoded_kernels=None, hardcoded_strides=None,
                 use_pointwise_conv=False, pointwise_channels=16,
                 # Unified sequence modeling parameters
                 sequence_mode="transformer", layer_config=None, d_model=384, num_layers=4, sequence_dropout=0.1, use_layer_norm=True,
                 hybrid_mode="sequential",
                 # Transformer parameters
                 transformer_heads=8, transformer_ff_dim=1024, use_positional_encoding=True, max_seq_len=10000,
                 init_from_pretrained=False, pretrained_model_name="bert-base-uncased", pretrained_model_type="auto",
                 init_layers_fraction=1.0, layer_init_strategy="middle", custom_layer_indices="",
                 # Mamba parameters
                 mamba_d_state=16, mamba_d_conv=4, mamba_expand=2,
                 # Token and pooling parameters
                 use_cls_token=False, use_covariate_tokens=True, covariate_embed_dim=64, 
                 covariate_token_strategy="combined", 
                 pooling_strategy="mean", pooling_kwargs=None,
                 # Intermediate pooling parameters
                 use_intermediate_pooling=False,
                 intermediate_pooling_strategy="mean",
                 intermediate_pooling_layers=None,
                 intermediate_pooling_kwargs=None):

        super(UnifiedGenomicModel, self).__init__()
        
        # Store basic parameters
        self.input_channels = 3
        self.use_covariates = use_covariates
        self.use_age = use_age
        self.use_gender = use_gender
        self.use_bmi = use_bmi
        self.num_diseases = num_diseases
        self.use_pooling = use_pooling
        self.pool_size = pool_size
        self.pool_type = pool_type
        self.use_multi_scale = use_multi_scale
        self.use_disease_attention = use_disease_attention
        self.use_separate_heads = use_separate_heads
        self.use_pointwise_conv = use_pointwise_conv
        self.pointwise_channels = pointwise_channels
        
        # Store sequence modeling parameters
        self.sequence_mode = sequence_mode

        # Store hybrid mode
        self.hybrid_mode = hybrid_mode

        self.d_model = d_model
        self.num_layers = num_layers
        self.use_covariate_tokens = use_covariate_tokens
        
        # Store multi-scale parameters
        self.multi_scale_kernels = multi_scale_kernels if multi_scale_kernels is not None else [15, 63, 255]
        self.multi_scale_strides = multi_scale_strides if multi_scale_strides is not None else [4, 16, 64]
        self.multi_scale_fusion = multi_scale_fusion
        self.multi_scale_mode = multi_scale_mode
        self.hardcoded_kernels = hardcoded_kernels
        self.hardcoded_strides = hardcoded_strides
        self.conv_channels = conv_channels
        self.input_size = input_size

        # Calculate total covariates
        self.total_covariates = 0
        if use_covariates:
            self.total_covariates += num_covariates
        if use_age:
            self.total_covariates += 1
        if use_gender:
            self.total_covariates += 1
        if use_bmi:
            self.total_covariates += 1

        # Create convolutional layers
        print(f"  Creating convolutional layers...")
        print(f"  Input sequence length: {input_size:,}")
        
        conv_layers = self._create_conv_layers(conv_channels, kernel_sizes, stride, dropout_rate, act)
        
        # Add final pooling if enabled
        # Add final pooling layer to the conv_layers sequential if enabled
        if self.use_pooling:
            if pool_type.lower() == "max":
                pooling_layer = nn.AdaptiveMaxPool1d(pool_size)
                print(f"Adding final AdaptiveMaxPool1d with pool_size={pool_size} to conv_layers")
            elif pool_type.lower() == "avg":
                pooling_layer = nn.AdaptiveAvgPool1d(pool_size)
                print(f"Adding final AdaptiveAvgPool1d with pool_size={pool_size} to conv_layers")
            else:
                raise ValueError(f"Invalid pool_type: {pool_type}. Must be 'max' or 'avg'")
            
            # Create a new sequential that includes the final pooling layer
            conv_layers_with_pool = nn.Sequential(*list(conv_layers), pooling_layer)
            self.conv_layers = conv_layers_with_pool
        else:
            print("Not using final adaptive pooling")
            self.conv_layers = conv_layers
        
        # Calculate output size after convolutions
        self.conv_output_info = self._get_conv_output_info(input_size)
        conv_output_size = self.conv_output_info['flattened_size']
        conv_seq_len = self.conv_output_info['seq_len']
        conv_channels_out = self.conv_output_info['channels']
        
        print(f"Convolutional output: channels={conv_channels_out}, seq_len={conv_seq_len}, flattened_size={conv_output_size}")
        
        # Add unified sequence processing
        print(f"\n  Adding Unified Sequence Processing:")
        print(f"  - Sequence mode: {sequence_mode}")
        print(f"  - Input to sequence block: (batch_size, {conv_channels_out}, {conv_seq_len})")
         # Get pooling configuration
        pooling_kwargs = pooling_kwargs or {}

        self.sequence_block = UnifiedSequenceBlock(
            input_dim=conv_channels_out,
            d_model=d_model,
            sequence_mode=sequence_mode,
            layer_config=parse_layer_config(layer_config) if layer_config else None,
            num_layers=num_layers,
            dropout=sequence_dropout,
            use_layer_norm=use_layer_norm,
            hybrid_mode=hybrid_mode,
            # Transformer parameters
            transformer_heads=transformer_heads,
            transformer_ff_dim=transformer_ff_dim,
            use_positional_encoding=use_positional_encoding,
            max_seq_len=max_seq_len,
            init_from_pretrained=init_from_pretrained,
            pretrained_model_name=pretrained_model_name,
            pretrained_model_type=pretrained_model_type,
            init_layers_fraction=init_layers_fraction,
            layer_init_strategy=layer_init_strategy,
            custom_layer_indices=custom_layer_indices,
            # Mamba parameters
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            # Token and pooling parameters
            use_cls_token=use_cls_token,
            use_covariate_tokens=use_covariate_tokens,
            covariate_embed_dim=covariate_embed_dim,
            covariate_token_strategy=covariate_token_strategy,
            pooling_strategy=pooling_strategy,
            use_age=use_age,
            use_gender=use_gender,
            use_bmi=use_bmi,
            use_pcs=use_covariates,
            # Intermediate pooling parameters
            use_intermediate_pooling=use_intermediate_pooling,
            intermediate_pooling_strategy=intermediate_pooling_strategy,
            intermediate_pooling_layers=intermediate_pooling_layers,
            intermediate_pooling_kwargs=intermediate_pooling_kwargs,
            **pooling_kwargs  # Pass pooling parameters
            )
        
        # Calculate feature size for fully connected layers

        if pooling_strategy == "concat":
            base_seq_len = conv_seq_len
        
            # Add tokens if enabled
            if use_cls_token:
                base_seq_len += 1
            
            if use_covariate_tokens:
                if covariate_token_strategy == "combined":
                    base_seq_len += 1
                elif covariate_token_strategy == "separate":
                    # Count individual covariate tokens
                    num_cov_tokens = sum([use_pcs, use_age, use_gender, use_bmi])
                    base_seq_len += num_cov_tokens
            
            feature_size_for_fc = base_seq_len * d_model
            self.sequence_block.final_output_dim = feature_size_for_fc

            print(f"  - Sequence output: (batch_size, {base_seq_len}, {d_model})")
            print(f"  - Flattened size after concat pooling: {feature_size_for_fc:,}")
        
        else:
            feature_size_for_fc = self.sequence_block.final_output_dim
            print(f"  - Sequence output dimension: {feature_size_for_fc:,} (from {pooling_strategy} pooling)")
       
            
        
        
        # # Calculate feature size for fully connected layers
        # if self.sequence_block.final_output_dim is not None:
        #     feature_size_for_fc = self.sequence_block.final_output_dim
        #     print(f"  - Sequence output dimension: {feature_size_for_fc:,} (from {pooling_strategy} pooling)")
        # else:
        #     # Fallback for concat strategy - calculate dynamically
        #     base_seq_len = conv_seq_len
        
        #     # Add tokens if enabled
        #     if use_cls_token:
        #         base_seq_len += 1
            
        #     if use_covariate_tokens:
        #         if covariate_token_strategy == "combined":
        #             base_seq_len += 1
        #         elif covariate_token_strategy == "separate":
        #             # Count individual covariate tokens
        #             num_cov_tokens = sum([use_pcs, use_age, use_gender, use_bmi])
        #             base_seq_len += num_cov_tokens
            
        #     feature_size_for_fc = base_seq_len * d_model
        #     print(f"  - Sequence output: (batch_size, {base_seq_len}, {d_model})")
        #     print(f"  - Flattened size after concat pooling: {feature_size_for_fc:,}")
#****************************
        # if use_covariate_tokens:
        #     if pooling_strategy == 'concat':
        #         # Need to calculate sequence length after adding tokens
        #         base_seq_len = conv_seq_len
        #         if use_cls_token:
        #             base_seq_len += 1
        #         if covariate_token_strategy == "combined":
        #             base_seq_len += 1
        #         elif covariate_token_strategy == "separate":
        #             # Count individual covariate tokens
        #             num_cov_tokens = sum([use_pcs, use_age, use_gender, use_bmi])
        #             base_seq_len += num_cov_tokens
                
        #         feature_size_for_fc = base_seq_len * d_model
        #         print(f"  - Sequence output: (batch_size, {base_seq_len}, {d_model})")
        #         print(f"  - Flattened size after sequence processing: {feature_size_for_fc:,}")
        #     else:
        #         feature_size_for_fc = d_model
        #         print(f"  - Sequence output: (batch_size, {d_model}) from {pooling_strategy} pooling")
        # else:
        #     if pooling_strategy == 'concat':
        #         base_seq_len = conv_seq_len
        #         if use_cls_token:
        #             base_seq_len += 1
        #         feature_size_for_fc = base_seq_len * d_model
        #         print(f"  - Sequence output: (batch_size, {base_seq_len}, {d_model})")
        #         print(f"  - Flattened size after sequence processing: {feature_size_for_fc:,}")
        #     else:
        #         feature_size_for_fc = d_model
        #         print(f"  - Sequence output: (batch_size, {d_model}) from {pooling_strategy} pooling")
        
        # Disease-specific attention mechanism
        if self.use_disease_attention:
            self.attention_input_dim = feature_size_for_fc
            self.attention_proj = nn.Linear(self.attention_input_dim, attention_dim)
            self.disease_attention = DiseaseSpecificAttention(attention_dim, num_diseases, attention_heads)
            self.attention_output_dim = attention_dim
            print(f"Using disease-specific attention with {attention_heads} heads and {attention_dim} dimensions")
        else:
            self.attention_output_dim = feature_size_for_fc
        
        # Shared feature layers and disease heads
        if self.use_separate_heads:
            shared_layers = []
            # Only add covariates if not using covariate tokens
            covariate_size = 0 if self.use_covariate_tokens else self.total_covariates
            in_features = self.attention_output_dim + covariate_size
            
            for i, out_features in enumerate(fc_layers[:-1]):
                shared_layers.extend([
                    nn.Linear(in_features, out_features),
                    nn.BatchNorm1d(out_features),
                    self.get_activation(act),
                    nn.Dropout(dropout_rate)
                ])
                in_features = out_features
            
            self.fc_shared = nn.Sequential(*shared_layers)
            shared_output_dim = in_features
            
            head_hidden_dims = [fc_layers[-1]] if len(fc_layers) > 1 else [64]
            self.disease_heads = nn.ModuleList([
                SeparateDiseaseHead(shared_output_dim, head_hidden_dims, dropout_rate, act)
                for _ in range(num_diseases)
            ])
            print(f"Using separate disease heads with shared feature dimension: {shared_output_dim}")
            
        else:
            fc_layers_list = []
            # Only add covariates if not using covariate tokens
            covariate_size = 0 if self.use_covariate_tokens else self.total_covariates
            in_features = self.attention_output_dim + covariate_size
            
            for i, out_features in enumerate(fc_layers):
                fc_layers_list.extend([
                    nn.Linear(in_features, out_features),
                    nn.BatchNorm1d(out_features),
                    self.get_activation(act),
                    nn.Dropout(dropout_rate)
                ])
                in_features = out_features
            
            # Create shared feature layers
            self.fc_shared = nn.Sequential(*fc_layers_list)
            self.disease_outputs = nn.Linear(in_features, num_diseases)
        
        # Print model summary
        self._print_model_summary()
    
    def _create_conv_layers(self, conv_channels, kernel_sizes, stride, dropout_rate, act):
        """Create convolutional layers based on multi-scale configuration"""
        if self.use_multi_scale:
            # Use multi-scale convolution
            return self._create_multi_scale_conv_layers(conv_channels, kernel_sizes, stride, dropout_rate, act)
        else:
            # Use standard convolution
            return self._create_standard_conv_layers(conv_channels, kernel_sizes, stride, dropout_rate, act)
    
    def _create_multi_scale_conv_layers(self, conv_channels, kernel_sizes, stride, dropout_rate, act):
        """Create multi-scale convolutional layers"""
        layers = nn.ModuleList()
        current_length = self.input_size
        
        for i in range(len(conv_channels)):
            is_final_layer = (i == len(conv_channels) - 1)
            
            if i == 0:
                in_channels = self.input_channels
            else:
                prev_is_final = (i-1 == len(conv_channels) - 1)
                if self.use_pointwise_conv and prev_is_final:
                    in_channels = len(self.multi_scale_kernels) * self.pointwise_channels
                else:
                    in_channels = len(self.multi_scale_kernels) * conv_channels[i-1]
            
            out_channels = conv_channels[i]
            
            # Get kernels and strides for this layer
            scale_kernels, scale_strides = self._get_layer_kernels_and_strides(i)
            
            multi_scale_block = MultiScaleConvBlock(
                in_channels, out_channels, scale_kernels, scale_strides, act, dropout_rate, self.pool_type,
                self.use_pointwise_conv, self.pointwise_channels, is_final_layer
            )
            layers.append(multi_scale_block)
        
        return nn.Sequential(*layers)
    
    def _create_standard_conv_layers(self, conv_channels, kernel_sizes, stride, dropout_rate, act):
        """Create standard sequential convolutional layers"""
        layers = []
        
        for i in range(len(conv_channels)):
            in_channels = self.input_channels if i == 0 else conv_channels[i-1]
            out_channels = conv_channels[i]
            kernel_size = kernel_sizes[i]
            stride_val = stride[i]
            padding = kernel_size // 2
            is_final_layer = (i == len(conv_channels) - 1)
            
            layers.append(nn.Conv1d(in_channels=in_channels, out_channels=out_channels,  
                                   kernel_size=kernel_size, stride=stride_val, padding=padding))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(self.get_activation(act))

            # Add pointwise convolution for final layer if enabled
            if self.use_pointwise_conv and is_final_layer:
                layers.append(nn.Conv1d(in_channels=out_channels, out_channels=self.pointwise_channels,
                                    kernel_size=1, stride=1, padding=0))
                layers.append(nn.BatchNorm1d(self.pointwise_channels))
                layers.append(self.get_activation(act))
        
        return nn.Sequential(*layers)
    
    def _get_layer_kernels_and_strides(self, layer_idx):
        """Get kernels and strides for a specific layer based on mode"""
        if self.multi_scale_mode == "hardcoded":
            # Use hardcoded values
            return self.hardcoded_kernels[layer_idx], self.hardcoded_strides[layer_idx]
        else:
            # Use progressive reduction (original behavior)
            if layer_idx == 0:
                return self.multi_scale_kernels.copy(), self.multi_scale_strides.copy()
            else:
                scale_kernels = []
                scale_strides = []
                
                for j in range(len(self.multi_scale_kernels)):
                    # Calculate kernel: original_kernel // (2^i)
                    new_kernel = max(1, self.multi_scale_kernels[j] // (2 ** layer_idx))
                    # Ensure kernel is odd for proper padding
                    if new_kernel % 2 == 0:
                        new_kernel += 1
                    
                    # Calculate stride: original_stride // (2^i)
                    new_stride = max(1, self.multi_scale_strides[j] // (2 ** layer_idx))
                    #new_stride = max(1, self.multi_scale_strides[j] // (1 ** layer_idx))
                    
                    scale_kernels.append(new_kernel)
                    scale_strides.append(new_stride)
                
                return scale_kernels, scale_strides

    def _get_conv_output_info(self, input_size):
        """Calculate output size after convolution layers"""
        try:
            x = torch.randn(1, 3, input_size, dtype=torch.float32)
            print(f"  - Input to conv layers: {x.shape}")
            
            x = self.conv_layers(x)
            print(f"  - Output from conv layers: {x.shape}")
            
            batch_size, channels, seq_len = x.shape
            flattened_size = x.numel() // x.size(0)
            
            return {
                'channels': channels,
                'seq_len': seq_len,
                'flattened_size': flattened_size
            }

        except Exception as e:
            print(f"Error in _get_conv_output_info: {e}")
            raise e
    
    def _print_model_summary(self):
        """Print model configuration summary"""
        print(f"\nUnified Genomic Model Summary:")
        print(f"  - Architecture: Multi-scale CNN + {self.sequence_mode.capitalize()} + Disease Prediction")
        print(f"  - Sequence mode: {self.sequence_mode}")
        print(f"  - Model dimension: {self.d_model}")
        print(f"  - Number of sequence layers: {self.num_layers}")
        print(f"  - Multi-scale convolutions: {self.use_multi_scale}")
        print(f"  - Disease-specific attention: {self.use_disease_attention}")
        print(f"  - Separate disease heads: {self.use_separate_heads}")
        print(f"  - Covariate tokens: {self.use_covariate_tokens}")
        print(f"  - Input size: {self.input_size:,} SNPs")
        print(f"  - Number of diseases: {self.num_diseases}")
        print(f"  - Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def get_activation(self, name):
        activations = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'leakyrelu': nn.LeakyReLU(0.01),
            'rrelu': nn.RReLU(0.125, 0.3333),
            'gelu': nn.GELU(),
            'silu': nn.SiLU()
        }
        return activations.get(name, nn.ReLU())

    def forward(self, x, covariates=None):
        # Input: x shape [batch_size, n_snps, 3]
        x = x.permute(0, 2, 1)  # -> [batch_size, 3, n_snps]
        
        # Convolutional processing
        x = self.conv_layers(x)  # Shape: (batch_size, channels, seq_len)
        
        # Sequence processing
        if self.use_covariate_tokens:
            # Process covariates as tokens within sequence block
            x = self.sequence_block(x, covariates)
            
            # Handle different output formats
            if self.sequence_block.pooling_strategy == 'concat':
                x = x.reshape(x.size(0), -1)
        else:
            # Sequence without covariate tokens
            x = self.sequence_block(x)
            
            if self.sequence_block.pooling_strategy == 'concat':
                x = x.reshape(x.size(0), -1)
        
        # Disease-specific attention processing
        if self.use_disease_attention:
            x_proj = self.attention_proj(x)
            x_seq = x_proj.unsqueeze(1)
            x_attended = self.disease_attention(x_seq)
            x = torch.mean(x_attended, dim=1)
        
        # Concatenate with covariates (only if not using covariate tokens)
        if covariates is not None and self.total_covariates > 0 and not self.use_covariate_tokens:
            x = torch.cat([x, covariates], dim=1)
        
        # Shared feature processing
        shared_features = self.fc_shared(x)
        
        if self.use_separate_heads:
            # Use separate heads for each disease
            if self.use_disease_attention:
                # Use disease-specific features for each head
                disease_outputs = []
                for i, head in enumerate(self.disease_heads):
                    disease_features = x_attended[:, i, :]
                    
                    if covariates is not None and self.total_covariates > 0 and not self.use_covariate_tokens:
                        disease_input = torch.cat([disease_features, covariates], dim=1)
                    else:
                        disease_input = disease_features
                    
                    disease_shared = self.fc_shared(disease_input)
                    disease_output = head(disease_shared)
                    disease_outputs.append(disease_output)
                
                return torch.cat(disease_outputs, dim=1)
            else:
                # Use shared features for all heads
                disease_outputs = []
                for head in self.disease_heads:
                    disease_output = head(shared_features)
                    disease_outputs.append(disease_output)
                
                return torch.cat(disease_outputs, dim=1)
        else:
            # Traditional shared output layer
            return self.disease_outputs(shared_features)

# Training and evaluation functions (shared from original scripts with minimal modifications)
def print_lr(optimizer):
    for param_group in optimizer.param_groups:
        print(f"Current Learning Rate: {param_group['lr']}")

def find_latest_checkpoint(dir_path, prefer_best=False):
    """Find the latest checkpoint in the directory"""
    if prefer_best:
        best_models = [f for f in os.listdir(dir_path) if f.startswith('best_model_')]
        if best_models:
            best_models.sort(key=lambda x: int(x.split('best_model_')[1].split('.pt')[0]))
            latest_best = os.path.join(dir_path, best_models[-1])
            print(f"Found best model checkpoint: {latest_best}")
            return latest_best

    old_best_model_path = os.path.join(dir_path, 'best_model.pt')
    if os.path.exists(old_best_model_path):
        print(f"Found best model checkpoint (old format): {old_best_model_path}")
        return old_best_model_path
    
    checkpoints = [f for f in os.listdir(dir_path) if f.startswith('checkpoint_epoch_')]
    if checkpoints:
        checkpoints.sort(key=lambda x: int(x.split('checkpoint_epoch_')[1].split('.pt')[0]))
        latest_checkpoint = os.path.join(dir_path, checkpoints[-1])
        print(f"Using latest epoch checkpoint: {latest_checkpoint}")
        return latest_checkpoint
    
    print("No checkpoints found")
    return None

def cleanup_old_checkpoints(dir_path, keep_last_n=3):
    """Remove old checkpoints, keeping only the most recent n"""
    checkpoints = [f for f in os.listdir(dir_path) if f.startswith('checkpoint_epoch_')]
    checkpoints.sort(key=lambda x: int(x.split('checkpoint_epoch_')[1].split('.pt')[0]))
    
    if len(checkpoints) > keep_last_n:
        for old_ckpt in checkpoints[:-keep_last_n]:
            old_path = os.path.join(dir_path, old_ckpt)
            print(f"Removing old checkpoint: {old_path}")
            os.remove(old_path)
    
    best_models = [f for f in os.listdir(dir_path) if f.startswith('best_model_')]
    if best_models:
        best_models.sort(key=lambda x: int(x.split('best_model_')[1].split('.pt')[0]))
        if len(best_models) > 1:
            for old_best in best_models[:-1]:
                old_path = os.path.join(dir_path, old_best)
                print(f"Removing old best model: {old_path}")
                os.remove(old_path)

def train_multilabel_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, disease_labels, device='cuda',
                         early_stopping=None, checkpoint_dir=None, 
                         start_epoch=0, keep_last_n=2, history=None, initial_best_loss=float('inf')):
    print(f"Training unified multilabel model on device: {device}")
    print(f"Disease labels: {disease_labels}")
    print(f"Starting with initial best loss: {initial_best_loss:.6f}")
    
    scaler = GradScaler('cuda')
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = initial_best_loss
    completed_epochs = start_epoch 

    num_diseases = len(disease_labels)

    # Initialize history if None or missing keys
    if history is None or 'train_loss' not in history or 'test_loss' not in history:
        print("Creating new history dictionary")
        history = {
            'train_loss': [], 'test_loss': [],
            'learning_rates': []
        }

        for disease in disease_labels:
            for phase in ['train', 'test']:
                history[f'{phase}_{disease}_acc'] = []
                history[f'{phase}_{disease}_auc'] = []
                history[f'{phase}_{disease}_pr_auc'] = []
                history[f'{phase}_{disease}_f1'] = []   
    
    # Verify history structure
    required_keys = ['train_loss', 'test_loss', 'learning_rates']
    for disease in disease_labels:
        for phase in ['train', 'test']:
            required_keys.extend([f'{phase}_{disease}_acc', f'{phase}_{disease}_auc',
                                    f'{phase}_{disease}_pr_auc', f'{phase}_{disease}_f1'])
    
    for key in required_keys:
        if key not in history:
            print(f"Adding missing key {key} to history")
            history[key] = []

    # Store metrics for current epoch
    phase_preds = {phase: {disease: [] for disease in disease_labels} for phase in ['train', 'test']}
    phase_labels = {phase: {disease: [] for disease in disease_labels} for phase in ['train', 'test']}
    
    for epoch in range(start_epoch, num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Reset predictions and labels for this epoch
        phase_preds = {phase: {disease: [] for disease in disease_labels} for phase in ['train', 'test']}
        phase_labels = {phase: {disease: [] for disease in disease_labels} for phase in ['train', 'test']}

        for phase in ['train', 'test']:
            start_time = time.time()
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = {disease: 0 for disease in disease_labels}
            total_samples = 0

            batch_count = len(dataloaders[phase])
            print(f"\nStarting {phase} phase: {batch_count} batches to process")

            # Create CUDA stream for async data transfer
            stream = torch.cuda.Stream()
            batch_iter = iter(dataloaders[phase])

            # Prefetch first batch
            try:
                inputs, covariates, labels = next(batch_iter)
                inputs = inputs.to(device, non_blocking=True)
                covariates = covariates.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
            except StopIteration:
                print(f"Warning: {phase} dataloader is empty!")
                continue

            batch_times = []

            # Process all batches
            for i in range(len(dataloaders[phase])):
                batch_start = time.time()

                # Async prefetch next batch
                try:
                    if i + 1 < len(dataloaders[phase]):
                        with torch.cuda.stream(stream):
                            next_inputs, next_covariates, next_labels = next(batch_iter)
                            next_inputs = next_inputs.to(device, non_blocking=True)
                            next_covariates = next_covariates.to(device, non_blocking=True)
                            next_labels = next_labels.to(device, non_blocking=True)
                except StopIteration:
                    pass

                torch.cuda.current_stream().wait_stream(stream)

                optimizer.zero_grad()

                with autocast('cuda'):
                    with torch.set_grad_enabled(phase == 'train'):
                        logits = model(inputs, covariates)
                        loss = criterion(logits, labels)

                        with torch.no_grad():
                            probs = torch.sigmoid(logits)
                            preds = (probs >= 0.5).float()
                        
                        if phase == 'train':
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()

                            # Step per-iteration schedulers
                            if isinstance(scheduler, (WarmupExponential, ExponentialDecay)):
                                scheduler.step()

                # Memory management
                if (i + 1) % 20 == 0:
                    torch.cuda.empty_cache()
                
                batch_size = labels.size(0)
                running_loss += loss.item() * batch_size

                # Store predictions and labels for each disease 
                for j, disease in enumerate(disease_labels):
                    running_corrects[disease] += torch.sum(preds[:, j] == labels[:, j])
                    phase_labels[phase][disease].extend(labels[:, j].cpu().numpy())
                    phase_preds[phase][disease].extend(probs[:, j].detach().cpu().numpy())
                
                total_samples += batch_size

                batch_end = time.time()
                batch_time = batch_end - batch_start
                batch_times.append(batch_time)

                # Print progress
                if (i + 1) % 100 == 0 or i == 0 or i == len(dataloaders[phase]) - 1:
                    avg_time = sum(batch_times) / len(batch_times)
                    eta = avg_time * (len(dataloaders[phase]) - i - 1)
                    
                    if eta > 60:
                        eta_str = f"{eta//60:.0f}m {eta%60:.0f}s"
                    else:
                        eta_str = f"{eta:.1f}s"
                        
                    print(f"{phase} Batch {i+1}/{len(dataloaders[phase])} | " 
                          f"Time: {batch_time:.2f}s | "
                          f"ETA: {eta_str} | "
                          f"LR: {optimizer.param_groups[0]['lr']:.6f}")

                # Prepare for next iteration
                try:
                    inputs, covariates, labels = next_inputs, next_covariates, next_labels
                except:
                    break

            # Calculate epoch metrics
            epoch_time = time.time() - start_time
            epoch_loss = running_loss / total_samples
            history[f'{phase}_loss'].append(epoch_loss)
            print(f'{phase} Loss: {epoch_loss:.4f} (Time: {epoch_time:.2f}s)')
            
            # Calculate and print metrics for each disease
            for i, disease in enumerate(disease_labels):
                epoch_acc = running_corrects[disease].double() / total_samples
                history[f'{phase}_{disease}_acc'].append(epoch_acc.item())

                y_true = np.array(phase_labels[phase][disease])
                y_pred_proba = np.array(phase_preds[phase][disease])
                y_pred = (y_pred_proba >= 0.5).astype(int)
                
                # Calculate metrics
                try:
                    epoch_auc = roc_auc_score(y_true, y_pred_proba)
                    history[f'{phase}_{disease}_auc'].append(epoch_auc)
                    auc_str = f" - ROC-AUC: {epoch_auc:.4f}"
                except Exception as e:
                    history[f'{phase}_{disease}_auc'].append(0.5)
                    auc_str = "ROC AUC: N/A (need both classes)"
                    
                try:
                    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
                    epoch_pr_auc = auc(recall_curve, precision_curve)
                    history[f'{phase}_{disease}_pr_auc'].append(epoch_pr_auc)
                    pr_auc_str = f" - PR-AUC: {epoch_pr_auc:.4f}"
                except Exception as e:
                    history[f'{phase}_{disease}_pr_auc'].append(0.5)
                    pr_auc_str = "PR AUC: N/A (need both classes)"
                    
                try:
                    epoch_f1 = f1_score(y_true, y_pred)
                    history[f'{phase}_{disease}_f1'].append(epoch_f1)
                    f1_str = f"- F1: {epoch_f1:.4f}"
                except Exception as e:
                    history[f'{phase}_{disease}_f1'].append(0.0)
                    f1_str = "F1: N/A (need both classes)"
                    
                print(f'  {disease}: Acc: {epoch_acc:.4f}, {auc_str}, {pr_auc_str}, {f1_str}')

            # Early stopping and model saving
            if phase == 'test':
                if epoch_loss < best_loss:
                    print(f"New best model! Validation loss improved from {best_loss:.6f} to {epoch_loss:.6f}")
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    
                    if checkpoint_dir:
                        best_model_path = os.path.join(checkpoint_dir, f'best_model_{epoch+1}.pt')
                        best_checkpoint = {
                            'epoch': epoch + 1,
                            'model_state_dict': best_model_wts,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_model_state_dict': best_model_wts,
                            'history': history,
                            'best_loss': best_loss,
                            'completed_epochs': epoch + 1
                        }
                        
                        if hasattr(scheduler, 'state_dict'):
                            best_checkpoint['scheduler_state_dict'] = scheduler.state_dict()
                        elif isinstance(scheduler, (WarmupExponential, ExponentialDecay)):
                            best_checkpoint['scheduler_state_dict'] = scheduler.state_dict()
                            
                        if early_stopping is not None:
                            best_checkpoint['early_stopping_state'] = early_stopping.state_dict()
                else:
                    print(f"Validation loss: {epoch_loss:.6f} (not improved from best: {best_loss:.6f})")
                
                # Update early stopping
                if early_stopping is not None:
                    early_stopping(epoch_loss, global_best_loss=best_loss)
                    
                    if early_stopping.early_stop:
                        print("Early stopping triggered")
                        completed_epochs = epoch + 1
                        
                        if checkpoint_dir:
                            final_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{completed_epochs}.pt')
                            final_checkpoint = {
                                'epoch': completed_epochs,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'best_model_state_dict': best_model_wts,
                                'history': history,
                                'best_loss': best_loss,
                                'completed_epochs': completed_epochs
                            }
                            
                            if hasattr(scheduler, 'state_dict'):
                                final_checkpoint['scheduler_state_dict'] = scheduler.state_dict()
                            elif isinstance(scheduler, (WarmupExponential, ExponentialDecay)):
                                final_checkpoint['scheduler_state_dict'] = scheduler.state_dict()
                                
                            final_checkpoint['early_stopping_state'] = early_stopping.state_dict()
                            
                            print(f"Saving final checkpoint to {final_checkpoint_path}")
                            torch.save(final_checkpoint, final_checkpoint_path)

                        model.load_state_dict(best_model_wts)
                        return model, history, compute_final_metrics(phase_labels, phase_preds, disease_labels), phase_preds, phase_labels, completed_epochs
            
        # Step epoch-level schedulers
        if scheduler is not None and not isinstance(scheduler, (WarmupExponential, ExponentialDecay)):
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(history['test_loss'][-1])
            else:
                scheduler.step()

        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        print_lr(optimizer)
        completed_epochs = epoch + 1
        
        # Save checkpoint at end of each epoch
        if checkpoint_dir:
            checkpoint = {
                'epoch': completed_epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_model_state_dict': best_model_wts,
                'history': history,
                'best_loss': best_loss,
                'completed_epochs': completed_epochs
            }
            
            if hasattr(scheduler, 'state_dict'):
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            elif isinstance(scheduler, (WarmupExponential, ExponentialDecay)):
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
                
            if early_stopping is not None:
                checkpoint['early_stopping_state'] = early_stopping.state_dict()
            
            regular_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{completed_epochs}.pt')
            print(f"Saving regular epoch checkpoint to {regular_checkpoint_path}")
            torch.save(checkpoint, regular_checkpoint_path)
            
            cleanup_old_checkpoints(checkpoint_dir, keep_last_n)

    print(f'Training completed. Best test loss: {best_loss:.4f}')
    model.load_state_dict(best_model_wts)
    return model, history, compute_final_metrics(phase_labels, phase_preds, disease_labels), phase_preds, phase_labels, completed_epochs

def compute_final_metrics(all_labels, all_preds, disease_labels):
    final_metrics = {}
    
    for phase in ['train', 'test']:
        phase_metrics = {}
        
        for disease in disease_labels:
            y_true = np.array(all_labels[phase][disease])
            y_pred_proba = np.array(all_preds[phase][disease])
            
            print(f"Computing metrics for {disease} in {phase} phase:")
            print(f"  - Number of samples: {len(y_true)}")
            print(f"  - Positive samples: {np.sum(y_true == 1)}")
            print(f"  - Negative samples: {np.sum(y_true == 0)}")

            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            try:
                cm = confusion_matrix(y_true, y_pred)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                    accuracy = (tp + tn) / (tp + tn + fp + fn)
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
                else:
                    sensitivity = specificity = accuracy = precision = f1 = 0
                
                try:
                    fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
                    roc_auc = auc(fpr, tpr)
                    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
                    pr_auc = auc(recall_curve, precision_curve)
                    f1_score_val = f1_score(y_true, y_pred)
                except Exception:
                    fpr, tpr, roc_thresholds = np.array([]), np.array([]), np.array([])
                    precision_curve, recall_curve, pr_thresholds = np.array([]), np.array([]), np.array([])
                    roc_auc = pr_auc = 0.5
                    f1_score_val = 0.0
                    
            except Exception:
                cm = np.zeros((2, 2))
                sensitivity = specificity = accuracy = precision = f1 = 0
                fpr, tpr, roc_thresholds = np.array([]), np.array([]), np.array([])
                precision_curve, recall_curve, pr_thresholds = np.array([]), np.array([]), np.array([])
                roc_auc = pr_auc = 0.5
                f1_score_val = 0.0

            disease_metrics = {
                'cm': cm,
                'sens': f'{sensitivity:.5f}',
                'spec': f'{specificity:.5f}',
                'acc': f'{accuracy:.5f}',
                'auc': roc_auc,
                'pr_auc': pr_auc,
                'f1': f1_score_val,
                'roc_curve': {
                    'fpr': fpr,
                    'tpr': tpr,
                    'thresholds': roc_thresholds
                },
                'pr_curve': {
                    'precision': precision_curve,
                    'recall': recall_curve,
                    'thresholds': pr_thresholds
                }
            }
            
            phase_metrics[disease] = disease_metrics
        
        final_metrics[phase] = phase_metrics
    
    return final_metrics

def plot_multilabel_metrics(history, disease_labels, save_dir, final_metrics=None):
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    plt.figure(figsize=(25, 16))
    
    # 1. Model Loss
    plt.subplot(2, 3, 1)
    for phase in ['train', 'test']:
        if f'{phase}_loss' in history and history[f'{phase}_loss']:
            plt.plot(history[f'{phase}_loss'], label=f'{phase}')
    plt.title('Model Loss', fontsize=16)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 2. Learning Rate
    plt.subplot(2, 3, 2)
    plt.plot(history['learning_rates'])
    plt.title('Learning Rate', fontsize=16)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(alpha=0.3)

    # 3. Average Accuracy
    plt.subplot(2, 3, 3)
    for phase in ['train', 'test']:
        avg_acc = []
        for epoch in range(len(history.get(f'train_loss', []))):
            epoch_values = []
            for disease in disease_labels:
                key = f'{phase}_{disease}_acc'
                if key in history and len(history[key]) > epoch:
                    epoch_values.append(history[key][epoch])
            if epoch_values:
                avg_acc.append(np.mean(epoch_values))
        if avg_acc:
            plt.plot(avg_acc, label=f'{phase} avg acc')
    plt.title('Average Accuracy Across All Diseases', fontsize=16)
    plt.xlabel('Epoch')
    plt.ylabel('Average Accuracy')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 4. Average ROC AUC
    plt.subplot(2, 3, 4)
    for phase in ['train', 'test']:
        avg_auc = []
        for epoch in range(len(history.get(f'train_loss', []))):
            epoch_values = []
            for disease in disease_labels:
                key = f'{phase}_{disease}_auc'
                if key in history and len(history[key]) > epoch:
                    epoch_values.append(history[key][epoch])
            if epoch_values:
                avg_auc.append(np.mean(epoch_values))
        if avg_auc:
            plt.plot(avg_auc, label=f'{phase} avg ROC AUC')
    plt.title('Average ROC AUC Across All Diseases', fontsize=16)
    plt.xlabel('Epoch')
    plt.ylabel('Average ROC AUC')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(alpha=0.3)

    # 5. Average PR AUC
    plt.subplot(2, 3, 5)
    for phase in ['train', 'test']:
        avg_pr_auc = []
        for epoch in range(len(history.get(f'train_loss', []))):
            epoch_values = []
            for disease in disease_labels:
                key = f'{phase}_{disease}_pr_auc'
                if key in history and len(history[key]) > epoch:
                    epoch_values.append(history[key][epoch])
            if epoch_values:
                avg_pr_auc.append(np.mean(epoch_values))
        if avg_pr_auc:
            plt.plot(avg_pr_auc, label=f'{phase} avg PR AUC')
    plt.title('Average PR AUC Across All Diseases', fontsize=16)
    plt.xlabel('Epoch')
    plt.ylabel('Average PR AUC')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 6. Average F1 Score
    plt.subplot(2, 3, 6)
    for phase in ['train', 'test']:
        avg_f1 = []
        for epoch in range(len(history.get(f'train_loss', []))):
            epoch_values = []
            for disease in disease_labels:
                key = f'{phase}_{disease}_f1'
                if key in history and len(history[key]) > epoch:
                    epoch_values.append(history[key][epoch])
            if epoch_values:
                avg_f1.append(np.mean(epoch_values))
        if avg_f1:
            plt.plot(avg_f1, label=f'{phase} avg F1')
    plt.title('Average F1 Score Across All Diseases', fontsize=16)
    plt.xlabel('Epoch')
    plt.ylabel('Average F1 Score')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.suptitle('Unified Genomic Disease Prediction Model Performance', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plt.savefig(os.path.join(plots_dir, 'combined_metrics_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Individual disease metrics plots
    metrics = ['acc', 'auc', 'pr_auc', 'f1'] 
    phases = ['train', 'test']
    
    for metric in metrics:
        plt.figure(figsize=(15, 10))
        for i, disease in enumerate(disease_labels):
            plt.subplot(3, 2, i+1)
            for phase in phases:
                key = f'{phase}_{disease}_{metric}'
                if key in history and history[key]:
                    plt.plot(history[key], label=f'{phase} {metric}')
            plt.title(f'{disease} {metric.upper()}')
            plt.xlabel('Epoch')
            plt.ylabel(metric.upper())
            plt.ylim(0, 1)
            plt.legend()
            plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{metric}_by_disease.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # ROC Curves by Disease
    if final_metrics:
        plt.figure(figsize=(15, 10))
        for i, disease in enumerate(disease_labels):
            plt.subplot(3, 2, i+1)
            
            for phase in ['train', 'test']:
                roc_data = final_metrics[phase][disease]['roc_curve']
                if len(roc_data['fpr']) > 0 and len(roc_data['tpr']) > 0:
                    auc_score = final_metrics[phase][disease]['auc']
                    plt.plot(roc_data['fpr'], roc_data['tpr'], 
                            label=f'{phase} (AUC={auc_score:.3f})')
            
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{disease} - ROC Curve')
            plt.legend()
            plt.grid(alpha=0.3)
            
        plt.suptitle('ROC Curves by Disease', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(plots_dir, 'roc_curves_by_disease.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Precision-Recall Curves by Disease (if final_metrics provided)
    if final_metrics:
        plt.figure(figsize=(15, 10))
        for i, disease in enumerate(disease_labels):
            plt.subplot(3, 2, i+1)
            
            for phase in ['train', 'test']:
                pr_data = final_metrics[phase][disease]['pr_curve']
                if len(pr_data['precision']) > 0 and len(pr_data['recall']) > 0:
                    pr_auc_score = final_metrics[phase][disease]['pr_auc']
                    plt.plot(pr_data['recall'], pr_data['precision'], 
                            label=f'{phase} (PR-AUC={pr_auc_score:.3f})')
            
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'{disease} - Precision-Recall Curve')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.ylim(0, 1)
            plt.xlim(0, 1)
            
        plt.suptitle('Precision-Recall Curves by Disease', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(plots_dir, 'pr_curves_by_disease.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Summary metrics comparison plot
    if final_metrics:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        metrics_to_plot = ['auc', 'pr_auc', 'f1', 'acc']
        metric_names = ['ROC AUC', 'PR AUC', 'F1 Score', 'Accuracy']
        
        for idx, (metric, name) in enumerate(zip(metrics_to_plot, metric_names)):
            ax = axes[idx // 2, idx % 2]
            
            train_vals = [final_metrics['train'][disease][metric] for disease in disease_labels]
            test_vals = [final_metrics['test'][disease][metric] for disease in disease_labels]
            
            x = np.arange(len(disease_labels))
            width = 0.35
            
            ax.bar(x - width/2, train_vals, width, label='Train', alpha=0.8)
            ax.bar(x + width/2, test_vals, width, label='Test', alpha=0.8)
            
            ax.set_title(f'{name} by Disease', fontsize=14)
            ax.set_xlabel('Disease')
            ax.set_ylabel(name)
            ax.set_xticks(x)
            ax.set_xticklabels(disease_labels, rotation=45)
            ax.legend()
            ax.grid(alpha=0.3)
            ax.set_ylim(0, 1)
        
        plt.suptitle('Performance Metrics Comparison', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(plots_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

def write_results(model, hyperparameters, final_metrics, disease_labels, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    # Write text results
    with open(os.path.join(save_dir, 'experiment_results.txt'), 'w') as f:
        f.write("Hyperparameters:\n")
        f.write("----------------\n")
        for key, value in hyperparameters.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\nResults by Disease:\n")
        f.write("------------------\n")
        
        for phase in ['train', 'test']:
            f.write(f"\n{phase.upper()} SET RESULTS:\n")
            
            for disease in disease_labels:
                metrics = final_metrics[phase][disease]
                f.write(f"\n{disease}:\n")
                f.write(f"  Accuracy:    {metrics['acc']}\n")
                f.write(f"  Sensitivity: {metrics['sens']}\n")
                f.write(f"  Specificity: {metrics['spec']}\n")
                f.write(f"  ROC AUC:     {metrics['auc']:.5f}\n")
                f.write(f"  PR AUC:      {metrics['pr_auc']:.5f}\n")
                f.write(f"  F1 Score:    {metrics['f1']:.5f}\n")
                f.write(f"  Confusion Matrix:\n    {metrics['cm']}\n")
    
    # Write CSV results
    with open(os.path.join(save_dir, 'experiment_results.csv'), 'w', newline='') as csvfile:
        fieldnames = list(hyperparameters.keys())
        
        for phase in ['train', 'test']:
            for disease in disease_labels:
                for metric in ['acc', 'sens', 'spec', 'auc', 'pr_auc', 'f1']:
                    fieldnames.append(f"{phase}_{disease}_{metric}")
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        row = hyperparameters.copy()
        
        for phase in ['train', 'test']:
            for disease in disease_labels:
                metrics = final_metrics[phase][disease]
                for metric in ['acc', 'sens', 'spec']:
                    row[f"{phase}_{disease}_{metric}"] = metrics[metric]
                
                for metric in ['auc', 'pr_auc', 'f1']:
                    row[f"{phase}_{disease}_{metric}"] = metrics[metric]    

        writer.writerow(row)

def get_scheduler(scheduler_name, optimizer, args, train_files):
    steps_per_epoch = len(train_files) // args.bs
    total_steps = steps_per_epoch * args.epochs
    warmup_percentage = 0.1
    wsteps = int(total_steps * warmup_percentage)

    if scheduler_name.lower() == "none" or scheduler_name is None:
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)
    elif scheduler_name == "warmup_exponential":
        return WarmupExponential(optimizer, start_lr=args.lr, peak_lr=args.peak_lr, 
                                 final_lr=args.final_lr, warmup_steps=wsteps, 
                                 t_total=total_steps, decay_factor=args.df)
    elif scheduler_name == "exponential_decay":
        return ExponentialDecay(optimizer, start_lr=args.lr, final_lr=args.final_lr, 
                                total_steps=total_steps, decay_factor=args.df)
    elif scheduler_name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, 
                                                    factor=0.1, threshold=0.0001)
    elif scheduler_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    elif scheduler_name == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_name == "multistep":
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
    elif scheduler_name == "explr":
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_name}")

def main():
    print("Starting Unified Transformer-Mamba Genomic Disease Prediction...")
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    args = parse_args()
    
    # Validate configurations
    if args.use_multi_scale and args.multi_scale_mode == "hardcoded":
        validate_hardcoded_parameters(args)
    
    validate_sequence_config(args)

    id = str(args.ID)
    print(f"Experiment ID: {id}")
    experiment_dir = os.path.join(args.exp_dir, id)

    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Results will be saved to: {experiment_dir}")

    # Parse disease labels
    disease_labels = args.disease_labels if isinstance(args.disease_labels, list) else args.disease_labels.split(',')
    disease_labels = [label.strip() for label in disease_labels]
    print(f"Using disease labels: {disease_labels}")

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load phenotype data
    phenotype_data = pd.read_excel(args.phenotype_file)
    print(f"Phenotype data loaded, shape: {phenotype_data.shape}")

    # Handle NaNs in phenotype data
    print("\nChecking for NaNs in phenotype data:")
    for column in ['Agexit', 'Sex', 'Bmi_C'] + [f'PC{i}' for i in range(1, 11)]:
        if column in phenotype_data.columns:
            nan_count = phenotype_data[column].isna().sum()
            print(f"Column '{column}': {nan_count} NaN values ({nan_count/len(phenotype_data):.2%})")
            
            # If NaNs exist, fill them
            if nan_count > 0:
                if column == 'Sex':
                    # For categorical, use mode
                    fill_value = phenotype_data[column].mode()[0]
                else:
                    # For numerical, use mean
                    fill_value = phenotype_data[column].mean()
                    
                print(f"  Filling NaNs with {fill_value}")
                phenotype_data[column] = phenotype_data[column].fillna(fill_value)

    # Get list of genotype files and sort numerically by sample number
    file_list = glob.glob(os.path.join(args.genotype_dir, "sample_*.npy"))
    file_list.sort(key=lambda x: int(x.split('sample_')[1].split('.npy')[0]))
    print(f"Number of genotype files found: {len(file_list)}")

    # Get the unique sample IDs from the phenotype data
    phenotype_samples = set(phenotype_data['new_order'].unique())
    print(f"Number of unique samples in phenotype data: {len(phenotype_samples)}")

    # Filter the file list to only include files for samples in the phenotype data
    filtered_file_list = []
    for file_path in file_list:
        # Extract sample ID from filename
        sample_id = int(file_path.split('sample_')[1].split('.npy')[0])
        
        # Check if this sample ID is in the phenotype data
        if sample_id in phenotype_samples:
            filtered_file_list.append(file_path)

    print(f"Number of filtered genotype files matching phenotype data: {len(filtered_file_list)}")

    # Get input size
    first_genotype_file = filtered_file_list[0] if filtered_file_list else None
    if first_genotype_file:
        print(f"First genotype file: {first_genotype_file}")
        input_size = get_input_size(first_genotype_file)
        print(f"Dynamically determined input size: {input_size}")
    else:
        print("No matching genotype files found!")
        return

    # Split data
    train_files, test_files = train_test_split(filtered_file_list, test_size=0.2, random_state=42)
    print(f"Data split: Train {len(train_files)}, Test {len(test_files)}")

    # Create unified model
    print(f"\nCreating Unified Model with sequence mode: {args.sequence_mode}")

    # Get pooling configuration
    pooling_kwargs = get_pooling_kwargs(args)
    
    # Parse intermediate pooling layers
    intermediate_pooling_layers = []
    if args.use_intermediate_pooling and args.intermediate_pooling_layers:
        intermediate_pooling_layers = [int(x.strip()) for x in args.intermediate_pooling_layers.split(',')]
        print(f"Intermediate pooling will be applied after layers: {intermediate_pooling_layers}")
    
    # Prepare intermediate pooling kwargs
    intermediate_pooling_kwargs = {
        'pool_factor': args.intermediate_pool_factor
    }
    if args.intermediate_pooling_strategy == "chunked":
        intermediate_pooling_kwargs['num_chunks'] = args.intermediate_chunked_num_chunks
        intermediate_pooling_kwargs['pool_type'] = args.intermediate_chunked_pool_type
    
    if args.intermediate_pooling_strategy == "conv":
        intermediate_pooling_kwargs['target_length'] = args.intermediate_conv_target_length
        
    model = UnifiedGenomicModel(
        input_size=input_size,
        num_diseases=len(disease_labels),
        kernel_sizes=args.kernel_sizes,
        stride=args.stride,
        conv_channels=args.conv_channels,
        fc_layers=args.fc_layers,
        act=args.act,
        dropout_rate=args.dropout,
        use_covariates=bool(args.cov),
        use_age=bool(args.use_age),
        use_gender=bool(args.use_gender),
        use_bmi=bool(args.use_bmi),
        num_covariates=10,
        use_pooling=bool(args.use_pooling),
        pool_size=args.pool_size,
        pool_type=args.pool_type,
        use_multi_scale=bool(args.use_multi_scale),
        use_disease_attention=bool(args.use_disease_attention),
        use_separate_heads=bool(args.use_separate_heads),
        attention_heads=args.attention_heads,
        attention_dim=args.attention_dim,
        multi_scale_kernels=args.multi_scale_kernels,
        multi_scale_strides=args.multi_scale_strides,
        multi_scale_fusion=args.multi_scale_fusion,
        multi_scale_mode=args.multi_scale_mode,
        hardcoded_kernels=args.hardcoded_kernels,
        hardcoded_strides=args.hardcoded_strides,
        use_pointwise_conv=bool(args.use_pointwise_conv),
        pointwise_channels=args.pointwise_channels,
        # Unified sequence modeling parameters
        sequence_mode=args.sequence_mode,
        layer_config=args.layer_config,
        d_model=args.d_model,
        num_layers=args.num_layers,
        sequence_dropout=args.sequence_dropout,
        use_layer_norm=bool(args.use_layer_norm),
        hybrid_mode=args.hybrid_mode, 
        # Transformer parameters
        transformer_heads=args.transformer_heads,
        transformer_ff_dim=args.transformer_ff_dim,
        use_positional_encoding=bool(args.use_positional_encoding),
        max_seq_len=args.max_seq_len,
        init_from_pretrained=bool(args.init_from_pretrained),
        pretrained_model_name=args.pretrained_model_name,
        pretrained_model_type=args.pretrained_model_type,
        init_layers_fraction=args.init_layers_fraction,
        layer_init_strategy=args.layer_init_strategy,
        custom_layer_indices=args.custom_layer_indices,
        # Mamba parameters
        mamba_d_state=args.mamba_d_state,
        mamba_d_conv=args.mamba_d_conv,
        mamba_expand=args.mamba_expand,
        # Token and pooling parameters
        use_cls_token=bool(args.use_cls_token),
        use_covariate_tokens=bool(args.use_covariate_tokens),
        covariate_embed_dim=args.covariate_embed_dim,
        covariate_token_strategy=args.covariate_token_strategy,
        pooling_strategy=args.pooling_strategy,
        pooling_kwargs=pooling_kwargs,
        # Intermediate pooling parameters
        use_intermediate_pooling=bool(args.use_intermediate_pooling),
        intermediate_pooling_strategy=args.intermediate_pooling_strategy,
        intermediate_pooling_layers=intermediate_pooling_layers,
        intermediate_pooling_kwargs=intermediate_pooling_kwargs
    )

    model = model.to(device)
    print("Model created and moved to device")
    
    # Save model architecture
    with open(os.path.join(experiment_dir, 'model_architecture.txt'), 'w') as file:
        file.write(str(model))
    print(model)

    # Set up loss function (BCELoss for multilabel)
    criterion = nn.BCEWithLogitsLoss()

    # Set up optimizer
    optimizer_map = {
        "adadelta": optim.Adadelta(model.parameters(), lr=args.lr),
        "adagrad": optim.Adagrad(model.parameters(), lr=args.lr),
        "adamw": optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd),
        "rmsprop": optim.RMSprop(model.parameters(), lr=args.lr),
        "sgd": optim.SGD(model.parameters(), lr=args.lr),
        "adam": optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    }
    optimizer = optimizer_map.get(args.opt.lower())
    if optimizer is None:
        raise ValueError(f"Optimizer {args.opt} not supported")
    
    # Initialize variables for checkpoint loading
    start_epoch = 0
    history = None
    best_loss = float('inf')

    # Check for existing checkpoints
    if args.resume:
        latest_checkpoint = find_latest_checkpoint(experiment_dir)
        if latest_checkpoint:
            print(f"Loading checkpoint: {latest_checkpoint}")
            try:
                checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
                
                # Load model state
                model.load_state_dict(checkpoint['model_state_dict'])
                print("Loaded model weights from checkpoint")                
                 # Load optimizer state
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Loaded optimizer state from checkpoint")
                
                # Get the start epoch and history
                start_epoch = checkpoint.get('epoch', 0)
                history = checkpoint.get('history', None)
                best_loss = checkpoint.get('best_loss', float('inf'))
                
                print(f"Resuming from epoch {start_epoch}")
                print(f"Best validation loss so far: {best_loss:.6f}")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Starting training from scratch.")
                start_epoch = 0
                history = None
                best_loss = float('inf')
        else:
            print("No checkpoints found. Starting training from scratch.")
    else:
        print("Resume flag is disabled. Starting training from scratch.")

    # Create datasets
    train_dataset = MultilabelGenotypeDataset(
        train_files, 
        phenotype_data, 
        disease_labels, 
        use_covariates=bool(args.cov),
        use_age=bool(args.use_age),
        use_gender=bool(args.use_gender),
        use_bmi=bool(args.use_bmi),
        norm_age=args.norm_age,
        norm_pcs=args.norm_pcs,
        norm_gender=args.norm_gender,
        norm_bmi=args.norm_bmi,
        fit_normalizers=True,
        normalizers=None)

    fitted_normalizers = train_dataset.get_normalizers()
    
    test_dataset = MultilabelGenotypeDataset(
        test_files,
        phenotype_data, 
        disease_labels, 
        use_covariates=bool(args.cov),
        use_age=bool(args.use_age),
        use_gender=bool(args.use_gender),
        use_bmi=bool(args.use_bmi),
        norm_age=args.norm_age,
        norm_pcs=args.norm_pcs,
        norm_gender=args.norm_gender,
        norm_bmi=args.norm_bmi,
        fit_normalizers=False,
        normalizers=fitted_normalizers
    )

    # Create dataloaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True),
        'test': DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    }
    print("DataLoaders created")
    
    # Create scheduler
    scheduler = get_scheduler(args.sch, optimizer, args, train_files)
    
    # Load scheduler state if resuming
    if args.resume and latest_checkpoint and 'scheduler_state_dict' in checkpoint:
        try:
            if isinstance(scheduler, (WarmupExponential, ExponentialDecay)):
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            elif hasattr(scheduler, 'load_state_dict'):
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Loaded scheduler state from checkpoint")
        except Exception as e:
            print(f"Warning: Failed to load scheduler state: {e}")

    print(f"Using optimizer: {args.opt}")
    print(f"Using scheduler: {args.sch}")

    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta,
        verbose=True
    )
    
    # Load early stopping state if resuming
    if args.resume and latest_checkpoint and 'early_stopping_state' in checkpoint:
        try:
            early_stopping.load_state_dict(checkpoint['early_stopping_state'])
            print("Loaded early stopping state from checkpoint")
        except Exception as e:
            print(f"Warning: Failed to load early stopping state: {e}")

    # Train model
    start_time = time.time()
    model, history, final_metrics, all_preds, all_labels, completed_epochs = train_multilabel_model(
        model, dataloaders, criterion, optimizer, scheduler, 
        args.epochs, disease_labels, device, early_stopping=early_stopping,
        checkpoint_dir=experiment_dir, start_epoch=start_epoch,
        keep_last_n=args.keep_checkpoints, history=history, initial_best_loss=best_loss
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Plot metrics
    plot_multilabel_metrics(history, disease_labels, experiment_dir, final_metrics)
    print("Metrics plotted")

    # Prepare hyperparameter dict for results
    hyperparameters = {
        'Exp_ID': id,
        'Sequence_Mode': args.sequence_mode,
        'Layer_Config': args.layer_config if args.sequence_mode == "hybrid" else 'N/A',
        'D_Model': args.d_model,
        'Num_Layers': args.num_layers,
        'Batch_Size': args.bs,
        'Epochs': args.epochs,
        'Completed_Epochs': completed_epochs,
        'Start_LR': args.lr,
        'Final_LR': optimizer.param_groups[0]["lr"],
        'Dropout': args.dropout,
        'Sequence_Dropout': args.sequence_dropout,
        'Act': args.act,
        'Opt': args.opt,
        'Sch': args.sch,
        'WD': args.wd,
        'DF': args.df,
        'Use_PCs': bool(args.cov),
        'Use_Age': bool(args.use_age),
        'Use_Gender': bool(args.use_gender),
        'Use_Bmi': bool(args.use_bmi),
        'Kernel_sizes': str(args.kernel_sizes),
        'Stride': str(args.stride),
        'Conv_channels': str(args.conv_channels),
        'Use_Pooling': bool(args.use_pooling),
        'Pool_size': args.pool_size if args.use_pooling else 'N/A',
        'Pool_type': args.pool_type if args.use_pooling else 'N/A',
        'FC_layers': str(args.fc_layers),
        'Use_Multi_Scale': bool(args.use_multi_scale),
        'Use_Disease_Attention': bool(args.use_disease_attention),
        'Use_Separate_Heads': bool(args.use_separate_heads),
        'Use_Covariate_Tokens': bool(args.use_covariate_tokens),
        'Pooling_Strategy': args.pooling_strategy,
        'Chunked_Num_Chunks': args.chunked_num_chunks if args.pooling_strategy == 'chunked' else 'N/A',
        'Chunked_Pool_Type': args.chunked_pool_type if args.pooling_strategy == 'chunked' else 'N/A',
        'MultiScale_Window_Sizes': str(args.multiscale_window_sizes) if args.pooling_strategy == 'multiscale' else 'N/A',
        'MultiHead_Num_Heads': args.multihead_num_heads if args.pooling_strategy == 'multihead' else 'N/A',
        'MultiHead_Head_Dim': args.multihead_head_dim if args.pooling_strategy == 'multihead' else 'N/A',
        'Conv_Target_Length': args.conv_target_length if args.pooling_strategy == 'conv' else 'N/A',
        'Conv_Num_Layers': args.conv_num_layers if args.pooling_strategy == 'conv' else 'N/A',
        'Hierarchical_Levels': str(args.hierarchical_levels) if args.pooling_strategy == 'hierarchical' else 'N/A',
        'Num_Diseases': len(disease_labels),
        'Disease_Labels': ','.join(disease_labels),
        # Transformer-specific
        'Transformer_Heads': args.transformer_heads if args.sequence_mode in ["transformer", "hybrid"] else 'N/A',
        'Transformer_FF_Dim': args.transformer_ff_dim if args.sequence_mode in ["transformer", "hybrid"] else 'N/A',
        'Use_Positional_Encoding': bool(args.use_positional_encoding) if args.sequence_mode in ["transformer", "hybrid"] else 'N/A',
        'Init_From_Pretrained': bool(args.init_from_pretrained) if args.sequence_mode in ["transformer", "hybrid"] else 'N/A',
        # Mamba-specific
        'Mamba_D_State': args.mamba_d_state if args.sequence_mode in ["mamba", "hybrid"] else 'N/A',
        'Mamba_D_Conv': args.mamba_d_conv if args.sequence_mode in ["mamba", "hybrid"] else 'N/A',
        'Mamba_Expand': args.mamba_expand if args.sequence_mode in ["mamba", "hybrid"] else 'N/A',
    }

    # Write results
    write_results(model, hyperparameters, final_metrics, disease_labels, experiment_dir)
    print("Results written to file")

    # Print final results summary
    for disease in disease_labels:
        print(f"\n{disease.upper()}:")
        print(f"Train - ROC AUC: {final_metrics['train'][disease]['auc']:.4f}, "
              f"PR AUC: {final_metrics['train'][disease]['pr_auc']:.4f}, "
              f"F1: {final_metrics['train'][disease]['f1']:.4f}, "
              f"Accuracy: {final_metrics['train'][disease]['acc']}")
        print(f"Test  - ROC AUC: {final_metrics['test'][disease]['auc']:.4f}, "
              f"PR AUC: {final_metrics['test'][disease]['pr_auc']:.4f}, "
              f"F1: {final_metrics['test'][disease]['f1']:.4f}, "
              f"Accuracy: {final_metrics['test'][disease]['acc']}")

if __name__ == '__main__':
    start_time = time.time()
    
    main()
    
    end_time = time.time()
    total_runtime = end_time - start_time
    
    print(f"\nTotal script runtime: {total_runtime:.2f} seconds")
    hours, rem = divmod(total_runtime, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Total runtime: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")