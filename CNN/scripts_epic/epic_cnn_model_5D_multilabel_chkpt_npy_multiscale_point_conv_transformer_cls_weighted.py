import os
import pandas as pd
import gzip
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import mmap
import copy
import glob
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer
from torch.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
import argparse
import csv
import time
import shutil
import torch.nn.functional as F
from transformers import ViTConfig, ViTModel, BertModel, BertConfig

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
    
def parse_args():
    parser = argparse.ArgumentParser(description="Multilabel Genotype Model Training")
    parser.add_argument("-ID", type=str, default="Exp_01_smooth_decay", help="ID of the experiment")
    parser.add_argument("-exp_dir", type=str, default='/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/5d_multilabel/multiscale', help="Directory to save experiment results")
    parser.add_argument("-genotype_dir", type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/sampled_data_5M_unq_npy', help="Directory containing genotype files")
    parser.add_argument("-phenotype_file", type=str, default='/vol/vssp/SF_ucdatasets/gwas/data_files/merged_v8_pcs_chip_added_Iqra_1_cleaned.xlsx', help="Path to phenotype file")

    # Model and training parameters
    parser.add_argument("-bs", type=int, default=5, help="Batch size for training")
    parser.add_argument("-dropout", type=float, default=0.5, help="Dropout rate for the model")
    parser.add_argument("-epochs", type=int, default=3, help="Number of epochs for training")
    parser.add_argument("-lr", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("-peak_lr", type=float, default=1e-2, help="Peak learning rate for WarmupExponential scheduler")
    parser.add_argument("-final_lr", type=float, default=1e-5, help="Final learning rate for custom schedulers")
    parser.add_argument("-act", type=str, default="gelu", choices=["tanh","relu","gelu"], help="Activation function for the model")
    parser.add_argument("-sch", type=str, default="exponential_decay", choices=["none","plateau", "cosine", "step","multistep","explr","warmup_exponential", "exponential_decay"], help="Learning rate scheduler")
    parser.add_argument("-df", type=float, default=2.0, help="Decay factor for custom schedulers")
    parser.add_argument("-opt", type=str, default="adamw", choices=["adam", "adamw", "sgd"], help="Optimizer to use")
    parser.add_argument("-wd", type=float, default=0.5, help="Weight decay for optimizer")

    # Model architecture
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
    parser.add_argument("-multi_scale_fusion", type=str, default="parallel", choices=["cross_scale", "parallel"], help="Multi-scale fusion strategy: cross_scale (branches see all scales) or parallel (independent branches)")

    # Multi-scale mode selection
    parser.add_argument("-multi_scale_mode", type=str, default="hardcoded", choices=["progressive", "hardcoded"], help="Multi-scale mode: 'progressive' (kernel//2^i, stride//2^i) or 'hardcoded' (explicit values for each layer)")
    
    # Hardcoded multi-scale parameters (used when multi_scale_mode="hardcoded")
    parser.add_argument("-hardcoded_kernels", type=parse_nested_int_list, default='16,128,1024;16,64,512;16,32,256', help="Hardcoded kernel sizes for all layers and branches. Format: 'layer1_branch1,layer1_branch2;layer2_branch1,layer2_branch2'. Example: '15,63,255;7,31,127;3,15,63'")
    parser.add_argument("-hardcoded_strides", type=parse_nested_int_list, default='16,16,16;16,16,16;16,16,16', help="Hardcoded stride values for all layers and branches. Format: 'layer1_branch1,layer1_branch2;layer2_branch1,layer2_branch2'. Example: '4,16,64;2,8,32;1,4,16'")

    # Pointwise convolution parameters
    parser.add_argument("-use_pointwise_conv", type=int, default=0, choices=[0, 1], help="Whether to use pointwise (1x1) convolution after each branch before concatenation (0: no, 1: yes)")
    parser.add_argument("-pointwise_channels", type=int, default=4, help="Number of output channels for pointwise convolution (applied to each branch)")

    # Data-specific parameters
    parser.add_argument("-cov", type=int, default=1, choices=[0, 1], help="Whether to include PC's as covariates in the model (0: no, 1: yes)")
    parser.add_argument("-use_age", type=int, default=1, choices=[0, 1], help="Whether to include age in covariates (0: no, 1: yes)")
    parser.add_argument("-use_gender", type=int, default=1, choices=[0, 1], help="Whether to include gender in covariates (0: no, 1: yes)")
    parser.add_argument("-use_bmi", type=int, default=1, choices=[0, 1], help="Whether to include BMI in covariates in the model (0: no, 1: yes)")

    # Early stopping parameters
    parser.add_argument("-patience", type=int, default=15, help="Patience for early stopping")
    parser.add_argument("-min_delta", type=float, default=1e-4, help="Minimum change for early stopping")

    # Normalization-related arguments
    parser.add_argument("-norm_age", type=str, default="none", choices=["none", "standard", "minmax", "robust", "quantile", "power"], help="Normalization method for age")
    parser.add_argument("-norm_pcs", type=str, default="none", choices=["none", "standard", "minmax", "robust", "quantile", "power"], help="Normalization method for PCs")
    parser.add_argument("-norm_gender", type=str, default="none", choices=["none", "minmax"], help="Normalization method for gender (usually keep as none)")
    parser.add_argument("-norm_bmi", type=str, default="none", choices=["none", "standard", "minmax", "robust", "quantile", "power"], help="Normalization method for BMI")
    parser.add_argument("-disease_labels", type=parse_str_list, default="pros01,panca,crc,breacancer,t2dm", help="Comma-separated list of disease column names in phenotype file")

    parser.add_argument("-use_pooling", type=int, default=0, choices=[0, 1], help="Whether to use Pooling after convolution layers (0: no, 1: yes)")
    parser.add_argument("-pool_size", type=int, default=256, help="Size of the adaptive pooling output")

    # Pool Type for all pooling if used
    parser.add_argument("-pool_type", type=str, default="max", choices=["max", "avg"], help="Type of adaptive pooling: 'max' for AdaptiveMaxPool1d, 'avg' for AdaptiveAvgPool1d")
    
    # Transformer-related arguments
    parser.add_argument("-use_transformer", type=int, default=1, choices=[0, 1], help="Whether to use transformer layers after convolution (0: no, 1: yes)")
    parser.add_argument("-transformer_layers", type=int, default=2, help="Number of transformer encoder layers")
    parser.add_argument("-transformer_heads", type=int, default=8, help="Number of attention heads in transformer")
    parser.add_argument("-transformer_dim", type=int, default=384, help="Transformer model dimension (d_model)")
    parser.add_argument("-transformer_ff_dim", type=int, default=1024, help="Transformer feedforward dimension")
    parser.add_argument("-transformer_dropout", type=float, default=0.1, help="Transformer dropout rate")
    parser.add_argument("-use_positional_encoding", type=int, default=1, choices=[0, 1], help="Whether to use positional encoding in transformer (0: no, 1: yes)")
    parser.add_argument("-max_seq_len", type=int, default=10000, help="Maximum sequence length for positional encoding")

    # Pretrained weight initialization arguments
    parser.add_argument("-init_from_pretrained", type=int, default=1, choices=[0, 1], help="Whether to initialize transformer weights from pretrained model (0: no, 1: yes)")
    parser.add_argument("-pretrained_model_type", type=str, default="bert", choices=["auto", "vit", "bert"], help="Type of pretrained model: 'auto' (detect from name), 'vit', or 'bert'")
    parser.add_argument("-pretrained_model_name", type=str, default="bert-base-uncased", choices=["WinKawaks/vit-small-patch16-224", "zhihan1996/DNABERT-2-117M", "bert-base-uncased"], help="Name of pretrained model for weight initialization")
    parser.add_argument("-init_layers_fraction", type=float, default=1.0, help="Fraction of transformer layers to initialize from pretrained (0.0-1.0)")

    # Layer selection strategy arguments
    parser.add_argument("-layer_init_strategy", type=str, default="middle", choices=["first", "middle", "last", "random", "custom"], help="Strategy for selecting which pretrained layers to use: first (0,1,2...), middle (center layers), last (end layers), random (random selection), custom (specify indices)")
    parser.add_argument("-custom_layer_indices", type=str, default="4,6", help="Comma-separated list of pretrained layer indices to use (e.g., '4,5,6'). Only used when layer_init_strategy='custom'")
    
    parser.add_argument("-use_cls_token", type=int, default=0, choices=[0, 1], help="Whether to use class token for classification (0: global pooling, 1: cls token)")
    parser.add_argument("-use_covariate_tokens", type=int, default=1, choices=[0, 1], help="Whether to process covariates as tokens in transformer (0: concatenate at end, 1: as tokens)")
    parser.add_argument("-covariate_token_strategy", type=str, default="combined", choices=["separate", "combined"], help="Covariate tokenization strategy: 'separate' (one token per covariate) or 'combined' (single token for all covariates)")
    parser.add_argument("-pooling_strategy", type=str, default="max", choices=["mean", "max", "attention", "concat"], help="Pooling strategy when not using cls token: 'mean' (avg pool), 'max' (max pool), 'attention' (learnable attention), 'concat' (flatten + project)")
    parser.add_argument("-covariate_embed_dim", type=int, default=64, help="Embedding dimension for covariate tokens")
    
    # Checkpoint-related parameters
    parser.add_argument("-resume", type=int, default=1, choices=[0, 1], help="Whether to resume from checkpoint if available (0: no, start fresh; 1: yes, resume if available)")
    parser.add_argument("-keep_checkpoints", type=int, default=1, help="Number of recent checkpoints to keep")
    
    # Class imbalance handling arguments
    parser.add_argument("-use_weighted_loss", type=int, default=1, choices=[0, 1], help="Whether to use weighted BCE loss to handle class imbalance (0: no, 1: yes)")
    parser.add_argument("-weight_strategy", type=str, default="balanced", choices=["inverse_freq", "balanced", "custom"], help="Strategy for calculating class weights")
    parser.add_argument("-custom_weights", type=str, default="1.0,1.0,1.0,1.0,1.0",  help="Custom weights for each disease (comma-separated, must match number of diseases)")
    parser.add_argument("-weight_smoothing", type=float, default=0.1, help="Smoothing factor for weight calculation (0.0 = no smoothing, 1.0 = uniform weights)")
    
    return parser.parse_args()

def calculate_class_weights(dataset, disease_labels, strategy="inverse_freq", custom_weights=None, smoothing=0.1):
    print(f"\nCalculating class weights using strategy: {strategy}")
    
    if strategy == "custom" and custom_weights is not None:
        weights = torch.tensor(custom_weights, dtype=torch.float32)
        print(f"Using custom weights: {weights.tolist()}")
        return weights
    
    # Collect all labels at once - much more efficient
    all_labels = []
    print("Collecting all labels...")
    for i in range(len(dataset)):
        _, _, labels = dataset[i]
        all_labels.append(labels)
    
    # Stack into a single tensor: [n_samples, n_diseases]
    all_labels_tensor = torch.stack(all_labels)
    total_samples = len(dataset)
    
    # Calculate weights using vectorized operations
    weights = []
    print("\nClass distribution and calculated weights:")
    print(f"{'Disease':<15} {'Positive':<8} {'Negative':<8} {'Pos%':<6} {'Weight':<8}")
    print("-" * 55)
    
    for j, disease in enumerate(disease_labels):
        pos_count = all_labels_tensor[:, j].sum().item()  # Sum of 1s = positive count
        neg_count = total_samples - pos_count             # Total - positives = negatives
        pos_ratio = pos_count / total_samples
        
        # Calculate weight based on strategy
        if strategy == "inverse_freq":
            weight = neg_count / pos_count if pos_count > 0 else 1.0
        elif strategy == "balanced":
            weight = total_samples / (2 * pos_count) if pos_count > 0 else 1.0
        
        # Apply smoothing
        if smoothing > 0:
            weight = weight * (1 - smoothing) + 1.0 * smoothing
            
        weights.append(weight)
        print(f"{disease:<15} {pos_count:<8} {neg_count:<8} {pos_ratio*100:<5.1f}% {weight:<8.2f}")
    
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    print(f"\nFinal weights tensor: {weights_tensor.tolist()}")
    return weights_tensor
    
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
        
        # Update multi_scale_kernels and strides to match the first layer of hardcoded values
        args.multi_scale_kernels = args.hardcoded_kernels[0]
        args.multi_scale_strides = args.hardcoded_strides[0]
        
        print(f"Hardcoded multi-scale mode validated:")
        print(f"  - Number of layers: {len(args.hardcoded_kernels)}")
        print(f"  - Number of branches per layer: {num_branches}")
        print(f"  - Hardcoded kernels: {args.hardcoded_kernels}")
        print(f"  - Hardcoded strides: {args.hardcoded_strides}")

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
                    demographics.append(f"Bmi: {bmi}")
                
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

class AttentionPooling(nn.Module):
    """Learnable attention-based pooling for transformer outputs"""
    
    def __init__(self, input_dim, hidden_dim=256):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, input_dim]
        Returns:
            pooled: [batch_size, input_dim]
        """
        # Compute attention weights
        attention_weights = self.attention(x)  # [batch_size, seq_len, 1]
        attention_weights = self.softmax(attention_weights)  # [batch_size, seq_len, 1]
        
        # Apply attention weights
        pooled = torch.sum(x * attention_weights, dim=1)  # [batch_size, input_dim]
        
        return pooled

class CovariateTokenEmbedder(nn.Module):
    """Convert covariates into tokens for transformer processing"""
    
    def __init__(self, transformer_dim, embed_dim=64, strategy="separate", use_age=True, use_gender=True, use_bmi=True, use_pcs=True):
        super(CovariateTokenEmbedder, self).__init__()
        
        self.use_age = use_age
        self.use_gender = use_gender  
        self.use_bmi = use_bmi
        self.use_pcs = use_pcs
        self.transformer_dim = transformer_dim
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
                    nn.Linear(embed_dim, transformer_dim)
                )
                
            if use_gender:
                self.embedders['gender'] = nn.Sequential(
                    nn.Linear(1, embed_dim),
                    nn.GELU(), 
                    nn.Linear(embed_dim, transformer_dim)
                )
                
            if use_bmi:
                self.embedders['bmi'] = nn.Sequential(
                    nn.Linear(1, embed_dim),
                    nn.GELU(),
                    nn.Linear(embed_dim, transformer_dim)
                )
                
            if use_pcs:
                self.embedders['pcs'] = nn.Sequential(
                    nn.Linear(10, embed_dim),  # 10 PCs
                    nn.GELU(),
                    nn.Linear(embed_dim, transformer_dim)
                )
            
            # Token type embeddings for separate tokens (optional - helps transformer distinguish token types)
            self.token_type_embeddings = nn.Embedding(5, transformer_dim)  # genomic, age, gender, bmi, pcs
        
        elif strategy == "combined":
            # Single embedder for all covariates combined
            self.combined_embedder = nn.Sequential(
                nn.Linear(total_cov_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, transformer_dim)
            )
            
            # Single token type embedding for combined token
            self.token_type_embeddings = nn.Embedding(2, transformer_dim)  # genomic, combined_covariates
        
        print(f"  CovariateTokenEmbedder created:")
        print(f"    - Strategy: {strategy}")
        if strategy == "separate":
            print(f"    - Age tokens: {'Yes' if use_age else 'No'}")
            print(f"    - Gender tokens: {'Yes' if use_gender else 'No'}")
            print(f"    - BMI tokens: {'Yes' if use_bmi else 'No'}")
            print(f"    - PC tokens: {'Yes' if use_pcs else 'No'}")
        else:  # combined
            print(f"    - Combined token dimensions: {total_cov_dim}")
            enabled_covs = []
            if use_pcs: enabled_covs.append("PCs(10)")
            if use_age: enabled_covs.append("Age(1)")
            if use_gender: enabled_covs.append("Gender(1)")
            if use_bmi: enabled_covs.append("BMI(1)")
            print(f"    - Included covariates: {', '.join(enabled_covs)}")
        print(f"    - Embed dim: {embed_dim} -> Transformer dim: {transformer_dim}")
    
    def forward(self, covariates_dict):
        """
        Args:
            covariates_dict: Dictionary with keys 'age', 'gender', 'bmi', 'pcs'
                           Each value is [batch_size, feature_dim]
        Returns:
            covariate_tokens: [batch_size, num_tokens, transformer_dim]
            token_types: [batch_size, num_tokens] for token type embeddings
        """
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
        """Create separate tokens for each covariate type"""
        tokens = []
        token_types = []
        token_idx = 1  # 0 reserved for genomic tokens
        
        if self.use_age and 'age' in covariates_dict:
            age_token = self.embedders['age'](covariates_dict['age'].unsqueeze(-1))  # [batch, transformer_dim]
            age_token = age_token.unsqueeze(1)  # [batch, 1, transformer_dim]
            tokens.append(age_token)
            token_types.append(torch.full((batch_size, 1), token_idx, device=age_token.device))
            token_idx += 1
            
        if self.use_gender and 'gender' in covariates_dict:
            gender_token = self.embedders['gender'](covariates_dict['gender'].unsqueeze(-1))  # [batch, transformer_dim]
            gender_token = gender_token.unsqueeze(1)  # [batch, 1, transformer_dim]
            tokens.append(gender_token)
            token_types.append(torch.full((batch_size, 1), token_idx, device=gender_token.device))
            token_idx += 1
            
        if self.use_bmi and 'bmi' in covariates_dict:
            bmi_token = self.embedders['bmi'](covariates_dict['bmi'].unsqueeze(-1))  # [batch, transformer_dim]
            bmi_token = bmi_token.unsqueeze(1)  # [batch, 1, transformer_dim]
            tokens.append(bmi_token)
            token_types.append(torch.full((batch_size, 1), token_idx, device=bmi_token.device))
            token_idx += 1
            
        if self.use_pcs and 'pcs' in covariates_dict:
            pcs_token = self.embedders['pcs'](covariates_dict['pcs'])  # [batch, transformer_dim]
            pcs_token = pcs_token.unsqueeze(1)  # [batch, 1, transformer_dim]
            tokens.append(pcs_token)
            token_types.append(torch.full((batch_size, 1), token_idx, device=pcs_token.device))
            token_idx += 1
        
        if tokens:
            covariate_tokens = torch.cat(tokens, dim=1)  # [batch_size, num_covariate_tokens, transformer_dim]
            token_type_ids = torch.cat(token_types, dim=1)  # [batch_size, num_covariate_tokens]
            
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
            combined_token = self.combined_embedder(combined_covariates)  # [batch_size, transformer_dim]
            combined_token = combined_token.unsqueeze(1)  # [batch_size, 1, transformer_dim]
            
            # Token type for combined covariate token
            token_type_ids = torch.full((batch_size, 1), 1, device=combined_token.device)  # 1 for combined covariates
            
            # Add token type embedding
            type_embeddings = self.token_type_embeddings(token_type_ids)
            combined_token = combined_token + type_embeddings
            
            return combined_token, token_type_ids
        else:
            return None, None

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
        if name == 'tanh':
            return nn.Tanh()
        elif name == 'relu':
            return nn.ReLU()
        elif name == 'gelu':
            return nn.GELU()
        else:
            return nn.ReLU()
    
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
                        elif self.pool_type.lower() == "avg":
                            self.branch_pools[i] = nn.AdaptiveAvgPool1d(target_length).to(output.device)
                        else:
                            self.branch_pools[i] = nn.AdaptiveMaxPool1d(target_length).to(output.device)
                
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

class ParallelMultiScaleConvBlock(nn.Module):
    """Parallel multi-scale convolution block that processes all branches and concatenates"""
    def __init__(self, in_channels, out_channels, kernel_sizes, strides, act, dropout_rate, layer_idx, is_final_layer, 
                pool_type="max", use_pointwise_conv=False, pointwise_channels=16):
        super(ParallelMultiScaleConvBlock, self).__init__()
        
        self.branches = nn.ModuleList()
        self.layer_idx = layer_idx
        self.is_final_layer = is_final_layer
        self.pool_type = pool_type
        self.num_branches = len(kernel_sizes)
        self.use_pointwise_conv = use_pointwise_conv
        self.pointwise_channels = pointwise_channels
        
        # Create parallel branches for this layer
        for branch_idx, (kernel_size, stride) in enumerate(zip(kernel_sizes, strides)):
            
            # Ensure proper padding
            padding = kernel_size // 2
            
            # Main convolution layers (no pointwise here)
            branch = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 
                         kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm1d(out_channels),
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
        
        # Initialize empty ModuleList for final pooling layers (only for final layer)
        if self.is_final_layer:
            self.final_pools = nn.ModuleList()
            self.final_pooling_needed = [False] * self.num_branches
            self._final_pools_initialized = False
    
    def get_activation(self, name):
        if name == 'tanh':
            return nn.Tanh()
        elif name == 'relu':
            return nn.ReLU()
        elif name == 'gelu':
            return nn.GELU()
        else:
            return nn.ReLU()
    
    def _initialize_final_pooling_layers(self, target_length, device):
        """Initialize final pooling layers only for branches that need them"""
        if not self.is_final_layer or self._final_pools_initialized:
            return
            
        # Clear any existing pooling layers
        self.final_pools = nn.ModuleList()
        
        # Add pooling layers only where needed
        for i in range(self.num_branches):
            if self.final_pooling_needed[i]:
                if self.pool_type.lower() == "max":
                    pool_layer = nn.AdaptiveMaxPool1d(target_length).to(device)
                elif self.pool_type.lower() == "avg":
                    pool_layer = nn.AdaptiveAvgPool1d(target_length).to(device)
                else:
                    pool_layer = nn.AdaptiveMaxPool1d(target_length).to(device)
                self.final_pools.append(pool_layer)
            else:
                # Add Identity layer as placeholder
                self.final_pools.append(nn.Identity())
        
        self._final_pools_initialized = True
    
    def forward(self, x):
        if isinstance(x, list):
            # x is a list of tensors from previous parallel layer
            branch_inputs = x
        else:
            # x is a single tensor (first layer), replicate for each branch
            branch_inputs = [x for _ in range(len(self.branches))]
        
        branch_outputs = []
        
        # Process each branch through main convolutions independently
        for branch_idx, (branch, branch_input) in enumerate(zip(self.branches, branch_inputs)):
            output = branch(branch_input)
            branch_outputs.append(output)
        
        if self.is_final_layer:
            # Final layer: apply pointwise conv first (if enabled), then standardize lengths and concatenate
            
            # Apply pointwise convolution if enabled
            if self.use_pointwise_conv:
                pointwise_outputs = []
                for i, output in enumerate(branch_outputs):
                    pointwise_output = self.pointwise_branches[i](output)
                    pointwise_outputs.append(pointwise_output)
                branch_outputs = pointwise_outputs
            
            # Standardize lengths
            output_lengths = [output.size(2) for output in branch_outputs]
            target_length = min(output_lengths)
            
            # Determine which branches need pooling
            for i, length in enumerate(output_lengths):
                self.final_pooling_needed[i] = (length != target_length)
            
            # Initialize final pooling layers if not done already
            if not self._final_pools_initialized:
                self._initialize_final_pooling_layers(target_length, branch_outputs[0].device)
            
            standardized_outputs = []
            for i, output in enumerate(branch_outputs):
                if self.final_pooling_needed[i]:
                    # Update pooling layer size if target length changed
                    if hasattr(self.final_pools[i], 'output_size'):
                        if self.final_pools[i].output_size != target_length:
                            if self.pool_type.lower() == "max":
                                self.final_pools[i] = nn.AdaptiveMaxPool1d(target_length).to(output.device)
                            elif self.pool_type.lower() == "avg":
                                self.final_pools[i] = nn.AdaptiveAvgPool1d(target_length).to(output.device)
                            else:
                                self.final_pools[i] = nn.AdaptiveMaxPool1d(target_length).to(output.device)
                    
                    output = self.final_pools[i](output)
                # If no pooling needed, use output as-is
                standardized_outputs.append(output)
            
            # Concatenate along channel dimension
            result = torch.cat(standardized_outputs, dim=1)
            return result
        else:
            # Intermediate layer: return list of tensors for next layer
            return branch_outputs

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
        if name == 'tanh':
            return nn.Tanh()
        elif name == 'relu':
            return nn.ReLU()
        elif name == 'gelu':
            return nn.GELU()
        else:
            return nn.ReLU()
    
    def forward(self, x):
        return self.head(x)

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer layers"""
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

class GenomicTransformerBlock(nn.Module):
    """Enhanced genomic transformer with class token and covariate integration supporting both ViT and BERT"""
    
    def __init__(self, input_dim, transformer_dim, num_layers=2, num_heads=4, 
                 ff_dim=1024, dropout=0.1, use_positional_encoding=True, max_seq_len=10000,
                 use_cls_token=True, use_covariate_tokens=False, 
                 covariate_embed_dim=64, covariate_token_strategy="separate", pooling_strategy="mean",
                 use_age=True, use_gender=True, use_bmi=True, use_pcs=True,
                 init_from_pretrained=False, pretrained_model_name="WinKawaks/vit-small-patch16-224",
                 pretrained_model_type="auto", 
                 init_layers_fraction=1.0, layer_init_strategy="middle", custom_layer_indices=""):
        super(GenomicTransformerBlock, self).__init__()
        
        self.input_dim = input_dim
        self.transformer_dim = transformer_dim
        self.num_layers = num_layers
        self.use_positional_encoding = use_positional_encoding
        self.init_from_pretrained = init_from_pretrained
        self.pretrained_model_name = pretrained_model_name
        self.pretrained_model_type = pretrained_model_type
        self.init_layers_fraction = init_layers_fraction
        self.layer_init_strategy = layer_init_strategy
        self.custom_layer_indices = custom_layer_indices

        self.use_cls_token = use_cls_token
        self.use_covariate_tokens = use_covariate_tokens
        self.covariate_token_strategy = covariate_token_strategy
        self.pooling_strategy = pooling_strategy
        
        print(f"  Creating Enhanced Genomic Transformer:")
        print(f"    - Class token: {'Enabled' if use_cls_token else 'Disabled (using ' + pooling_strategy + ' pooling)'}")
        print(f"    - Covariate tokens: {'Enabled' if use_covariate_tokens else 'Disabled (concatenate at end)'}")
        if use_covariate_tokens:
            print(f"    - Covariate strategy: {covariate_token_strategy}")
        if not use_cls_token:
            print(f"    - Pooling strategy: {pooling_strategy}")
        
        # Project input to transformer dimension if needed
        self.input_projection = None
        if input_dim != transformer_dim:
            self.input_projection = nn.Linear(input_dim, transformer_dim)
            print(f"  - Adding input projection: {input_dim} → {transformer_dim}")
        
        # Class token 
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, transformer_dim))
            print(f"    - Class token initialized: [1, 1, {transformer_dim}]")
        
        # Pooling strategy (only used when not using cls token)
        if not use_cls_token:
            if pooling_strategy == "attention":
                self.attention_pooling = AttentionPooling(transformer_dim)
                print(f"    - Attention pooling initialized")
            elif pooling_strategy == "concat":
                print(f"    - Concat pooling")
        
        # Covariate token embedder
        if use_covariate_tokens:
            self.covariate_embedder = CovariateTokenEmbedder(
                transformer_dim, covariate_embed_dim, covariate_token_strategy, use_age, use_gender, use_bmi, use_pcs
            )
        
        # Positional encoding
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(transformer_dim, max_seq_len, dropout)
            print(f"  - Adding positional encoding (max_len={max_seq_len})")
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=False  # (seq_len, batch_size, features)
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(transformer_dim)
        self.final_output_dim = transformer_dim
        
        # Initialize weights from pretrained model if requested
        if init_from_pretrained:
            self._initialize_from_pretrained()
        
        print(f"GenomicTransformerBlock initialized:")
        print(f"  - Input dimension: {input_dim}")
        print(f"  - Output dimension: {self.final_output_dim}")
        print(f"  - Transformer dimension: {transformer_dim}")
        print(f"  - Number of layers: {num_layers}")
        print(f"  - Number of heads: {num_heads}")
        print(f"  - Feedforward dimension: {ff_dim}")
        print(f"  - Dropout: {dropout}")
        print(f"  - Using positional encoding: {use_positional_encoding}")
        print(f"  - Initialize from pretrained: {init_from_pretrained}")
        if init_from_pretrained:
            print(f"  - Pretrained model: {pretrained_model_name}")
            print(f"  - Model type: {pretrained_model_type}")
            print(f"  - Layer selection strategy: {layer_init_strategy}")
    
    def _detect_model_type(self, model_name, specified_type):
        """Automatically detect model type from name or use specified type"""
        if specified_type.lower() != "auto":
            return specified_type.lower()
        
        model_name_lower = model_name.lower()
        
        # Detection patterns
        if any(pattern in model_name_lower for pattern in ['vit', 'vision']):
            return "vit"
        elif any(pattern in model_name_lower for pattern in ['bert', 'dnabert', 'biobert']):
            return "bert"
        else:
            # Default fallback - could also raise an error
            print(f"Warning: Could not auto-detect model type for '{model_name}'. Defaulting to 'bert'")
            return "bert"
    
    def _initialize_from_pretrained(self):
        """Initialize transformer weights from pretrained model (ViT or BERT)"""
        model_type = self._detect_model_type(self.pretrained_model_name, self.pretrained_model_type)
        
        print(f"\n  Initializing transformer weights from pretrained {model_type.upper()} model: {self.pretrained_model_name}")
        print(f"  Model type: {model_type} ({'auto-detected' if self.pretrained_model_type == 'auto' else 'manually specified'})")
        
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
        """Parse the concatenated covariates tensor into individual components"""
        if covariates_tensor is None or covariates_tensor.numel() == 0:
            return {}
        
        covariates_dict = {}
        start_idx = 0
        
        # Extract PCs (10 dimensions)
        if hasattr(self.covariate_embedder, 'use_pcs') and self.covariate_embedder.use_pcs:
            if start_idx + 10 <= covariates_tensor.size(1):
                covariates_dict['pcs'] = covariates_tensor[:, start_idx:start_idx+10]
                start_idx += 10
            else:
                print(f"DEBUG: Cannot extract PCs - need indices {start_idx}:{start_idx+10} but tensor only has {covariates_tensor.size(1)} features")
        
        # Extract age (1 dimension)
        if hasattr(self.covariate_embedder, 'use_age') and self.covariate_embedder.use_age:
            if start_idx < covariates_tensor.size(1):
                covariates_dict['age'] = covariates_tensor[:, start_idx]
                start_idx += 1
            else:
                print(f"DEBUG: Cannot extract age - need index {start_idx} but tensor only has {covariates_tensor.size(1)} features")
                
        # Extract gender (1 dimension)
        if hasattr(self.covariate_embedder, 'use_gender') and self.covariate_embedder.use_gender:
            if start_idx < covariates_tensor.size(1):
                covariates_dict['gender'] = covariates_tensor[:, start_idx]
                start_idx += 1
            else:
                print(f"DEBUG: Cannot extract gender - need index {start_idx} but tensor only has {covariates_tensor.size(1)} features")
                
        # Extract BMI (1 dimension)
        if hasattr(self.covariate_embedder, 'use_bmi') and self.covariate_embedder.use_bmi:
            if start_idx < covariates_tensor.size(1):
                covariates_dict['bmi'] = covariates_tensor[:, start_idx]
                start_idx += 1
            else:
                print(f"DEBUG: Cannot extract BMI - need index {start_idx} but tensor only has {covariates_tensor.size(1)} features")
        
        return covariates_dict
        
    def forward(self, x, covariates=None):
        """
        Args:
            x: Tensor of shape (batch_size, channels, seq_len) from conv layers
            covariates: Tensor of shape (batch_size, num_covariates) or None
        Returns:
            If use_cls_token=True: cls_token features [batch_size, transformer_dim]
            If use_cls_token=False: pooled features [batch_size, transformer_dim]
        """
        batch_size, channels, seq_len = x.shape
        
        # Reshape: (batch_size, channels, seq_len) → (batch_size, seq_len, channels)
        x = x.transpose(1, 2)
        
        # Project to transformer dimension if needed
        if self.input_projection is not None:
            x = self.input_projection(x)  # [batch_size, seq_len, transformer_dim]
        
        # Add class token if enabled
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, transformer_dim]
            x = torch.cat([cls_tokens, x], dim=1)  # [batch_size, 1+seq_len, transformer_dim]
            seq_len += 1  # Account for cls token
        
        # Add covariate tokens if enabled
        covariate_seq_len = 0
        if self.use_covariate_tokens and covariates is not None:
            covariates_dict = self._parse_covariates_tensor(covariates)

            covariate_result = self.covariate_embedder(covariates_dict)
            
            if covariate_result is not None:
                covariate_tokens, _ = covariate_result
                if covariate_tokens is not None:
                    x = torch.cat([x, covariate_tokens], dim=1)  # [batch_size, seq_len+num_cov_tokens, transformer_dim]
                    covariate_seq_len = covariate_tokens.size(1)
        
        # Reshape for transformer: (batch_size, seq_len, features) → (seq_len, batch_size, features)
        x = x.transpose(0, 1) # (seq_len, batch_size, transformer_dim)
        
        # Add positional encoding
        if self.use_positional_encoding:
            x = self.pos_encoding(x)
        
        # Apply transformer layers
        x = self.transformer_encoder(x)  # (total_seq_len, batch_size, transformer_dim)
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Extract features for classification
        if self.use_cls_token:
            # Use cls token (first token) for classification
            cls_features = x[0]  # [batch_size, transformer_dim]
            return cls_features
        else:
            # Reshape back to batch-first for pooling: (seq_len, batch_size, transformer_dim) → (batch_size, seq_len, transformer_dim)
            x_batch_first = x.transpose(0, 1)
            
            # Apply different pooling strategies
            if self.pooling_strategy == "mean":
                # Global average pooling over ALL tokens
                pooled_features = torch.mean(x_batch_first, dim=1)  # [batch_size, transformer_dim]
                
            elif self.pooling_strategy == "max":
                # Global max pooling over ALL tokens
                pooled_features, _ = torch.max(x_batch_first, dim=1)  # [batch_size, transformer_dim]
                    
            elif self.pooling_strategy == "attention":
                # Learnable attention-based pooling
                pooled_features = self.attention_pooling(x_batch_first)  # [batch_size, transformer_dim]
                
            elif self.pooling_strategy == "concat":
                pooled_features = x_batch_first
                # # Create projection layer if not exists (dynamic sizing)
                # if self.concat_projection is None:
                #     input_size = seq_len * transformer_dim
                #     self.concat_projection = nn.Linear(input_size, transformer_dim).to(x_batch_first.device)
                #     print(f"    - Created concat projection: {input_size} → {transformer_dim}")
                
                #pooled_features = self.concat_projection(flattened)  # [batch_size, transformer_dim]
                
            else:
                # Fallback to mean pooling
                print(f"Warning: Unknown pooling strategy '{self.pooling_strategy}', using mean pooling")
                pooled_features = torch.mean(x_batch_first, dim=1)
            
            return pooled_features

class MultilabelGenotypeModel(nn.Module):
    def __init__(self, input_size, num_diseases, kernel_sizes, stride, conv_channels, fc_layers, act, dropout_rate, 
                 use_covariates=True, use_age=True, use_gender=True, use_bmi=True, num_covariates=10, use_pooling=True, pool_size=16, pool_type="max",
                 use_multi_scale=True, use_disease_attention=True, use_separate_heads=True, 
                 attention_heads=8, attention_dim=256, multi_scale_kernels=None, multi_scale_strides=None,
                 multi_scale_fusion="cross_scale", multi_scale_mode="progressive", hardcoded_kernels=None, hardcoded_strides=None,
                 use_pointwise_conv=False, pointwise_channels=16,
                 use_transformer=False, transformer_layers=2, transformer_heads=4, transformer_dim=256, 
                 transformer_ff_dim=1024, transformer_dropout=0.1, use_positional_encoding=True, max_seq_len=10000,
                 pretrained_model_type="auto",
                 init_from_pretrained=False, pretrained_model_name="WinKawaks/vit-small-patch16-224",
                 init_layers_fraction=1.0, layer_init_strategy="middle", custom_layer_indices="",
                 use_cls_token=True, use_covariate_tokens=False, covariate_embed_dim=64, covariate_token_strategy="separate", pooling_strategy="mean"):

        super(MultilabelGenotypeModel, self).__init__()
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

        # Store transformer parameters
        self.use_transformer = use_transformer
        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads
        self.transformer_dim = transformer_dim
        self.transformer_ff_dim = transformer_ff_dim
        self.transformer_dropout = transformer_dropout
        self.use_positional_encoding = use_positional_encoding
        self.max_seq_len = max_seq_len

        # Store pretrained initialization parameters
        self.init_from_pretrained = init_from_pretrained
        self.pretrained_model_name = pretrained_model_name
        self.pretrained_model_type = pretrained_model_type
        self.init_layers_fraction = init_layers_fraction
        self.layer_init_strategy = layer_init_strategy
        self.custom_layer_indices = custom_layer_indices

        self.use_cls_token = use_cls_token
        self.use_covariate_tokens = use_covariate_tokens
        self.covariate_embed_dim = covariate_embed_dim
        self.covariate_token_strategy = covariate_token_strategy

        # Store multi-scale parameters
        self.multi_scale_kernels = multi_scale_kernels if multi_scale_kernels is not None else [15, 63, 255]
        self.multi_scale_strides = multi_scale_strides if multi_scale_strides is not None else [4, 16, 64]
        self.multi_scale_fusion = multi_scale_fusion
        self.multi_scale_mode = multi_scale_mode
        self.hardcoded_kernels = hardcoded_kernels
        self.hardcoded_strides = hardcoded_strides
        
        # Store conv_channels for later reference
        self.conv_channels = conv_channels
        
        # Store input size for sequence length calculations
        self.input_size = input_size

        # Calculate total number of covariates
        self.total_covariates = 0
        if use_covariates:
            self.total_covariates += num_covariates  # PCs
        if use_age:
            self.total_covariates += 1  # Age
        if use_gender:
            self.total_covariates += 1  # Gender
        if use_bmi:
            self.total_covariates += 1  # BMI

        # Create convolutional layers based on architecture choice
        print(f"  Creating convolutional layers...")
        print(f"  Input sequence length: {input_size:,}")
        
        if self.use_multi_scale:
            print(f"  Multi-scale processing enabled with {self.multi_scale_fusion} fusion")
            print(f"  Multi-scale mode: {self.multi_scale_mode}")
            print(f"  Multi-scale Pool Type: {self.pool_type}")
            print(f"  Pointwise convolution: {'Enabled' if use_pointwise_conv else 'Disabled'}")
            if use_pointwise_conv:
                print(f"  Pointwise channels: {pointwise_channels}")
            
            if self.multi_scale_fusion == "cross_scale":
                print(f"  Using cross-scale fusion strategy")
                conv_layers = self._create_multi_scale_conv_layers_cross_scale(conv_channels, kernel_sizes, stride, dropout_rate, act)
            else:  # parallel
                print(f"  Using parallel processing strategy with independent channels")
                conv_layers = self._create_multi_scale_conv_layers_parallel(conv_channels, kernel_sizes, stride, dropout_rate, act)
            
            # Update the final conv output channels accounting for multi-scale concatenation and pointwise conv
            if use_pointwise_conv:
                final_conv_channels = len(self.multi_scale_kernels) * pointwise_channels
            else:
                final_conv_channels = len(self.multi_scale_kernels) * conv_channels[-1]
        else:
            print(f"  Using standard single-scale convolution")
            print(f"  Pointwise convolution: {'Enabled' if use_pointwise_conv else 'Disabled'}")
            if use_pointwise_conv:
                print(f"  Pointwise channels: {pointwise_channels}")
            conv_layers = self._create_conv_layers(conv_channels, kernel_sizes, stride, dropout_rate, act)
            
            # Update final conv channels
            if use_pointwise_conv:
                final_conv_channels = pointwise_channels
            else:
                final_conv_channels = conv_channels[-1]

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
        
        # Calculate output size after convolutions (but before transformer)
        self.conv_output_info = self._get_conv_output_info(input_size)
        conv_output_size = self.conv_output_info['flattened_size']
        conv_seq_len = self.conv_output_info['seq_len']
        conv_channels_out = self.conv_output_info['channels']
        
        print(f"Convolutional output: channels={conv_channels_out}, seq_len={conv_seq_len}, flattened_size={conv_output_size}")
        
        # Add transformer layers if enabled
        if self.use_transformer:
            print(f"\n  Adding Transformer Processing:")
            print(f"  - Class token: {'Enabled' if use_cls_token else 'Disabled (using global pooling)'}")
            print(f"  - Covariate tokens: {'Enabled' if use_covariate_tokens else 'Disabled (concatenate at end)'}")
            print(f"  - Initialize from pretrained: {init_from_pretrained}")
            if init_from_pretrained:
                print(f"  - Pretrained model: {pretrained_model_name}")
                #print(f"  - Layers to initialize: {init_layers_fraction * 100:.1f}%")
            
            print(f"  - Input to transformer: (batch_size, {conv_channels_out}, {conv_seq_len})")
            
            self.transformer_block = GenomicTransformerBlock(
                input_dim=conv_channels_out,
                transformer_dim=transformer_dim,
                num_layers=transformer_layers,
                num_heads=transformer_heads,
                ff_dim=transformer_ff_dim,
                dropout=transformer_dropout,
                use_positional_encoding=use_positional_encoding,
                max_seq_len=max_seq_len,
                use_cls_token=use_cls_token,
                use_covariate_tokens=use_covariate_tokens,
                covariate_embed_dim=covariate_embed_dim,
                covariate_token_strategy=covariate_token_strategy,
                pooling_strategy=pooling_strategy,
                use_age=use_age,
                use_gender=use_gender,
                use_bmi=use_bmi,
                use_pcs=use_covariates,
                # Pretrained initialization parameters
                init_from_pretrained=init_from_pretrained,
                pretrained_model_name=pretrained_model_name,
                pretrained_model_type=pretrained_model_type,
                init_layers_fraction=init_layers_fraction,
                layer_init_strategy=layer_init_strategy,
                custom_layer_indices=custom_layer_indices
            )
            
            # Calculate output size after transformer
            #if use_cls_token or use_covariate_tokens:
            if use_cls_token or use_covariate_tokens:
                if use_covariate_tokens and self.transformer_block.pooling_strategy == 'concat' and self.transformer_block.covariate_token_strategy == 'combined':
                    # Transformer returns [batch_size, seq_len+1, transformer_dim], needs flattening
                    self.post_transformer_output_size = (conv_seq_len + 1) * transformer_dim
                    feature_size_for_fc = self.post_transformer_output_size
                    print(f"  - Transformer output: (batch_size, {conv_seq_len+1}, {transformer_dim})")
                    print(f"  - Flattened size after transformer: {self.post_transformer_output_size:,}")
                
                elif use_covariate_tokens and self.transformer_block.pooling_strategy == 'concat' and self.transformer_block.covariate_token_strategy == 'separate':
                    # Transformer returns [batch_size, seq_len+4, transformer_dim], needs flattening
                    self.post_transformer_output_size = (conv_seq_len + 4) * transformer_dim
                    feature_size_for_fc = self.post_transformer_output_size
                    print(f"  - Transformer output: (batch_size, {conv_seq_len+4}, {transformer_dim})")
                    print(f"  - Flattened size after transformer: {self.post_transformer_output_size:,}")
                
                else:
                    # Transformer returns [batch_size, transformer_dim] directly
                    self.post_transformer_output_size = transformer_dim
                    feature_size_for_fc = self.post_transformer_output_size
                    print(f"  - Transformer output: (batch_size, {transformer_dim}) from cls token or Max/Avg pooling")
            else:
                if self.transformer_block.pooling_strategy == 'mean' or self.transformer_block.pooling_strategy == 'max':
                    # Transformer returns [batch_size, transformer_dim] directly
                    self.post_transformer_output_size = transformer_dim
                    feature_size_for_fc = self.post_transformer_output_size
                    print(f"  - Transformer output: (batch_size, {transformer_dim}) from Max/Avg pooling")
                
                else:
                    # Transformer returns [batch_size, seq_len, transformer_dim], needs flattening
                    self.post_transformer_output_size = conv_seq_len * transformer_dim
                    feature_size_for_fc = self.post_transformer_output_size
                    print(f"  - Transformer output: (batch_size, {conv_seq_len}, {transformer_dim})")
                    print(f"  - Flattened size after transformer: {self.post_transformer_output_size:,}")
        else:
            print(f"\n  Transformer processing: Disabled")
            self.transformer_block = None
            feature_size_for_fc = conv_output_size
        
        # Disease-specific attention mechanism
        if self.use_disease_attention:
            self.attention_input_dim = feature_size_for_fc
            self.attention_proj = nn.Linear(self.attention_input_dim, attention_dim)
            self.disease_attention = DiseaseSpecificAttention(attention_dim, num_diseases, attention_heads)
            self.attention_output_dim = attention_dim
            print(f"Using disease-specific attention with {attention_heads} heads and {attention_dim} dimensions")
        else:
            self.attention_output_dim = feature_size_for_fc
            
        # Shared feature layers
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
        
        # Print model configuration
        architecture_info = []
        architecture_info.append(f"Multi-scale convolutions: {use_multi_scale}")
        architecture_info.append(f"Final pointwise convolutions: {use_pointwise_conv}")
        if use_pointwise_conv:
            architecture_info.append(f"Final pointwise channels: {pointwise_channels}") 
        # Transformer info
        architecture_info.append(f"Transformer layers: {use_transformer}")
        if use_transformer:
            architecture_info.append(f"Enhanced transformer: {transformer_layers} layers, {transformer_heads} heads, dim={transformer_dim}")
            architecture_info.append(f"Class token: {use_cls_token}")
            architecture_info.append(f"Covariate tokens: {use_covariate_tokens}")
            architecture_info.append(f"Initialize from pretrained: {init_from_pretrained}")
            if init_from_pretrained:
                architecture_info.append(f"Pretrained model: {pretrained_model_name}")
        architecture_info.append(f"Disease-specific attention: {use_disease_attention}")
        architecture_info.append(f"Separate disease heads: {use_separate_heads}")
        architecture_info.append(f"Using PC's: {use_covariates}, age: {use_age}, gender: {use_gender}, BMI:{use_bmi}")
        architecture_info.append(f"Using final pooling: {use_pooling} ({pool_type} pool, size={pool_size})" if use_pooling else "Using final pooling: False")
        architecture_info.append(f"Multi-scale internal pooling: {pool_type} (for length standardization)" if use_multi_scale else "Internal pooling: N/A")
        
        print(f"\nEnhanced MultilabelGenotypeModel initialized:")
        for info in architecture_info:
            print(f"  - {info}")
        
        # Print architecture details
        if use_multi_scale:
            print(f"  - Multi-scale fusion strategy: {multi_scale_fusion}")
            print(f"  - Multi-scale mode: {multi_scale_mode}")
            if multi_scale_mode == "progressive":
                print(f"  - Multi-scale kernel sizes for first layer: {self.multi_scale_kernels}")
                print(f"  - Multi-scale strides for first layer: {self.multi_scale_strides}")
                print(f"  - Subsequent layers use progressive reduction (kernel//2^i, stride//2^i)")
            else:  # hardcoded
                print(f"  - Using hardcoded kernels and strides for all layers")
                print(f"  - Hardcoded kernels: {self.hardcoded_kernels}")
                print(f"  - Hardcoded strides: {self.hardcoded_strides}")
            print(f"  - Number of branches per layer: {len(self.multi_scale_kernels)}")
            if multi_scale_fusion == "cross_scale":
                print(f"  - Cross-scale fusion: Each branch sees all previous scales")
            else:
                print(f"  - Parallel processing: Each branch maintains independent path")
            
            if use_pointwise_conv:
                print(f"  - Final layer pointwise conv: {conv_channels[-1]} → {pointwise_channels} channels per branch")
                print(f"  - Final conv output channels: {len(self.multi_scale_kernels)} * {pointwise_channels} = {final_conv_channels}")
            else:
                print(f"  - Each branch outputs full conv_channels[i] value")
                print(f"  - Final conv output channels: {len(self.multi_scale_kernels)} * {conv_channels[-1]} = {final_conv_channels}")
        else:
            print(f"  - Single-scale architecture details:")
            print(f"  - Kernel sizes: {kernel_sizes}")
            print(f"  - Strides: {stride}")
            print(f"  - Conv channels: {conv_channels}")
            if use_pointwise_conv:
                print(f"  - Final layer pointwise conv: {conv_channels[-1]} → {pointwise_channels} channels")
                print(f"  - Final conv output channels: {pointwise_channels}")
            else:
                print(f"  - Final conv output channels: {conv_channels[-1]}")
        
        if use_disease_attention:
            print(f"  - Attention: {attention_heads} heads, {attention_dim} dimensions")
        if use_separate_heads:
            print(f"  - Separate heads: {num_diseases} disease-specific prediction heads")
        
        print(f"  - Input size: {input_size:,} SNPs")
        print(f"  - Total parameters: {sum(p.numel() for p in self.parameters()):,}")

    def _calculate_conv_output_length(self, input_length, kernel_size, stride, padding):
        """Helper method to calculate output sequence length after convolution"""
        return (input_length + 2 * padding - kernel_size) // stride + 1

    def _print_sequence_progression(self, layer_idx, branch_idx, branch_name, input_length, kernel_size, stride, padding, channels_in, channels_out, is_final_layer=False):
        """Helper method to print sequence length progression"""
        output_length = self._calculate_conv_output_length(input_length, kernel_size, stride, padding)

        if self.use_pointwise_conv and is_final_layer:
            pointwise_out = self.pointwise_channels
            print(f"      Branch {branch_idx} ({branch_name}): "
                  f"sequence {input_length:,} → {output_length:,} "
                  f"(kernel={kernel_size}, stride={stride}, padding={padding}) "
                  f"channels {channels_in} → {channels_out} → {pointwise_out} (final pointwise)")
        else:
            print(f"      Branch {branch_idx} ({branch_name}): "
                  f"sequence {input_length:,} → {output_length:,} "
                  f"(kernel={kernel_size}, stride={stride}, padding={padding}) "
                  f"channels {channels_in} → {channels_out}")
        return output_length

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

    def _create_multi_scale_conv_layers_cross_scale(self, conv_channels, kernel_sizes, stride, dropout_rate, act):
        """Create multi-scale convolutional layers with cross-scale fusion"""
        layers = nn.ModuleList()
        
        print(f"  Sequence length progression:")
        
        current_length = self.input_size
        
        for i in range(len(conv_channels)):
            print(f"\n  Layer {i}:")
            print(f"    Input sequence length: {current_length:,}")
            
            # Determine if this is the final layer
            is_final_layer = (i == len(conv_channels) - 1)
            
            # Calculate input channels considering multi-scale concatenation and pointwise conv
            if i == 0:
                in_channels = self.input_channels
            else:
                # Previous layer output channels depend on whether pointwise was applied to previous layer
                prev_is_final = (i-1 == len(conv_channels) - 1)
                if self.use_pointwise_conv and prev_is_final:
                    in_channels = len(self.multi_scale_kernels) * self.pointwise_channels
                else:
                    in_channels = len(self.multi_scale_kernels) * conv_channels[i-1]
            
            # Each branch gets the FULL conv_channels[i] value
            out_channels = conv_channels[i]
            
            # Get kernels and strides for this layer
            scale_kernels, scale_strides = self._get_layer_kernels_and_strides(i)
            
            if self.multi_scale_mode == "progressive":
                if i == 0:
                    print(f"    Genomic multi-scale kernels={scale_kernels}, strides={scale_strides}")
                    branch_names = ['Local LD blocks', 'Gene-level regions', 'Long-range domains']
                else:
                    print(f"    Progressive multi-scale kernels={scale_kernels}, strides={scale_strides}")
                    print(f"    Reduction factor: kernels ÷ {2**i}, strides ÷ {2**i}")
                    branch_names = ['Local', 'Gene', 'Domain']
            else:  # hardcoded
                print(f"    Hardcoded multi-scale kernels={scale_kernels}, strides={scale_strides}")
                branch_names = [f'Branch_{j}' for j in range(len(scale_kernels))]
            
            # Calculate output lengths for each branch
            branch_outputs = []
            for j, (kernel_size, stride_val) in enumerate(zip(scale_kernels, scale_strides)):
                padding = kernel_size // 2
                branch_name = branch_names[j] if j < len(branch_names) else f'Scale_{j}'
                output_length = self._print_sequence_progression(
                    i, j, branch_name, current_length, kernel_size, stride_val, padding, in_channels, out_channels, is_final_layer
                )
                branch_outputs.append(output_length)
            
            # After multi-scale block, length is minimum of all branches
            #final_length = min(branch_outputs)
            final_length = branch_outputs[1] if len(branch_outputs) >=2 else branch_outputs[0]
            print(f"    After concatenation: sequence length = {final_length:,}")
            
            if self.use_pointwise_conv and is_final_layer:
                print(f"    Output channels: {len(scale_kernels)} * {self.pointwise_channels} = {len(scale_kernels) * self.pointwise_channels} (after final pointwise)")
            else:
                print(f"    Output channels: {len(scale_kernels)} * {out_channels} = {len(scale_kernels) * out_channels}")
            print(f"    Internal {self.pool_type} pooling applied for length standardization")
            
            multi_scale_block = MultiScaleConvBlock(
                in_channels, out_channels, scale_kernels, scale_strides, act, dropout_rate, self.pool_type,
                 self.use_pointwise_conv, self.pointwise_channels, is_final_layer
            )
            layers.append(multi_scale_block)
            
            # Update current length for next layer
            current_length = final_length
        
        print(f"\n  Final sequence length after all conv layers: {current_length:,}")
        return nn.Sequential(*layers)
    
    def _create_multi_scale_conv_layers_parallel(self, conv_channels, kernel_sizes, stride, dropout_rate, act):
        """Create parallel multi-scale convolutional layers in sequential manner"""
        layers = nn.ModuleList()
        
        print(f"  Using PARALLEL fusion strategy (Sequential Implementation)")
        print(f"  Sequence length progression:")
        
        # Track sequence lengths for each branch independently
        branch_lengths = [self.input_size] * len(self.multi_scale_kernels)
        branch_names = ['Local LD blocks', 'Gene-level regions', 'Long-range domains']
        
        for i in range(len(conv_channels)):
            print(f"\n  Layer {i}:")

            # Determine if this is the final layer
            is_final_layer = (i == len(conv_channels) - 1)
            
            # Input channels are same as specified (no concatenation from previous layer)
            in_channels = self.input_channels if i == 0 else conv_channels[i-1]
            out_channels = conv_channels[i]
            
            # Get kernels and strides for this layer
            scale_kernels, scale_strides = self._get_layer_kernels_and_strides(i)
            
            if self.multi_scale_mode == "progressive":
                if i == 0:
                    print(f"    Genomic multi-scale (independent channels)")
                else:
                    print(f"    Progressive parallel multi-scale (independent channels)")
                    print(f"    Reduction factor: kernels ÷ {2**i}, strides ÷ {2**i}")
            else:  # hardcoded
                print(f"    Hardcoded parallel multi-scale (independent channels)")
            
            if self.use_pointwise_conv and is_final_layer:
                print(f"    Each branch: {in_channels} → {out_channels} → {self.pointwise_channels} channels (final pointwise)")
            else:
                print(f"    Each branch: {in_channels} → {out_channels} channels")
            
            # Calculate sequence progression for each branch
            for j, (kernel_size, stride_val) in enumerate(zip(scale_kernels, scale_strides)):
                padding = kernel_size // 2
                branch_name = branch_names[j] if j < len(branch_names) else f'Branch_{j}'
                
                new_length = self._print_sequence_progression(
                    i, j, branch_name, branch_lengths[j], kernel_size, stride_val, padding, in_channels, out_channels, is_final_layer
                )
                branch_lengths[j] = new_length
            
            if is_final_layer:
                # Final layer: show concatenation info
                #final_length = min(branch_lengths)
                final_length = branch_lengths[1] if len(branch_lengths) >=2 else branch_lengths[0]
                print(f"    Final concatenation: min sequence length = {final_length:,}")
                
                if self.use_pointwise_conv:
                    print(f"    Total output channels: {len(self.multi_scale_kernels)} * {self.pointwise_channels} = {len(self.multi_scale_kernels) * self.pointwise_channels} (after final pointwise)")
                else:
                    print(f"    Total output channels: {len(self.multi_scale_kernels)} * {out_channels} = {len(self.multi_scale_kernels) * out_channels}")
                print(f"    Final layer {self.pool_type} pooling applied for length standardization")
            
            # Create parallel multi-scale block for this layer
            parallel_block = ParallelMultiScaleConvBlock(
                in_channels, out_channels, scale_kernels, scale_strides, act, dropout_rate, i, is_final_layer, self.pool_type,
                self.use_pointwise_conv, self.pointwise_channels
            )
            layers.append(parallel_block)
        
        final_sequence_length = min(branch_lengths)
        print(f"\n  Final sequence length after all conv layers: {final_sequence_length:,}")
        
        # Return as Sequential, just like cross-scale mode
        return nn.Sequential(*layers)

    def _create_conv_layers(self, conv_channels, kernel_sizes, stride, dropout_rate, act):
        """Create traditional sequential convolutional layers"""
        layers = []
        current_length = self.input_size
        
        print(f"  Standard convolution sequence length progression:")
        
        for i in range(len(conv_channels)):
            in_channels = self.input_channels if i == 0 else conv_channels[i-1]
            out_channels = conv_channels[i]
            kernel_size = kernel_sizes[i]
            stride_val = stride[i]
            padding = kernel_size // 2

            # Determine if this is the final layer
            is_final_layer = (i == len(conv_channels) - 1)
            
            # Calculate output length
            output_length = self._calculate_conv_output_length(current_length, kernel_size, stride_val, padding)
            
            # Print progression info
            if self.use_pointwise_conv and is_final_layer:
                print(f"    Layer {i}: sequence {current_length:,} → {output_length:,} "
                    f"(kernel={kernel_size}, stride={stride_val}, padding={padding}) "
                    f"channels {in_channels} → {out_channels} → {self.pointwise_channels} (final pointwise)")
            else:
                print(f"    Layer {i}: sequence {current_length:,} → {output_length:,} "
                  f"(kernel={kernel_size}, stride={stride_val}, padding={padding}) "
                  f"channels {in_channels} → {out_channels}")
            
            layers.append(nn.Conv1d(in_channels=in_channels,
                                   out_channels=out_channels,  
                                   kernel_size=kernel_size,
                                   stride=stride_val,
                                   padding=padding))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(self.get_activation(act))

            # Add pointwise convolution for final layer if enabled
            if self.use_pointwise_conv and is_final_layer:
                layers.append(nn.Conv1d(in_channels=out_channels,
                                    out_channels=self.pointwise_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0))
                layers.append(nn.BatchNorm1d(self.pointwise_channels))
                layers.append(self.get_activation(act))
                print(f"    Added pointwise conv: {out_channels} → {self.pointwise_channels} channels")

            current_length = output_length
        
        print(f"  Final sequence length after standard conv layers: {current_length:,}")
        return nn.Sequential(*layers)

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

    def forward(self, x, covariates=None):
        # Input: x shape [batch_size, n_snps, 3]
        x = x.permute(0, 2, 1)  # -> [batch_size, 3, n_snps]
        
        # Convolutional processing (includes final pooling if enabled)
        x = self.conv_layers(x)  # Shape: (batch_size, channels, seq_len)
        
        # Transformer processing if enabled
        if self.use_transformer:
            if self.use_covariate_tokens:
                # Process covariates as tokens within transformer
                x = self.transformer_block(x, covariates)  # Returns [batch_size, transformer_dim]
                
                # For cls token: x is already the right shape
                # For pooling: x is already pooled
                if self.transformer_block.pooling_strategy == 'concat':
                    x = x.reshape(x.size(0), -1)
                
            else:
                # Original behavior: transformer without covariate tokens
                x = self.transformer_block(x)  # Transform: (batch_size, channels, seq_len) -> (batch_size, seq_len, transformer_dim) or (batch_size, transformer_dim) if cls token
                
                # Flatten transformer output for fully connected layers if not using cls token
                if not self.use_cls_token:
                    x = x.reshape(x.size(0), -1)  # [batch_size, seq_len * transformer_dim]
                # If using cls token, x is already [batch_size, transformer_dim]
        else:
            # No transformer: flatten conv output for fully connected layers
            x = x.reshape(x.size(0), -1)  # [batch_size, flattened_features]
        
        # Disease-specific attention processing
        if self.use_disease_attention:
            # Project to attention dimension and add sequence dimension
            x_proj = self.attention_proj(x)  # [batch_size, attention_dim]
            x_seq = x_proj.unsqueeze(1)  # [batch_size, 1, attention_dim]
            
            # Apply disease-specific attention
            x_attended = self.disease_attention(x_seq)  # [batch_size, num_diseases, attention_dim]
            
            # For now, use mean pooling across diseases for shared processing
            x = torch.mean(x_attended, dim=1)  # [batch_size, attention_dim]
        
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
                    # Get disease-specific features
                    disease_features = x_attended[:, i, :]  # [batch_size, attention_dim]
                    
                    # Concatenate with covariates (only if not using covariate tokens)
                    if covariates is not None and self.total_covariates > 0 and not self.use_covariate_tokens:
                        disease_input = torch.cat([disease_features, covariates], dim=1)
                    else:
                        disease_input = disease_features
                    
                    # Process through shared layers first
                    disease_shared = self.fc_shared(disease_input)
                    
                    # Then through disease-specific head
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
                
    def get_activation(self, name):
        if name == 'tanh':
            return nn.Tanh()
        elif name == 'relu':
            return nn.ReLU()
        elif name == 'leakyrelu':
            return nn.LeakyReLU(0.01)
        elif name == 'rrelu':
            return nn.RReLU(0.125, 0.3333)
        elif name == 'gelu':
            return nn.GELU()
        elif name == 'silu':
            return nn.SiLU()
        else:
            raise NotImplementedError("Activation function not implemented.")

def print_lr(optimizer):
    for param_group in optimizer.param_groups:
        print(f"Current Learning Rate: {param_group['lr']}")

def save_checkpoint(state, is_best, filename, best_filename):
    print(f"Saving checkpoint to {filename}")
    torch.save(state, filename)
    if is_best:
        print(f"This is the best model so far. Saving to {best_filename}")
        shutil.copyfile(filename, best_filename)

def find_latest_checkpoint(dir_path, prefer_best=False):
    """Find the latest checkpoint in the directory"""

    # First try to find the best model checkpoint if preferred
    if prefer_best:
        best_models = [f for f in os.listdir(dir_path) if f.startswith('best_model_')]
        best_models.sort(key=lambda x: int(x.split('best_model_')[1].split('.pt')[0]))

        latest_best = os.path.join(dir_path, best_models[-1])
        print(f"Found best model checkpoint: {latest_best}")
        return latest_best

    # Fallback: check for old naming scheme (best_model.pt)
    old_best_model_path = os.path.join(dir_path, 'best_model.pt')
    if os.path.exists(old_best_model_path):
        print(f"Found best model checkpoint (old format): {old_best_model_path}")
        return old_best_model_path
    
    checkpoints = [f for f in os.listdir(dir_path) if f.startswith('checkpoint_epoch_')]
    checkpoints.sort(key=lambda x: int(x.split('checkpoint_epoch_')[1].split('.pt')[0]))
    if checkpoints:
        latest_checkpoint = os.path.join(dir_path, checkpoints[-1])
        print(f"Using latest epoch checkpoint: {latest_checkpoint}")
        return latest_checkpoint
    
    print("No checkpoints found")
    return None

def cleanup_old_checkpoints(dir_path, keep_last_n=3):
    """Remove old checkpoints and best models, keeping only the most recent n checkpoints and 1 best model"""
    
    # Handle regular checkpoints
    checkpoints = [f for f in os.listdir(dir_path) if f.startswith('checkpoint_epoch_')]
    checkpoints.sort(key=lambda x: int(x.split('checkpoint_epoch_')[1].split('.pt')[0]))
    
    if len(checkpoints) > keep_last_n:
        for old_ckpt in checkpoints[:-keep_last_n]:
            old_path = os.path.join(dir_path, old_ckpt)
            print(f"Removing old checkpoint: {old_path}")
            os.remove(old_path)
    
    # Handle best model files
    best_models = [f for f in os.listdir(dir_path) if f.startswith('best_model_')]
    best_models.sort(key=lambda x: int(x.split('best_model_')[1].split('.pt')[0]))
        
    if len(best_models) > 1:
        for old_best in best_models[:-1]:
            old_path = os.path.join(dir_path, old_best)
            print(f"Removing old best model: {old_path}")
            os.remove(old_path)

def train_multilabel_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, disease_labels, device='cuda',
                         early_stopping=None, checkpoint_dir=None, 
                         start_epoch=0, keep_last_n=2, history=None, initial_best_loss=float('inf')):
    print(f"Training multilabel model on device: {device}")
    print(f"Disease labels: {disease_labels}")
    print(f"Starting with initial best loss: {initial_best_loss:.6f}")
    
    print(f"DEBUG: History at start of train_multilabel_model: {'None' if history is None else 'Present'}")
    
    scaler = GradScaler('cuda')
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = initial_best_loss
    completed_epochs = start_epoch 

    num_diseases = len(disease_labels)

    # Always initialize history if None or missing keys
    if history is None or 'train_loss' not in history or 'test_loss' not in history:
        print("Creating new history dictionary")
        history = {
            'train_loss': [], 'test_loss': [],
            'learning_rates': []
        }

        # Add metrics for each disease
        for disease in disease_labels:
            for phase in ['train', 'test']:
                history[f'{phase}_{disease}_acc'] = []
                history[f'{phase}_{disease}_auc'] = []
                history[f'{phase}_{disease}_pr_auc'] = []
                history[f'{phase}_{disease}_f1'] = []   
    
    # Verify history structure is complete
    required_keys = ['train_loss', 'test_loss', 'learning_rates']
    for disease in disease_labels:
        for phase in ['train', 'test']:
            required_keys.extend([f'{phase}_{disease}_acc', f'{phase}_{disease}_auc',
                                    f'{phase}_{disease}_pr_auc', f'{phase}_{disease}_f1'])
    
    for key in required_keys:
        if key not in history:
            print(f"Adding missing key {key} to history")
            history[key] = []
            
    print(f"History structure verified with keys: {list(history.keys())}")

    # Store final metrics for the last epoch only
    # Phase-specific predictions and labels (for the current epoch only)
    phase_preds = {phase: {disease: [] for disease in disease_labels} for phase in ['train', 'test']}
    phase_labels = {phase: {disease: [] for disease in disease_labels} for phase in ['train', 'test']}
    
    # Track if we have a new best model in the current epoch
    new_best_model = False
    
    for epoch in range(start_epoch, num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Reset predictions and labels for this epoch
        phase_preds = {phase: {disease: [] for disease in disease_labels} for phase in ['train', 'test']}
        phase_labels = {phase: {disease: [] for disease in disease_labels} for phase in ['train', 'test']}

        # Reset the flag at the start of each epoch
        new_best_model = False

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

            # Create a CUDA stream for asynchronous data transfer
            stream = torch.cuda.Stream()

            # Get iterator for the dataloader
            batch_iter = iter(dataloaders[phase])

            # Prefetch the first batch
            try:
                inputs, covariates, labels = next(batch_iter)
                inputs = inputs.to(device, non_blocking=True)
                covariates = covariates.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
            except StopIteration:
                print(f"Warning: {phase} dataloader is empty!")
                continue

            # Initialize batch metrics for logging
            batch_times = []

            # Process all batches
            for i in range(len(dataloaders[phase])):
                batch_start = time.time()

                # Asynchronously prefetch the next batch
                try:
                    if i + 1 < len(dataloaders[phase]):
                        with torch.cuda.stream(stream):
                            next_inputs, next_covariates, next_labels = next(batch_iter)
                            next_inputs = next_inputs.to(device, non_blocking=True)
                            next_covariates = next_covariates.to(device, non_blocking=True)
                            next_labels = next_labels.to(device, non_blocking=True)
                except StopIteration:
                    pass

                # Wait for the current batch to be ready
                torch.cuda.current_stream().wait_stream(stream)

                # Process the current batch
                optimizer.zero_grad()

                with autocast('cuda'):
                    with torch.set_grad_enabled(phase == 'train'):
                        logits = model(inputs, covariates)
                        loss = criterion(logits, labels)

                        # Convert logits to probabilities for metrics
                        with torch.no_grad():
                            probs = torch.sigmoid(logits)
                            preds = (probs >= 0.5).float()
                        
                        if phase == 'train':
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()

                             # Step the scheduler if it's a per-iteration scheduler
                            if isinstance(scheduler, (WarmupExponential, ExponentialDecay)):
                                scheduler.step()

                # Clear cache periodically
                if (i + 1) % 20 == 0:
                    torch.cuda.empty_cache()
                
                batch_size = labels.size(0)
                running_loss += loss.item() * batch_size

                # Store predictions and labels for each disease 
                for j, disease in enumerate(disease_labels):
                    running_corrects[disease] += torch.sum(preds[:, j] == labels[:, j])

                    # Store predictions and true labels for AUC calculation
                    phase_labels[phase][disease].extend(labels[:, j].cpu().numpy())
                    phase_preds[phase][disease].extend(probs[:, j].detach().cpu().numpy())
                
                total_samples += batch_size

                # Calculate batch processing time
                batch_end = time.time()
                batch_time = batch_end - batch_start
                batch_times.append(batch_time)

                # Print batch progress every 20 batches (adjust as needed)
                if (i + 1) % 30 == 0 or i == 0 or i == len(dataloaders[phase]) - 1:
                    avg_time = sum(batch_times) / len(batch_times)
                    eta = avg_time * (len(dataloaders[phase]) - i - 1)
                    
                    # Format times for readability
                    if eta > 60:
                        eta_str = f"{eta//60:.0f}m {eta%60:.0f}s"
                    else:
                        eta_str = f"{eta:.1f}s"
                        
                    print(f"{phase} Batch {i+1}/{len(dataloaders[phase])} | " 
                          f"Time: {batch_time:.2f}s | "
                          f"ETA: {eta_str} | "
                          f"LR: {optimizer.param_groups[0]['lr']:.6f}")

                # Prepare for the next iteration
                try:
                    inputs, covariates, labels = next_inputs, next_covariates, next_labels
                except:
                    break

            # Calculate epoch time
            epoch_time = time.time() - start_time
            epoch_loss = running_loss / total_samples
            history[f'{phase}_loss'].append(epoch_loss)
            # Print overall loss
            print(f'{phase} Loss: {epoch_loss:.4f} (Time: {epoch_time:.2f}s)')
            
            # Calculate and print metrics for each disease
            for i, disease in enumerate(disease_labels):
                epoch_acc = running_corrects[disease].double() / total_samples
                history[f'{phase}_{disease}_acc'].append(epoch_acc.item())

                # Compute epoch metrics
                y_true = np.array(phase_labels[phase][disease])
                y_pred_proba = np.array(phase_preds[phase][disease])
                y_pred = (y_pred_proba >= 0.5).astype(int)
                
                # Calculate AUC if possible (requires both classes to be present)
                try:
                    epoch_auc = roc_auc_score(y_true, y_pred_proba)
                    history[f'{phase}_{disease}_auc'].append(epoch_auc)
                    auc_str = f" - ROC-AUC: {epoch_auc:.4f}"
                except Exception as e:
                    print(f"Warning: Could not calculate ROC AUC for {disease} in {phase} phase: {str(e)}")
                    history[f'{phase}_{disease}_auc'].append(0.5)
                    auc_str = "ROC AUC: N/A (need both classes)"
                    
                # Compute PR-AUC
                try:
                    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
                    epoch_pr_auc = auc(recall_curve, precision_curve)
                    history[f'{phase}_{disease}_pr_auc'].append(epoch_pr_auc)
                    pr_auc_str = f" - PR-AUC: {epoch_pr_auc:.4f}"
                except Exception as e:
                    print(f"Warning: Could not calculate PR-AUC for {disease} in {phase} phase: {str(e)}")
                    history[f'{phase}_{disease}_pr_auc'].append(0.5)
                    pr_auc_str = "PR AUC: N/A (need both classes)"
                    
                # Compute F1 Score
                try:
                    epoch_f1 = f1_score(y_true, y_pred)
                    history[f'{phase}_{disease}_f1'].append(epoch_f1)
                    f1_str = f"- F1: {epoch_f1:.4f}"
                except Exception as e:
                    print(f"Warning: Could not calculate F1 for {disease} in {phase} phase: {str(e)}")
                    history[f'{phase}_{disease}_f1'].append(0.0)
                    f1_str = "F1: N/A (need both classes)"
                    
                
                print(f'  {disease}: Acc: {epoch_acc:.4f}, {auc_str}, {pr_auc_str}, {f1_str}')

            # Early stopping check based on test loss
            if phase == 'test':
                # Check if this is the best model so far before updating early stopping
                if epoch_loss < best_loss:
                    print(f"New best model! Validation loss improved from {best_loss:.6f} to {epoch_loss:.6f}")
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    new_best_model = True
                    
                    # If we have a checkpoint directory, save the best model immediately
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
                        
                        print(f"Saving best model to {best_model_path}")
                        #torch.save(best_checkpoint, best_model_path)
                else:
                    print(f"Validation loss: {epoch_loss:.6f} (not improved from best: {best_loss:.6f})")
                
                # Update early stopping after checking for best model
                if early_stopping is not None:
                    early_stopping(epoch_loss, global_best_loss=best_loss)
                    
                    if early_stopping.early_stop:
                        print("Early stopping triggered")
                        completed_epochs = epoch + 1  # Save the number of completed epochs
                        
                        # Save final checkpoint before early stopping
                        if checkpoint_dir:
                            final_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{completed_epochs}.pt')
                            
                            # Create final checkpoint with current model state
                            final_checkpoint = {
                                'epoch': completed_epochs,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'best_model_state_dict': best_model_wts,
                                'history': history,
                                'best_loss': best_loss,
                                'completed_epochs': completed_epochs
                            }
                            
                            # Add scheduler state if it exists
                            if hasattr(scheduler, 'state_dict'):
                                final_checkpoint['scheduler_state_dict'] = scheduler.state_dict()
                            elif isinstance(scheduler, (WarmupExponential, ExponentialDecay)):
                                final_checkpoint['scheduler_state_dict'] = scheduler.state_dict()
                                
                            # Add early stopping state
                            final_checkpoint['early_stopping_state'] = early_stopping.state_dict()
                            
                            print(f"Saving final checkpoint to {final_checkpoint_path}")
                            torch.save(final_checkpoint, final_checkpoint_path)
                            
                            print("Note: Best model has already been saved separately and is NOT being updated by early stopping")

                        # Load best model before returning
                        model.load_state_dict(best_model_wts)
                        # When early stopping is triggered, return the metrics for the last completed epoch
                        return model, history, compute_final_metrics(phase_labels, phase_preds, disease_labels), phase_preds, phase_labels, completed_epochs
            
        # Step schedulers that work on epoch-level
        if scheduler is not None and not isinstance(scheduler, (WarmupExponential, ExponentialDecay)):
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(history['test_loss'][-1])
            else:
                scheduler.step()

        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        print_lr(optimizer)
        completed_epochs = epoch + 1
        
        # Save checkpoint at the end of each epoch
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
            
            # Add scheduler state if it exists
            if hasattr(scheduler, 'state_dict'):
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            elif isinstance(scheduler, (WarmupExponential, ExponentialDecay)):
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
                
            # Add early stopping state
            if early_stopping is not None:
                checkpoint['early_stopping_state'] = early_stopping.state_dict()
            
            # Save regular epoch checkpoint
            regular_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{completed_epochs}.pt')
            print(f"Saving regular epoch checkpoint to {regular_checkpoint_path}")

            torch.save(checkpoint, regular_checkpoint_path)
            
            # Cleanup old checkpoints (keep only the last n)
            cleanup_old_checkpoints(checkpoint_dir, keep_last_n)

    print(f'Training completed. Best test loss: {best_loss:.4f}')
    # Load best model
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
            
            disease_metrics = {}
            
            try:
                cm = confusion_matrix(y_true, y_pred)
                print(f"  - Raw confusion matrix shape: {cm.shape}")
                print(f"  - Confusion matrix:\n{cm}")
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
                    # ROC curve
                    fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
                    roc_auc = auc(fpr, tpr)

                    # Precision-Recall curve
                    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
                    pr_auc = auc(recall_curve, precision_curve)
                    
                    # F1 score
                    f1_score_val = f1_score(y_true, y_pred)
                    
                except Exception as e:
                    print(f"Error calculating curves for {disease} in {phase}: {str(e)}")
                    fpr, tpr, roc_thresholds = np.array([]), np.array([]), np.array([])
                    precision_curve, recall_curve, pr_thresholds = np.array([]), np.array([]), np.array([])
                    roc_auc = pr_auc = 0.5
                    f1_score_val = 0.0
            except Exception as e:
                print(f"Error calculating metrics for {disease} in {phase}: {str(e)}")
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
                # Store curve data for plotting
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

    # 5. Average PR AUC (epoch-wise progression)
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
    
    # 6. Average F1 Score (epoch-wise progression)
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
    
    plt.suptitle('Enhanced Multilabel Disease Prediction Model Performance', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plt.savefig(os.path.join(plots_dir, 'combined_metrics_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Individual disease metrics plots (accuracy, ROC AUC, PR AUC, and F1)
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

    # ROC Curves by Disease (if final_metrics provided)
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
                                 t_total=total_steps, decay_factor= args.df)
    elif scheduler_name == "exponential_decay":
        return ExponentialDecay(optimizer, start_lr=args.lr, final_lr=args.final_lr, 
                                total_steps=total_steps, decay_factor= args.df)
    elif scheduler_name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, 
                                                    factor=0.1, threshold=0.0001)
    elif scheduler_name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs-10)
    elif scheduler_name == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_name == "multistep":
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
    elif scheduler_name == "explr":
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_name}")

def main():
    print("Starting multilabel disease prediction model...")
    
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    args = parse_args()
    
    # Validate hardcoded parameters if using hardcoded mode
    if args.use_multi_scale and args.multi_scale_mode == "hardcoded":
        validate_hardcoded_parameters(args)

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

    # Check for NaNs in phenotype data
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

    # Continue with analysis using filtered_file_list 
    first_genotype_file = filtered_file_list[0] if filtered_file_list else None
    if first_genotype_file:
        print(f"First genotype file after filtering is: {first_genotype_file}")
        input_size = get_input_size(first_genotype_file)
        print(f"Dynamically determined input size: {input_size}")
    else:
        print("No matching genotype files found!")
        return

    if len(filtered_file_list) != len(phenotype_samples):
        print(f"Warning: Number of files ({len(filtered_file_list)}) does not match number of samples ({len(phenotype_samples)}) in phenotype data.")

    # Split data
    train_files, test_files = train_test_split(filtered_file_list, test_size=0.2, random_state=42)
    print(f"Data split: Train {len(train_files)}, Test {len(test_files)}")

    # Create model
    model = MultilabelGenotypeModel(
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
        use_transformer=bool(args.use_transformer),
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        transformer_dim=args.transformer_dim,
        transformer_ff_dim=args.transformer_ff_dim,
        transformer_dropout=args.transformer_dropout,
        use_positional_encoding=bool(args.use_positional_encoding),
        max_seq_len=args.max_seq_len,
        init_from_pretrained=bool(args.init_from_pretrained),
        pretrained_model_name=args.pretrained_model_name,
        init_layers_fraction=args.init_layers_fraction,
        layer_init_strategy=args.layer_init_strategy,
        custom_layer_indices=args.custom_layer_indices,
        use_cls_token=bool(args.use_cls_token),
        use_covariate_tokens=bool(args.use_covariate_tokens),
        covariate_embed_dim=args.covariate_embed_dim,
        covariate_token_strategy=args.covariate_token_strategy,
        pooling_strategy=args.pooling_strategy
    )

    model = model.to(device)
    print("Model created and moved to device")
    
    with open(os.path.join(experiment_dir, 'model_architecture.txt'), 'w') as file:
        file.write(str(model))
    print(model)

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
    best_model_wts = None

    # Check for existing checkpoints if -resume is enabled
    if args.resume:
        latest_checkpoint = find_latest_checkpoint(experiment_dir)
        if latest_checkpoint:
            print(f"Loading checkpoint: {latest_checkpoint}")
            try:
                checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
                
                # Load model state
                model.load_state_dict(checkpoint['model_state_dict'])
                print("Loaded model weights from checkpoint")
                
                # Keep track of best weights separately if available
                if 'best_model_state_dict' in checkpoint and checkpoint['best_model_state_dict'] is not None:
                    best_model_wts = checkpoint['best_model_state_dict']
                    print("Loaded best model weights from checkpoint")
                
                # Load optimizer state
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Loaded optimizer state from checkpoint")
                
                # Get the start epoch and history
                start_epoch = checkpoint.get('epoch', 0)

                # Check history in checkpoint
                if 'history' in checkpoint and checkpoint['history'] is not None:
                    history = checkpoint['history']
                    print(f"Loaded history from checkpoint with keys: {list(history.keys())}")
                else:
                    print("No valid history found in checkpoint, will create new history")
                    history = None

                best_loss = checkpoint.get('best_loss', float('inf'))
                
                print(f"Resuming from epoch {start_epoch} (total epochs completed so far: {start_epoch})")
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

    if history is not None:
        print(f"DEBUG: History loaded from checkpoint with {len(history.get('train_loss', []))} epochs of data")
    else:
        print("DEBUG: History is None after loading checkpoint")

    # Create datasets (after potential checkpoint loading)
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

    # Get the fitted normalizers from training dataset
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

    # Set up loss function with optional weighting
    if args.use_weighted_loss:
        print(f"\nUsing weighted BCE loss with strategy: {args.weight_strategy}")
        
        # Parse custom weights if provided
        custom_weights = None
        if args.weight_strategy == "custom":
            try:
                custom_weights = [float(w.strip()) for w in args.custom_weights.split(',')]
                if len(custom_weights) != len(disease_labels):
                    raise ValueError(f"Number of custom weights ({len(custom_weights)}) must match number of diseases ({len(disease_labels)})")
            except Exception as e:
                print(f"Error parsing custom weights: {e}")
                print("Falling back to inverse_freq strategy")
                args.weight_strategy = "inverse_freq"
                custom_weights = None
        
        # Calculate class weights
        class_weights = calculate_class_weights(
            train_dataset, 
            disease_labels, 
            strategy=args.weight_strategy,
            custom_weights=custom_weights,
            smoothing=args.weight_smoothing
        )
        
        # Move weights to device
        pos_weights = class_weights.to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        
        print(f"Weighted BCE loss initialized with pos_weights: {pos_weights.cpu().tolist()}")
        
    else:
        print("Using standard BCE loss (no class weighting)")
        criterion = nn.BCEWithLogitsLoss()


    # Create dataloaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True),
        'test': DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    }
    print("DataLoaders created")
    
    # Create scheduler (after potential checkpoint loading)
    scheduler = get_scheduler(args.sch, optimizer, args, train_files)
    
    # Load scheduler state if it exists in checkpoint
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
    
    # Load early stopping state if it exists in checkpoint
    if args.resume and latest_checkpoint and 'early_stopping_state' in checkpoint:
        try:
            early_stopping.load_state_dict(checkpoint['early_stopping_state'])
            print("Loaded early stopping state from checkpoint")
        except Exception as e:
            print(f"Warning: Failed to load early stopping state: {e}")

    # Initialize history dict if starting fresh
    if history is None:
        history = {
            'train_loss': [], 'test_loss': [],
            'learning_rates': []
        }
        
        # Add metrics for each disease
        for disease in disease_labels:
            for phase in ['train', 'test']:
                history[f'{phase}_{disease}_acc'] = []
                history[f'{phase}_{disease}_auc'] = []
                history[f'{phase}_{disease}_pr_auc'] = []  
                history[f'{phase}_{disease}_f1'] = []     

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

    # Prepare hyperparameter dict
    hyperparameters = {
        'Exp_ID': id,
        'Batch_Size': args.bs,
        'Epochs': args.epochs,
        'Completed_Epochs': completed_epochs,
        'Start_LR': args.lr,
        'Peak_LR': args.peak_lr,
        'Final_LR': optimizer.param_groups[0]["lr"],
        'Dropout': args.dropout,
        'Act': args.act,
        'Opt': args.opt,
        'Sch': args.sch,
        'WD': args.wd,
        'DF': args.df,
        'Use_PCs': bool(args.cov),
        'norm_PCs': args.norm_pcs,
        'Use_Age': bool(args.use_age),
        'norm_Age': args.norm_age,
        'Use_Gender': bool(args.use_gender),
        'norm_Gender': args.norm_gender,
        'Use_Bmi': bool(args.use_bmi),
        'norm_Bmi': args.norm_bmi,
        'Kernel_sizes': str(args.kernel_sizes),
        'Stride': str(args.stride),
        'Conv_channels': str(args.conv_channels),
        'Use_Pooling': bool(args.use_pooling),
        'Pool_size': args.pool_size if args.use_pooling else 'N/A',
        'Pool_type': args.pool_type if args.use_pooling else 'N/A',
        'FC_layers': str(args.fc_layers),
        'Num_Diseases': len(disease_labels),
        'Disease_Labels': ','.join(disease_labels),
        'Use_Multi_Scale': bool(args.use_multi_scale),
        'Multi_Scale_Mode': args.multi_scale_mode if args.use_multi_scale else 'N/A',
        'Multi_Scale_Fusion': args.multi_scale_fusion if args.use_multi_scale else 'N/A',
        'use_pointwise_conv': bool(args.use_pointwise_conv),
        'Pointwise_Channels': args.pointwise_channels if args.use_pointwise_conv else 'N/A',
        'Use_Disease_Attention': bool(args.use_disease_attention),
        'Use_Separate_Heads': bool(args.use_separate_heads),
        'Attention_Heads': args.attention_heads if args.use_disease_attention else 'N/A',
        'Attention_Dim': args.attention_dim if args.use_disease_attention else 'N/A',
        'Multi_Scale_Kernels': str(args.multi_scale_kernels) if args.use_multi_scale and args.multi_scale_mode == 'progressive' else 'N/A',
        'Multi_Scale_Strides': str(args.multi_scale_strides) if args.use_multi_scale and args.multi_scale_mode == 'progressive' else 'N/A',
        'Hardcoded_Kernels': str(args.hardcoded_kernels) if args.use_multi_scale and args.multi_scale_mode == 'hardcoded' else 'N/A',
        'Hardcoded_Strides': str(args.hardcoded_strides) if args.use_multi_scale and args.multi_scale_mode == 'hardcoded' else 'N/A',
        'Use_Transformer': bool(args.use_transformer),
        'Transformer_Layers': args.transformer_layers if args.use_transformer else 'N/A',
        'Transformer_Heads': args.transformer_heads if args.use_transformer else 'N/A',
        'Transformer_Dim': args.transformer_dim if args.use_transformer else 'N/A',
        'Transformer_FF_Dim': args.transformer_ff_dim if args.use_transformer else 'N/A',
        'Transformer_Dropout': args.transformer_dropout if args.use_transformer else 'N/A',
        'Use_Positional_Encoding': bool(args.use_positional_encoding) if args.use_transformer else 'N/A',
        'Max_Seq_Len': args.max_seq_len if args.use_transformer else 'N/A',
        'Init_From_Pretrained': bool(args.init_from_pretrained) if args.use_transformer else 'N/A',
        'Pretrained_Model_Name': args.pretrained_model_name if (args.use_transformer and args.init_from_pretrained) else 'N/A',
        'Init_Layers_Fraction': args.init_layers_fraction if (args.use_transformer and args.init_from_pretrained) else 'N/A',
        'Use_CLS_Token': bool(args.use_cls_token) if args.use_transformer else 'N/A',
        'Use_Covariate_Tokens': bool(args.use_covariate_tokens) if args.use_transformer else 'N/A',
        'Covariate_Token_Strategy': args.covariate_token_strategy if (args.use_transformer and args.use_covariate_tokens) else 'N/A',
        'Pooling_Strategy': args.pooling_strategy if (args.use_transformer and not args.use_cls_token) else 'N/A',
        'Covariate_Embed_Dim': args.covariate_embed_dim if (args.use_transformer and args.use_covariate_tokens) else 'N/A',
        'Use_Weighted_Loss': bool(args.use_weighted_loss),
        'Weight_Strategy': args.weight_strategy if args.use_weighted_loss else 'N/A',
        'Weight_Smoothing': args.weight_smoothing if args.use_weighted_loss else 'N/A',
        'Custom_Weights': args.custom_weights if (args.use_weighted_loss and args.weight_strategy == 'custom') else 'N/A',
    }

    # Write results
    write_results(model, hyperparameters, final_metrics, disease_labels, experiment_dir)
    print("Results written to file")

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