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
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
import argparse
import csv
import time
import shutil
from collections import OrderedDict
import random
import torch.nn.functional as F

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
    parser = argparse.ArgumentParser(description="Enhanced CNN-Based Genotype Model Training")
    parser.add_argument("-ID", type=str, default="Exp_01", help="ID of the experiment")
    parser.add_argument("-exp_dir", type=str, default='/mnt/fast/nobackup/scratch4weeks/if00208/disease_wise_singlescale', help="Directory to save experiment results")
    parser.add_argument("-genotype_dir", type=str, default='/mnt/fast/datasets/ucdatasets/gwas/gwas_mono_rm/sampled_data_5M_unq_npy_dup/sampled_data_5M_unq_npy', help="Directory containing genotype files")
    parser.add_argument("-phenotype_file", type=str, default='/mnt/fast/datasets/ucdatasets/gwas/data_files/merged_v8_pcs_chip_added_Iqra_1_cleaned.xlsx', help="Path to phenotype file")

    # Model and training parameters
    parser.add_argument("-bs", type=int, default=5, help="Batch size for training")
    parser.add_argument("-dropout", type=float, default=0.5, help="Dropout rate for the model")
    parser.add_argument("-epochs", type=int, default=2, help="Number of epochs for training")
    parser.add_argument("-lr", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("-peak_lr", type=float, default=1e-2, help="Peak learning rate for WarmupExponential scheduler")
    parser.add_argument("-final_lr", type=float, default=1e-5, help="Final learning rate for custom schedulers")
    parser.add_argument("-act", type=str, default="gelu", choices=["tanh","relu","gelu"], help="Activation function for the model")
    parser.add_argument("-sch", type=str, default="exponential_decay", choices=["none","plateau", "cosine", "step","multistep","explr","warmup_exponential", "exponential_decay"], help="Learning rate scheduler")
    parser.add_argument("-df", type=float, default=0.1, help="Decay factor for custom schedulers")
    parser.add_argument("-opt", type=str, default="adamw", choices=["adam", "adamw", "sgd"], help="Optimizer to use")
    parser.add_argument("-wd", type=float, default=0.5, help="Weight decay for optimizer")

    # Model architecture
    parser.add_argument("-kernel_sizes", type=parse_int_list, default=[128,64,32], help="Convolution Kernel Size")
    parser.add_argument("-stride", type=parse_int_list, default=[16,16,16], help="Convolution Stride")
    parser.add_argument("-conv_channels", type=parse_int_list, default=[4,8,16], help="Convolution channels")
    parser.add_argument("-fc_layers", type=parse_int_list, default=[128,64], help="Fully connected layers")

    # Enhanced architecture parameters
    parser.add_argument("-use_multi_scale", type=int, default=0, choices=[0, 1], help="Whether to use multi-scale convolutions (0: no, 1: yes)")
    
    # Multi-scale configuration
    parser.add_argument("-multi_scale_kernels", type=parse_int_list, default=[15,127], help="Multi-scale kernel sizes for first layer")
    parser.add_argument("-multi_scale_strides", type=parse_int_list, default=[8,16], help="Multi-scale strides for first layer")
    parser.add_argument("-multi_scale_fusion", type=str, default="parallel", choices=["cross_scale", "parallel"], help="Multi-scale fusion strategy: cross_scale (branches see all scales) or parallel (independent branches)")

    # Multi-scale mode selection
    parser.add_argument("-multi_scale_mode", type=str, default="hardcoded", choices=["progressive", "hardcoded"], help="Multi-scale mode: 'progressive' (kernel//2^i, stride//2^i) or 'hardcoded' (explicit values for each layer)")
    
    # Hardcoded multi-scale parameters (used when multi_scale_mode="hardcoded")
    parser.add_argument("-hardcoded_kernels", type=parse_nested_int_list, default='128,1024;64,512;32,256', help="Hardcoded kernel sizes for all layers and branches. Format: 'layer1_branch1,layer1_branch2;layer2_branch1,layer2_branch2'. Example: '15,63,255;7,31,127;3,15,63'")
    parser.add_argument("-hardcoded_strides", type=parse_nested_int_list, default='16,16;16,16;16,16', help="Hardcoded stride values for all layers and branches. Format: 'layer1_branch1,layer1_branch2;layer2_branch1,layer2_branch2'. Example: '4,16,64;2,8,32;1,4,16'")

    # Pointwise convolution parameters
    parser.add_argument("-use_pointwise_conv", type=int, default=1, choices=[0, 1], help="Whether to use pointwise (1x1) convolution after each branch before concatenation (0: no, 1: yes)")
    parser.add_argument("-pointwise_channels", type=int, default=4, help="Number of output channels for pointwise convolution (applied to each branch)")

    # Data-specific parameters
    parser.add_argument("-cov", type=int, default=0, choices=[0, 1], help="Whether to include PC's as covariates in the model (0: no, 1: yes)")
    parser.add_argument("-use_age", type=int, default=0, choices=[0, 1], help="Whether to include age in covariates (0: no, 1: yes)")
    parser.add_argument("-use_gender", type=int, default=0, choices=[0, 1], help="Whether to include gender in covariates (0: no, 1: yes)")
    parser.add_argument("-use_bmi", type=int, default=0, choices=[0, 1], help="Whether to include BMI in covariates in the model (0: no, 1: yes)")
    
    # Early stopping parameters
    parser.add_argument("-patience", type=int, default=15, help="Patience for early stopping")
    parser.add_argument("-min_delta", type=float, default=1e-4, help="Minimum change for early stopping")

    # Normalization-related arguments
    parser.add_argument("-norm_age", type=str, default="none", choices=["none", "standard", "minmax", "robust", "quantile", "power"], help="Normalization method for age")
    parser.add_argument("-norm_pcs", type=str, default="none", choices=["none", "standard", "minmax", "robust", "quantile", "power"], help="Normalization method for PCs")
    parser.add_argument("-norm_gender", type=str, default="none", choices=["none", "minmax"], help="Normalization method for gender (usually keep as none)")
    parser.add_argument("-norm_bmi", type=str, default="none", choices=["none", "standard", "minmax", "robust", "quantile", "power"], help="Normalization method for BMI")
    parser.add_argument("-label_col", type=str, default="t2dm", help="Column name in phenotype file to use as label")

    parser.add_argument("-use_pooling", type=int, default=0, choices=[0, 1], help="Whether to use Pooling after convolution layers (0: no, 1: yes)")
    parser.add_argument("-pool_size", type=int, default=256, help="Size of the adaptive pooling output")

    # Pool Type for all pooling if used
    parser.add_argument("-pool_type", type=str, default="max", choices=["max", "avg"], help="Type of adaptive pooling: 'max' for AdaptiveMaxPool1d, 'avg' for AdaptiveAvgPool1d")
    
    # Model type argument
    parser.add_argument("-model_type", type=str, default="snps_only", choices=["full", "snps_only"], help="Type of model to use")
    
    # Class weighting for imbalanced data
    parser.add_argument("-class_weight", type=int, default=0, choices=[0, 1], help="Whether to use class weighting (0: no, 1: yes)")
    parser.add_argument("-pos_weight_scale", type=float, default=1.2, help="Additional scaling factor for positive class weight")
    
    # Random seed for reproducibility
    parser.add_argument("-random_seed", type=int, default=42, help="Random seed for train-test split and model initialization")
    
    # Class imbalance handling methods
    parser.add_argument("-sampling", type=str, default="none", choices=["none", "weighted", "balanced_batch"], help="Sampling/weighting method to handle class imbalance")
    parser.add_argument("-sampling_ratio", type=float, default=0.8, help="Desired ratio of minority to majority class after sampling")
    
    # Threshold adjustment
    parser.add_argument("-threshold", type=float, default=0.5, help="Classification threshold (lower values favor recall)")
    
    # Loss function choice
    parser.add_argument("-loss_fn", type=str, default="bce", choices=["bce", "focal"], help="Loss function to use (bce: Binary Cross Entropy, focal: Focal Loss)")
    parser.add_argument("-focal_alpha", type=float, default=0.25, help="Alpha parameter for Focal Loss")
    parser.add_argument("-focal_gamma", type=float, default=2.0, help="Gamma parameter for Focal Loss")
    
    
    # Checkpoint-related parameters
    parser.add_argument("-resume", type=int, default=1, choices=[0, 1], help="Whether to resume from checkpoint if available (0: no, start fresh; 1: yes, resume if available)")
    parser.add_argument("-keep_checkpoints", type=int, default=1, help="Number of recent checkpoints to keep")
    
    return parser.parse_args()

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

# Focal Loss implementation for handling class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        else:
            return F_loss

class GenotypeDataset(Dataset):
    def __init__(self, file_list, phenotype_data, label_column, use_covariates=True, use_age=False, 
                 use_gender=False, use_bmi=False, norm_age="none", norm_pcs="none", norm_gender="none", norm_bmi="none",
                 fit_normalizers=True, normalizers=None):
        self.file_list = file_list
        self.phenotype_data = phenotype_data
        self.label_column = label_column
        self.use_covariates = use_covariates
        self.use_age = use_age
        self.use_gender = use_gender
        self.use_bmi = use_bmi
        self.label_tensor = torch.tensor([self.phenotype_data.loc[self.phenotype_data['new_order'] == int(os.path.basename(f).replace("sample_", "").replace(".npy", "")),
                        self.label_column].values[0] for f in self.file_list], dtype=torch.float32)

        # Verify that the label column exists in the phenotype data
        if self.label_column not in self.phenotype_data.columns:
            raise ValueError(f"Label column '{self.label_column}' not found in phenotype data. "
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
        print(f"- Label column: {label_column}")
        print(f"- Using PCs: {use_covariates} (normalization: {norm_pcs})")
        print(f"- Using age: {use_age} (normalization: {norm_age})")
        print(f"- Using gender: {use_gender} (normalization: {norm_gender})")
        print(f"- Using BMI: {use_bmi} (normalization: {norm_bmi})")

    def _fit_normalizers(self):
        """Fit all normalizers on training data"""
        if self.use_covariates:
            # Get all PC data as a matrix (n_samples, n_pcs)
            pc_data = np.array([self.phenotype_data[f'PC{i}'].values for i in range(1, 11)]).T
            self.pcs_normalizer.fit(pc_data)
        
        if self.use_age and 'Agexit' in self.phenotype_data.columns:
            age_data = self.phenotype_data['Agexit'].values
            self.age_normalizer.fit(age_data)
        
        if self.use_gender and 'Sex' in self.phenotype_data.columns:
            gender_data = self.phenotype_data['Sex'].values
            self.gender_normalizer.fit(gender_data)
        
        if self.use_bmi and 'Bmi_C' in self.phenotype_data.columns:
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
        
        # Use the configured label column
        label = self.phenotype_data.loc[self.phenotype_data['new_order'] == sample_id, self.label_column].values[0]
        
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
        if self.use_age and 'Agexit' in self.phenotype_data.columns:
            # Get and normalize age
            age = self.phenotype_data.loc[self.phenotype_data['new_order'] == sample_id, 'Agexit'].values[0]
            normalized_age = self.age_normalizer.transform(np.array([[age]])).flatten()
            covariates_list.append(normalized_age)
        
        # Gender
        if self.use_gender and 'Sex' in self.phenotype_data.columns:
            # Get and normalize gender
            gender = self.phenotype_data.loc[self.phenotype_data['new_order'] == sample_id, 'Sex'].values[0]
            normalized_gender = self.gender_normalizer.transform(np.array([[gender]])).flatten()
            covariates_list.append(normalized_gender)
        
        # BMI
        if self.use_bmi and 'Bmi_C' in self.phenotype_data.columns:
            # Get and normalize BMI
            bmi = self.phenotype_data.loc[self.phenotype_data['new_order'] == sample_id, 'Bmi_C'].values[0]
            normalized_bmi = self.bmi_normalizer.transform(np.array([[bmi]])).flatten()
            covariates_list.append(normalized_bmi)
        
        # Combine all covariates
        if covariates_list:
            covariates = np.concatenate(covariates_list)
        else:
            covariates = np.array([])
        
        covariates_tensor = torch.tensor(covariates, dtype=torch.float32)

        if '.npy' in genotype_file:
            genotype_data = np.load(genotype_file)  # Shape: (5M, 3)
            genotype_tensor = torch.from_numpy(genotype_data).float()

        label_tensor = torch.tensor(label, dtype=torch.float32)

        return genotype_tensor, covariates_tensor, label_tensor

    
            
# Creating balanced sampler for weighted sampling
def create_balanced_sampler(train_dataset, sampling_ratio=0.8):
    print(f"\nCreating balanced sampler with target ratio {sampling_ratio}")
    
    # Get labels for all samples
    all_labels = []
    for i in range(len(train_dataset)):
        _, _, label = train_dataset[i]
        all_labels.append(label.item())
    
    # Count class distribution
    class_counts = {}
    for label in all_labels:
        label_int = int(label)
        if label_int not in class_counts:
            class_counts[label_int] = 0
        class_counts[label_int] += 1
    
    print(f"Class distribution: {class_counts}")
    
    # Determine majority and minority classes
    if len(class_counts) != 2:
        print(f"Warning: Expected 2 classes, found {len(class_counts)}. Using default weights.")
        weights = torch.ones(len(all_labels))
    else:
        # Find majority and minority class
        class_0_count = class_counts.get(0, 0)
        class_1_count = class_counts.get(1, 0)
        
        if class_0_count >= class_1_count:
            majority_class, majority_count = 0, class_0_count
            minority_class, minority_count = 1, class_1_count
        else:
            majority_class, majority_count = 1, class_1_count
            minority_class, minority_count = 0, class_0_count
        
        print(f"Majority class: {majority_class} (count: {majority_count})")
        print(f"Minority class: {minority_class} (count: {minority_count})")
        print(f"Current ratio: {minority_count / majority_count:.4f}")
        
        # Calculate target count for minority class
        target_minority_count = int(majority_count * sampling_ratio)
        print(f"Target minority count: {target_minority_count}")
        
        # Calculate weights for each class
        # Weight = 1.0 for majority class
        # Weight = target_minority_count / actual_minority_count for minority class
        weight_for_minority = target_minority_count / minority_count if minority_count > 0 else 1.0
        weight_for_majority = 1.0
        
        print(f"Weight for majority class: {weight_for_majority}")
        print(f"Weight for minority class: {weight_for_minority}")
        
        # Create sample weights
        weights = torch.ones(len(all_labels))
        for i, label in enumerate(all_labels):
            if int(label) == minority_class:
                weights[i] = weight_for_minority
            else:
                weights[i] = weight_for_majority
    
    # Create a sampler with replacement
    from torch.utils.data import WeightedRandomSampler
    
    # The number of samples to draw equals the dataset size
    # We use replacement to enable oversampling
    num_samples = len(all_labels)
    
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples,
        replacement=True
    )
    
    print(f"Created WeightedRandomSampler with {num_samples} samples per epoch")
    return sampler

# Balanced batch sampler for balanced mini-batches
class BalancedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, pos_ratio=0.5, pos_class=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.pos_ratio = pos_ratio
        self.pos_class = pos_class
        
        # Group sample indices by class
        self.pos_indices = []
        self.neg_indices = []
        
        print("Analyzing dataset for balanced batch sampling...")
        for i in range(len(dataset)):
            _, _, label = dataset[i]
            if int(label.item()) == pos_class:
                self.pos_indices.append(i)
            else:
                self.neg_indices.append(i)
        
        # Calculate number of samples of each class per batch
        self.pos_per_batch = max(1, int(batch_size * pos_ratio))
        self.neg_per_batch = batch_size - self.pos_per_batch
        
        # Calculate number of batches per epoch
        pos_batches = len(self.pos_indices) // self.pos_per_batch
        neg_batches = len(self.neg_indices) // self.neg_per_batch
        self.batches_per_epoch = min(pos_batches, neg_batches)
        
        if self.batches_per_epoch == 0:
            # If we can't create even one batch, adjust the ratio
            print(f"Warning: Cannot create batches with {pos_ratio} positive ratio.")
            print(f"Adjusting to use all available positive samples.")
            self.pos_per_batch = min(len(self.pos_indices), batch_size - 1)
            self.neg_per_batch = batch_size - self.pos_per_batch
            self.batches_per_epoch = 1
            
        samples_per_epoch = self.batches_per_epoch * batch_size
        
        print(f"Balanced Batch Sampler initialized:")
        print(f"- Positive samples: {len(self.pos_indices)}")
        print(f"- Negative samples: {len(self.neg_indices)}")
        print(f"- Positive samples per batch: {self.pos_per_batch} ({self.pos_per_batch/batch_size:.1%})")
        print(f"- Negative samples per batch: {self.neg_per_batch} ({self.neg_per_batch/batch_size:.1%})")
        print(f"- Batches per epoch: {self.batches_per_epoch}")
        print(f"- Samples per epoch: {samples_per_epoch} (vs. {len(dataset)} total)")
        
    def __iter__(self):
        # Shuffle indices for each class
        pos_indices = self.pos_indices.copy()
        neg_indices = self.neg_indices.copy()
        random.shuffle(pos_indices)
        random.shuffle(neg_indices)
        
        # For smaller positive set, cycle through if needed
        if len(pos_indices) < self.batches_per_epoch * self.pos_per_batch:
            # Calculate how many times to repeat positive indices
            repeats = (self.batches_per_epoch * self.pos_per_batch + len(pos_indices) - 1) // len(pos_indices)
            pos_indices = pos_indices * repeats
            print(f"Repeating positive samples {repeats} times to fill batches")
        
        # Generate batches with fixed class ratio
        for i in range(self.batches_per_epoch):
            batch = []
            
            # Add positive samples
            start_idx = i * self.pos_per_batch
            end_idx = (i + 1) * self.pos_per_batch
            batch.extend(pos_indices[start_idx:end_idx])
            
            # Add negative samples
            start_idx = i * self.neg_per_batch
            end_idx = (i + 1) * self.neg_per_batch
            batch.extend(neg_indices[start_idx:end_idx])
            
            # Shuffle within batch for randomness
            random.shuffle(batch)
            
            # Yield entire batch as one list
            yield batch
    
    def __len__(self):
        return self.batches_per_epoch * self.batch_size

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

class EnhancedGenotypeModel(nn.Module):
    def __init__(self, input_size, kernel_sizes, stride, conv_channels, fc_layers, act, dropout_rate, 
                use_covariates=True, use_age=False, use_gender=False, use_bmi=False, num_pc_covariates=10, 
                use_pooling=True, pool_size=16, pool_type="max",
                use_multi_scale=True, multi_scale_kernels=None, multi_scale_strides=None,
                multi_scale_fusion="cross_scale", multi_scale_mode="progressive", 
                hardcoded_kernels=None, hardcoded_strides=None,
                use_pointwise_conv=False, pointwise_channels=16):
        super(EnhancedGenotypeModel, self).__init__()
        self.input_channels = 3
        self.use_covariates = use_covariates
        self.use_age = use_age
        self.use_gender = use_gender
        self.use_bmi = use_bmi
        self.use_pooling = use_pooling
        self.pool_size = pool_size
        self.pool_type = pool_type
        self.use_multi_scale = use_multi_scale
        self.use_pointwise_conv = use_pointwise_conv
        self.pointwise_channels = pointwise_channels
        
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
        self.num_covariates = 0
        if use_covariates:
            self.num_covariates += num_pc_covariates  # PCs
        if use_age:
            self.num_covariates += 1  # Age
        if use_gender:
            self.num_covariates += 1  # Gender
        if use_bmi:
            self.num_covariates += 1  # BMI

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
            
            # Update for multi-scale concatenation and pointwise conv
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
        
        # Calculate output size after convolutions (including pooling if enabled)
        self.conv_output_size = self._get_conv_output_size(input_size)
        print(f"Convolutional output size: {self.conv_output_size}")
        
        # Create fully connected layers
        fc_layers_list = []
        in_features = self.conv_output_size + self.num_covariates
        
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
        self.final_output = nn.Linear(in_features, 1)
        
        # Print model configuration
        architecture_info = []
        architecture_info.append(f"Multi-scale convolutions: {use_multi_scale}")
        if use_multi_scale:
            architecture_info.append(f"Final pointwise convolutions: {use_pointwise_conv}")
        else:
            architecture_info.append(f"Single-scale pointwise convolutions: {use_pointwise_conv}")
        if use_pointwise_conv:
            architecture_info.append(f"Pointwise channels: {pointwise_channels}") 
        architecture_info.append(f"Using PC's: {use_covariates}, age: {use_age}, gender: {use_gender}, BMI:{use_bmi}")
        architecture_info.append(f"Using final pooling: {use_pooling} ({pool_type} pool, size={pool_size})" if use_pooling else "Using final pooling: False")
        
        print(f"\nEnhanced GenotypeModel initialized:")
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

    def _get_conv_output_size(self, input_size):
        """Calculate output size after convolution layers (including pooling if enabled)"""
        try:
            x = torch.randn(1, 3, input_size, dtype=torch.float32)
            print(f"  - Input to conv layers: {x.shape}")
            
            x = self.conv_layers(x)
            print(f"  - Output from conv layers: {x.shape}")
            
            flattened_size = x.numel() // x.size(0)
            print(f"  - Flattened size: {flattened_size:,}")
            return flattened_size
            
        except Exception as e:
            print(f"Error in _get_conv_output_size: {e}")
            print(f"Input size: {input_size}")
            print(f"Conv layers: {self.conv_layers}")
            raise e

    def forward(self, x, covariates=None):
        # Input: x shape [batch_size, n_snps, 3]
        x = x.permute(0, 2, 1)  # -> [batch_size, 3, n_snps]
        
        # Convolutional processing
        x = self.conv_layers(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # [batch_size, flattened_features]
        
        # Concatenate with covariates
        if covariates is not None and self.num_covariates > 0:
            x = torch.cat([x, covariates], dim=1)
        
        # Shared feature processing
        shared_features = self.fc_shared(x)
        return self.final_output(shared_features).squeeze(1)
                
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
        best_model_path = os.path.join(dir_path, 'best_model.pt')
        if os.path.exists(best_model_path):
            print(f"Found best model checkpoint: {best_model_path}")
            return best_model_path

    checkpoints = [f for f in os.listdir(dir_path) if f.startswith('checkpoint_epoch_')]
    checkpoints.sort(key=lambda x: int(x.split('checkpoint_epoch_')[1].split('.pt')[0]))
    if checkpoints:
        latest_checkpoint = os.path.join(dir_path, checkpoints[-1])
        print(f"No best model checkpoint found, using latest epoch checkpoint: {latest_checkpoint}")
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

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, device='cuda', 
                early_stopping=None, classification_threshold=0.5, checkpoint_dir=None, 
                start_epoch=0, keep_last_n=2, history=None, initial_best_loss=float('inf')):
    print(f"Training model on device: {device}")
    print(f"Classification threshold: {classification_threshold}")
    print(f"Starting with initial best loss: {initial_best_loss:.6f}")
    
    scaler = GradScaler()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = initial_best_loss
    completed_epochs = start_epoch

    if history is None:
        history = {
            'train_loss': [], 'train_acc': [], 'train_auc': [], 'train_f1': [], 'train_pr_auc': [],
            'test_loss': [], 'test_acc': [], 'test_auc': [], 'test_f1': [], 'test_pr_auc': [],
            'learning_rates': []
        }

    for epoch in range(start_epoch, num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        all_preds = {phase: [] for phase in ['train', 'test']}
        all_labels = {phase: [] for phase in ['train', 'test']}

        # Track if we have a new best model in the current epoch
        new_best_model = False

        for phase in ['train', 'test']:
            start_time = time.time()
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
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
            for i in range(batch_count):
                batch_start = time.time()
                
                # Asynchronously prefetch the next batch
                try:
                    if i + 1 < batch_count:
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
                
                with autocast():
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs, covariates)
                        loss = criterion(outputs, labels)
                        
                        # Convert logits to probabilities for metrics
                        with torch.no_grad():
                            probs = torch.sigmoid(outputs)
                            preds = (probs >= classification_threshold).float()

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
                running_corrects += torch.sum(preds == labels.data)

                all_labels[phase].extend(labels.cpu().numpy())
                all_preds[phase].extend(probs.detach().cpu().numpy())
                total_samples += batch_size

                # Calculate batch processing time
                batch_end = time.time()
                batch_time = batch_end - batch_start
                batch_times.append(batch_time)

                # Print batch progress every 20 batches (adjust as needed)
                if (i + 1) % 30 == 0 or i == 0 or i == batch_count - 1:
                    avg_time = sum(batch_times) / len(batch_times)
                    eta = avg_time * (batch_count - i - 1)
                    
                    if eta > 60:
                        eta_str = f"{eta//60:.0f}m {eta%60:.0f}s"
                    else:
                        eta_str = f"{eta:.1f}s"
                        
                    print(f"{phase} Batch {i+1}/{batch_count} | " 
                          f"Time: {batch_time:.2f}s | "
                          f"ETA: {eta_str} | "
                          f"LR: {optimizer.param_groups[0]['lr']:.6f}")

                # Prepare for the next iteration
                try:
                    inputs, covariates, labels = next_inputs, next_covariates, next_labels
                except:
                    break

            epoch_time = time.time() - start_time
            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples
            
            # Compute epoch metrics
            y_true = np.array(all_labels[phase])
            y_pred_proba = np.array(all_preds[phase])
            y_pred = (y_pred_proba >= classification_threshold).astype(int)
            
            # Calculate AUC if possible (requires both classes to be present)
            try:
                epoch_auc = roc_auc_score(y_true, y_pred_proba)
                auc_str = f" - ROC-AUC: {epoch_auc:.4f}"
            except Exception as e:
                print(f"Warning: Could not calculate AUC for {phase} phase: {str(e)}")
                epoch_auc = 0.5  # Default value if AUC can't be calculated
                auc_str = "ROC AUC: N/A (need both classes)"
            
            # Compute PR-AUC
            try:
                precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
                epoch_pr_auc = auc(recall, precision)
                pr_auc_str = f" - PR-AUC: {epoch_pr_auc:.4f}"
            except Exception as e:
                print(f"Warning: Could not calculate PR-AUC for {phase} phase: {str(e)}")
                epoch_pr_auc = 0.5
                pr_auc_str = "- PR-AUC: N/A"
            
            # Compute F1 Score
            try:
                epoch_f1 = f1_score(y_true, y_pred)
                f1_str = f"- F1: {epoch_f1:.4f}"
            except Exception as e:
                print(f"Warning: Could not calculate F1 score for {phase} phase: {str(e)}")
                epoch_f1 = 0.0
                f1_str = " - F1: N/A"
            
            

            print(f'{phase} Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}{auc_str}{f1_str}{pr_auc_str} (Time: {epoch_time:.2f}s)')

            # Store metrics in history
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())
            history[f'{phase}_auc'].append(epoch_auc)
            history[f'{phase}_f1'].append(epoch_f1)
            history[f'{phase}_pr_auc'].append(epoch_pr_auc)

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
                        best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
                        
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
                        return model, history, compute_final_metrics(all_labels, all_preds, classification_threshold), all_preds, all_labels, completed_epochs

        # Step the scheduler if it's an epoch-wise scheduler
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
    return model, history, compute_final_metrics(all_labels, all_preds, classification_threshold), all_preds, all_labels, completed_epochs

def compute_final_metrics(all_labels, all_preds, threshold=0.5):
    final_metrics = {}

    for phase in ['train', 'test']:
        y_true = np.array(all_labels[phase])
        y_pred_proba = np.array(all_preds[phase])

        # Convert probabilities to binary predictions
        y_pred = (y_pred_proba >= threshold).astype(int)
        
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
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
                # Precision-Recall curve
                precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
                pr_auc = auc(recall, precision)
                
                # F1 score
                f1 = f1_score(y_true, y_pred)
            except Exception as e:
                print(f"Error calculating curves for {phase}: {str(e)}")
                fpr, tpr = np.array([]), np.array([])
                precision, recall = np.array([]), np.array([])
                roc_auc = pr_auc = 0.5
                f1 = 0.0
        except Exception as e:
            print(f"Error calculating metrics for {phase}: {str(e)}")
            cm = np.zeros((2, 2))
            sensitivity = specificity = accuracy = precision = f1 = 0
            fpr, tpr = np.array([]), np.array([])
            roc_auc = pr_auc = 0.5

        final_metrics[phase] = {
            'confusion_matrix': cm,
            'sensitivity': f'{sensitivity:.5f}',
            'specificity': f'{specificity:.5f}',
            'f1_score': f'{f1:.5f}',
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'fpr': fpr,
            'tpr': tpr,
            'precision': precision,
            'recall': recall
        }
    return final_metrics

def plot_all_metrics(history, final_metrics, save_dir):
    phases = ['train', 'test']
    metrics = ['loss', 'acc', 'auc', 'f1', 'pr_auc']
    
    # Create a 3x2 grid for all metrics including PR curve
    fig, axs = plt.subplots(3, 2, figsize=(20, 18))
    fig.suptitle('Model Performance Metrics', fontsize=16)
    
    # Plot the first 5 metrics (loss, acc, auc, f1, pr_auc)
    for i, metric in enumerate(metrics):
        row, col = i // 2, i % 2
        for phase in phases:
            axs[row, col].plot(history[f'{phase}_{metric}'], label=f'{phase}')
        axs[row, col].set_title(f'{metric.upper() if metric in ["auc", "pr_auc"] else metric.capitalize()}')
        axs[row, col].set_xlabel('Epoch')
        axs[row, col].set_ylabel(metric.upper() if metric in ["auc", "pr_auc"] else metric.capitalize())
        axs[row, col].legend()
    
    # Plot learning rate in the remaining subplot
    axs[2, 1].plot(history['learning_rates'])
    axs[2, 1].set_title('Learning Rate')
    axs[2, 1].set_xlabel('Step')
    axs[2, 1].set_ylabel('Learning Rate')
    axs[2, 1].set_yscale('log')  # Use log scale for better visualization
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_plot.png'))
    plt.close()
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    for phase in phases:
        if len(final_metrics[phase]['fpr']) > 0 and len(final_metrics[phase]['tpr']) > 0:
            plt.plot(
                final_metrics[phase]['fpr'], 
                final_metrics[phase]['tpr'], 
                label=f'{phase} (AUC = {final_metrics[phase]["roc_auc"]:.2f})'
            )
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
    plt.close()
    
    # Plot Precision-Recall curves
    plt.figure(figsize=(10, 8))
    for phase in phases:
        if len(final_metrics[phase]['precision']) > 0 and len(final_metrics[phase]['recall']) > 0:
            plt.plot(
                final_metrics[phase]['recall'], 
                final_metrics[phase]['precision'], 
                label=f'{phase} (PR-AUC = {final_metrics[phase]["pr_auc"]:.2f})'
            )
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'pr_curve.png'))
    plt.close()

def write_results(model, hyperparameters, results, save_dir):
    with open(os.path.join(save_dir, 'experiment_results.txt'), 'w') as f:
        f.write("Hyperparameters:\n")
        f.write("----------------\n")
        for key, value in hyperparameters.items():
            f.write(f"{key}: {value}\n")
        f.write("\nResults:\n")
        f.write("--------\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    
    csv_file = os.path.join(save_dir, 'experiment_results.csv')
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=hyperparameters.keys())
        writer.writeheader()
        writer.writerow(hyperparameters)

def append_metrics_to_csv(save_dir, results):
    csv_file = os.path.join(save_dir, 'experiment_results.csv')
    
    # Read existing data
    with open(csv_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        existing_data = next(reader)  
    
    # Update existing data with new metrics
    existing_data.update(results)
    
    # Write updated data back to CSV
    with open(csv_file, 'w', newline='') as csvfile:
        fieldnames = list(existing_data.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(existing_data)

def gradient_based_feature_importance(model, train_loader, target_class, device):
    model.eval()
    total_gradients = None
    total_samples = 0
    
    for batch_input, covariates, batch_target in train_loader:
        batch_size = batch_input.size(0)
        batch_input = batch_input.to(device)
        covariates = covariates.to(device)
        batch_input.requires_grad_(True)
        
        # Forward pass
        output = model(batch_input, covariates)

        # For binary classification with single output
        target = torch.full_like(output, float(target_class))

        # Compute loss
        loss = nn.BCEWithLogitsLoss()(output, target)
        
        # Backward pass
        loss.backward()
        
        # Accumulate gradients
        if batch_input.grad is not None:
            gradients = batch_input.grad.detach().cpu().numpy()
            if total_gradients is None:
                total_gradients = np.zeros((batch_size, *gradients.shape[1:]))
            total_gradients[:batch_size] += gradients
        
        total_samples += batch_size
        
        # Clear gradients for the next iteration
        model.zero_grad()
        batch_input.grad = None
    
    print("total_samples is:", total_samples)
    if total_gradients is not None:
        print("total_gradients shape is:", total_gradients.shape)
        # Compute average gradients
        average_gradients = total_gradients / total_samples
        return average_gradients
    else:
        print("No gradients were computed. Check if your model parameters require gradients.")
        return None

def get_scheduler(scheduler_name, optimizer, args, train_files):
    steps_per_epoch = len(train_files) // args.bs
    total_steps = steps_per_epoch * args.epochs
    warmup_percentage = 0.1
    wsteps = int(total_steps * warmup_percentage)

    if scheduler_name.lower() == "none" or scheduler_name is None:
        # Return a dummy scheduler that doesn't change the learning rate
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
    print("Starting Enhanced CNN-Based Genotype Model Training...")
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
    # Validate hardcoded parameters if using hardcoded mode
    if args.use_multi_scale and args.multi_scale_mode == "hardcoded":
        validate_hardcoded_parameters(args)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
        
    id = str(args.ID)
    print(f"Experiment ID is: {id}\n")
    experiment_dir = os.path.join(args.exp_dir, id)

    genotype_dir = args.genotype_dir
    phenotype_file = args.phenotype_file
    batch_size = args.bs
    dropout_rate = args.dropout

    num_epochs = args.epochs
    learning_rate = args.lr
    act = args.act
    opt = args.opt
    sch = args.sch
    wd = args.wd
    kernel_sizes = args.kernel_sizes
    stride = args.stride
    conv_channels = args.conv_channels
    fc_layers = args.fc_layers    

    # Create experiment folder
    if not os.path.exists(experiment_dir):
        print(f"Result path did not exist but is made now.\nResult Path is {experiment_dir}")
        os.makedirs(experiment_dir)
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load phenotype data
    phenotype_data = pd.read_excel(phenotype_file)
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

    # Get list of genotype files
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

    # Continue with analysis using filtered_file_list instead of file_list
    first_genotype_file = filtered_file_list[0] if filtered_file_list else None
    if first_genotype_file:
        print(f"First genotype file after filtering is: {first_genotype_file}")
        input_size = get_input_size(first_genotype_file)
        print(f"Dynamically determined input size: {input_size}")
    else:
        print("No matching genotype files found!")
        return
    
    if len(filtered_file_list) != len(phenotype_samples):
        print(f"Warning: Number of files ({len(filtered_file_list)}) in {genotype_dir} does not match number of samples ({len(phenotype_samples)}) in phenotype data.")
    
    # # Split data into train and test
    # train_files, test_files = train_test_split(
    #     filtered_file_list, test_size=0.2, random_state=args.random_seed
    # )
    # print(f"Data split: Train {len(train_files)}, Test {len(test_files)}")

    # Extract labels in the same order as filtered_file_list
    sample_ids = [int(os.path.basename(path).split("sample_")[1].split(".npy")[0]) for path in filtered_file_list]
    sample_id_to_label = dict(zip(phenotype_data['new_order'], phenotype_data[args.label_col]))
    labels = [sample_id_to_label[sid] for sid in sample_ids]
    
    # Create stratified splitter
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.random_seed)
    train_idx, test_idx = next(sss.split(filtered_file_list, labels))
    
    # Split the file paths
    train_files = [filtered_file_list[i] for i in train_idx]
    test_files = [filtered_file_list[i] for i in test_idx]
    
    print(f"Data split (stratified): Train {len(train_files)}, Test {len(test_files)}")
    
    # Create datasets with appropriate flags
    train_dataset = GenotypeDataset(
        train_files, 
        phenotype_data, 
        label_column=args.label_col,
        use_covariates=bool(args.cov) and args.model_type == "full",
        use_age=bool(args.use_age) and args.model_type == "full",
        use_gender=bool(args.use_gender) and args.model_type == "full",
        use_bmi=bool(args.use_bmi) and args.model_type == "full",
        norm_age=args.norm_age,
        norm_pcs=args.norm_pcs,
        norm_gender=args.norm_gender,
        norm_bmi=args.norm_bmi,
        fit_normalizers=True,
        normalizers=None)
    
    # Get the fitted normalizers from training dataset
    fitted_normalizers = train_dataset.get_normalizers()
    
    test_dataset = GenotypeDataset(
        test_files, 
        phenotype_data, 
        label_column=args.label_col,
        use_covariates=bool(args.cov) and args.model_type == "full",
        use_age=bool(args.use_age) and args.model_type == "full",
        use_gender=bool(args.use_gender) and args.model_type == "full",
        use_bmi=bool(args.use_bmi) and args.model_type == "full",
        norm_age=args.norm_age,
        norm_pcs=args.norm_pcs,
        norm_gender=args.norm_gender,
        norm_bmi=args.norm_bmi,
        fit_normalizers=False,
        normalizers=fitted_normalizers
    )

    # # Count class distribution in training data
    # train_labels = []
    # for i in range(len(train_dataset)):
    #     _, _, label = train_dataset[i]
    #     train_labels.append(label.item())
    
    # n_samples = len(train_labels)
    # n_positive = sum(train_labels)
    # n_negative = n_samples - n_positive
    
    # pos_ratio = n_positive / n_samples if n_samples > 0 else 0.5
    # neg_ratio = n_negative / n_samples if n_samples > 0 else 0.5
     
    train_labels = train_dataset.label_tensor  # shape: (N,)
    n_samples = train_labels.size(0)
    n_positive = train_labels.sum().item()
    n_negative = n_samples - n_positive
    
    pos_ratio = n_positive / n_samples if n_samples > 0 else 0.5
    neg_ratio = n_negative / n_samples if n_samples > 0 else 0.5

    print(f"Class distribution in training data: Positive: {n_positive} ({pos_ratio:.2%}), Negative: {n_negative} ({neg_ratio:.2%})")

    test_labels = test_dataset.label_tensor  # shape: (N,)
    n_samples_test = test_labels.size(0)
    n_positive_test = test_labels.sum().item()
    n_negative_test = n_samples_test - n_positive_test
    
    pos_ratio_test = n_positive_test / n_samples_test if n_samples_test > 0 else 0.5
    neg_ratio_test = n_negative_test / n_samples_test if n_samples_test > 0 else 0.5

    print(f"Class distribution in test data: Positive: {n_positive_test} ({pos_ratio_test:.2%}), Negative: {n_negative_test} ({neg_ratio_test:.2%})")

    # Create DataLoaders with appropriate sampling strategy
    if args.sampling == "weighted":
        # Use weighted sampling
        print(f"Using weighted sampling with ratio {args.sampling_ratio}")
        train_sampler = create_balanced_sampler(train_dataset, sampling_ratio=args.sampling_ratio)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=4, 
            pin_memory=True
        )
    elif args.sampling == "balanced_batch":
        # Use balanced batch sampling
        print(f"Using balanced batch sampling with ratio {args.sampling_ratio}")
        train_sampler = BalancedBatchSampler(
            train_dataset, 
            batch_size=batch_size,
            pos_ratio=args.sampling_ratio / (1 + args.sampling_ratio)
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_sampler=train_sampler,
            num_workers=4, 
            pin_memory=True
        )
    else:
        # Standard approach without class balancing
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4, 
            pin_memory=True, 
            prefetch_factor=2
        )
    
    # Create test dataloader (never use sampling for test data)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True, 
        prefetch_factor=2
    )
    
    dataloaders = {
        'train': train_loader,
        'test': test_loader
    }

    # Print information for sample items from each dataset
    print("\nSample items from datasets:")
    for dataset_name, dataset in [("Train", train_dataset), ("Test", test_dataset)]:
        print(f"\n{dataset_name} dataset samples:")
        for i in range(3):  # Print info for 3 items from each dataset
            genotype, covariates, label = dataset[i]
            print(f"Item {i}:")
            print(f"  Label: {label.item()}")
            print(f"  Genotype shape: {genotype.shape}")
            print(f"  Covariates shape: {covariates.shape}")
    print("\nDataLoaders created")

    # Calculate number of covariates
    num_covariates = 0
    if bool(args.cov) and args.model_type == "full":
        num_covariates += 10  # PCs
    if bool(args.use_age) and args.model_type == "full":
        num_covariates += 1  # Age
    if bool(args.use_gender) and args.model_type == "full":
        num_covariates += 1  # Gender

    # Create enhanced model with new architecture options including pointwise convolution
    model = EnhancedGenotypeModel(
        input_size=input_size,
        kernel_sizes=kernel_sizes,
        stride=stride,
        conv_channels=conv_channels,
        fc_layers=fc_layers,
        act=act,
        dropout_rate=dropout_rate,
        use_covariates=bool(args.cov) and args.model_type == "full",
        use_age=bool(args.use_age) and args.model_type == "full",
        use_gender=bool(args.use_gender) and args.model_type == "full",
        use_bmi=bool(args.use_bmi) and args.model_type == "full",
        num_pc_covariates=10,
        use_pooling=bool(args.use_pooling),
        pool_size=args.pool_size,
        pool_type=args.pool_type,
        use_multi_scale=bool(args.use_multi_scale),
        multi_scale_kernels=args.multi_scale_kernels,
        multi_scale_strides=args.multi_scale_strides,
        multi_scale_fusion=args.multi_scale_fusion,
        multi_scale_mode=args.multi_scale_mode,
        hardcoded_kernels=args.hardcoded_kernels,
        hardcoded_strides=args.hardcoded_strides,
        use_pointwise_conv=bool(args.use_pointwise_conv),
        pointwise_channels=args.pointwise_channels
    )

    model = model.to(device)
    print("Model created and moved to device")
    
    with open(experiment_dir + '/model_architecture.txt', 'w') as file:
        file.write(str(model))
        print(model)


    # Calculate class weights
    weight_for_0 = 1.0
    weight_for_1 = (n_negative / n_positive) * args.pos_weight_scale if n_positive > 0 else 30.0 * args.pos_weight_scale
    
    # Set up loss function
    if args.loss_fn == "focal":
        # Use Focal Loss
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
        print(f"Using Focal Loss with alpha={args.focal_alpha}, gamma={args.focal_gamma}")
    elif args.class_weight:
        # Use weighted BCE loss
        pos_weight = torch.tensor([weight_for_1], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Class weights: 0: {weight_for_0:.4f}, 1: {weight_for_1:.4f}")
        print(f"Using weighted BCEWithLogitsLoss with pos_weight={weight_for_1:.4f}")
    else:
        # Standard BCE loss
        criterion = nn.BCEWithLogitsLoss()
        print("Using standard BCEWithLogitsLoss (no class weighting)")

    # Set up optimizer
    optimizer = {
        "adadelta": optim.Adadelta(model.parameters(), lr=args.lr),
        "adagrad": optim.Adagrad(model.parameters(), lr=args.lr),
        "adamw": optim.AdamW(model.parameters(), lr=args.lr, weight_decay=wd),
        "rmsprop": optim.RMSprop(model.parameters(), lr=args.lr),
        "sgd": optim.SGD(model.parameters(), lr=args.lr),
        "adam": optim.Adam(model.parameters(), lr=args.lr, weight_decay=wd)
    }
    optimizer = optimizer.get(opt.lower())
    if optimizer is None:
        raise ValueError(f"Optimizer {opt} not supported")

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
                checkpoint = torch.load(latest_checkpoint, map_location=device)
                
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

    # Create scheduler (after potential checkpoint loading)
    scheduler = get_scheduler(sch, optimizer, args, train_files)
    
    # Load scheduler state if it exists in checkpoint
    if args.resume and 'latest_checkpoint' in locals() and latest_checkpoint and 'scheduler_state_dict' in checkpoint:
        try:
            if isinstance(scheduler, (WarmupExponential, ExponentialDecay)):
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            elif hasattr(scheduler, 'load_state_dict'):
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Loaded scheduler state from checkpoint")
        except Exception as e:
            print(f"Warning: Failed to load scheduler state: {e}")

    print(f"Using optimizer: {opt}")
    print(f"Using scheduler: {sch}")

    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta,
        verbose=True
    )
    
    if args.resume and 'latest_checkpoint' in locals() and latest_checkpoint and 'early_stopping_state' in checkpoint:
        try:
            early_stopping.load_state_dict(checkpoint['early_stopping_state'])
            print("Loaded early stopping state from checkpoint")
        except Exception as e:
            print(f"Warning: Failed to load early stopping state: {e}")

    # Initialize history dict if starting fresh
    if history is None:
        history = {
            'train_loss': [], 'train_acc': [], 'train_auc': [], 'train_f1': [], 'train_pr_auc': [],
            'test_loss': [], 'test_acc': [], 'test_auc': [], 'test_f1': [], 'test_pr_auc': [],
            'learning_rates': []
        }

    start_time = time.time()
    model, history, final_metrics, all_preds, all_labels, completed_epochs = train_model(
        model, dataloaders, criterion, optimizer, scheduler, num_epochs, 
        device=device, early_stopping=early_stopping,
        classification_threshold=args.threshold, checkpoint_dir=experiment_dir,
        start_epoch=start_epoch, keep_last_n=args.keep_checkpoints, 
        history=history, initial_best_loss=best_loss
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Plot metrics
    plot_all_metrics(history, final_metrics, experiment_dir)

    print("Model training completed")

    # Update results dictionary
    results = {
        'train_acc': round(history['train_acc'][-1], 4),
        'train_auc': round(history['train_auc'][-1], 4),
        'train_f1': round(history['train_f1'][-1], 4),
        'train_pr_auc': round(history['train_pr_auc'][-1], 4),
        'test_acc': round(history['test_acc'][-1], 4),
        'test_auc': round(history['test_auc'][-1], 4),
        'test_f1': round(history['test_f1'][-1], 4),
        'test_pr_auc': round(history['test_pr_auc'][-1], 4)
    }

    # Add final metrics for each phase
    for phase in ['train', 'test']:
        results.update({
            f'{phase}_sens': final_metrics[phase]['sensitivity'],
            f'{phase}_spec': final_metrics[phase]['specificity'],
            f'{phase}_f1_score': final_metrics[phase]['f1_score'],
            f'{phase}_CM': str(final_metrics[phase]['confusion_matrix']),
        })

    # Update hyperparameters dictionary
    hyperparameters = {
        'Exp_ID': id,
        'BS': batch_size,
        'Epochs': completed_epochs,
        'Completed_Epochs': completed_epochs,
        'Start_LR': learning_rate,
        'Peak_LR': args.peak_lr,
        'Final_LR': optimizer.param_groups[0]["lr"],
        'Dropout': dropout_rate,
        'Act': act,
        'Opt': opt,
        'Sch': sch,
        'WD': wd,
        'DF': args.df,
        'Label_Column': args.label_col,
        'Use_PCs': bool(args.cov) and args.model_type == "full",
        'norm_PCs': args.norm_pcs,
        'Use_Age': bool(args.use_age) and args.model_type == "full",
        'norm_Age': args.norm_age,
        'Use_Gender': bool(args.use_gender) and args.model_type == "full",
        'norm_Gender': args.norm_gender,
        'Use_Bmi': bool(args.use_bmi),
        'norm_Bmi': args.norm_bmi,
        'model_type': args.model_type,
        'loss_fn': args.loss_fn,
        'threshold': args.threshold,
        'Kernel_sizes': str(kernel_sizes),
        'Stride': str(stride),
        'Conv_channels': str(conv_channels),
        'Use_Pooling': bool(args.use_pooling),
        'Pool_size': args.pool_size if args.use_pooling else 'N/A',
        'Pool_type': args.pool_type if args.use_pooling else 'N/A',
        'FC_layers': str(fc_layers),
        'class_weight': bool(args.class_weight),
        'pos_weight_scale': args.pos_weight_scale if bool(args.class_weight) else None,
        'sampling': args.sampling,
        'sampling_ratio': args.sampling_ratio if args.sampling != "none" else None,
        'random_seed': args.random_seed,
        # Enhanced architecture parameters
        'Use_Multi_Scale': bool(args.use_multi_scale),
        'Use_Pointwise_Conv': bool(args.use_pointwise_conv),
        'Pointwise_Channels': args.pointwise_channels if args.use_pointwise_conv else 'N/A',
        'Multi_Scale_Mode': args.multi_scale_mode if args.use_multi_scale else 'N/A',
        'Multi_Scale_Fusion': args.multi_scale_fusion if args.use_multi_scale else 'N/A',
        'Multi_Scale_Kernels': str(args.multi_scale_kernels) if args.use_multi_scale and args.multi_scale_mode == 'progressive' else 'N/A',
        'Multi_Scale_Strides': str(args.multi_scale_strides) if args.use_multi_scale and args.multi_scale_mode == 'progressive' else 'N/A',
        'Hardcoded_Kernels': str(args.hardcoded_kernels) if args.use_multi_scale and args.multi_scale_mode == 'hardcoded' else 'N/A',
        'Hardcoded_Strides': str(args.hardcoded_strides) if args.use_multi_scale and args.multi_scale_mode == 'hardcoded' else 'N/A',
    }

    # Write results
    write_results(model, hyperparameters, results, experiment_dir)
    append_metrics_to_csv(experiment_dir, results)

    print(f"\nTraining complete. Results saved to {experiment_dir}")
    
    print("\nFinal performance metrics:")
    print(f"Train accuracy: {results['train_acc']}, AUC: {results['train_auc']}, F1: {results['train_f1']}, PR-AUC: {results['train_pr_auc']}")
    print(f"Test accuracy: {results['test_acc']}, AUC: {results['test_auc']}, F1: {results['test_f1']}, PR-AUC: {results['test_pr_auc']}")
    
    print("\nConfusion matrices:")
    for phase in ['train', 'test']:
        print(f"{phase.capitalize()}:")
        print(final_metrics[phase]['confusion_matrix'])

    # # Feature importance analysis
    # print("\nPerforming gradient-based feature importance analysis...")
    # model.eval()
    # target_class = 1
    
    # grad_importance = gradient_based_feature_importance(model, dataloaders['train'], target_class, device)
    # if grad_importance is not None:
    #     print(f"Shape of grad_importance: {grad_importance.shape}")

    #     # Compute the importance score for each feature
    #     feature_importance = np.mean(np.abs(grad_importance), axis=0)
    #     feature_importance = np.mean(np.abs(feature_importance), axis=0)
    #     print(f"Shape of feature_importance: {feature_importance.shape}")

    #     # Get indices of top features
    #     top_features_indices = np.argsort(feature_importance.flatten())[::-1]
    #     print("Number of top features identified:", len(top_features_indices))
        
    #     # Create a DataFrame with the top feature indices
    #     df = pd.DataFrame({'SNP_Index': top_features_indices})

    #     # Save the DataFrame to a CSV file
    #     top_csv_path = os.path.join(experiment_dir,f'top_{len(top_features_indices)}_indices.csv')
    #     df.to_csv(top_csv_path, index=False)

    #     print(f"Top {len(top_features_indices)} indices saved to {top_csv_path}")
    # else:
    #     print("Could not compute feature importance.")

if __name__ == '__main__':
    start_time = time.time()
    
    main()
    
    end_time = time.time()
    total_runtime = end_time - start_time
    
    print(f"\nTotal script runtime: {total_runtime:.2f} seconds")
    hours, rem = divmod(total_runtime, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Total runtime: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")