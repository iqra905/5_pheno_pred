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
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
import argparse
from sklearn.metrics import confusion_matrix, roc_curve, auc
import csv
import time
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer
import random
import torch.nn.functional as F

def parse_int_list(s):
    return [int(x) for x in s.split(',')]

def parse_args():
    parser = argparse.ArgumentParser(description="Residual Genotype Model Training")
    parser.add_argument("-ID", type=str, default="0_best_01_1_32_0.5_100_0.001_tanh_warmup_exponential_0.1_adamw_0.5_cov", help="ID of the experiment")
    parser.add_argument("-exp_dir", type=str, default='/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/disease_wise_setting_01/pruned/col/chr_wise', help="Directory to save experiment results")
    parser.add_argument("-genotype_dir", type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/col_can_pruned', help="Directory containing genotype files")
    parser.add_argument("-phenotype_file", type=str, default='/vol/vssp/SF_ucdatasets/gwas/data_files/disease_pheno/col_can.xlsx', help="Path to phenotype file")

    parser.add_argument("-bs", type=int, default=8, help="Batch size for training")
    parser.add_argument("-dropout", type=float, default=0.5, help="Dropout rate for the model")
    parser.add_argument("-epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("-lr", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("-act", type=str, default="tanh", choices=["tanh","relu","gelu"], help="Activation function for the model")
    parser.add_argument("-opt", type=str, default="adamw", choices=["adam", "adamw", "sgd"], help="Optimizer to use")
    parser.add_argument("-sch", type=str, default="warmup_exponential", choices=["none","plateau", "cosine", "step","multistep","explr","warmup_exponential", "exponential_decay"], help="Learning rate scheduler")
    parser.add_argument("-peak_lr", type=float, default=1e-2, help="Peak learning rate for WarmupExponential scheduler")
    parser.add_argument("-final_lr", type=float, default=1e-5, help="Final learning rate for custom schedulers")
    parser.add_argument("-wd", type=float, default=0.5, help="Weight decay for optimizer")
    parser.add_argument("-df", type=float, default=0.1, help="Decay factor for custom schedulers")

    parser.add_argument("-kernel_sizes", type=parse_int_list, default=[4096,1,1], help="Convolution Kernel Size")
    parser.add_argument("-stride", type=parse_int_list, default=[2048,1,1], help="Convolution Stride")
    parser.add_argument("-conv_channels", type=parse_int_list, default=[64,128,256], help="Convolution channels")
    parser.add_argument("-fc_layers", type=parse_int_list, default=[64,128], help="Fully connected layers")
    
    parser.add_argument("-cov", type=int, default=1, choices=[0, 1], help="Whether to include PC covariates in the model (0: no, 1: yes)")
    parser.add_argument("-use_age", type=int, default=1, choices=[0, 1], help="Whether to include age in covariates (0: no, 1: yes)")
    parser.add_argument("-use_gender", type=int, default=1, choices=[0, 1], help="Whether to include gender in covariates (0: no, 1: yes)")
    parser.add_argument("-label_col", type=str, default="crc", help="Column name in phenotype file to use as label")
    
    # Early stopping parameters
    parser.add_argument("-patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("-min_delta", type=float, default=1e-4, help="Minimum change for early stopping")

    # Normalization-related arguments
    parser.add_argument("-norm_age", type=str, default="none", choices=["none", "standard", "minmax", "robust", "quantile", "power"], help="Normalization method for age")
    parser.add_argument("-norm_pcs", type=str, default="none", choices=["none", "standard", "minmax", "robust", "quantile", "power"], help="Normalization method for PCs")
    parser.add_argument("-norm_gender", type=str, default="none", choices=["none", "minmax"], help="Normalization method for gender (usually keep as none)")
    
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
    
    parser.add_argument("-pool_size", type=int, default=64, help="Size of the adaptive pooling output (smaller values = more aggressive pooling)")
    
    return parser.parse_args()

def get_input_size(genotype_file):
    with gzip.open(genotype_file, 'rt') as f:
        return sum(1 for line in f) 

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

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.verbose:
                print(f'Initial validation loss: {val_loss:.6f}')
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            if self.verbose:
                print(f'Validation loss decreased to {val_loss:.6f}')
            self.counter = 0

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
    def __init__(self, file_list, phenotype_data, label_column, 
                 use_covariates=True, use_age=False, use_gender=False,
                 norm_age="none", norm_pcs="none", norm_gender="none",
                 fit_normalizers=True, normalizers=None):
        self.file_list = file_list
        self.phenotype_data = phenotype_data
        self.label_column = label_column
        self.use_covariates = use_covariates
        self.use_age = use_age
        self.use_gender = use_gender
        
        # Verify that the label column exists in the phenotype data
        if self.label_column not in self.phenotype_data.columns:
            raise ValueError(f"Label column '{self.label_column}' not found in phenotype data. "
                           f"Available columns are: {', '.join(self.phenotype_data.columns)}")
        
        if normalizers is None:
            self.age_normalizer = CovariateNormalizer(norm_age)
            self.pcs_normalizer = CovariateNormalizer(norm_pcs)
            self.gender_normalizer = CovariateNormalizer(norm_gender)

            # Fit normalizers if this is training set
            if fit_normalizers:
                self._fit_normalizers()
        else:
            # Use the provided normalizers: for Test data
            self.age_normalizer = normalizers['age']
            self.pcs_normalizer = normalizers['pcs']
            self.gender_normalizer = normalizers['gender']
        
        self.file_handles = {}
        for file in file_list:
            f = open(file, 'rb')
            self.file_handles[file] = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        # Log initialization
        print(f"\nDataset Initialization:")
        print(f"- Number of files: {len(file_list)}")
        print(f"- Label column: {label_column}")
        print(f"- Using PCs: {use_covariates} (normalization: {norm_pcs})")
        print(f"- Using age: {use_age} (normalization: {norm_age})")
        print(f"- Using gender: {use_gender} (normalization: {norm_gender})")

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
    
    def get_normalizers(self):
        """Return the fitted normalizers"""
        return {
            'age': self.age_normalizer,
            'pcs': self.pcs_normalizer,
            'gender': self.gender_normalizer
        }

    def __getitem__(self, idx):
        genotype_file = self.file_list[idx]
        sample_id_str = os.path.basename(genotype_file).replace("sample_", "").replace(".gen.gz", "")
        sample_id = int(sample_id_str)
        
        # Use the configured label column
        label = self.phenotype_data.loc[self.phenotype_data['new_order'] == sample_id, self.label_column].values[0]
        
        # Process covariates with normalization if needed
        covariates_list = []
        if self.use_covariates:
            # Get PC values and normalize
            pc_data = np.array([
                self.phenotype_data.loc[self.phenotype_data['new_order'] == sample_id, f'PC{i}'].values[0] 
                for i in range(1, 11)
            ]).reshape(1, -1)
            
            normalized_pcs = self.pcs_normalizer.transform(pc_data).flatten()
            covariates_list.append(normalized_pcs)
            
        if self.use_age and 'Agexit' in self.phenotype_data.columns:
            # Get and normalize age
            age = self.phenotype_data.loc[self.phenotype_data['new_order'] == sample_id, 'Agexit'].values[0]
            normalized_age = self.age_normalizer.transform(np.array([[age]])).flatten()
            covariates_list.append(normalized_age)
        
        if self.use_gender and 'Sex' in self.phenotype_data.columns:
            # Get and normalize gender
            gender = self.phenotype_data.loc[self.phenotype_data['new_order'] == sample_id, 'Sex'].values[0]
            normalized_gender = self.gender_normalizer.transform(np.array([[gender]])).flatten()
            covariates_list.append(normalized_gender)
        
        # Combine all covariates
        if covariates_list:
            covariates = np.concatenate(covariates_list)
        else:
            covariates = np.array([])
        
        covariates_tensor = torch.tensor(covariates, dtype=torch.float32)

        # Use memory-mapped file for faster access
        mmap_file = self.file_handles[genotype_file]
        mmap_file.seek(0)
        with gzip.GzipFile(fileobj=mmap_file) as f:
            data = pd.read_csv(f, sep=r'\s+', header=None)

        genotype_tensor = torch.tensor(data.values.T, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return genotype_tensor, covariates_tensor, label_tensor

    def __len__(self):
        return len(self.file_list)
    
    def __del__(self):
        for handle in self.file_handles.values():
            handle.close()

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
            
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout_prob, act):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = self.get_activation(act)
        #self.dropout = nn.Dropout(dropout_prob)
        
        # 1x1 convolution for residual connection if dimensions change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv(x)
        out = self.bn(out)
        out += residual
        out = self.activation(out)
        #out = self.dropout(out)
        return out

    @staticmethod
    def get_activation(name):
        if name == 'tanh':
            return nn.Tanh()
        elif name == 'relu':
            return nn.ReLU()
        elif name == 'gelu':
            return nn.GELU()
        else:
            raise NotImplementedError("Activation function not implemented.")

class GenotypeModel(nn.Module):
    def __init__(self, input_size, kernel_sizes, stride, conv_channels, fc_layers, act, dropout_rate, 
                 use_covariates=True, use_age=False, use_gender=False, num_pc_covariates=10, pool_size=16):
        super(GenotypeModel, self).__init__()
        self.input_channels = 3
        self.use_covariates = use_covariates
        self.use_age = use_age
        self.use_gender = use_gender
        self.pool_size = pool_size


        # Calculate total number of covariates
        self.num_covariates = 0
        if use_covariates:
            self.num_covariates += num_pc_covariates  # PCs
        if use_age:
            self.num_covariates += 1  # Age
        if use_gender:
            self.num_covariates += 1  # Gender

        self.conv_layers = self._create_conv_layers(conv_channels, kernel_sizes, stride, dropout_rate, act)

        # Add pooling layer
        self.pool = nn.AdaptiveMaxPool1d(pool_size)  # Adaptive pooling to fixed output size
        
        self.conv_output_size = self._get_conv_output_size(input_size)
        print(f"Convolutional output size: {self.conv_output_size}")
        
        fc_layers_list = []
        # Adjust input features based on covariate usage
        in_features = self.conv_output_size + self.num_covariates
        print(f"FC input features: {in_features}")
        
        for i, out_features in enumerate(fc_layers):
            fc_layers_list.extend([
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                self.get_activation(act),
                nn.Dropout(dropout_rate)
            ])
            in_features = out_features
        
        fc_layers_list.append(nn.Linear(in_features, 1, bias=False))
        
        self.fc = nn.Sequential(*fc_layers_list)
        print(f"GenotypeModel initialized (using covariates: {use_covariates}, age: {use_age}, gender: {use_gender})")
        print(f"Total number of covariates: {self.num_covariates}")

    def _create_conv_layers(self, conv_channels, kernel_sizes, stride, dropout_rate, act):
        layers = []
        in_channels = self.input_channels
        for i, out_channels in enumerate(conv_channels):
            layers.append(ResidualBlock(in_channels, out_channels, kernel_sizes[i], stride[i], dropout_rate, act))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def _get_conv_output_size(self, input_size):
        x = torch.randn(1, self.input_channels, input_size)
        x = self.conv_layers(x)
        # Apply pooling to the test tensor
        x = self.pool(x)
        return x.numel() // x.size(0)

    def forward(self, x, covariates=None):
        x = self.conv_layers(x)
        # Apply pooling before flattening
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        
        if self.num_covariates > 0 and covariates is not None and covariates.numel() > 0:
            x = torch.cat([x, covariates], dim=1)
            
        x = self.fc(x)
        return x.squeeze(1)

    @staticmethod
    def get_activation(name):
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
 
def gradient_based_feature_importance(model, train_loader, target_class, device):
    """
    Calculate gradient-based feature importance scores.
    
    Args:
        model: The trained GenotypeModel
        train_loader: DataLoader containing training data
        target_class: Target class for importance analysis (typically 0 or 1 for binary classification)
        device: Device to run the computation on ('cuda' or 'cpu')
    
    Returns:
        average_gradients: Array of importance scores for each feature
    """
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
                total_gradients = np.zeros((train_loader.batch_size, *gradients.shape[1:]))
            total_gradients[:batch_size] += gradients
        
        total_samples += batch_size
        
        # Clear gradients for the next iteration
        model.zero_grad()
        batch_input.grad = None
    
    print("Total samples analyzed:", total_samples)
    if total_gradients is not None:
        print("Gradient shape:", total_gradients.shape)
        # Compute average gradients
        average_gradients = total_gradients / total_samples
        return average_gradients
    else:
        print("Warning: No gradients were computed. Check if your model parameters require gradients.")
        return None

def analyze_and_save_feature_importance(model, train_loader, experiment_dir, device, n_classes=2):
    """
    Perform feature importance analysis and save results.
    
    Args:
        model: The trained GenotypeModel
        train_loader: DataLoader containing training data
        experiment_dir: Directory to save results
        device: Device to run computation on
        n_classes: Number of classes (default 2 for binary classification)
    """
    print("\nPerforming feature importance analysis...")
    
    # Calculate importance scores for each class
    importance_scores = []
    for target_class in range(n_classes):
        print(f"\nAnalyzing feature importance for class {target_class}")
        grad_importance = gradient_based_feature_importance(model, train_loader, target_class, device)
        
        if grad_importance is not None:
            # Compute the importance score for each feature
            feature_importance = np.mean(np.abs(grad_importance), axis=0)
            feature_importance = np.mean(np.abs(feature_importance), axis=0)
            importance_scores.append(feature_importance)
            
            # Get indices of top features
            top_features_indices = np.argsort(feature_importance.flatten())[::-1]
            
            # Save indices to CSV
            df = pd.DataFrame({
                'SNP_Index': top_features_indices,
                'Importance_Score': feature_importance.flatten()[top_features_indices]
            })
            
            csv_path = os.path.join(experiment_dir, f'feature_importance_class_{target_class}.csv')
            df.to_csv(csv_path, index=False)
            print(f"Feature importance scores for class {target_class} saved to {csv_path}")
            
            # Create visualization
            plt.figure(figsize=(12, 6))
            plt.plot(feature_importance)
            plt.title(f'Feature Importance Scores (Class {target_class})')
            plt.xlabel('Feature Index')
            plt.ylabel('Importance Score')
            plt.savefig(os.path.join(experiment_dir, f'feature_importance_plot_class_{target_class}.png'))
            plt.close()
    
    return importance_scores

def print_lr(optimizer):
    for param_group in optimizer.param_groups:
        print(f"Current Learning Rate: {param_group['lr']}")

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, device='cuda', 
                early_stopping=None, classification_threshold=0.5):
    print(f"Training on device: {device}")
    print(f"Classification threshold: {classification_threshold}")
    
    scaler = GradScaler()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    completed_epochs = 0

    history = {
        'train_loss': [], 'train_acc': [], 'train_auc': [], 'train_f1': [], 'train_pr_auc': [],
        'test_loss': [], 'test_acc': [], 'test_auc': [], 'test_f1': [], 'test_pr_auc': [],
        'learning_rates': []
    }

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        all_preds = {phase: [] for phase in ['train', 'test']}
        all_labels = {phase: [] for phase in ['train', 'test']}

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            # Create a CUDA stream for asynchronous data transfer
            stream = torch.cuda.Stream()

            # Get iterator for the dataloader
            batch_iter = iter(dataloaders[phase])
            
            # Process all batches
            for i in range(len(dataloaders[phase])):
                try:
                    # Get the current batch
                    inputs, covariates, labels = next(batch_iter)
                except StopIteration:
                    # If we ran out of batches, break
                    break
                
                # Move data to device
                inputs = inputs.to(device, non_blocking=True)
                covariates = covariates.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # Process the current batch
                optimizer.zero_grad()
                
                with autocast():
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs, covariates)
                        loss = criterion(outputs, labels)
                        
                        # Convert logits to probabilities for metrics
                        probs = torch.sigmoid(outputs)
                        preds = (probs >= classification_threshold).float()

                        if phase == 'train':
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()

                            # Step the scheduler if it's a per-iteration scheduler
                            if isinstance(scheduler, (WarmupExponential, ExponentialDecay)):
                                scheduler.step()

                running_loss += loss.item() * labels.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += labels.size(0)

                all_labels[phase].extend(labels.cpu().numpy())
                all_preds[phase].extend(probs.detach().cpu().numpy())

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples
            
            # Calculate metrics
            y_true = np.array(all_labels[phase])
            y_pred_proba = np.array(all_preds[phase])
            y_pred = (y_pred_proba >= classification_threshold).astype(int)
            
            # Calculate AUC if possible (requires both classes to be present)
            try:
                epoch_auc = roc_auc_score(y_true, y_pred_proba)
                auc_str = f" - AUC: {epoch_auc:.4f}"
            except Exception as e:
                print(f"Warning: Could not calculate AUC for {phase} phase: {str(e)}")
                epoch_auc = 0.5  # Default value if AUC can't be calculated
                auc_str = " - AUC: N/A (need both classes)"
            
            # Calculate F1 score
            try:
                epoch_f1 = f1_score(y_true, y_pred)
                f1_str = f" - F1: {epoch_f1:.4f}"
            except Exception as e:
                print(f"Warning: Could not calculate F1 score for {phase} phase: {str(e)}")
                epoch_f1 = 0.0
                f1_str = " - F1: N/A"
            
            # Calculate PR-AUC
            try:
                precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
                epoch_pr_auc = auc(recall, precision)
                pr_auc_str = f" - PR-AUC: {epoch_pr_auc:.4f}"
            except Exception as e:
                print(f"Warning: Could not calculate PR-AUC for {phase} phase: {str(e)}")
                epoch_pr_auc = 0.5
                pr_auc_str = " - PR-AUC: N/A"

            print(f'{phase} Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}{auc_str}{f1_str}{pr_auc_str}')

            # Store metrics in history
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())
            history[f'{phase}_auc'].append(epoch_auc)
            history[f'{phase}_f1'].append(epoch_f1)
            history[f'{phase}_pr_auc'].append(epoch_pr_auc)

            # Early stopping check based on test loss
            if phase == 'test' and early_stopping is not None:
                early_stopping(epoch_loss)
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    completed_epochs = epoch + 1  # Save the number of completed epochs
                    model.load_state_dict(best_model_wts)
                    return model, history, compute_final_metrics(all_labels, all_preds, classification_threshold), all_preds, all_labels, completed_epochs
                
            # Save best model
            if phase == 'test' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        # Step the scheduler if it's an epoch-wise scheduler
        if scheduler is not None and not isinstance(scheduler, (WarmupExponential, ExponentialDecay)):
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(history['test_loss'][-1])
            else:
                scheduler.step()

        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        print_lr(optimizer)
        completed_epochs = epoch + 1  # Update completed epochs counter

    print(f'Best test loss: {best_loss:.4f}')
    model.load_state_dict(best_model_wts)
    return model, history, compute_final_metrics(all_labels, all_preds, classification_threshold), all_preds, all_labels, completed_epochs

def compute_final_metrics(all_labels, all_preds, threshold=0.5):
    final_metrics = {}
    for phase in ['train', 'test']:
        y_true = np.array(all_labels[phase])
        y_pred_proba = np.array(all_preds[phase])

        # Convert probabilities to binary predictions
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate confusion matrix and derived metrics
        try:
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):  # Only calculate if we have a proper 2x2 confusion matrix
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            else:
                sensitivity = 0
                specificity = 0
                
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
                roc_auc = 0.5
                pr_auc = 0.5
                f1 = 0.0
        except Exception as e:
            print(f"Error calculating metrics for {phase}: {str(e)}")
            cm = np.zeros((2, 2))
            sensitivity = 0
            specificity = 0
            fpr, tpr = np.array([]), np.array([])
            precision, recall = np.array([]), np.array([])
            roc_auc = 0.5
            pr_auc = 0.5
            f1 = 0.0

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
    print("Parsing Arguments...")
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
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
    for column in ['Agexit', 'Sex'] + [f'PC{i}' for i in range(1, 11)]:
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
    file_list = glob.glob(os.path.join(genotype_dir, "sample_*.gen.gz"))
    file_list.sort(key=lambda x: int(x.split('sample_')[1].split('.gen.gz')[0]))
    print(f"Number of genotype files found: {len(file_list)}")

    # Get the unique sample IDs from the phenotype data
    phenotype_samples = set(phenotype_data['new_order'].unique())
    print(f"Number of unique samples in phenotype data: {len(phenotype_samples)}")

    # Filter the file list to only include files for samples in the phenotype data
    filtered_file_list = []
    for file_path in file_list:
        # Extract sample ID from filename
        sample_id = int(file_path.split('sample_')[1].split('.gen.gz')[0])
        
        # Check if this sample ID is in the phenotype data
        if sample_id in phenotype_samples:
            filtered_file_list.append(file_path)

    print(f"Number of filtered genotype files matching phenotype data: {len(filtered_file_list)}")

    # Continue with your analysis using filtered_file_list instead of file_list
    first_genotype_file = filtered_file_list[0] if filtered_file_list else None
    if first_genotype_file:
        print(f"First genotype file after filtering is: {first_genotype_file}")
    else:
        print("No matching genotype files found!")
    
    # For raw SNP data, get the number of SNPs from the first file
    orig_input_size = get_input_size(first_genotype_file)
    print(f"Dynamically determined original input size: {orig_input_size}")
    input_size = orig_input_size
    
    if len(filtered_file_list) != len(phenotype_samples):
        raise ValueError(f"Number of files ({len(filtered_file_list)}) in {genotype_dir} does not match number of samples ({len(phenotype_samples)}) in phenotype data.")

    # Split data into train and test only
    train_files, test_files = train_test_split(
        filtered_file_list, test_size=0.2, random_state=args.random_seed
    )
    print(f"Data split: Train {len(train_files)}, Test {len(test_files)}")

    # Create datasets with appropriate flags
    train_dataset = GenotypeDataset(
        train_files, 
        phenotype_data, 
        label_column=args.label_col,
        use_covariates=bool(args.cov) and args.model_type == "full",
        use_age=bool(args.use_age) and args.model_type == "full",
        use_gender=bool(args.use_gender) and args.model_type == "full",
        norm_age=args.norm_age,
        norm_pcs=args.norm_pcs,
        norm_gender=args.norm_gender,
        fit_normalizers=True
    )
    
    # Get the fitted normalizers from training dataset
    fitted_normalizers = train_dataset.get_normalizers()
    
    test_dataset = GenotypeDataset(
        test_files, 
        phenotype_data, 
        label_column=args.label_col,
        use_covariates=bool(args.cov) and args.model_type == "full",
        use_age=bool(args.use_age) and args.model_type == "full",
        use_gender=bool(args.use_gender) and args.model_type == "full",
        norm_age=args.norm_age,
        norm_pcs=args.norm_pcs,
        norm_gender=args.norm_gender,
        fit_normalizers=False,
        normalizers=fitted_normalizers
    )

    # Count class distribution in training data
    train_labels = []
    for i in range(len(train_dataset)):
        _, _, label = train_dataset[i]
        train_labels.append(label.item())
    
    n_samples = len(train_labels)
    n_positive = sum(train_labels)
    n_negative = n_samples - n_positive
    
    pos_ratio = n_positive / n_samples if n_samples > 0 else 0.5
    neg_ratio = n_negative / n_samples if n_samples > 0 else 0.5
    
    print(f"Class distribution in training data: Positive: {n_positive} ({pos_ratio:.2%}), Negative: {n_negative} ({neg_ratio:.2%})")

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

    model = GenotypeModel(
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
        num_pc_covariates=10,
        pool_size=args.pool_size
    )
    model = model.to(device)

    with open(experiment_dir + '/model_architecture.txt', 'w') as file:
        file.write(str(model))
        print(model)

    print("Model created and moved to device")

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
    }.get(opt, None)

    if optimizer is None:
        raise NotImplementedError(f"Optimizer {opt} not implemented.")

    scheduler = get_scheduler(sch, optimizer, args, train_files)

    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta,
        verbose=True
    )

    # Train the model
    model, history, final_metrics, all_preds, all_labels, completed_epochs = train_model(
        model, dataloaders, criterion, optimizer, scheduler, num_epochs, 
        device=device, early_stopping=early_stopping,
        classification_threshold=args.threshold
    )
    
    # Plot metrics
    plot_all_metrics(history, final_metrics, experiment_dir)

    # Save the trained model
    #torch.save(model.state_dict(), os.path.join(experiment_dir, 'final_model.pth'))
    print("Model training completed and saved")

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
        'Use_Age': bool(args.use_age) and args.model_type == "full",
        'Use_Gender': bool(args.use_gender) and args.model_type == "full",
        'model_type': args.model_type,
        'loss_fn': args.loss_fn,
        'threshold': args.threshold,
        'Kernel_sizes': str(kernel_sizes),
        'Stride': str(stride),
        'conv_channels': str(conv_channels),
        'fc_layers': str(fc_layers),
        'class_weight': bool(args.class_weight),
        'pos_weight_scale': args.pos_weight_scale if bool(args.class_weight) else None,
        'sampling': args.sampling,
        'sampling_ratio': args.sampling_ratio if args.sampling != "none" else None,
        'random_seed': args.random_seed
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

    # After model training and results saving, add:
    print("\nStarting feature importance analysis...")
    importance_scores = analyze_and_save_feature_importance(
        model=model,
        train_loader=dataloaders['train'],
        experiment_dir=experiment_dir,
        device=device
    )
    
    # Save importance scores summary
    if importance_scores:
        importance_summary = {
            'mean_importance': np.mean(importance_scores, axis=1),
            'max_importance': np.max(importance_scores, axis=1),
            'min_importance': np.min(importance_scores, axis=1)
        }
        
        with open(os.path.join(experiment_dir, 'feature_importance_summary.txt'), 'w') as f:
            f.write("Feature Importance Summary\n")
            f.write("------------------------\n")
            for metric, values in importance_summary.items():
                f.write(f"{metric}:\n")
                for class_idx, value in enumerate(values):
                    f.write(f"  Class {class_idx}: {value}\n")

    print("Feature importance analysis completed.")


if __name__ == '__main__':
    start_time = time.time()
    
    main()
    
    end_time = time.time()
    total_runtime = end_time - start_time
    
    print(f"\nTotal script runtime: {total_runtime:.2f} seconds")
    hours, rem = divmod(total_runtime, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Total runtime: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")