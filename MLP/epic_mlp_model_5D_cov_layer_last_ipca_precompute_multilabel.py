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
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
import argparse
from sklearn.metrics import confusion_matrix, roc_curve, auc
import csv
import time
from collections import OrderedDict


def parse_int_list(s):
    return [int(x) for x in s.split(',')]

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Disease Genotype Model Training")
    parser.add_argument("-ID", type=str, default="000", help="ID of the experiment")
    parser.add_argument("-exp_dir", type=str, default='/vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/5d_multilabel', help="Directory to save experiment results")
    parser.add_argument("-genotype_dir", type=str, default='/vol/research/ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/pros_can', help="Directory containing genotype files")
    parser.add_argument("-phenotype_file", type=str, default='/vol/research/ucdatasets/gwas/data_files/merged_v8_pcs_chip_added_Iqra_1_cleaned_pros_multi.xlsx', help="Path to combined phenotype file with all disease labels")
    parser.add_argument("-bs", type=int, default=32, help="Batch size for training")
    parser.add_argument("-dropout", type=float, default=0.5, help="Dropout rate for the model")

    parser.add_argument("-epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("-lr", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("-act", type=str, default="gelu", choices=["tanh","relu","gelu"], help="Activation function for the model")
    parser.add_argument("-opt", type=str, default="adamw", choices=["adam", "adamw", "sgd"], help="Optimizer to use")
    parser.add_argument("-sch", type=str, default="explr", choices=["none","plateau", "cosine", "step","multistep","explr","warmup_exponential", "exponential_decay"], help="Learning rate scheduler")
    parser.add_argument("-peak_lr", type=float, default=1e-2, help="Peak learning rate for WarmupExponential scheduler")
    parser.add_argument("-final_lr", type=float, default=1e-5, help="Final learning rate for custom schedulers")
    parser.add_argument("-wd", type=float, default=0.5, help="Weight decay for optimizer")
    parser.add_argument("-df", type=float, default=0.1, help="Decay factor for custom schedulers")

    parser.add_argument("-hidden_sizes", type=parse_int_list, default=[8,16,32], help="Hidden layer sizes for MLP")

    parser.add_argument("-cov", type=int, default=0, choices=[0, 1], help="Whether to include PC's in covariates in the model (0: no, 1: yes)")
    parser.add_argument("-use_age", type=int, default=0, choices=[0, 1], help="Whether to include age in covariates in the model (0: no, 1: yes)")
    parser.add_argument("-use_gender", type=int, default=0, choices=[0, 1], help="Whether to include gender in covariates in the model (0: no, 1: yes)")
    
    # Define disease label columns
    parser.add_argument("-disease_labels", type=str, nargs='+', 
                       default=["pros01", "panca", "crc", "breacancer", "t2dm"], 
                       help="Column names in phenotype file to use as disease labels")

    # Early stopping parameters
    parser.add_argument("-patience", type=int, default=15, help="Patience for early stopping")
    parser.add_argument("-min_delta", type=float, default=1e-4, help="Minimum change for early stopping")

    # Normalization-related arguments
    parser.add_argument("-norm_age", type=str, default="none", choices=["none", "standard", "minmax", "robust", "quantile", "power"], help="Normalization method for age")
    parser.add_argument("-norm_pcs", type=str, default="none", choices=["none", "standard", "minmax", "robust", "quantile", "power"], help="Normalization method for PCs")
    parser.add_argument("-norm_gender", type=str, default="none", choices=["none", "minmax"], help="Normalization method for gender (usually keep as none)")

    # Model type and PCA arguments
    parser.add_argument("-model_type", type=str, default="snps_only", choices=["full", "covariates_only", "snps_only"], help="Type of model to use")
    parser.add_argument("-use_pca", type=int, default=0, choices=[0, 1], help="Whether to use pre-computed PCA features (0: no, 1: yes)")
    parser.add_argument("-n_components", type=int, default=100, help="Number of PCA components to use")
    parser.add_argument("-pca_features_dir", type=str, default='./pca_features', help="Directory containing pre-computed PCA features")
    
    # Class weighting for imbalanced data
    parser.add_argument("-class_weight", type=int, default=0, choices=[0, 1], help="Whether to use class weighting (0: no, 1: yes)")
    
    # Random seed for reproducibility
    parser.add_argument("-random_seed", type=int, default=42, help="Random seed for train-test split and model initialization")
    
    return parser.parse_args()

def get_input_size(genotype_file):
    with gzip.open(genotype_file, 'rt') as f:
        first_line = next(f)
        values = first_line.strip().split()
        if len(values) == 3:  # If format is 3 columns (AA, AB, BB probabilities)
            return sum(1 for _ in f) + 1  # Add 1 for the first line we already read
    
    # If we reach here, reopen the file and count lines
    with gzip.open(genotype_file, 'rt') as f:
        return sum(1 for _ in f)

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

# Model for covariates-only prediction with multiple disease outputs
class CovariatesOnlyModel(nn.Module):
    def __init__(self, hidden_sizes, dropout_rate, act, use_covariates=True, use_age=True, use_gender=True, 
                 num_covariates=10, num_diseases=5):
        super(CovariatesOnlyModel, self).__init__()
        self.use_covariates = use_covariates
        self.use_age = use_age
        self.use_gender = use_gender
        self.num_diseases = num_diseases

        # Calculate input size based on enabled covariates
        input_size = 0
        if use_covariates:
            input_size += num_covariates  # PCs
        if use_age:
            input_size += 1  # Age
        if use_gender:
            input_size += 1  # Gender

        if input_size == 0:
            raise ValueError("At least one type of covariate must be enabled")

        # Create main layers
        layers = []
        current_size = input_size
        
        # Hidden layers
        for i in range(len(hidden_sizes)):
            layers.extend([
                (f'linear_{i}', nn.Linear(current_size, hidden_sizes[i])),
                (f'batchnorm_{i}', nn.BatchNorm1d(hidden_sizes[i])),
                (f'activation_{i}', self.get_activation(act)),
                (f'dropout_{i}', nn.Dropout(dropout_rate))
            ])
            current_size = hidden_sizes[i]
        
        # Output layer - now outputs predictions for multiple diseases
        layers.append(('output', nn.Linear(current_size, num_diseases)))
        
        self.model = nn.Sequential(OrderedDict(layers))
        
        print(f"CovariatesOnlyModel initialized with {num_diseases} disease outputs "
              f"(using PCs: {use_covariates}, age: {use_age}, gender: {use_gender})")

    def forward(self, covariates):
        return self.model(covariates)
    
    def get_activation(self, name):
        if name == 'tanh':
            return nn.Tanh()
        elif name == 'relu':
            return nn.ReLU()
        elif name == 'gelu':
            return nn.GELU()
        else:
            raise NotImplementedError(f"Activation function {name} not implemented.")

# Model for SNPs-only with or without PCA - multiple disease outputs
class SNPsOnlyModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rate, act, use_pca=True, num_diseases=5):
        super(SNPsOnlyModel, self).__init__()
        self.use_pca = use_pca
        self.num_diseases = num_diseases
        
        # If not using PCA, we need to handle the 3-channel SNP data
        if not use_pca:
            # For raw SNP data with shape (n_snps, 3), first apply a pointwise convolution
            self.pointwise_conv = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1)
            # After conv, shape becomes (n_snps, 1), so we flatten it
            self.flattened_size = input_size  # This is n_snps
        else:
            # For PCA features, we use them directly
            self.flattened_size = input_size  # This is n_components
        
        # Create main layers
        layers = []
        current_size = self.flattened_size
        
        # Hidden layers
        for i in range(len(hidden_sizes)):
            layers.extend([
                (f'linear_{i}', nn.Linear(current_size, hidden_sizes[i])),
                (f'batchnorm_{i}', nn.BatchNorm1d(hidden_sizes[i])),
                (f'activation_{i}', self.get_activation(act)),
                (f'dropout_{i}', nn.Dropout(dropout_rate))
            ])
            current_size = hidden_sizes[i]
        
        # Output layer - now outputs predictions for multiple diseases
        layers.append(('output', nn.Linear(current_size, num_diseases)))
        
        self.model = nn.Sequential(OrderedDict(layers))
        
        print(f"SNPsOnlyModel initialized with input size: {input_size}, "
              f"use_pca: {use_pca}, num_diseases: {num_diseases}")

    def forward(self, x):
        # If not using PCA and data is in format [batch_size, n_snps, 3]
        if not self.use_pca and x.dim() == 3 and x.size(2) == 3:
            # Reshape input from [batch_size, n_snps, 3] to [batch_size, 3, n_snps]
            x = x.permute(0, 2, 1)
            # Apply pointwise convolution to get [batch_size, 1, n_snps]
            x = self.pointwise_conv(x)
            # Flatten to [batch_size, n_snps]
            x = x.squeeze(1)
        elif x.dim() == 3 and x.size(1) == 1:
            # For PCA features with shape [batch_size, 1, n_components]
            x = x.squeeze(1)        
        return self.model(x)
    
    def get_activation(self, name):
        if name == 'tanh':
            return nn.Tanh()
        elif name == 'relu':
            return nn.ReLU()
        elif name == 'gelu':
            return nn.GELU()
        else:
            raise NotImplementedError(f"Activation function {name} not implemented.")

# Model for combined SNPs and covariates - multiple disease outputs
class GenotypeModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rate, act, use_covariates=True, 
                 use_age=True, use_gender=True, num_covariates=10, use_pca=True, num_diseases=5):
        super(GenotypeModel, self).__init__()
        self.use_covariates = use_covariates
        self.use_age = use_age
        self.use_gender = use_gender
        self.use_pca = use_pca
        self.num_diseases = num_diseases

        # Calculate total number of covariates
        self.total_covariates = 0
        if use_covariates:
            self.total_covariates += num_covariates  # PCs
        if use_age:
            self.total_covariates += 1  # Age
        if use_gender:
            self.total_covariates += 1  # Gender

        # If not using PCA, handle the 3-channel SNP data
        if not use_pca:
            self.pointwise_conv = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1)
            # After conv, input_size remains the same (n_snps)
        
        # Create main layers
        layers = []
        current_size = input_size
        
        # All hidden layers except the last one
        for i in range(len(hidden_sizes)):
            layers.extend([
                (f'linear_{i}', nn.Linear(current_size, hidden_sizes[i])),
                (f'batchnorm_{i}', nn.BatchNorm1d(hidden_sizes[i])),
                (f'activation_{i}', self.get_activation(act)),
                (f'dropout_{i}', nn.Dropout(dropout_rate))
            ])
            current_size = hidden_sizes[i]
        
        self.main_layers = nn.Sequential(OrderedDict(layers))
        
        # Last hidden layer (will concatenate covariates to its output)
        last_hidden_size = hidden_sizes[-1]
        concat_size = last_hidden_size + self.total_covariates
        
        # Process concatenated features
        self.final_processing = nn.Sequential(OrderedDict([
            ('final_linear', nn.Linear(concat_size, last_hidden_size)),
            ('final_bn', nn.BatchNorm1d(last_hidden_size)),
            ('final_activation', self.get_activation(act)),
            ('final_dropout', nn.Dropout(dropout_rate))
        ]))
        
        # Output layer - now outputs predictions for multiple diseases
        self.output_layer = nn.Linear(last_hidden_size, num_diseases)
        
        print(f"GenotypeModel initialized with {num_diseases} disease outputs "
              f"(using covariates: {use_covariates}, age: {use_age}, gender: {use_gender}, pca: {use_pca})")

    def forward(self, x, covariates=None):
        # Handle different input formats
        if not self.use_pca and x.dim() == 3 and x.size(2) == 3:
            # Raw SNP data [batch_size, n_snps, 3]
            x = x.permute(0, 2, 1)  # -> [batch_size, 3, n_snps]
            x = self.pointwise_conv(x).squeeze(1)  # -> [batch_size, n_snps]
        elif x.dim() == 3 and x.size(1) == 3:
            # Another possible raw data format [batch_size, 3, n_snps]
            x = self.pointwise_conv(x).squeeze(1)  # -> [batch_size, n_snps]
        elif x.dim() == 3 and x.size(1) == 1:
            # For PCA features with shape [batch_size, 1, n_components]
            x = x.squeeze(1)  # -> [batch_size, n_components]
        
        # Main layers
        x = self.main_layers(x)
        
        # Concatenate with covariates
        if covariates is not None and self.total_covariates > 0:
            x = torch.cat([x, covariates], dim=1)
        
        # Process concatenated features
        x = self.final_processing(x)
        
        # Output layer
        return self.output_layer(x)
    
    def get_activation(self, name):
        if name == 'tanh':
            return nn.Tanh()
        elif name == 'relu':
            return nn.ReLU()
        elif name == 'gelu':
            return nn.GELU()
        else:
            raise NotImplementedError(f"Activation function {name} not implemented.")

class MultiDiseaseGenotypeDataset(Dataset):
    def __init__(self, file_list, phenotype_data, disease_labels, use_covariates=True, use_age=True, 
                 use_gender=True, norm_age="standard", norm_pcs="standard", norm_gender="none", 
                 fit_normalizers=True, normalizers=None, covariates_only=False, 
                 snps_only=False, use_pca=False, pca_features_dir=None):
        self.file_list = file_list
        self.phenotype_data = phenotype_data
        self.disease_labels = disease_labels
        self.use_covariates = use_covariates
        self.use_age = use_age
        self.use_gender = use_gender
        self.covariates_only = covariates_only
        self.snps_only = snps_only
        self.use_pca = use_pca
        self.pca_features_dir = pca_features_dir
        
        # Verify that the label columns exist in the phenotype data
        for label in self.disease_labels:
            if label not in self.phenotype_data.columns:
                raise ValueError(f"Disease label column '{label}' not found in phenotype data. "
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
        if not (self.use_pca and self.pca_features_dir is not None):
            # Only open file handles if not using pre-computed PCA features
            for file in file_list:
                f = open(file, 'rb')
                self.file_handles[file] = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        
        # Log initialization
        print(f"\nDataset Initialization:")
        print(f"- Number of files: {len(file_list)}")
        print(f"- Disease labels: {', '.join(disease_labels)}")
        print(f"- Model mode: {'Covariates-only' if covariates_only else 'SNPs-only' if snps_only else 'Full'}")
        if not snps_only:
            print(f"- Using PCs: {use_covariates} (normalization: {norm_pcs})")
            print(f"- Using age: {use_age} (normalization: {norm_age})")
            print(f"- Using gender: {use_gender} (normalization: {norm_gender})")
        if not covariates_only:
            print(f"- Using PCA for SNPs: {use_pca}")
            if use_pca and pca_features_dir:
                print(f"- Using pre-computed PCA features from: {pca_features_dir}")

        # Print disease prevalence information
        self._print_disease_statistics()

        # Print sample information for the first few samples
        self._print_sample_examples(3)

    def _print_sample_examples(self, num_samples=3):
        """Print detailed information about the first few samples in the dataset"""
        print(f"\nSample Examples (first {num_samples}):")
        for i in range(min(num_samples, len(self.file_list))):
            file_path = self.file_list[i]
            sample_id = int(file_path.split('sample_')[1].split('.gen.gz')[0])
            
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
            sample_id = int(file_path.split('sample_')[1].split('.gen.gz')[0])
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
    
    def get_normalizers(self):
        """Return the fitted normalizers"""
        return {
            'age': self.age_normalizer,
            'pcs': self.pcs_normalizer,
            'gender': self.gender_normalizer
        }
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        genotype_file = self.file_list[idx]
        sample_id_str = os.path.basename(genotype_file).replace("sample_", "").replace(".gen.gz", "")
        sample_id = int(sample_id_str)
        
        # Get labels for all diseases
        labels = []
        for disease in self.disease_labels:
            label = self.phenotype_data.loc[self.phenotype_data['new_order'] == sample_id, disease].values[0]
            labels.append(float(label))
        
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        
        # Process covariates if needed
        if not self.snps_only:
            covariates_list = []
            if self.use_covariates:
                # Get PC values as matrix (1, n_pcs)
                pc_data = np.array([
                    self.phenotype_data.loc[self.phenotype_data['new_order'] == sample_id, f'PC{i}'].values[0] for i in range(1, 11)]).reshape(1, -1)
                
                # Transform PCs
                normalized_pcs = self.pcs_normalizer.transform(pc_data).flatten()
                covariates_list.append(normalized_pcs)
                
            if self.use_age:
                # Get and normalize age
                age = self.phenotype_data.loc[self.phenotype_data['new_order'] == sample_id, 'Agexit'].values[0]
                normalized_age = self.age_normalizer.transform(np.array([[age]])).flatten()
                covariates_list.append(normalized_age)
            
            if self.use_gender:
                # Get and normalize gender
                gender = self.phenotype_data.loc[self.phenotype_data['new_order'] == sample_id, 'Sex'].values[0]
                normalized_gender = self.gender_normalizer.transform(np.array([[gender]])).flatten()
                covariates_list.append(normalized_gender)
            
            # Combine all covariates
            covariates = np.concatenate(covariates_list) if covariates_list else np.array([])
            covariates_tensor = torch.tensor(covariates, dtype=torch.float32)
        
        # Return only covariates for covariates-only mode
        if self.covariates_only:
            return covariates_tensor, labels_tensor
        
        # Check if we're using pre-computed PCA features
        if self.use_pca and self.pca_features_dir is not None:
            # Load pre-computed PCA features
            pca_file = os.path.join(self.pca_features_dir, f"sample_{sample_id}_pca.npy")
            if os.path.exists(pca_file):
                pca_features = np.load(pca_file)
                # Handle different array shapes 
                if pca_features.ndim == 3 and pca_features.shape[0] == 1:
                    pca_features = pca_features.reshape(pca_features.shape[1:])
                elif pca_features.ndim == 2 and pca_features.shape[0] == 1:
                    # Keep as is - this is the expected shape (1, n_components)
                    pass
                
                genotype_tensor = torch.tensor(pca_features, dtype=torch.float32)
            else:
                raise FileNotFoundError(f"Pre-computed PCA file not found: {pca_file}")
        else:
            # Process raw genotype data 
            mmap_file = self.file_handles[genotype_file]
            mmap_file.seek(0)
            with gzip.GzipFile(fileobj=mmap_file) as f:
                data = pd.read_csv(f, sep=r'\s+', header=None)
            
            # Raw genotype data - keep the format with 3 values per SNP
            genotype_tensor = torch.tensor(data.values, dtype=torch.float32)
        
        # Return based on mode
        if self.snps_only:
            return genotype_tensor, labels_tensor
        else:
            # For full model, include both genotype and covariates
            return genotype_tensor, covariates_tensor, labels_tensor

    def __del__(self):
        # Clean up file handles
        for handle in self.file_handles.values():
            handle.close()

def print_lr(optimizer):
    for param_group in optimizer.param_groups:
        print(f"Current Learning Rate: {param_group['lr']}")

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, disease_labels, 
                device='cuda', save_dir=None, early_stopping=None, covariates_only=False, snps_only=False):
    print(f"Training on device: {device}")
    print(f"Training mode: {'covariates-only' if covariates_only else 'SNPs-only' if snps_only else 'full'}")
    
    # Set up CUDA streams for prefetching if using CUDA
    use_prefetching = device.type == 'cuda'
    if use_prefetching:
        print("Using data prefetching optimization")
        main_stream = torch.cuda.Stream()
        prefetch_stream = torch.cuda.Stream()
    else:
        print("Prefetching disabled (not using CUDA)")
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    completed_epochs = 0  

    num_diseases = len(disease_labels)

    # Initialize history dictionary with per-disease metrics
    history = {
        'train_loss': [], 'test_loss': [],
        'learning_rates': []
    }
    
    # Add per-disease metrics
    for disease in disease_labels:
        for phase in ['train', 'test']:
            history[f'{phase}_{disease}_acc'] = []
            history[f'{phase}_{disease}_auc'] = []
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Store predictions and true labels for each disease and phase
        all_preds = {phase: {disease: [] for disease in disease_labels} for phase in ['train', 'test']}
        all_labels = {phase: {disease: [] for disease in disease_labels} for phase in ['train', 'test']}

        for phase in ['train', 'test']:
            start_time = time.time()
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = {disease: 0 for disease in disease_labels}
            total_samples = 0

            

            # Initialize prefetch variables
            prefetched_batch = None
            
            if use_prefetching:
                # Create iterator for dataloader
                batch_iter = iter(dataloaders[phase])
                num_batches = len(dataloaders[phase])
                
                # Prefetch first batch
                with torch.cuda.stream(prefetch_stream):
                    try:
                        if covariates_only:
                            prefetch_covariates, prefetch_labels = next(batch_iter)
                            prefetch_covariates = prefetch_covariates.to(device, non_blocking=True)
                            prefetch_labels = prefetch_labels.to(device, non_blocking=True)
                            prefetched_batch = (prefetch_covariates, prefetch_labels)
                        elif snps_only:
                            prefetch_inputs, prefetch_labels = next(batch_iter)
                            prefetch_inputs = prefetch_inputs.to(device, non_blocking=True)
                            prefetch_labels = prefetch_labels.to(device, non_blocking=True)
                            prefetched_batch = (prefetch_inputs, prefetch_labels)
                        else:
                            prefetch_inputs, prefetch_covariates, prefetch_labels = next(batch_iter)
                            prefetch_inputs = prefetch_inputs.to(device, non_blocking=True)
                            prefetch_covariates = prefetch_covariates.to(device, non_blocking=True)
                            prefetch_labels = prefetch_labels.to(device, non_blocking=True)
                            prefetched_batch = (prefetch_inputs, prefetch_covariates, prefetch_labels)
                    except StopIteration:
                        prefetched_batch = None
                
                # Process batches with prefetching
                for batch_idx in range(num_batches):
                    # Wait for the prefetched batch to be ready
                    torch.cuda.current_stream().wait_stream(prefetch_stream)
                    
                    # Get the current batch from prefetched data
                    current_batch = prefetched_batch
                    
                    # Start prefetching the next batch
                    if batch_idx + 1 < num_batches:
                        with torch.cuda.stream(prefetch_stream):
                            try:
                                if covariates_only:
                                    prefetch_covariates, prefetch_labels = next(batch_iter)
                                    prefetch_covariates = prefetch_covariates.to(device, non_blocking=True)
                                    prefetch_labels = prefetch_labels.to(device, non_blocking=True)
                                    prefetched_batch = (prefetch_covariates, prefetch_labels)
                                elif snps_only:
                                    prefetch_inputs, prefetch_labels = next(batch_iter)
                                    prefetch_inputs = prefetch_inputs.to(device, non_blocking=True)
                                    prefetch_labels = prefetch_labels.to(device, non_blocking=True)
                                    prefetched_batch = (prefetch_inputs, prefetch_labels)
                                else:
                                    prefetch_inputs, prefetch_covariates, prefetch_labels = next(batch_iter)
                                    prefetch_inputs = prefetch_inputs.to(device, non_blocking=True)
                                    prefetch_covariates = prefetch_covariates.to(device, non_blocking=True)
                                    prefetch_labels = prefetch_labels.to(device, non_blocking=True)
                                    prefetched_batch = (prefetch_inputs, prefetch_covariates, prefetch_labels)
                            except StopIteration:
                                prefetched_batch = None
                    
                    # Process the current batch
                    if current_batch is not None:
                        with torch.cuda.stream(main_stream):
                            if covariates_only:
                                covariates, labels = current_batch
                            elif snps_only:
                                inputs, labels = current_batch
                            else:
                                inputs, covariates, labels = current_batch
                            
                            # Run the model and compute loss
                            optimizer.zero_grad()
                            with torch.set_grad_enabled(phase == 'train'):
                                if covariates_only:
                                    logits = model(covariates)
                                elif snps_only:
                                    logits = model(inputs)
                                else:
                                    logits = model(inputs, covariates)

                                loss = criterion(logits, labels)

                                # Convert logits to probabilities for metrics
                                probs = torch.sigmoid(logits)
                                preds = (probs >= 0.5).float()

                                if phase == 'train':
                                    loss.backward()
                                    optimizer.step()
                                    
                                    # Step the scheduler if it's a per-iteration scheduler
                                    if isinstance(scheduler, (WarmupExponential, ExponentialDecay)):
                                        scheduler.step()

                            batch_size = labels.size(0)
                            running_loss += loss.item() * batch_size
                            
                            # Calculate correct predictions for each disease
                            for i, disease in enumerate(disease_labels):
                                running_corrects[disease] += torch.sum(preds[:, i] == labels[:, i])
                                
                                # Store predictions and true labels for AUC calculation
                                all_labels[phase][disease].extend(labels[:, i].cpu().numpy())
                                all_preds[phase][disease].extend(probs[:, i].detach().cpu().numpy())
                            
                            total_samples += batch_size
                        
                        # Wait for the main stream to finish processing
                        torch.cuda.current_stream().wait_stream(main_stream)
                
            else:
                # Standard processing without prefetching
                for batch in dataloaders[phase]:
                    if covariates_only:
                        covariates, labels = batch
                        covariates = covariates.to(device)
                        labels = labels.to(device)
                    elif snps_only:
                        inputs, labels = batch
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                    else:
                        inputs, covariates, labels = batch
                        inputs = inputs.to(device)
                        covariates = covariates.to(device)
                        labels = labels.to(device)

                    # Process the current batch
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        if covariates_only:
                            logits = model(covariates)
                        elif snps_only:
                            logits = model(inputs)
                        else:
                            logits = model(inputs, covariates)

                        loss = criterion(logits, labels)

                        # Convert logits to probabilities for metrics
                        probs = torch.sigmoid(logits)
                        preds = (probs >= 0.5).float()

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            
                            # Step the scheduler if it's a per-iteration scheduler
                            if isinstance(scheduler, (WarmupExponential, ExponentialDecay)):
                                scheduler.step()

                    batch_size = labels.size(0)
                    running_loss += loss.item() * batch_size
                    
                    # Calculate correct predictions for each disease
                    for i, disease in enumerate(disease_labels):
                        running_corrects[disease] += torch.sum(preds[:, i] == labels[:, i])
                        
                        # Store predictions and true labels for AUC calculation
                        all_labels[phase][disease].extend(labels[:, i].cpu().numpy())
                        all_preds[phase][disease].extend(probs[:, i].detach().cpu().numpy())
                    
                    total_samples += batch_size

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
                
                # Calculate AUC if possible
                try:
                    epoch_auc = roc_auc_score(all_labels[phase][disease], all_preds[phase][disease])
                    history[f'{phase}_{disease}_auc'].append(epoch_auc)
                    auc_str = f"AUC: {epoch_auc:.4f}"
                except Exception as e:
                    history[f'{phase}_{disease}_auc'].append(0.5)  # Default value
                    auc_str = "AUC: N/A (need both classes)"
                    print(f"Warning: Could not calculate AUC for {disease} in {phase} phase: {str(e)}")
                
                print(f'  {disease}: Acc: {epoch_acc:.4f}, {auc_str}')

            # Early stopping check based on validation loss
            if phase == 'test' and early_stopping is not None:
                early_stopping(epoch_loss)
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    completed_epochs = epoch + 1  # Save the number of completed epochs
                    return model, history, compute_final_metrics(all_labels, all_preds, disease_labels), all_preds, all_labels, completed_epochs
                
            # Save best model
            if phase == 'test' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                
                # # Save the best model
                # if save_dir:
                #     torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
                #     print(f"Saved new best model with validation loss: {best_loss:.4f}")

        # Step schedulers that work on epoch-level
        if scheduler is not None and not isinstance(scheduler, (WarmupExponential, ExponentialDecay)):
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(history['test_loss'][-1])
            else:
                scheduler.step()

        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        print_lr(optimizer)
        completed_epochs = epoch + 1  # Update completed epochs counter

    # Load best model
    model.load_state_dict(best_model_wts)
    return model, history, compute_final_metrics(all_labels, all_preds, disease_labels), all_preds, all_labels, completed_epochs


def compute_final_metrics(all_labels, all_preds, disease_labels):
    final_metrics = {}
    
    for phase in ['train', 'test']:
        phase_metrics = {}
        
        for disease in disease_labels:
            y_true = np.array(all_labels[phase][disease])
            y_pred_proba = np.array(all_preds[phase][disease])

            # Convert probabilities to binary predictions
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            disease_metrics = {}
            
            try:
                cm = confusion_matrix(y_true, y_pred)
                if cm.shape == (2, 2):  # Only calculate if we have a proper 2x2 confusion matrix
                    tn, fp, fn, tp = cm.ravel()
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                    accuracy = (tp + tn) / (tp + tn + fp + fn)
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
                else:
                    sensitivity = specificity = accuracy = precision = f1 = 0
                
                try:
                    # ROC Curve
                    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                    roc_auc = auc(fpr, tpr)
                    
                    # PR Curve
                    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
                    pr_auc = average_precision_score(y_true, y_pred_proba)
                except Exception as e:
                    print(f"Error calculating curves for {disease} in {phase}: {str(e)}")
                    fpr, tpr = np.array([]), np.array([])
                    precision_curve, recall_curve = np.array([]), np.array([])
                    roc_auc = pr_auc = 0.5
            except Exception as e:
                print(f"Error calculating metrics for {disease} in {phase}: {str(e)}")
                cm = np.zeros((2, 2))
                sensitivity = specificity = accuracy = precision = f1 = 0
                fpr, tpr = np.array([]), np.array([])
                precision_curve, recall_curve = np.array([]), np.array([])
                roc_auc = pr_auc = 0.5

            disease_metrics = {
                'confusion_matrix': cm,
                'sensitivity': f'{sensitivity:.5f}',
                'specificity': f'{specificity:.5f}',
                'accuracy': f'{accuracy:.5f}',
                'precision': f'{precision:.5f}',
                'f1_score': f'{f1:.5f}',
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'fpr': fpr,
                'tpr': tpr,
                'precision_curve': precision_curve,
                'recall_curve': recall_curve
            }
            
            phase_metrics[disease] = disease_metrics
        
        final_metrics[phase] = phase_metrics
    
    return final_metrics

def plot_all_metrics(history, final_metrics, disease_labels, save_dir):
    # Create directory for plots
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create a 2x2 subplot figure for combined metrics
    plt.figure(figsize=(20, 16))
    
    # 1. Plot overall loss (top left)
    plt.subplot(2, 2, 1)
    for phase in ['train', 'test']:
        if f'{phase}_loss' in history and history[f'{phase}_loss']:
            plt.plot(history[f'{phase}_loss'], label=f'{phase}')
    plt.title('Model Loss', fontsize=16)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 2. Plot learning rate (top right)
    plt.subplot(2, 2, 2)
    plt.plot(history['learning_rates'])
    plt.title('Learning Rate', fontsize=16)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(alpha=0.3)
    
    # 3. Plot average accuracy across diseases (bottom left)
    plt.subplot(2, 2, 3)
    for phase in ['train', 'test']:
        # Calculate average accuracy across all diseases for each epoch
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
    
    # 4. Plot average AUC across diseases (bottom right)
    plt.subplot(2, 2, 4)
    for phase in ['train', 'test']:
        # Calculate average AUC across all diseases for each epoch
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
            plt.plot(avg_auc, label=f'{phase} avg auc')
    plt.title('Average AUC Across All Diseases', fontsize=16)
    plt.xlabel('Epoch')
    plt.ylabel('Average AUC')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Add a main title for the entire figure
    plt.suptitle('Multilabel Disease Prediction Model Performance', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    
    # Save the combined plot
    plt.savefig(os.path.join(plots_dir, 'combined_metrics_plot.png'), dpi=300)
    plt.close()
    
    # Still plot metrics for each disease individually (keeping this part)
    metrics = ['acc', 'auc']
    phases = ['train', 'test']
    
    # Plot accuracy and AUC for each disease
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
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{metric}_by_disease.png'))
        plt.close()

# def plot_all_metrics(history, final_metrics, disease_labels, save_dir):
#     # Create metrics directory if it doesn't exist
#     metrics_dir = os.path.join(save_dir, 'metrics_plots')
#     os.makedirs(metrics_dir, exist_ok=True)
    
#     # Plot overall loss
#     plt.figure(figsize=(12, 8))
#     plt.plot(history['train_loss'], label='Train')
#     plt.plot(history['test_loss'], label='Test')
#     plt.title('Overall Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(os.path.join(metrics_dir, 'overall_loss.png'))
#     plt.close()
    
#     # Plot learning rate
#     plt.figure(figsize=(12, 8))
#     plt.plot(history['learning_rates'])
#     plt.title('Learning Rate')
#     plt.xlabel('Epoch')
#     plt.ylabel('Learning Rate')
#     plt.yscale('log')
#     plt.grid(True)
#     plt.savefig(os.path.join(metrics_dir, 'learning_rate.png'))
#     plt.close()
    
#     # Plot per-disease metrics
#     for disease in disease_labels:
#         # Accuracy
#         plt.figure(figsize=(12, 8))
#         plt.plot(history[f'train_{disease}_acc'], label='Train')
#         plt.plot(history[f'test_{disease}_acc'], label='Test')
#         plt.title(f'{disease} - Accuracy')
#         plt.xlabel('Epoch')
#         plt.ylabel('Accuracy')
#         plt.legend()
#         plt.grid(True)
#         plt.savefig(os.path.join(metrics_dir, f'{disease}_accuracy.png'))
#         plt.close()
        
#         # AUC
#         plt.figure(figsize=(12, 8))
#         plt.plot(history[f'train_{disease}_auc'], label='Train')
#         plt.plot(history[f'test_{disease}_auc'], label='Test')
#         plt.title(f'{disease} - AUC')
#         plt.xlabel('Epoch')
#         plt.ylabel('AUC')
#         plt.legend()
#         plt.grid(True)
#         plt.savefig(os.path.join(metrics_dir, f'{disease}_auc.png'))
#         plt.close()
    
#     # Plot ROC curves for each disease
#     phases = ['train', 'test']
    
#     for disease in disease_labels:
#         plt.figure(figsize=(12, 10))
#         for phase in phases:
#             metrics = final_metrics[phase][disease]
#             if len(metrics['fpr']) > 0 and len(metrics['tpr']) > 0:
#                 plt.plot(
#                     metrics['fpr'], 
#                     metrics['tpr'], 
#                     label=f'{phase} (AUC = {metrics["roc_auc"]:.3f})'
#                 )
        
#         plt.plot([0, 1], [0, 1], 'k--')
#         plt.xlabel('False Positive Rate')
#         plt.ylabel('True Positive Rate')
#         plt.title(f'{disease} - ROC Curve')
#         plt.legend(loc='lower right')
#         plt.grid(True)
#         plt.savefig(os.path.join(metrics_dir, f'{disease}_roc_curve.png'))
#         plt.close()
        
#         # Plot PR curves for each disease
#         plt.figure(figsize=(12, 10))
#         for phase in phases:
#             metrics = final_metrics[phase][disease]
#             if len(metrics['precision_curve']) > 0 and len(metrics['recall_curve']) > 0:
#                 plt.plot(
#                     metrics['recall_curve'], 
#                     metrics['precision_curve'], 
#                     label=f'{phase} (AP = {metrics["pr_auc"]:.3f})'
#                 )
        
#         plt.xlabel('Recall')
#         plt.ylabel('Precision')
#         plt.title(f'{disease} - Precision-Recall Curve')
#         plt.legend(loc='lower left')
#         plt.grid(True)
#         plt.savefig(os.path.join(metrics_dir, f'{disease}_pr_curve.png'))
#         plt.close()
    
#     # Combined ROC and PR plots for test set
#     plt.figure(figsize=(15, 12))
#     for disease in disease_labels:
#         metrics = final_metrics['test'][disease]
#         if len(metrics['fpr']) > 0 and len(metrics['tpr']) > 0:
#             plt.plot(
#                 metrics['fpr'], 
#                 metrics['tpr'], 
#                 label=f'{disease} (AUC = {metrics["roc_auc"]:.3f})'
#             )
    
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('ROC Curves for All Diseases (Test Set)')
#     plt.legend(loc='lower right')
#     plt.grid(True)
#     plt.savefig(os.path.join(metrics_dir, 'all_diseases_roc_curve.png'))
#     plt.close()
    
#     plt.figure(figsize=(15, 12))
#     for disease in disease_labels:
#         metrics = final_metrics['test'][disease]
#         if len(metrics['precision_curve']) > 0 and len(metrics['recall_curve']) > 0:
#             plt.plot(
#                 metrics['recall_curve'], 
#                 metrics['precision_curve'], 
#                 label=f'{disease} (AP = {metrics["pr_auc"]:.3f})'
#             )
    
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title('Precision-Recall Curves for All Diseases (Test Set)')
#     plt.legend(loc='lower left')
#     plt.grid(True)
#     plt.savefig(os.path.join(metrics_dir, 'all_diseases_pr_curve.png'))
#     plt.close()

def write_results(model, hyperparameters, final_metrics, disease_labels, save_dir):
    # Create results directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Write hyperparameters and overall results to text file
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
                f.write(f"  Accuracy:    {metrics['accuracy']}\n")
                f.write(f"  Sensitivity: {metrics['sensitivity']}\n")
                f.write(f"  Specificity: {metrics['specificity']}\n")
                f.write(f"  Precision:   {metrics['precision']}\n")
                f.write(f"  F1 Score:    {metrics['f1_score']}\n")
                f.write(f"  ROC AUC:     {metrics['roc_auc']:.5f}\n")
                f.write(f"  PR AUC:      {metrics['pr_auc']:.5f}\n")
                f.write(f"  Confusion Matrix:\n    {metrics['confusion_matrix']}\n")
    
    # Write results to CSV
    with open(os.path.join(save_dir, 'experiment_results.csv'), 'w', newline='') as csvfile:
        # Start with hyperparameters
        fieldnames = list(hyperparameters.keys())
        
        # Add metrics fields for each disease and phase
        for phase in ['train', 'test']:
            for disease in disease_labels:
                for metric in ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1_score', 'roc_auc', 'pr_auc']:
                    fieldnames.append(f"{phase}_{disease}_{metric}")
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Create row with all results
        row = hyperparameters.copy()
        
        # Add metrics for each disease and phase
        for phase in ['train', 'test']:
            for disease in disease_labels:
                metrics = final_metrics[phase][disease]
                for metric in ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1_score']:
                    row[f"{phase}_{disease}_{metric}"] = metrics[metric]
                
                row[f"{phase}_{disease}_roc_auc"] = metrics['roc_auc']
                row[f"{phase}_{disease}_pr_auc"] = metrics['pr_auc']
        
        writer.writerow(row)

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
    
    # Disease labels
    disease_labels = args.disease_labels
    num_diseases = len(disease_labels)
    print(f"Multi-disease classification with {num_diseases} diseases: {', '.join(disease_labels)}")
      
    # Create experiment folder
    if not os.path.exists(experiment_dir):
        print(f"Result path did not exist but is made now.\nResult Path is {experiment_dir}")
        os.makedirs(experiment_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load phenotype data
    phenotype_data = pd.read_excel(phenotype_file)
    print(f"Phenotype data loaded, shape: {phenotype_data.shape}")

    # Get list of genotype files and sort numerically by sample number
    file_list = glob.glob(os.path.join(args.genotype_dir, "sample_*.gen.gz"))
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

    first_genotype_file = filtered_file_list[0] if filtered_file_list else None
    if first_genotype_file:
        print(f"First genotype file after filtering is: {first_genotype_file}")
    else:
        print("No matching genotype files found!")
        return
    
    # Determine input size based on the model type and whether we're using PCA
    if args.use_pca and args.pca_features_dir is not None:
        # Try to determine the number of components from the PCA model
        pca_model_path = os.path.join(args.pca_features_dir, "pca_model.pkl")
        if os.path.exists(pca_model_path):
            print(f"Loading PCA model from {pca_model_path}")
            with open(pca_model_path, 'rb') as f:
                pca_model = pickle.load(f)
                
            # Handle both standard PCA and IncrementalPCA
            if hasattr(pca_model, 'n_components_'):
                input_size = pca_model.n_components_
            else:
                input_size = pca_model.n_components
                
            print(f"Using pre-computed PCA features with {input_size} components")
        else:
            # If PCA model not found, use specified number of components
            input_size = args.n_components
            print(f"PCA model not found, using specified {input_size} components")
    else:
        # For raw SNP data, get the number of SNPs from the first file
        orig_input_size = get_input_size(first_genotype_file)
        print(f"Dynamically determined original input size: {orig_input_size}")
        input_size = orig_input_size

    if len(filtered_file_list) != len(phenotype_samples):
        print(f"Warning: Number of files ({len(filtered_file_list)}) does not match number of samples ({len(phenotype_samples)}) in phenotype data.")

    # Split the data
    train_files, test_files = train_test_split(
        filtered_file_list, test_size=0.2, random_state=args.random_seed
    )
    print(f"Data split: Train {len(train_files)}, Test {len(test_files)}")

    # Determine model type
    covariates_only = (args.model_type == "covariates_only")
    snps_only = (args.model_type == "snps_only")
    
    # Create datasets with appropriate mode
    train_dataset = MultiDiseaseGenotypeDataset(
        train_files, 
        phenotype_data,
        disease_labels=disease_labels,
        use_covariates=bool(args.cov),
        use_age=bool(args.use_age),
        use_gender=bool(args.use_gender),
        norm_age=args.norm_age,
        norm_pcs=args.norm_pcs,
        norm_gender=args.norm_gender,
        fit_normalizers=True,
        normalizers=None,
        covariates_only=covariates_only,
        snps_only=snps_only,
        use_pca=bool(args.use_pca),
        pca_features_dir=args.pca_features_dir
    )
    
    # Get the fitted normalizers from training dataset
    fitted_normalizers = train_dataset.get_normalizers()

    # Create test dataset with the fitted normalizers
    test_dataset = MultiDiseaseGenotypeDataset(
        test_files, 
        phenotype_data,
        disease_labels=disease_labels,
        use_covariates=bool(args.cov),
        use_age=bool(args.use_age),
        use_gender=bool(args.use_gender),
        norm_age=args.norm_age,
        norm_pcs=args.norm_pcs,
        norm_gender=args.norm_gender,
        fit_normalizers=False,  # Don't fit new normalizers
        normalizers=fitted_normalizers,  # Use normalizers from training set
        covariates_only=covariates_only,
        snps_only=snps_only,
        use_pca=bool(args.use_pca),
        pca_features_dir=args.pca_features_dir
    )

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    }
    print("DataLoaders created")
    
    # Create the appropriate model based on model_type
    if covariates_only:
        model = CovariatesOnlyModel(
            hidden_sizes=args.hidden_sizes,
            dropout_rate=args.dropout,
            act=args.act,
            use_covariates=bool(args.cov),
            use_age=bool(args.use_age),
            use_gender=bool(args.use_gender),
            num_covariates=10,
            num_diseases=num_diseases
        )
    elif snps_only:
        model = SNPsOnlyModel(
            input_size=input_size,
            hidden_sizes=args.hidden_sizes,
            dropout_rate=args.dropout,
            act=args.act,
            use_pca=bool(args.use_pca),
            num_diseases=num_diseases
        )
    else:
        model = GenotypeModel(
            input_size=input_size,
            hidden_sizes=args.hidden_sizes,
            dropout_rate=args.dropout,
            act=args.act,
            use_covariates=bool(args.cov),
            use_age=bool(args.use_age),
            use_gender=bool(args.use_gender),
            num_covariates=10,
            use_pca=bool(args.use_pca),
            num_diseases=num_diseases
        )

    model = model.to(device)
    with open(experiment_dir + '/model_architecture.txt', 'w') as file:
        file.write(str(model))
        print(model)

    print("Model created and moved to device")

    # For multilabel classification, we use BCEWithLogitsLoss
    if args.class_weight:
        # If using class weighting, calculate weight for each disease
        disease_weights = []
        for disease in disease_labels:
            pos_count = phenotype_data[disease].sum()
            neg_count = len(phenotype_data) - pos_count
            
            if pos_count > 0 and neg_count > 0:
                # Calculate weight inversely proportional to class frequency
                weight = len(phenotype_data) / (2 * pos_count)
                disease_weights.append(weight)
            else:
                disease_weights.append(1.0)
        
        # Set pos_weight for BCEWithLogitsLoss
        print(f"Disease weights: {dict(zip(disease_labels, disease_weights))}")
        pos_weight = torch.tensor(disease_weights, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

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
        model, dataloaders, criterion, optimizer, scheduler, num_epochs, disease_labels,
        device=device, save_dir=experiment_dir, early_stopping=early_stopping,
        covariates_only=covariates_only, snps_only=snps_only
    )

    # Plot metrics
    plot_all_metrics(history, final_metrics, disease_labels, experiment_dir)

    # Save the trained model
    #torch.save(model.state_dict(), os.path.join(experiment_dir, 'final_model.pth'))
    print("Model training completed and saved")

    # Update hyperparameters dictionary
    hyperparameters = {
        'Exp_ID': id,
        'BS': batch_size,
        'Epochs': completed_epochs,
        'Start_LR': learning_rate,
        'Final_LR': optimizer.param_groups[0]["lr"],
        'Dropout': dropout_rate,
        'Act': act,
        'Opt': opt,
        'Sch': sch,
        'WD': wd,
        'DF': args.df,
        'Disease_Labels': ','.join(disease_labels),
        'Num_Diseases': num_diseases,
        'Use_PCs': bool(args.cov),
        'norm_PCs': args.norm_pcs,
        'Use_Age': bool(args.use_age),
        'norm_Age': args.norm_age,
        'Use_Gender': bool(args.use_gender),
        'norm_Gender': args.norm_gender,
        'hidden_sizes': str(args.hidden_sizes),
        'model_type': args.model_type,
        'use_pca': bool(args.use_pca),
        'n_components': input_size if args.use_pca else None,
        'pca_features_dir': args.pca_features_dir,
        'class_weight': bool(args.class_weight),
        'random_seed': args.random_seed
    }

    # Write results
    write_results(model, hyperparameters, final_metrics, disease_labels, experiment_dir)

    print(f"\nTraining complete. Results saved to {experiment_dir}")
    
    print("\nFinal performance metrics:")
    for disease in disease_labels:
        print(f"\n{disease.upper()}:")
        print(f"Train - AUC: {final_metrics['train'][disease]['roc_auc']:.4f}, "
              f"Accuracy: {final_metrics['train'][disease]['accuracy']}")
        print(f"Test  - AUC: {final_metrics['test'][disease]['roc_auc']:.4f}, "
              f"Accuracy: {final_metrics['test'][disease]['accuracy']}")


if __name__ == '__main__':
    start_time = time.time()
    
    main()
    
    end_time = time.time()
    total_runtime = end_time - start_time
    
    print(f"\nTotal script runtime: {total_runtime:.2f} seconds")
    hours, rem = divmod(total_runtime, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Total runtime: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")