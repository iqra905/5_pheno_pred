import os
import pandas as pd
import gzip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import mmap
import copy
import glob
import math
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
import matplotlib.pyplot as plt
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
import argparse
from sklearn.metrics import confusion_matrix, roc_curve, auc
import csv
import time
from torch.nn import TransformerEncoder, TransformerEncoderLayer

def parse_int_list(s):
    return [int(x) for x in s.split(',')]

def parse_args():
    parser = argparse.ArgumentParser(description="Genotype Model Training")
    parser.add_argument("-ID", type=str, default="Exp_01", help="ID of the experiment")
    parser.add_argument("-exp_dir", type=str, default='/vol/research/fmodal_mmmed/Codes/5_disease_experiments/Transformer/results_scratch/t2d/full/', help="Directory to save experiment results")
    parser.add_argument("-genotype_dir", type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/sampled_data_5M_unq_npy', help="Directory containing genotype files")
    parser.add_argument("-phenotype_file", type=str, default='/vol/vssp/SF_ucdatasets/gwas/data_files/disease_pheno/t2d.xlsx', help="Path to phenotype file")

    # Model architecture parameters
    parser.add_argument("-num_transformer_layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("-nhead", type=int, default=8, help="Number of transformer heads")
    parser.add_argument("-d_model", type=int, default=384, help="Dimension of transformer model")
    parser.add_argument("-fc_layers", type=parse_int_list, default=[128,32], help="Sizes of fully connected layers")
    parser.add_argument("-seq_len", type=int, default=32, help="Sequence length after pooling for transformer input")

    # Add pooling type argument
    parser.add_argument("-pooling", type=str, default="cls", choices=["mean", "cls", "weighted"], help="Type of pooling to use in transformer")
    
    # Data parameters
    parser.add_argument("-label_col", type=str, default="t2dm", help="Column name for labels")
    parser.add_argument("-use_pcs", type=int, default=0, choices=[0, 1], help="Whether to use PC covariates (0 for No, 1 for Yes)")
    
    # Add arguments for Age and Sex
    parser.add_argument("-use_age", type=int, default=0, choices=[0, 1], help="Whether to use Age covariate (0 for No, 1 for Yes)")
    parser.add_argument("-use_sex", type=int, default=0, choices=[0, 1], help="Whether to use Sex covariate (0 for No, 1 for Yes)")
    
    parser.add_argument("-age_col", type=str, default="Agexit", help="Column name for age in phenotype file")
    parser.add_argument("-sex_col", type=str, default="Sex", help="Column name for sex in phenotype file")

    # Training parameters
    parser.add_argument("-bs", type=int, default=2, help="Batch size for training")
    parser.add_argument("-dropout", type=float, default=0.5, help="Dropout rate for the model")
    parser.add_argument("-epochs", type=int, default=30, help="Number of epochs for training")
    parser.add_argument("-act", type=str, default="gelu", choices=["tanh","relu","gelu"], help="Activation function")
    parser.add_argument("-opt", type=str, default="adamw", choices=["adam", "adamw", "sgd"], help="Optimizer to use")
    parser.add_argument("-sch", type=str, default="explr", choices=["none","plateau", "cosine", "step","multistep","explr","warmup_exponential", "exponential_decay"], help="Learning rate scheduler")
    parser.add_argument("-wd", type=float, default=0.5, help="Weight decay for optimizer")
    parser.add_argument("-df", type=float, default=0.1, help="Decay factor for custom schedulers")
    parser.add_argument("-lr", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("-peak_lr", type=float, default=1e-2, help="Peak learning rate for WarmupExponential scheduler")
    parser.add_argument("-final_lr", type=float, default=1e-5, help="Final learning rate for custom schedulers")

    parser.add_argument("-kernel_sizes", type=parse_int_list, default=[127,31,7], help="Convolution Kernel Size")
    parser.add_argument("-stride", type=parse_int_list, default=[64,16,4], help="Convolution Stride")
    parser.add_argument("-conv_channels", type=parse_int_list, default=[2,4,8], help="Convolution channels")

    # Early stopping parameters
    parser.add_argument("-patience", type=int, default=30, help="Number of epochs to wait before early stopping")
    parser.add_argument("-min_delta", type=float, default=0.001, help="Minimum change in monitored quantity to qualify as an improvement")
    # Random seed for reproducibility
    parser.add_argument("-random_seed", type=int, default=42, help="Random seed for train-test split and model initialization")
    # Threshold adjustment
    parser.add_argument("-threshold", type=float, default=0.5, help="Classification threshold (lower values favor recall)")
    
    # Class weighting for imbalanced data
    parser.add_argument("-class_weight", type=int, default=1, choices=[0, 1], help="Whether to use class weighting (0: no, 1: yes)")
    parser.add_argument("-pos_weight_scale", type=float, default=1.2, help="Additional scaling factor for positive class weight")
    
    # Loss function choice
    parser.add_argument("-loss_fn", type=str, default="bce", choices=["bce", "focal"], 
                      help="Loss function to use (bce: Binary Cross Entropy, focal: Focal Loss)")
    parser.add_argument("-focal_alpha", type=float, default=0.25, help="Alpha parameter for Focal Loss")
    parser.add_argument("-focal_gamma", type=float, default=2.0, help="Gamma parameter for Focal Loss")
    
    # Checkpoint-related parameters
    parser.add_argument("-resume", type=int, default=1, choices=[0, 1], help="Whether to resume from checkpoint if available (0: no, start fresh; 1: yes, resume if available)")
    parser.add_argument("-keep_checkpoints", type=int, default=1, help="Number of recent checkpoints to keep")
    
    return parser.parse_args()

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
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience} (no improvement)')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            prev_best = self.best_loss
            self.best_loss = val_loss
            self.counter = 0
            
            if self.verbose:
                if global_best_loss is not None and val_loss > global_best_loss:
                    # Don't show "improved" message if it's not a global improvement
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


class GenotypeDataset(Dataset):
    def __init__(self, file_list, phenotype_data, label_col, use_pcs=0, use_age=0, use_sex=0, 
                 age_col="Agexit", sex_col="Sex"):
        self.file_list = file_list
        self.phenotype_data = phenotype_data
        self.label_col = label_col
        self.use_pcs = use_pcs
        self.use_age = use_age
        self.use_sex = use_sex
        self.age_col = age_col
        self.sex_col = sex_col
        
        print(f"GenotypeDataset initialized with {len(file_list)} files")
        print(f"Using covariates - PCs: {bool(use_pcs)}, Age: {bool(use_age)}, Sex: {bool(use_sex)}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        genotype_file = self.file_list[idx]
        sample_id_str = os.path.basename(genotype_file).replace("sample_", "").replace(".npy", "")
        sample_id = int(sample_id_str)
        
        label = self.phenotype_data.loc[self.phenotype_data['new_order'] == sample_id, self.label_col].values[0]
        
        # Initialize covariates list
        covariates_list = []
        
        # Add PCs if requested
        if self.use_pcs:
            pcs = self.phenotype_data.loc[self.phenotype_data['new_order'] == sample_id, 'PC1':'PC10'].values[0]
            covariates_list.extend(pcs)
        
        # Add Age if requested
        if self.use_age:
            age = self.phenotype_data.loc[self.phenotype_data['new_order'] == sample_id, self.age_col].values[0]
            covariates_list.append(age)
        
        # Add Sex if requested
        if self.use_sex:
            sex = self.phenotype_data.loc[self.phenotype_data['new_order'] == sample_id, self.sex_col].values[0]
            covariates_list.append(sex)
        
        # Convert to numpy array
        covariates = np.array(covariates_list)

        if '.npy' in genotype_file:
            genotype_data = np.load(genotype_file)  # Shape: (5M, 3)
            genotype_tensor = torch.from_numpy(genotype_data).float().T

        label_tensor = torch.tensor(label, dtype=torch.float32)
        covariates_tensor = torch.tensor(covariates, dtype=torch.float32)

        return genotype_tensor, covariates_tensor, label_tensor

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)  # Changed shape for batch_first
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]  # Adjusted indexing
        return self.dropout(x)

class TransformerPooling(nn.Module):
    def __init__(self, d_model, pooling_type='mean'):
        super().__init__()
        self.pooling_type = pooling_type
        if pooling_type == 'cls':
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls_token, std=0.02)
        elif pooling_type == 'weighted':
            self.attention = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.Tanh(),
                nn.Linear(d_model // 2, 1)
            )

    def forward(self, x, attention_mask=None):
        if self.pooling_type == 'cls':
            # Add CLS token
            if self.training:  # Only during training phase
                cls_token = self.cls_token.expand(x.shape[0], -1, -1)
                x = torch.cat((cls_token, x), dim=1)
            return x[:, 0]  # Return CLS token representation
            
        elif self.pooling_type == 'mean':
            x = x.mean(dim=1)
            return x
            # if attention_mask is not None:
            #     # Masked mean
            #     mask = attention_mask.unsqueeze(-1).float()
            #     x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            # else:
            #     # Simple mean
            #     x = x.mean(dim=1)
            # return x
            
        elif self.pooling_type == 'weighted':
            # Compute attention weights
            weights = self.attention(x)  # (B, L, 1)
            weights = F.softmax(weights, dim=1)
            # Apply weights
            x = (x * weights).sum(dim=1)
            return x

class GenotypeModelWithTransformer(nn.Module):
    def __init__(self, input_size, kernel_sizes, stride, conv_channels, act, dropout_rate, 
                 use_pcs=0, use_age=0, use_sex=0, num_transformer_layers=3, d_model=384, 
                 nhead=8, fc_layers=[128], pooling_type='mean', print_dimensions=False,
                 sequence_length=128):
        super().__init__()
        self.print_dimensions = print_dimensions
        self.has_printed_dimensions = False
        self.use_pcs = use_pcs
        self.use_age = use_age
        self.use_sex = use_sex
        self.pooling_type = pooling_type
        self.d_model = d_model
        self.sequence_length = sequence_length

        # Calculate number of covariates
        self.num_covariates = (10 if use_pcs else 0) + (1 if use_age else 0) + (1 if use_sex else 0)
        
        # Directly use 3 input channels
        self.input_channels = 3
        self.conv_layers = self._create_conv_layers(conv_channels, kernel_sizes, stride, dropout_rate, act)

        self.conv_output_size = self._get_conv_output_size(input_size)
        print(f"Convolutional output size: {self.conv_output_size}")

        # Add MaxPool1d to fix the SEQUENCE LENGTH
        self.max_pool = nn.AdaptiveMaxPool1d(sequence_length)

        # Add layer norm after maxpool (will be applied after permutation)
        #self.post_pool_norm = nn.LayerNorm(conv_channels[-1])

        # Check if projection is needed to adjust the FEATURE DIMENSION
        self.needs_projection = conv_channels[-1] != d_model
        if self.needs_projection:
            self.projection = nn.Linear(conv_channels[-1], d_model)
            #self.projection_norm = nn.LayerNorm(d_model)  # Add normalization after projection
            print(f"Added projection layer: {conv_channels[-1]} -> {d_model} features")
        else:
            print(f"No projection needed: CNN output dimension ({conv_channels[-1]}) matches transformer dimension ({d_model})")

        print(f"Sequence length after pooling: {sequence_length}")
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout_rate)

        # # Pre-transformer normalization
        # self.pre_transformer_norm = nn.LayerNorm(d_model)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout_rate, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # # Post-transformer normalization
        # self.post_transformer_norm = nn.LayerNorm(d_model)

        # Pooling layer
        self.pooling = TransformerPooling(d_model, pooling_type)

        # # Normalization after pooling
        # self.post_pooling_norm = nn.LayerNorm(d_model)
        
        # Fully connected layers
        fc_input_size = d_model + self.num_covariates

        # Add Layer Norm before classifier to stabilize inputs
        #self.pre_classifier_norm = nn.LayerNorm(fc_input_size) if self.num_covariates > 0 else nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            self._create_fc_layers(fc_input_size, fc_layers, dropout_rate, act),
            nn.Linear(fc_layers[-1], 1)
        )
       
    def _create_conv_layers(self, conv_channels, kernel_sizes, stride, dropout_rate, act):
        layers = []
        for i in range(len(conv_channels)):
            layers.append(nn.Conv1d(in_channels=self.input_channels if i == 0 else conv_channels[i-1],
                                    out_channels=conv_channels[i],  
                                    kernel_size=kernel_sizes[i],
                                    stride=stride[i]))
            layers.append(nn.BatchNorm1d(conv_channels[i]))
            layers.append(self.get_activation(act))
            # Add gradient clipping via a custom layer when using GELU
            # if act == 'gelu':
            #     layers.append(nn.Dropout(dropout_rate * 0.5))  # Reduced dropout for GELU
        return nn.Sequential(*layers)

    def _create_fc_layers(self, input_size, fc_sizes, dropout_rate, act):
        layers = []
        for i, fc_size in enumerate(fc_sizes):
            layers.append(nn.Linear(input_size if i == 0 else fc_sizes[i-1], fc_size))
            #layers.append(nn.BatchNorm1d(fc_size))
            #layers.append(nn.LayerNorm(fc_size))
            layers.append(self.get_activation(act))
            layers.append(nn.Dropout(dropout_rate))
        return nn.Sequential(*layers)

    def _get_conv_output_size(self, input_size):
        x = torch.randn(1, 3, input_size, dtype=torch.float32)
        x = self.conv_layers(x)
        return x.shape[2]  # Return the sequence length after convolutions

    def forward(self, x, covariates):
        if self.print_dimensions and not self.has_printed_dimensions:
            print(f"Input shape: {x.shape}")
        
        # Direct convolution on 3-channel input
        x = self.conv_layers(x)
        if self.print_dimensions and not self.has_printed_dimensions:
            print(f"After conv layers: {x.shape}")
        
        # Apply maxpool to fix the sequence length
        x = self.max_pool(x)
        if self.print_dimensions and not self.has_printed_dimensions:
            print(f"After maxpool: {x.shape}")
        
        # Convert (B, C, L) to (B, L, C)
        x = x.permute(0, 2, 1)
        if self.print_dimensions and not self.has_printed_dimensions:
            print(f"After permute: {x.shape}")
            
        # Apply LayerNorm after permute for stability
        #x = self.post_pool_norm(x)
        
        # Apply projection ONLY if FEATURE dimensions don't match with d_model
        if self.needs_projection:
            x = self.projection(x)
            #x = self.projection_norm(x)  # Apply normalization after projection
            if self.print_dimensions and not self.has_printed_dimensions:
                print(f"After projection: {x.shape}")  # [B, sequence_length, d_model]
        
        # Add positional encoding
        x = self.pos_encoder(x)

        # # Apply pre-transformer norm
        # x = self.pre_transformer_norm(x)
        
        # Apply transformer
        x = self.transformer(x)
        if self.print_dimensions and not self.has_printed_dimensions:
            print(f"After transformer layers: {x.shape}")
        
        # # Apply post-transformer norm
        # x = self.post_transformer_norm(x)
        
        # Apply pooling
        x = self.pooling(x)
        if self.print_dimensions and not self.has_printed_dimensions:
            print(f"After pooling: {x.shape}")
        
        # # Apply post-pooling norm
        # x = self.post_pooling_norm(x)
            
        # Concatenate with covariates if using them
        if self.num_covariates > 0:
            #print(f"Covariates shape: {covariates.shape}")
            x = torch.cat([x, covariates], dim=1)
            if self.print_dimensions and not self.has_printed_dimensions:
                print(f"After concatenating covariates: {x.shape}")
        
        # Apply pre-classifier norm
        #x = self.pre_classifier_norm(x)
        
        if self.print_dimensions and not self.has_printed_dimensions:
                print(f"Final output before classifier layer: {x.shape}")
        
        x = self.classifier(x)
        if self.print_dimensions and not self.has_printed_dimensions:
            print(f"Final output: {x.shape}")
            self.has_printed_dimensions = True
        
        return x.squeeze(1)
    
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
    """Save checkpoint to file"""
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
            os.remove(old_path)

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, device='cuda',
                early_stopping=None, classification_threshold=0.5, checkpoint_dir=None, 
                start_epoch=0, keep_last_n=2, history=None, initial_best_loss=float('inf')):
    print(f"Training on device: {device}")
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

            # Prefetch the first batch
            batch_iter = iter(dataloaders[phase])
            try:
                inputs, covariates, labels = next(batch_iter)
                inputs = inputs.to(device, non_blocking=True)
                covariates = covariates.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
            except StopIteration:
                print(f"Warning: {phase} dataloader is empty!")
                continue

            for i in range(batch_count):
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

                        # Apply sigmoid for predictions and metrics
                        probs = torch.sigmoid(outputs)       # Convert to probabilities
                        preds = (probs >= classification_threshold).float()       # Binary predictions

                        if phase == 'train':
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()

                            # Step the scheduler if it's a per-iteration scheduler
                            if isinstance(scheduler, (WarmupExponential, ExponentialDecay)):
                                scheduler.step()

                batch_size = labels.size(0)
                running_loss += loss.item() * batch_size
                running_corrects += torch.sum(preds == labels.data)
                total_samples += batch_size

                all_labels[phase].extend(labels.cpu().numpy())
                all_preds[phase].extend(probs.detach().cpu().numpy())

                # Prepare for the next iteration
                try:
                    inputs, covariates, labels = next_inputs, next_covariates, next_labels
                except:
                    break
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
            if phase == 'test':
                # Check if this is the best model so far before updating early stopping
                if epoch_loss < best_loss:
                    print(f"New best model! Validation loss improved from {best_loss:.6f} to {epoch_loss:.6f}")
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    new_best_model = True  # Flag that we have a new best model
                    
                    # If we have a checkpoint directory, save the best model immediately
                    if checkpoint_dir:
                        best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
                        
                        # Create a checkpoint with the best model weights
                        best_checkpoint = {
                            'epoch': epoch + 1,
                            'model_state_dict': best_model_wts,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_model_state_dict': best_model_wts,
                            'history': history,
                            'best_loss': best_loss,
                            'completed_epochs': epoch + 1
                        }
                        
                        # Add scheduler state if it exists
                        if hasattr(scheduler, 'state_dict'):
                            best_checkpoint['scheduler_state_dict'] = scheduler.state_dict()
                        elif isinstance(scheduler, (WarmupExponential, ExponentialDecay)):
                            best_checkpoint['scheduler_state_dict'] = scheduler.state_dict()
                            
                        # Add early stopping state if it exists
                        if early_stopping is not None:
                            best_checkpoint['early_stopping_state'] = early_stopping.state_dict()
                        
                        print(f"Saving best model to {best_model_path}")
                        torch.save(best_checkpoint, best_model_path)
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
                                'best_model_state_dict': best_model_wts,  # Include best weights separately
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
        completed_epochs = epoch + 1  # Update completed epochs counter
        
        # Save checkpoint at the end of each epoch
        if checkpoint_dir:
            checkpoint = {
                'epoch': completed_epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_model_state_dict': best_model_wts,  # Always include best weights
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
    #print(f"Steps per epoch: {steps_per_epoch}")
    total_steps = steps_per_epoch * args.epochs
    #print(f"Total Steps: {total_steps}")
    warmup_percentage = 0.1
    wsteps = int(total_steps * warmup_percentage)
    #print(f"Warmup Steps: {wsteps}")

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
    file_list = glob.glob(os.path.join(genotype_dir, "sample_*.npy"))
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
        print(f"Warning: Number of files ({len(filtered_file_list)}) does not match number of samples ({len(phenotype_samples)}) in phenotype data.")

    # Split data into train and test only
    train_files, test_files = train_test_split(
        filtered_file_list, test_size=0.2, random_state=args.random_seed
    )
    print(f"Data split: Train {len(train_files)}, Test {len(test_files)}")

    # Convert binary flags to boolean
    use_pcs = bool(args.use_pcs)
    use_age = bool(args.use_age)
    use_sex = bool(args.use_sex)
    
    # Create datasets
    train_dataset = GenotypeDataset(
        train_files, phenotype_data, args.label_col, 
        use_pcs=args.use_pcs, use_age=args.use_age, use_sex=args.use_sex,
        age_col=args.age_col, sex_col=args.sex_col
    )
    test_dataset = GenotypeDataset(
        test_files, phenotype_data, args.label_col, 
        use_pcs=args.use_pcs, use_age=args.use_age, use_sex=args.use_sex,
        age_col=args.age_col, sex_col=args.sex_col
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

    # Create dataloaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, prefetch_factor=2),
        'test': DataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, prefetch_factor=2)
    }

    print("DataLoaders created")

    # Calculate number of covariates
    num_covariates = (10 if use_pcs else 0) + (1 if use_age else 0) + (1 if use_sex else 0)
    print(f"Using {num_covariates} covariates")

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

    # Create model
    model = GenotypeModelWithTransformer(
        input_size=input_size,
        kernel_sizes=args.kernel_sizes,
        stride=args.stride,
        conv_channels=args.conv_channels,
        act=act,
        dropout_rate=dropout_rate,
        use_pcs=args.use_pcs,
        use_age=args.use_age,
        use_sex=args.use_sex,
        num_transformer_layers=args.num_transformer_layers,
        d_model=args.d_model,
        nhead=args.nhead,
        fc_layers=args.fc_layers,
        pooling_type=args.pooling, 
        print_dimensions=True,
        sequence_length=args.seq_len
    )
    model = model.to(device)

    with open(os.path.join(experiment_dir, 'model_architecture.txt'), 'w') as file:
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

    start_epoch = 0
    history = None
    best_loss = float('inf')
    best_model_wts = None

    # NEW: Check for existing checkpoints if -resume is enabled
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

    # NEW: Load scheduler state if it exists in checkpoint
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

     # Train the model with checkpointing
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

    # Save the trained model
    # torch.save(model.state_dict(), os.path.join(experiment_dir, 'final_model.pth'))
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
        'Start_LR': args.lr,
        'Peak_LR': args.peak_lr,
        'Final_LR': optimizer.param_groups[0]["lr"],
        'Dropout': dropout_rate,
        'Act': act,
        'Opt': opt,
        'Sch': sch,
        'WD': wd,
        'DF': args.df,
        'Kernel_sizes': args.kernel_sizes,
        'Stride': args.stride,
        'Conv_channels': args.conv_channels,
        'Seq_len' : args.seq_len,
        'Num_transformer_layers': args.num_transformer_layers,
        'nhead': args.nhead,
        'd_model': args.d_model,
        'FC_layers': args.fc_layers,
        'Trans_pool': args.pooling,
        'Label_col': args.label_col,
        'Use_PCs': args.use_pcs,
        'Use_Age': args.use_age,
        'Use_Sex': args.use_sex,
        'Age_col': args.age_col,
        'Sex_col': args.sex_col,
        'loss_fn': args.loss_fn,
        'class_weight': bool(args.class_weight),
        'pos_weight_scale': args.pos_weight_scale if bool(args.class_weight) else None,
        'threshold': args.threshold,
        'random_seed': args.random_seed
    }

    # Update hyperparameters dictionary with early stopping info
    hyperparameters.update({
        'ES_patience': args.patience,
        'ES_min_delta': args.min_delta,
        'Early_stopped_epoch': len(history['train_loss'])
    })

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

if __name__ == '__main__':
    start_time = time.time()
    
    main()
    
    end_time = time.time()
    total_runtime = end_time - start_time
    
    print(f"\nTotal script runtime: {total_runtime:.2f} seconds")
    hours, rem = divmod(total_runtime, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Total runtime: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")