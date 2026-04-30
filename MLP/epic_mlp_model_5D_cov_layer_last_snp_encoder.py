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
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score
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
    parser = argparse.ArgumentParser(description="Genotype Model Training")
    parser.add_argument("-ID", type=str, default="Exp_only_snps_no_covs_snp_encoder", help="ID of the experiment")
    parser.add_argument("-exp_dir", type=str, default='/vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/pros/pros_full/full_covariates/Last_layer/exp_no_snps/', help="Directory to save experiment results")
    parser.add_argument("-genotype_dir", type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/pros_can', help="Directory containing genotype files")
    #parser.add_argument("-genotype_dir", type=str, default='/vol/research/ucdatasets/gwas/gwas_mono_rm/stat_test_exp_split/pros/geno_ml_filtered_splits_random/split_80', help="Directory containing genotype files")
    parser.add_argument("-phenotype_file", type=str, default='/vol/vssp/SF_ucdatasets/gwas/data_files/disease_pheno/pros_can.xlsx', help="Path to phenotype file")

    parser.add_argument("-bs", type=int, default=32, help="Batch size for training")
    parser.add_argument("-dropout", type=float, default=0.5, help="Dropout rate for the model")

    parser.add_argument("-epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("-lr", type=float, default=0.0001, help="Learning rate for optimizer")
    parser.add_argument("-act", type=str, default="gelu", choices=["tanh","relu","gelu"], help="Dropout rate for the model")
    parser.add_argument("-opt", type=str, default="adamw", choices=["adam", "adamw", "sgd"], help="Optimizer to use")
    parser.add_argument("-sch", type=str, default="explr", choices=["none","plateau", "cosine", "step","multistep","explr","warmup_exponential", "exponential_decay"], help="Learning rate scheduler")
    parser.add_argument("-peak_lr", type=float, default=1e-3, help="Peak learning rate for WarmupExponential scheduler")
    parser.add_argument("-final_lr", type=float, default=1e-6, help="Final learning rate for custom schedulers")
    parser.add_argument("-wd", type=float, default=0.5, help="Weight decay for optimizer")
    parser.add_argument("-df", type=float, default=0.1, help="Decay factor for custom schedulers")

    parser.add_argument("-hidden_sizes", type=parse_int_list, default=[256,64,32], help="Hidden layer sizes for MLP")

    parser.add_argument("-cov", type=int, default=0, choices=[0, 1], help="Whether to include PC's in covariates in the model (0: no, 1: yes)")
    parser.add_argument("-use_age", type=int, default=0, choices=[0, 1], help="Whether to include age in covariates in the model (0: no, 1: yes)")
    parser.add_argument("-use_gender", type=int, default=0, choices=[0, 1], help="Whether to include gender in covariates in the model (0: no, 1: yes)")
    parser.add_argument("-label_col", type=str, default="pros01", help="Column name in phenotype file to use as label (e.g., 'pan01', etc.)")

    # Add early stopping parameters
    parser.add_argument("-patience", type=int, default=15, help="Patience for early stopping")
    parser.add_argument("-min_delta", type=float, default=1e-4, help="Minimum change for early stopping")

    # Add new normalization-related arguments
    parser.add_argument("-norm_age", type=str, default="none", choices=["none", "standard", "minmax", "robust", "quantile", "power"], help="Normalization method for age")
    parser.add_argument("-norm_pcs", type=str, default="none", choices=["none", "standard", "minmax", "robust", "quantile", "power"], help="Normalization method for PCs")
    parser.add_argument("-norm_gender", type=str, default="none", choices=["none", "minmax"], help="Normalization method for gender (usually keep as none)")

    # New argument for model type
    parser.add_argument("-model_type", type=str, default="full", choices=["full", "covariates_only"], help="Type of model to use (full: SNPs + covariates, covariates_only: only covariates)")
    
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
        """Fit the scaler to the data"""
        if self.method != "none" and data is not None and self.scaler is not None:
            # Ensure data is 2D for sklearn
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            self.scaler.fit(data)
    
    def transform(self, data):
        """Transform the data using the fitted scaler"""
        if self.method != "none" and data is not None and self.scaler is not None:
            # Ensure data is 2D for sklearn
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            return self.scaler.transform(data)
        return data

# Add a new model class for covariates-only prediction
class CovariatesOnlyModel(nn.Module):
    def __init__(self, hidden_sizes, dropout_rate, act, use_covariates=True, use_age=True, use_gender=True, num_covariates=10):
        super(CovariatesOnlyModel, self).__init__()
        self.use_covariates = use_covariates
        self.use_age = use_age
        self.use_gender = use_gender

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
        
        # Output layer
        layers.append(('output', nn.Linear(current_size, 1)))
        
        self.model = nn.Sequential(OrderedDict(layers))
        
        print(f"CovariatesOnlyModel initialized (using PCs: {use_covariates}, age: {use_age}, gender: {use_gender})")

    def forward(self, covariates):
        return self.model(covariates).squeeze(1)
    
    def get_activation(self, name):
        if name == 'tanh':
            return nn.Tanh()
        elif name == 'relu':
            return nn.ReLU()
        elif name == 'gelu':
            return nn.GELU()
        else:
            raise NotImplementedError("Activation function not implemented.")

class GenotypeDataset(Dataset):
    def __init__(self, file_list, phenotype_data, label_column, use_covariates=True, use_age=True, 
                 use_gender=True, norm_age="standard", norm_pcs="standard", norm_gender="none", 
                 fit_normalizers=True, normalizers=None, covariates_only=False):
        self.file_list = file_list
        self.phenotype_data = phenotype_data
        self.label_column = label_column
        self.use_covariates = use_covariates
        self.use_age = use_age
        self.use_gender = use_gender
        self.covariates_only = covariates_only

        # Verify that the label column exists in the phenotype data
        if self.label_column not in self.phenotype_data.columns:
            raise ValueError(f"Label column '{self.label_column}' not found in phenotype data. "
                           f"Available columns are: {', '.join(self.phenotype_data.columns)}")
            
        # Initialize or use provided normalizers
        if normalizers is None:
            # For training set: Create new normalizers
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
        print(f"- Fitting normalizers: {fit_normalizers}")

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
        
        # Use the configured label column
        label = self.phenotype_data.loc[self.phenotype_data['new_order'] == sample_id, self.label_column].values[0]
        
        # Process covariates
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
        
        if self.covariates_only:
            return covariates_tensor, torch.tensor(label, dtype=torch.float32)
        
        # Get genotype data only if not in covariates_only mode
        mmap_file = self.file_handles[genotype_file]
        mmap_file.seek(0)
        with gzip.GzipFile(fileobj=mmap_file) as f:
            data = pd.read_csv(f, sep=r'\s+', header=None)
        
        genotype_tensor = torch.tensor(data.values.T, dtype=torch.float32)
        return genotype_tensor, covariates_tensor, torch.tensor(label, dtype=torch.float32)

    def __del__(self):
        #Clean up file handles
        for handle in self.file_handles.values():
            handle.close()

class GenotypeModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rate, act, use_covariates=True, use_age=True, use_gender=True, num_covariates=10):
        super(GenotypeModel, self).__init__()
        self.use_covariates = use_covariates
        self.use_age = use_age
        self.use_gender = use_gender

        # Calculate total number of covariates
        self.total_covariates = 0
        if use_covariates:
            self.total_covariates += num_covariates  # PCs
        if use_age:
            self.total_covariates += 1  # Age
        if use_gender:
            self.total_covariates += 1  # Gender

        # SNP Encoder
        self.snp_encoder = nn.Sequential(
            # First layer: Process each SNP's probabilities
            nn.Conv1d(3, 16, kernel_size=1),  # Process each SNP independently
            nn.BatchNorm1d(16),
            self.get_activation(act),
            nn.Dropout(dropout_rate),
            
            # Second layer: Local patterns across nearby SNPs
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            self.get_activation(act),
            nn.Dropout(dropout_rate),
            
            # Third layer: Broader patterns
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            self.get_activation(act),
            nn.Dropout(dropout_rate),
            
            # Global pooling
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # Calculate size after SNP encoding
        snp_encoded_size = 64  # This matches the output channels of last conv layer
        
        # Create main layers
        layers = []
        current_size = snp_encoded_size
        
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
        
        # Output layer
        self.output_layer = nn.Linear(last_hidden_size, 1)
        
        print(f"GenotypeModel initialized with enhanced SNP encoder")
        print(f"Using covariates: {use_covariates}, age: {use_age}, gender: {use_gender}")

    def forward(self, x, covariates=None):
        # x shape: (batch_size, 3, num_snps)
        # Process SNP data through the encoder
        x = self.snp_encoder(x)
        
        # Main layers
        x = self.main_layers(x)
        
        # Concatenate with covariates
        if covariates is not None and self.total_covariates > 0:
            x = torch.cat([x, covariates], dim=1)
        
        # Process concatenated features
        x = self.final_processing(x)
        
        # Output layer
        return self.output_layer(x).squeeze(1)

    def forward_lrp(self, x, covariates=None):
        """Special forward pass for LRP analysis"""
        # Process SNP data through the encoder
        x = self.snp_encoder(x)
        
        # Main layers
        x = self.main_layers(x)
        
        # Concatenate with covariates
        if covariates is not None and self.total_covariates > 0:
            x = torch.cat([x, covariates], dim=1)
        
        # Process concatenated features
        x = self.final_processing(x)
        
        # Output layer
        return self.output_layer(x).squeeze(1)
    
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
 
def custom_lrp(model, data, covariates, target=1, epsilon=1e-9):
    # Perform custom Layer-wise Relevance Propagation analysis
    
    model.train() 
    
    # Enable gradients for inputs
    data = data.requires_grad_()
    if covariates is not None:
        covariates = covariates.requires_grad_()
    
    # Forward pass
    outputs = model(data, covariates)
    
    # Backward pass
    model.zero_grad()
    outputs.sum().backward()
    
    # Get gradients for the input
    input_grad = data.grad
    
    if input_grad is None:
        print("Warning: No gradients computed")
        return torch.zeros_like(data[0, 0]).cpu().numpy()
    
    # Calculate feature importance
    # Element-wise multiply input with its gradient and sum across batch and channel dimensions
    importance = (data * input_grad).sum(dim=(0, 1)).abs().detach().cpu().numpy()
    
    # Normalize importance scores
    importance = importance / (importance.sum() + epsilon)
    
    return importance

def get_top_snps(snp_importance, snp_ids, top_n=1000):
 
    # Sort SNPs by importance
    sorted_indices = np.argsort(snp_importance)[::-1]
    
    # Get top N SNPs
    #top_snps = [(snp_ids[i], snp_importance[i]) for i in sorted_indices[:top_n]]
    top_snps = [(snp_ids[i], snp_importance[i]) for i in sorted_indices]
    
    return top_snps

def save_snp_info_to_csv(top_snps, output_dir):
    """Save SNP importance information to CSV file"""
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.join(output_dir, "top_snps.csv")
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['SNP_ID', 'Importance'])
        for snp_id, importance in top_snps:
            writer.writerow([snp_id, importance])
    print(f"Saved top SNPs to {filename}")

def visualize_snp_importance(snp_importance, save_path):
    """Create visualization of SNP importance scores"""
    plt.figure(figsize=(15, 10))
    plt.plot(range(len(snp_importance)), sorted(snp_importance, reverse=True))
    plt.xlabel('SNP Index (sorted by importance)')
    plt.ylabel('Importance Score')
    plt.title('SNP Importance Distribution')
    plt.yscale('log')  # Use log scale for better visualization
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()



def print_lr(optimizer):
    for param_group in optimizer.param_groups:
        print(f"Current Learning Rate: {param_group['lr']}")

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, device='cuda', save_dir=None, early_stopping=None, covariates_only=False):
    # print(f"Criterion is: {criterion}\n")
    # print(f"Optimizer is: {optimizer}\n")
    # print(f"Scheduler is: {scheduler.__class__.__name__}\n")
    # print(f"num_epochs is: {num_epochs}\n")
    print(f"Training on device: {device}")
    print(f"Training in {'covariates-only' if covariates_only else 'full'} mode")
    
    scaler = GradScaler()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    completed_epochs = 0  

    history = {
        'train_loss': [], 'train_acc': [], 'train_auc': [],
        'test_loss': [], 'test_acc': [], 'test_auc': [],
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

            # Prefetch the first batch
            batch_iter = iter(dataloaders[phase])
            try:
                if covariates_only:
                    covariates, labels = next(batch_iter)
                    covariates = covariates.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                else:
                    inputs, covariates, labels = next(batch_iter)
                    inputs = inputs.to(device, non_blocking=True)
                    covariates = covariates.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
            except StopIteration:
                print(f"Warning: Empty dataloader for {phase} phase")
                continue

            for i in range(len(dataloaders[phase])):
                # Asynchronously prefetch the next batch
                if i + 1 < len(dataloaders[phase]):
                    try:
                        with torch.cuda.stream(stream):
                            if covariates_only:
                                next_covariates, next_labels = next(batch_iter)
                                next_covariates = next_covariates.to(device, non_blocking=True)
                                next_labels = next_labels.to(device, non_blocking=True)
                            else:
                                next_inputs, next_covariates, next_labels = next(batch_iter)
                                next_inputs = next_inputs.to(device, non_blocking=True)
                                next_covariates = next_covariates.to(device, non_blocking=True)
                                next_labels = next_labels.to(device, non_blocking=True)
                    except StopIteration:
                        break

                # Wait for the current batch to be ready
                torch.cuda.current_stream().wait_stream(stream)

                # Process the current batch
                optimizer.zero_grad()
                with autocast():
                    with torch.set_grad_enabled(phase == 'train'):
                        if covariates_only:
                            logits = model(covariates)
                        else:
                            logits = model(inputs, covariates)

                        loss = criterion(logits, labels)

                        # Convert logits to probabilities for metrics
                        probs = torch.sigmoid(logits)
                        preds = (probs >= 0.5).float()

                        #preds = torch.round(outputs)

                        if phase == 'train':
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()

                            # Step the scheduler if it's a per-iteration scheduler
                            if isinstance(scheduler, (WarmupExponential, ExponentialDecay)):
                                #old_lr = optimizer.param_groups[0]['lr']
                                scheduler.step()
                                #new_lr = optimizer.param_groups[0]['lr']
                                # if old_lr != new_lr:
                                #     print(f"Scheduler stepped. Old LR: {old_lr}, New LR: {new_lr}")
                                #history['learning_rates'].append(new_lr)

                running_loss += loss.item() * (covariates.size(0) if covariates_only else inputs.size(0))
                running_corrects += torch.sum(preds == labels.data)
                total_samples += (covariates.size(0) if covariates_only else inputs.size(0))

                all_labels[phase].extend(labels.cpu().numpy())
                all_preds[phase].extend(probs.detach().cpu().numpy())

                # Prepare for the next iteration
                if covariates_only:
                    covariates, labels = next_covariates, next_labels
                else:
                    inputs, covariates, labels = next_inputs, next_covariates, next_labels

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples
            epoch_auc = roc_auc_score(all_labels[phase], all_preds[phase])

            print(f'{phase} Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f} - AUC: {epoch_auc:.4f}')

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())
            history[f'{phase}_auc'].append(epoch_auc)

            # Early stopping check based on validation loss
            if phase == 'test' and early_stopping is not None:
                early_stopping(epoch_loss)
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    completed_epochs = epoch + 1  # Save the number of completed epochs
                    return model, history, compute_final_metrics(all_labels, all_preds), all_preds, all_labels, completed_epochs


        if scheduler is not None and not isinstance(scheduler, (WarmupExponential, ExponentialDecay)):
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(history['test_loss'][-1])
            else:
                scheduler.step()

        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        print_lr(optimizer)
        completed_epochs = epoch + 1  # Update completed epochs counter

    return model, history, compute_final_metrics(all_labels, all_preds), all_preds, all_labels, completed_epochs

def compute_final_metrics(all_labels, all_preds):
    final_metrics = {}
    for phase in ['train', 'test']:
        y_true = np.array(all_labels[phase])
        y_pred_proba = np.array(all_preds[phase])

        # Convert probabilities to binary predictions
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        #y_pred = np.array(all_preds[phase])

        #cm = confusion_matrix(y_true, y_pred.round())
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        final_metrics[phase] = {
            'confusion_matrix': cm,
            'sensitivity': f'{sensitivity:.5f}',
            'specificity': f'{specificity:.5f}',
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr
        }
    return final_metrics

def plot_all_metrics(history, final_metrics, save_dir):
    phases = ['train', 'test']
    metrics = ['loss', 'acc', 'auc']
    
    fig, axs = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('Model Performance Metrics', fontsize=16)
    
    for i, metric in enumerate(metrics):
        for phase in phases:
            axs[i//2, i%2].plot(history[f'{phase}_{metric}'], label=f'{phase}')
        axs[i//2, i%2].set_title(f'{metric.capitalize()}')
        axs[i//2, i%2].set_xlabel('Epoch')
        axs[i//2, i%2].set_ylabel(metric.capitalize())
        axs[i//2, i%2].legend()
    
    # Add learning rate subplot
    axs[1, 1].plot(history['learning_rates'])
    axs[1, 1].set_title('Learning Rate')
    axs[1, 1].set_xlabel('Step')
    axs[1, 1].set_ylabel('Learning Rate')
    axs[1, 1].set_yscale('log')  # Use log scale for better visualization
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_plot.png'))
    plt.close()

def perform_lrp_analysis(model, dataloader, device, args):
    print("Performing LRP analysis...")
    
    batch = next(iter(dataloader['test']))
    genotype_data, covariates, _ = batch
    
    genotype_data = genotype_data.to(device)
    covariates = covariates.to(device)
    
    snp_importance = custom_lrp(model, genotype_data, covariates)
    
    snp_ids = [i for i in range(len(snp_importance))]
    top_snps = get_top_snps(snp_importance, snp_ids, top_n=1000)
    
    output_dir = os.path.join(args.exp_dir, args.ID, 'lrp_analysis')
    save_snp_info_to_csv(top_snps, output_dir)
    
    vis_path = os.path.join(output_dir, 'snp_importance_distribution.png')
    visualize_snp_importance(snp_importance, vis_path)
    
    print("\nTop 10 Most Important SNPs:")
    for snp_id, importance in top_snps[:10]:
        print(f"SNP ID: {snp_id}, Importance: {importance:.4f}")
    
    return snp_importance, top_snps

def write_results(model, hyperparameters, results, save_dir):
    with open(os.path.join(save_dir, 'experiment_results.txt'), 'w') as f:
        f.write("Hyperparameters:\n")
        f.write("----------------\n")
        for key, value in hyperparameters.items():
            f.write(f"{key}: {value}\n")
            #f.write("{}: {}\n".format(key,value))
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

    id = str(args.ID)
    print(f"Experiment ID is:{id}\n")
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

    # Get list of genotype files and sort numerically by sample number
    file_list = glob.glob(os.path.join(args.genotype_dir, "sample_*.gen.gz"))
    file_list.sort(key=lambda x: int(x.split('sample_')[1].split('.gen.gz')[0]))
    print(f"Number of genotype files found: {len(file_list)}")

    num_samples = len(phenotype_data['new_order'].unique())
    print(f"Number of unique samples in phenotype data: {num_samples}")

    # Get input size dynamically
    first_genotype_file = file_list[0]
    print(f"First genotype file of the directory is: {first_genotype_file}")
    input_size = get_input_size(first_genotype_file)
    print(f"Dynamically determined input size: {input_size}")

    if len(file_list) != num_samples:
        raise ValueError(f"Number of files ({len(file_list)}) in {genotype_dir} does not match number of samples ({num_samples}) in phenotype data.")


    # Select training and testing files based on specified indices
    #train_files = file_list[:610]
    #test_files = file_list[-163:]

    train_files, test_files = train_test_split(file_list, test_size=0.2, random_state=42)
    print(f"Data split: Train {len(train_files)}, Test {len(test_files)}")

    # Create datasets with appropriate mode
    train_dataset = GenotypeDataset(
        train_files, 
        phenotype_data,
        label_column=args.label_col,
        use_covariates=bool(args.cov),
        use_age=bool(args.use_age),
        use_gender=bool(args.use_gender),
        norm_age=args.norm_age,
        norm_pcs=args.norm_pcs,
        norm_gender=args.norm_gender,
        fit_normalizers=True,
        normalizers=None,
        covariates_only=(args.model_type == "covariates_only")
    )
    
    # Get the fitted normalizers from training dataset
    fitted_normalizers = train_dataset.get_normalizers()

    # Create test dataset with the fitted normalizers
    test_dataset = GenotypeDataset(
        test_files, 
        phenotype_data,
        label_column=args.label_col,
        use_covariates=bool(args.cov),
        use_age=bool(args.use_age),
        use_gender=bool(args.use_gender),
        norm_age=args.norm_age,
        norm_pcs=args.norm_pcs,
        norm_gender=args.norm_gender,
        fit_normalizers=False,  # Don't fit new normalizers
        normalizers=fitted_normalizers,  # Use normalizers from training set
        covariates_only=(args.model_type == "covariates_only")
    )

    #print information for a few items from each dataset
    print("Sampling a few items from each dataset:")
    if args.model_type == "covariates_only":
        for dataset_name, dataset in [("Train", train_dataset), ("Test", test_dataset)]:
            print(f"\n{dataset_name} dataset sample:")
            for i in range(3):  # Print info for 3 items from each dataset
                covariates, label = dataset[i]
                print(f"Item {i}: Covariates shape: {covariates.shape}, Label: {label}")
    else:
        for dataset_name, dataset in [("Train", train_dataset), ("Test", test_dataset)]:
            print(f"\n{dataset_name} dataset sample:")
            for i in range(3):  # Print info for 3 items from each dataset
                genotype, covariates, label = dataset[i]
                print(f"Item {i}: Genotype shape: {genotype.shape}, Covariates shape: {covariates.shape}, Label: {label}")

    dataloaders = {
    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2),
    'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2)
}
    print("DataLoaders created")
    
    # Create the appropriate model based on model_type
    if args.model_type == "covariates_only":
        model = CovariatesOnlyModel(
            hidden_sizes=args.hidden_sizes,
            dropout_rate=args.dropout,
            act=args.act,
            use_covariates=bool(args.cov),
            use_age=bool(args.use_age),
            use_gender=bool(args.use_gender),
            num_covariates=10
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
            num_covariates=10
        )

    model = model.to(device)
    with open(experiment_dir + '/model_architecture.txt', 'w') as file:
        file.write(str(model))
        print(model)

    print("Model created and moved to device")

    # Set up loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = {
        "adadelta": optim.Adadelta(model.parameters(), lr=args.lr),
        "adagrad": optim.Adagrad(model.parameters(), lr=args.lr),
        "adamw": optim.AdamW(model.parameters(), lr=args.lr, weight_decay=wd),
        "rmsprop": optim.RMSprop(model.parameters(), lr=args.lr),
        "sgd": optim.SGD(model.parameters(), lr=args.lr),
        "adam": optim.Adam(model.parameters(), lr=args.lr, weight_decay=wd)
    }.get(opt, None)

    if optimizer is None:
        raise NotImplementedError("Optimizer not implemented.")

    scheduler = get_scheduler(sch, optimizer, args, train_files)

    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta,
        verbose=True
    )

    #pass scheduler to the train_model function
    model, history, final_metrics, all_preds, all_labels, completed_epochs = train_model(
        model, dataloaders, criterion, optimizer, scheduler, num_epochs, 
        device=device, save_dir=experiment_dir, early_stopping=early_stopping,
        covariates_only=(args.model_type == "covariates_only") 
    )

    #Plot metrics
    plot_all_metrics(history, final_metrics, experiment_dir)

    # Save the trained model
    #torch.save(model.state_dict(), os.path.join(experiment_dir, 'trained_genotype_model.pth'))
    print("Model training completed and saved")

    # Update results dictionary
    results = {
        #'Train_loss': history['train_loss'][-1],
        'train_acc': round(history['train_acc'][-1],4),
        'train_auc': round(history['train_auc'][-1],4),
        #'Test_loss': history['test_loss'][-1],
        'test_acc': round(history['test_acc'][-1],4),
        'test_auc': round(history['test_auc'][-1],4)
    }

    # Add final metrics for each phase
    for phase in ['train', 'test']:
        results.update({
            f'{phase}_sens': final_metrics[phase]['sensitivity'],
            f'{phase}_spec': final_metrics[phase]['specificity'],
            f'{phase}_CM': final_metrics[phase]['confusion_matrix'],
            #f'{phase}_roc_auc': final_metrics[phase]['roc_auc']
        })

    # Update hyperparameters dictionary
    hyperparameters = {
        'Exp_ID': id,
        'BS': batch_size,
        'Epochs': completed_epochs,
        'Start_LR': learning_rate,
        #'Peak_LR': args.peak_lr,
        'Final_LR':optimizer.param_groups[0]["lr"],
        'Dropout': dropout_rate,
        'Act': act,
        'Opt': opt,
        'Sch': sch,
        'WD': wd,
        'DF': args.df,
        'Label_Column': args.label_col,
        'Use_PCs': bool(args.cov),
        'norm_PCs': args.norm_pcs,
        'Use_Age': bool(args.use_age),
        'norm_Age': args.norm_age,
        'Use_Gender': bool(args.use_gender),
        'norm_Gender': args.norm_gender,
        'hidden_sizes':args.hidden_sizes 
    }

    # Perform LRP analysis
    print("\nStarting LRP analysis...")
    #snp_importance, top_snps = perform_lrp_analysis(model, dataloaders, device, args)
    
    # # Add LRP results to the results dictionary
    # results.update({
    #     'lrp_analysis_completed': True,
    #     'top_snp_importance': top_snps[0][1] if top_snps else None  
    # })

    # Write results
    write_results(model, hyperparameters, results, experiment_dir)
    append_metrics_to_csv(experiment_dir, results)


if __name__ == '__main__':
    start_time = time.time()
    
    main()
    
    end_time = time.time()
    total_runtime = end_time - start_time
    
    print(f"\nTotal script runtime: {total_runtime:.2f} seconds")
    hours, rem = divmod(total_runtime, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Total runtime: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")
