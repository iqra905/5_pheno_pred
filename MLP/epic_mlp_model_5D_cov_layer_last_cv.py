import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import gzip
import mmap
import copy
import glob
import math
import time
import csv
from collections import OrderedDict
import argparse
from datetime import datetime

def parse_int_list(s):
    """Parse comma-separated integers into a list"""
    return [int(x) for x in s.split(',')]

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Genotype Model Training with Cross-Validation")
    parser.add_argument("-ID", type=str, default="Exp_01", help="ID of the experiment")
    parser.add_argument("-exp_dir", type=str, default='/vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/pros/pros_full/full_covariates/Last_layer_cv/', help="Directory to save experiment results")
    parser.add_argument("-genotype_dir", type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/pros_can', help="Directory containing genotype files")
    parser.add_argument("-phenotype_file", type=str, default='/vol/vssp/SF_ucdatasets/gwas/data_files/disease_pheno/pros_can.xlsx', help="Path to phenotype file")
    
    parser.add_argument("-bs", type=int, default=32, help="Batch size for training")
    parser.add_argument("-dropout", type=float, default=0.5, help="Dropout rate for the model")

    parser.add_argument("-epochs", type=int, default=30, help="Number of epochs for training")
    parser.add_argument("-lr", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("-act", type=str, default="gelu", choices=["tanh","relu","gelu"], help="Dropout rate for the model")
    parser.add_argument("-opt", type=str, default="adamw", choices=["adam", "adamw", "sgd"], help="Optimizer to use")
    parser.add_argument("-sch", type=str, default="warmup_exponential", choices=["none","plateau", "cosine", "step","multistep","explr","warmup_exponential", "exponential_decay"], help="Learning rate scheduler")
    parser.add_argument("-peak_lr", type=float, default=1e-2, help="Peak learning rate for WarmupExponential scheduler")
    parser.add_argument("-final_lr", type=float, default=1e-5, help="Final learning rate for custom schedulers")
    parser.add_argument("-wd", type=float, default=0.01, help="Weight decay for optimizer")
    parser.add_argument("-df", type=float, default=0.1, help="Decay factor for custom schedulers")
    
    parser.add_argument("-hidden_sizes", type=parse_int_list, default=[128,128,128], help="Hidden layer sizes for MLP")
        
    parser.add_argument("-cov", type=int, default=1, choices=[0, 1], help="Whether to include covariates in the model (0: no, 1: yes)")
    parser.add_argument("-use_age", type=int, default=1, choices=[0, 1], help="Whether to include age in covariates in the model (0: no, 1: yes)")
    parser.add_argument("-use_gender", type=int, default=1, choices=[0, 1], help="Whether to include gender in covariates in the model (0: no, 1: yes)")
    parser.add_argument("-label_col", type=str, default="pros01", help="Column name in phenotype file to use as label (e.g., 'pan01', etc.)")

    # Add early stopping parameters
    parser.add_argument("-patience", type=int, default=15, help="Patience for early stopping")
    parser.add_argument("-min_delta", type=float, default=1e-4, help="Minimum change for early stopping")
    
    parser.add_argument("-cv_folds", type=int, default=5, help="Number of cross-validation folds")
    return parser.parse_args()

def get_input_size(genotype_file):
    """Get the input size from the first genotype file"""
    with gzip.open(genotype_file, 'rt') as f:
        return sum(1 for line in f)

class GenotypeModel(nn.Module):
    """Neural network model for genotype data"""
    
    def __init__(self, input_size, hidden_sizes, dropout_rate, act, 
                 use_covariates=True, use_age=True, use_gender=True, num_covariates=10):
        super(GenotypeModel, self).__init__()
        
        self.use_covariates = use_covariates
        self.use_age = use_age
        self.use_gender = use_gender
        
        # Calculate total covariates
        self.total_covariates = 0
        if use_covariates:
            self.total_covariates += num_covariates  # PCs
        if use_age:
            self.total_covariates += 1  # Age
        if use_gender:
            self.total_covariates += 1  # Gender
        
        # Pointwise convolution layer
        self.pointwise_conv = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1)
        
        # Create main layers
        layers = []
        current_size = input_size
        
        # Hidden layers
        for i, hidden_size in enumerate(hidden_sizes):
            layers.extend([
                (f'linear_{i}', nn.Linear(current_size, hidden_size)),
                (f'batchnorm_{i}', nn.BatchNorm1d(hidden_size)),
                (f'activation_{i}', self.get_activation(act)),
                (f'dropout_{i}', nn.Dropout(dropout_rate))
            ])
            current_size = hidden_size
        
        self.main_layers = nn.Sequential(OrderedDict(layers))
        
        # Final processing layers
        last_hidden_size = hidden_sizes[-1]
        concat_size = last_hidden_size + self.total_covariates
        
        self.final_processing = nn.Sequential(OrderedDict([
            ('final_linear', nn.Linear(concat_size, last_hidden_size)),
            ('final_bn', nn.BatchNorm1d(last_hidden_size)),
            ('final_activation', self.get_activation(act)),
            ('final_dropout', nn.Dropout(dropout_rate))
        ]))
        
        # Output layer
        self.output_layer = nn.Linear(last_hidden_size, 1)
        self.output_activation = nn.Sigmoid()

    def forward(self, x, covariates=None):
        # Apply pointwise convolution
        x = self.pointwise_conv(x).squeeze(1)
        
        # Main layers
        x = self.main_layers(x)
        
        # Concatenate with covariates if available
        if covariates is not None and self.total_covariates > 0:
            x = torch.cat([x, covariates], dim=1)
        
        # Final processing
        x = self.final_processing(x)
        
        # Output
        x = self.output_layer(x)
        return self.output_activation(x).squeeze(1)

    def get_activation(self, name):
        """Get activation function by name"""
        activations = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'gelu': nn.GELU()
        }
        return activations.get(name, nn.GELU())

class EarlyStopping:
    """Early stopping handler"""
    
    def __init__(self, patience=7, min_delta=0, mode='max', verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.min_delta = min_delta
        self.mode = mode
        self.delta = -min_delta if mode == 'min' else min_delta

    def __call__(self, score):
        if self.mode == 'min':
            score = -score
            
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

class WarmupExponentialScheduler:
    """Custom learning rate scheduler with warmup and exponential decay"""
    
    def __init__(self, optimizer, start_lr, peak_lr, final_lr, warmup_steps, 
                 total_steps, decay_factor):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.peak_lr = peak_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.decay_factor = decay_factor
        self.current_step = 0

    def step(self):
        if self.current_step < self.total_steps:
            self.current_step += 1
            lr = self.get_lr()
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def get_lr(self):
        if self.current_step >= self.total_steps:
            return self.final_lr
        elif self.current_step < self.warmup_steps:
            # Linear warmup
            warmup_progress = self.current_step / self.warmup_steps
            return self.start_lr + (self.peak_lr - self.start_lr) * warmup_progress
        else:
            # Exponential decay
            decay_progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            decay = decay_progress ** self.decay_factor
            return self.peak_lr * (self.final_lr / self.peak_lr) ** decay

def get_optimizer(args, model):
    """Create optimizer based on arguments"""
    optimizers = {
        "adamw": optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd),
        "adam": optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd),
        "sgd": optim.SGD(model.parameters(), lr=args.lr)
    }
    return optimizers.get(args.opt)

def get_scheduler(args, optimizer, train_size):
    """Create learning rate scheduler based on arguments"""
    steps_per_epoch = train_size // args.bs
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * 0.1)  # 10% warmup

    if args.sch == "warmup_exponential":
        return WarmupExponentialScheduler(
            optimizer=optimizer,
            start_lr=args.lr,
            peak_lr=args.peak_lr,
            final_lr=args.final_lr,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            decay_factor=args.df
        )
    elif args.sch == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.sch == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    elif args.sch == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    return None

class GenotypeDataset(Dataset):
    """Dataset class for handling genotype data"""
    
    def __init__(self, file_list, phenotype_data, label_column, 
                 use_covariates=True, use_age=True, use_gender=True):
        self.file_list = file_list
        self.phenotype_data = phenotype_data
        self.label_column = label_column
        self.use_covariates = use_covariates
        self.use_age = use_age
        self.use_gender = use_gender
        
        # Verify label column exists
        if self.label_column not in self.phenotype_data.columns:
            raise ValueError(f"Label column '{self.label_column}' not found in phenotype data. "
                           f"Available columns: {', '.join(self.phenotype_data.columns)}")
        
        # Initialize file handles for memory mapping
        self.file_handles = {}
        for file in file_list:
            f = open(file, 'rb')
            self.file_handles[file] = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        
        print(f"GenotypeDataset initialized with {len(file_list)} files")
        print(f"Using covariates: {use_covariates}, age: {use_age}, gender: {use_gender}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Get file and extract sample ID
        genotype_file = self.file_list[idx]
        sample_id = int(os.path.basename(genotype_file).split('_')[1].split('.')[0])
        
        # Get label
        label = self.phenotype_data.loc[
            self.phenotype_data['new_order'] == sample_id, 
            self.label_column
        ].values[0]
        
        # Prepare covariates
        covariates_list = []
        if self.use_covariates:
            pcs = self.phenotype_data.loc[
                self.phenotype_data['new_order'] == sample_id, 
                'PC1':'PC10'
            ].values[0]
            covariates_list.append(pcs)
            
        if self.use_age:
            age = self.phenotype_data.loc[
                self.phenotype_data['new_order'] == sample_id, 
                'Agexit'
            ].values[0]
            covariates_list.append([age])
            
        if self.use_gender:
            gender = self.phenotype_data.loc[
                self.phenotype_data['new_order'] == sample_id, 
                'Sex'
            ].values[0]
            covariates_list.append([gender])

        # Combine all covariates
        if covariates_list:
            covariates = np.concatenate(covariates_list)
            covariates_tensor = torch.tensor(covariates, dtype=torch.float32)
        else:
            covariates_tensor = torch.tensor([], dtype=torch.float32)

        # Read genotype data
        mmap_file = self.file_handles[genotype_file]
        mmap_file.seek(0)
        with gzip.GzipFile(fileobj=mmap_file) as f:
            data = pd.read_csv(f, sep=r'\s+', header=None)

        # Convert to tensors
        genotype_tensor = torch.tensor(data.values.T, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return genotype_tensor, covariates_tensor, label_tensor

    def __del__(self):
        # Clean up file handles
        for handle in self.file_handles.values():
            handle.close()

def prepare_cross_validation(file_list, phenotype_data, label_column, n_splits=5, random_state=42):
    """Prepare stratified k-fold cross-validation splits"""
    
    # Extract sample IDs and labels
    sample_ids = []
    labels = []
    
    for file in file_list:
        sample_id = int(os.path.basename(file).split('_')[1].split('.')[0])
        label = phenotype_data.loc[phenotype_data['new_order'] == sample_id,label_column].values[0]
        
        sample_ids.append(sample_id)
        labels.append(label)
    
    # Convert to numpy arrays
    X = np.array(sample_ids).reshape(-1, 1)
    y = np.array(labels)
    
    # Create stratified k-fold splitter
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Prepare splits
    cv_splits = []
    for train_idx, val_idx in skf.split(X, y):
        train_sample_ids = X[train_idx].flatten()
        val_sample_ids = X[val_idx].flatten()
        
        # Get corresponding files for each split
        train_files = [f for f in file_list if int(os.path.basename(f).split('_')[1].split('.')[0]) in train_sample_ids]
        val_files = [f for f in file_list if int(os.path.basename(f).split('_')[1].split('.')[0]) in val_sample_ids]
        
        cv_splits.append((train_files, val_files))
    
    return cv_splits

def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, scaler):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []
    
    for inputs, covariates, labels in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        covariates = covariates.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Use automatic mixed precision
        with autocast():
            outputs = model(inputs, covariates)
            loss = criterion(outputs, labels)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if scheduler is not None:
            scheduler.step()
        
        # Compute metrics
        preds = torch.round(outputs)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Calculate epoch metrics
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    epoch_auc = roc_auc_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc.item(), epoch_auc, all_preds, all_labels

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, covariates, labels in dataloader:
            inputs = inputs.to(device)
            covariates = covariates.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs, covariates)
            loss = criterion(outputs, labels)
            
            preds = torch.round(outputs)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate epoch metrics
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    epoch_auc = roc_auc_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc.item(), epoch_auc, all_preds, all_labels

def train_fold(fold_idx, train_files, val_files, args, phenotype_data, device):
    """Train model on one fold"""
    print(f"\nTraining Fold {fold_idx + 1}")
    print("-" * 50)
    
    # Create datasets
    train_dataset = GenotypeDataset(
        train_files, phenotype_data,
        label_column=args.label_col,
        use_covariates=bool(args.cov),
        use_age=bool(args.use_age),
        use_gender=bool(args.use_gender)
    )
    
    val_dataset = GenotypeDataset(
        val_files, phenotype_data,
        label_column=args.label_col,
        use_covariates=bool(args.cov),
        use_age=bool(args.use_age),
        use_gender=bool(args.use_gender)
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, 
                            num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, 
                          num_workers=4, pin_memory=True)
    
    # Initialize model
    input_size = get_input_size(train_files[0])
    model = GenotypeModel(
        input_size=input_size,
        hidden_sizes=args.hidden_sizes,
        dropout_rate=args.dropout,
        act=args.act,
        use_covariates=bool(args.cov),
        use_age=bool(args.use_age),
        use_gender=bool(args.use_gender),
        num_covariates=10
    ).to(device)
    
    # Setup training components
    criterion = nn.BCEWithLogitsLoss()
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer, len(train_dataset))
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta, 
                                 mode='max', verbose=True)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 'train_auc': [],
        'val_loss': [], 'val_acc': [], 'val_auc': []
    }
    
    # Training loop
    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1}/{args.epochs}')
        
        # Train
        train_loss, train_acc, train_auc, train_preds, train_labels = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, scaler
        )
        
        # Validate
        val_loss, val_acc, val_auc, val_preds, val_labels = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_auc'].append(train_auc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} AUC: {train_auc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f}')
        
        # Early stopping
        early_stopping(val_auc)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    return model, history

def train_cross_validation(args, file_list, phenotype_data, device):
    """Perform k-fold cross-validation"""
    cv_splits = prepare_cross_validation(
        file_list, phenotype_data, args.label_col, args.cv_folds
    )
    
    # Store results for each fold
    cv_results = {
        'train_loss': [], 'train_acc': [], 'train_auc': [],
        'val_loss': [], 'val_acc': [], 'val_auc': [],
        'models': [], 'histories': []
    }
    
    # Train each fold
    for fold_idx, (train_files, val_files) in enumerate(cv_splits):
        model, history = train_fold(fold_idx, train_files, val_files, 
                                  args, phenotype_data, device)
        
        # Store results
        cv_results['models'].append(model)
        cv_results['histories'].append(history)
        
        # Store final metrics
        cv_results['train_loss'].append(history['train_loss'][-1])
        cv_results['train_acc'].append(history['train_acc'][-1])
        cv_results['train_auc'].append(history['train_auc'][-1])
        cv_results['val_loss'].append(history['val_loss'][-1])
        cv_results['val_acc'].append(history['val_acc'][-1])
        cv_results['val_auc'].append(history['val_auc'][-1])
    
    return cv_results

def compute_cv_metrics(cv_results):
    """Compute aggregate metrics across all folds"""
    metrics = {}
    
    # Calculate mean and std for each metric
    for metric in ['train_acc', 'train_auc', 'val_acc', 'val_auc']:
        values = cv_results[metric]
        metrics[f'mean_{metric}'] = np.mean(values)
        metrics[f'std_{metric}'] = np.std(values)
    
    return metrics

# Part 5: Utilities and Visualization Functions

def plot_cv_learning_curves(cv_results, save_dir):
    """Plot learning curves for all folds"""
    metrics = ['loss', 'acc', 'auc']
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Find the maximum length among all histories
        max_epochs = max(len(history[f'train_{metric}']) for history in cv_results['histories'])
        
        # Initialize arrays for mean calculation
        train_values = np.zeros((len(cv_results['histories']), max_epochs)) * np.nan
        val_values = np.zeros((len(cv_results['histories']), max_epochs)) * np.nan
        
        # Plot each fold and fill arrays
        for fold_idx, history in enumerate(cv_results['histories']):
            train_data = history[f'train_{metric}']
            val_data = history[f'val_{metric}']
            epochs = len(train_data)
            
            # Plot individual fold data
            ax.plot(range(epochs), train_data, 
                   alpha=0.3, color='blue', 
                   label=f'Train Fold {fold_idx+1}' if fold_idx == 0 else None)
            ax.plot(range(epochs), val_data, 
                   alpha=0.3, color='red', 
                   label=f'Val Fold {fold_idx+1}' if fold_idx == 0 else None)
            
            # Fill arrays for mean calculation
            train_values[fold_idx, :epochs] = train_data
            val_values[fold_idx, :epochs] = val_data
        
        # Calculate means ignoring NaN values
        train_mean = np.nanmean(train_values, axis=0)
        val_mean = np.nanmean(val_values, axis=0)
        
        # Plot mean curves
        epochs_range = range(max_epochs)
        ax.plot(epochs_range, train_mean, color='blue', linewidth=2, label='Train (mean)')
        ax.plot(epochs_range, val_mean, color='red', linewidth=2, label='Validation (mean)')
        
        ax.set_title(f'{metric.upper()} vs. Epoch')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.upper())
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cv_learning_curves.png'))
    plt.close()

def plot_cv_boxplots(cv_results, save_dir):
    """Create boxplots of final metrics across folds"""
    metrics = ['acc', 'auc']
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        data = [
            cv_results[f'train_{metric}'],
            cv_results[f'val_{metric}']
        ]
        
        ax.boxplot(data, labels=['Train', 'Validation'])
        ax.set_title(f'{metric.upper()} Distribution Across Folds')
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cv_boxplots.png'))
    plt.close()

def save_cv_results(cv_results, metrics, args, save_dir):
    """Save cross-validation results and configuration"""
    results_file = os.path.join(save_dir, 'cv_results.txt')
    
    with open(results_file, 'w') as f:
        # Write configuration
        f.write("=== Configuration ===\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
        
        f.write("\n=== Cross-Validation Results ===\n")
        
        # Write fold-wise results
        f.write("\nPer-fold final metrics:\n")
        for fold_idx in range(len(cv_results['models'])):
            f.write(f"\nFold {fold_idx + 1}:\n")
            f.write(f"Train Acc: {cv_results['train_acc'][fold_idx]:.4f}\n")
            f.write(f"Train AUC: {cv_results['train_auc'][fold_idx]:.4f}\n")
            f.write(f"Val Acc: {cv_results['val_acc'][fold_idx]:.4f}\n")
            f.write(f"Val AUC: {cv_results['val_auc'][fold_idx]:.4f}\n")
        
        # Write aggregate metrics
        f.write("\nAggregate metrics:\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")

def setup_experiment_dir(args):
    """Create experiment directory and save configuration"""
    # Create main experiment directory
    exp_dir = os.path.join(args.exp_dir, args.ID)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(os.path.join(exp_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'models'), exist_ok=True)
    
    # Save configuration
    config_file = os.path.join(exp_dir, 'config.txt')
    with open(config_file, 'w') as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
    
    return exp_dir

def main():
    # Parse arguments
    args = parse_args()
    
    # Setup experiment directory
    exp_dir = setup_experiment_dir(args)
    print(f"Experiment directory: {exp_dir}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load phenotype data
    phenotype_data = pd.read_excel(args.phenotype_file)
    print(f"Phenotype data loaded, shape: {phenotype_data.shape}")
    
    # Get list of genotype files
    file_list = glob.glob(os.path.join(args.genotype_dir, "sample_*.gen.gz"))
    file_list.sort(key=lambda x: int(x.split('sample_')[1].split('.gen.gz')[0]))
    print(f"Number of genotype files found: {len(file_list)}")
    
    # Verify data
    if len(file_list) != len(phenotype_data['new_order'].unique()):
        raise ValueError("Number of files does not match number of samples in phenotype data")
    
    # Perform cross-validation
    print("\nStarting cross-validation...")
    cv_results = train_cross_validation(args, file_list, phenotype_data, device)
    
    # Compute aggregate metrics
    print("\nComputing aggregate metrics...")
    metrics = compute_cv_metrics(cv_results)
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_cv_learning_curves(cv_results, os.path.join(exp_dir, 'plots'))
    plot_cv_boxplots(cv_results, os.path.join(exp_dir, 'plots'))
    
    # Save results
    print("\nSaving results...")
    save_cv_results(cv_results, metrics, args, exp_dir)
    
    # Print final results
    print("\nCross-validation completed!")
    print("\nFinal Results:")
    print("-" * 50)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    start_time = time.time()
    
    try:
        main()
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise
    finally:
        end_time = time.time()
        total_time = end_time - start_time
        hours, rem = divmod(total_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"\nTotal runtime: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")