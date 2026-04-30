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
    parser.add_argument("-ID", type=str, default="Exp_01", help="ID of the experiment")
    parser.add_argument("-exp_dir", type=str, default='/vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/pros/pros_full/full_covariates/2nd_layer/', help="Directory to save experiment results")
    parser.add_argument("-genotype_dir", type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/pros_can', help="Directory containing genotype files")
    parser.add_argument("-phenotype_file", type=str, default='/vol/vssp/SF_ucdatasets/gwas/data_files/disease_pheno/pros_can.xlsx', help="Path to phenotype file")

    parser.add_argument("-bs", type=int, default=32, help="Batch size for training")
    parser.add_argument("-dropout", type=float, default=0.5, help="Dropout rate for the model")

    parser.add_argument("-epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("-lr", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("-act", type=str, default="gelu", choices=["tanh","relu","gelu"], help="Dropout rate for the model")
    parser.add_argument("-opt", type=str, default="adamw", choices=["adam", "adamw", "sgd"], help="Optimizer to use")
    parser.add_argument("-sch", type=str, default="warmup_exponential", choices=["none","plateau", "cosine", "step","multistep","explr","warmup_exponential", "exponential_decay"], help="Learning rate scheduler")
    parser.add_argument("-peak_lr", type=float, default=1e-2, help="Peak learning rate for WarmupExponential scheduler")
    parser.add_argument("-final_lr", type=float, default=1e-5, help="Final learning rate for custom schedulers")
    parser.add_argument("-wd", type=float, default=0.5, help="Weight decay for optimizer")
    parser.add_argument("-df", type=float, default=0.1, help="Decay factor for custom schedulers")

    parser.add_argument("-hidden_sizes", type=parse_int_list, default=[128,128,128], help="Hidden layer sizes for MLP")

    parser.add_argument("-cov", type=int, default=1, choices=[0, 1], help="Whether to include covariates in the model (0: no, 1: yes)")
    parser.add_argument("-use_age", type=int, default=1, choices=[0, 1], help="Whether to include age in covariates in the model (0: no, 1: yes)")
    parser.add_argument("-use_gender", type=int, default=1, choices=[0, 1], help="Whether to include gender in covariates in the model (0: no, 1: yes)")
    parser.add_argument("-label_col", type=str, default="pros01", help="Column name in phenotype file to use as label (e.g., 'pan01', etc.)")

    # Add early stopping parameters
    parser.add_argument("-patience", type=int, default=15, help="Patience for early stopping")
    parser.add_argument("-min_delta", type=float, default=1e-4, help="Minimum change for early stopping")
    
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
    """Early stops the training if validation metric doesn't improve after a given patience."""
    def __init__(self, patience=7, min_delta=0, mode='min', verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation metric improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            mode (str): 'min' for loss, 'max' for metrics like accuracy
            verbose (bool): If True, prints a message for each improvement
        """
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
            if self.verbose:
                print(f'Initial best score: {score}')
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.verbose:
                print(f'Score improved to {score}')
            self.counter = 0

class GenotypeDataset(Dataset):
    def __init__(self, file_list, phenotype_data, label_column, use_covariates=True, use_age=True, use_gender=True):
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
            
        print(f"GenotypeDataset initialized with {len(file_list)} files")
        print(f"Using label column: {label_column}")
        print(f"Using covariates: {use_covariates}")
        print(f"Using age: {use_age}")
        print(f"Using gender: {use_gender}")
        
        self.file_handles = {}
        for file in file_list:
            f = open(file, 'rb')
            self.file_handles[file] = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        genotype_file = self.file_list[idx]
        sample_id_str = os.path.basename(genotype_file).replace("sample_", "").replace(".gen.gz", "")
        sample_id = int(sample_id_str)
        
        # Use the configured label column
        label = self.phenotype_data.loc[self.phenotype_data['new_order'] == sample_id, self.label_column].values[0]
        
        covariates_list = []
        if self.use_covariates:
            pc_covariates = self.phenotype_data.loc[self.phenotype_data['new_order'] == sample_id, 'PC1':'PC10'].values[0]
            covariates_list.append(pc_covariates)
            
        if self.use_age:
            age = self.phenotype_data.loc[self.phenotype_data['new_order'] == sample_id, 'Agexit'].values[0]
            ## Normalize age (you might want to adjust the normalization method)
            #age_normalized = (age - self.phenotype_data['Agexit'].mean()) / self.phenotype_data['Agexit'].std()
            #covariates_list.append([age_normalized])
            covariates_list.append([age])
            
        if self.use_gender:
            gender = self.phenotype_data.loc[self.phenotype_data['new_order'] == sample_id, 'Sex'].values[0]
            # # Convert gender to binary (assuming it's coded as 1 and 2 or M and F)
            # if isinstance(gender, str):
            #     gender_binary = 1 if gender.upper() == 'M' else 0
            # else:
            #     gender_binary = 1 if gender == 1 else 0
            #covariates_list.append([gender_binary])
            covariates_list.append([gender])

        if covariates_list:
            covariates = np.concatenate(covariates_list)
            covariates_tensor = torch.tensor(covariates, dtype=torch.float32)
        else:
            covariates_tensor = torch.tensor([], dtype=torch.float32)

        mmap_file = self.file_handles[genotype_file]
        mmap_file.seek(0)
        with gzip.GzipFile(fileobj=mmap_file) as f:
            data = pd.read_csv(f, sep=r'\s+', header=None)

        genotype_tensor = torch.tensor(data.values.T, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return genotype_tensor, covariates_tensor, label_tensor

    def __del__(self):
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
            
        # Pointwise convolution layer
        self.pointwise_conv = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1)
        
        # Create first two layers using Sequential and for loop
        initial_layers = []
        layer_sizes = [input_size] + hidden_sizes[:2]  # Only first two hidden layers
        
        for i in range(2):  # Create first two layers
            initial_layers.extend([
                (f'linear_{i+1}', nn.Linear(layer_sizes[i], layer_sizes[i+1])),
                (f'bn_{i+1}', nn.BatchNorm1d(layer_sizes[i+1])),
                (f'act_{i+1}', self.get_activation(act)),
                (f'dropout_{i+1}', nn.Dropout(dropout_rate))
            ])
        
        self.initial_layers = nn.Sequential(OrderedDict(initial_layers))
        
        # Third layer (after covariates)
        third_input_size = hidden_sizes[1] + (self.total_covariates if self.total_covariates > 0 else 0)
        self.third_linear = nn.Linear(third_input_size, hidden_sizes[2])
        self.third_bn = nn.BatchNorm1d(hidden_sizes[2])
        self.third_activation = self.get_activation(act)
        self.third_dropout = nn.Dropout(dropout_rate)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_sizes[2], 1)
        self.output_activation = nn.Sigmoid()
        
        print(f"GenotypeModel initialized (using covariates: {use_covariates}, age: {use_age}, gender: {use_gender})")

    def forward(self, x, covariates=None):
        # Apply pointwise convolution
        x = self.pointwise_conv(x).squeeze(1)
        
        # First two layers through Sequential
        x = self.initial_layers(x)
        
        # Concatenate with covariates after the second layer
        if covariates is not None and self.total_covariates > 0:
            x = torch.cat([x, covariates], dim=1)
        
        # Third layer
        x = self.third_linear(x)
        x = self.third_bn(x)
        x = self.third_activation(x)
        x = self.third_dropout(x)
        
        # Output layer
        x = self.output_layer(x)
        return self.output_activation(x).squeeze(1)

    def forward_lrp(self, x, covariates=None):
        """Special forward pass for LRP analysis that skips the pointwise convolution"""
        # First two layers through Sequential
        x = self.initial_layers(x)
        
        # Concatenate with covariates after the second layer
        if covariates is not None and self.total_covariates > 0:
            x = torch.cat([x, covariates], dim=1)
        
        # Third layer
        x = self.third_linear(x)
        x = self.third_bn(x)
        x = self.third_activation(x)
        x = self.third_dropout(x)
        
        # Output layer
        x = self.output_layer(x)
        return self.output_activation(x).squeeze(1)
    
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

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, device='cuda', save_dir=None, early_stopping=None):
    # print(f"Criterion is: {criterion}\n")
    # print(f"Optimizer is: {optimizer}\n")
    # print(f"Scheduler is: {scheduler.__class__.__name__}\n")
    # print(f"num_epochs is: {num_epochs}\n")
    print(f"Training on device: {device}")
    
    scaler = GradScaler()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    completed_epochs = 0  # Add counter for completed epochs

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
            inputs, covariates, labels = next(batch_iter)
            inputs = inputs.to(device, non_blocking=True)
            covariates = covariates.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            for i in range(len(dataloaders[phase])):
                # Asynchronously prefetch the next batch
                if i + 1 < len(dataloaders[phase]):
                    with torch.cuda.stream(stream):
                        next_inputs, next_covariates, next_labels = next(batch_iter)
                        next_inputs = next_inputs.to(device, non_blocking=True)
                        next_covariates = next_covariates.to(device, non_blocking=True)
                        next_labels = next_labels.to(device, non_blocking=True)

                # Wait for the current batch to be ready
                torch.cuda.current_stream().wait_stream(stream)

                # Process the current batch
                optimizer.zero_grad()
                with autocast():
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs, covariates)
                        loss = criterion(outputs, labels)
                        preds = torch.round(outputs)

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

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)

                all_labels[phase].extend(labels.cpu().numpy())
                all_preds[phase].extend(outputs.detach().cpu().numpy())

                # Prepare for the next iteration
                inputs, covariates, labels = next_inputs, next_covariates, next_labels

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples
            epoch_auc = roc_auc_score(all_labels[phase], all_preds[phase])

            print(f'{phase} Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f} - AUC: {epoch_auc:.4f}')

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())
            history[f'{phase}_auc'].append(epoch_auc)

            # Early stopping check based on validation AUC
            if phase == 'test' and early_stopping is not None:
                early_stopping(epoch_auc)
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
        y_pred = np.array(all_preds[phase])

        cm = confusion_matrix(y_true, y_pred.round())
        tn, fp, fn, tp = cm.ravel()

        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        fpr, tpr, _ = roc_curve(y_true, y_pred)
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
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
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

    # Create datasets with covariate and label column flags
    train_dataset = GenotypeDataset(
        train_files, phenotype_data,
        label_column=args.label_col,
        use_covariates=bool(args.cov),
        use_age=bool(args.use_age),
        use_gender=bool(args.use_gender)
    )
    test_dataset = GenotypeDataset(
        test_files, phenotype_data,
        label_column=args.label_col,
        use_covariates=bool(args.cov),
        use_age=bool(args.use_age),
        use_gender=bool(args.use_gender)
    )

    #print information for a few items from each dataset
    print("Sampling a few items from each dataset:")
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
    
    model = GenotypeModel(
        input_size=input_size,
        hidden_sizes=args.hidden_sizes,
        dropout_rate=dropout_rate,
        act=act,
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
        mode='max',  # We're monitoring AUC, so we use 'max'
        verbose=True
    )

    #pass scheduler to the train_model function
    model, history, final_metrics, all_preds, all_labels, completed_epochs = train_model(
        model, dataloaders, criterion, optimizer, scheduler, num_epochs, 
        device=device, save_dir=experiment_dir, early_stopping=early_stopping
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
        'Use_Covariates': bool(args.cov),
        'Use_Age': bool(args.use_age),
        'Use_Gender': bool(args.use_gender),
        'hidden_sizes':args.hidden_sizes     
    }

    # Perform LRP analysis
    print("\nStarting LRP analysis...")
    snp_importance, top_snps = perform_lrp_analysis(model, dataloaders, device, args)
    
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
