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
from captum.attr import LayerLRP



def parse_int_list(s):
    return [int(x) for x in s.split(',')]

def parse_args():
    parser = argparse.ArgumentParser(description="Genotype Model Training")
    parser.add_argument("-ID", type=str, default="1_1", help="ID of the experiment")
    parser.add_argument("-exp_dir", type=str, default='/vol/research/fmodal_mmmed/Codes/5_disease_experiments/MLP/results/col/col_0.1/chr_wise/random_test_labels/', help="Directory to save experiment results")
    parser.add_argument("-genotype_dir", type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/col_can_0.1', help="Directory containing genotype files")
    parser.add_argument("-phenotype_file", type=str, default='/vol/vssp/SF_ucdatasets/gwas/data_files/disease_pheno/col_can.xlsx', help="Path to phenotype file")
    parser.add_argument("-chr_file", type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/meta_data/first_5_columns_col_can_0.1.gen.gz', help="Path to phenotype file")

    parser.add_argument("-bs", type=int, default=32, help="Batch size for training")
    parser.add_argument("-dropout", type=float, default=0.5, help="Dropout rate for the model")

    parser.add_argument("-epochs", type=int, default=2, help="Number of epochs for training")
    parser.add_argument("-lr", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("-act", type=str, default="gelu", choices=["tanh","relu","gelu"], help="Dropout rate for the model")
    parser.add_argument("-opt", type=str, default="adamw", choices=["adam", "adamw", "sgd"], help="Optimizer to use")
    parser.add_argument("-sch", type=str, default="warmup_exponential", choices=["none","plateau", "cosine", "step","multistep","explr","warmup_exponential", "exponential_decay"], help="Learning rate scheduler")
    parser.add_argument("-peak_lr", type=float, default=1e-2, help="Peak learning rate for WarmupExponential scheduler")
    parser.add_argument("-final_lr", type=float, default=1e-5, help="Final learning rate for custom schedulers")
    parser.add_argument("-wd", type=float, default=0.5, help="Weight decay for optimizer")
    parser.add_argument("-df", type=float, default=0.1, help="Decay factor for custom schedulers")

    parser.add_argument("-hidden_sizes", type=parse_int_list, default=[128,128,128], help="Hidden layer sizes for MLP")
    parser.add_argument("-cov", type=int, default=0, choices=[0, 1], help="Whether to include covariates in the model (0: no, 1: yes)")
    parser.add_argument("-label_col", type=str, default="crc", help="Column name in phenotype file to use as label (e.g., 'col01', etc.)")
    
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


def read_chromosome_info(file_path):
    chromosome_info = {}
    with gzip.open(file_path, 'rt') as f:
        for line in f:
            chrom, snp_id, bp, ref, alt = line.strip().split()
            if chrom not in chromosome_info:
                chromosome_info[chrom] = []
            chromosome_info[chrom].append(snp_id)
            #print(chromosome_info)
    return chromosome_info


class GenotypeDataset(Dataset):
    def __init__(self, file_list, phenotype_data, chromosome_info, label_column, use_covariates=True, is_test=False):
        self.file_list = file_list
        self.phenotype_data = phenotype_data
        self.chromosome_info = chromosome_info
        self.label_column = label_column
        self.use_covariates = use_covariates
        self.is_test = is_test

        # Generate random labels for test set
        if self.is_test:
            print("Initializing test dataset with random labels")
            self.random_labels = torch.randint(0, 2, (len(file_list),), dtype=torch.float32)
        
        # Verify that the label column exists in the phenotype data
        if self.label_column not in self.phenotype_data.columns:
            raise ValueError(f"Label column '{self.label_column}' not found in phenotype data. "
                           f"Available columns are: {', '.join(self.phenotype_data.columns)}")
        
        print(f"GenotypeDataset initialized with {len(file_list)} files")
        print(f"Using label column: {label_column}")
        print(f"Using covariates: {use_covariates}")
        print(f"Is test set: {is_test}")
        
        self.file_handles = {}
        for file in file_list:
            f = open(file, 'rb')
            self.file_handles[file] = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        
        # Calculate cumulative SNP counts for each chromosome
        self.chrom_cumulative_snps = {}
        total_snps = 0
        for chrom, snps in self.chromosome_info.items():
            self.chrom_cumulative_snps[chrom] = total_snps
            total_snps += len(snps)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        genotype_file = self.file_list[idx]
        sample_id_str = os.path.basename(genotype_file).replace("sample_", "").replace(".gen.gz", "")
        sample_id = int(sample_id_str)
        
        # Use the configured label column
        if self.is_test:
            label = self.random_labels[idx].clone().detach() 
        else:
            label = self.phenotype_data.loc[self.phenotype_data['new_order'] == sample_id, self.label_column].values[0]
            label = torch.tensor(label, dtype=torch.float32)
                
        if self.use_covariates:
            covariates = self.phenotype_data.loc[self.phenotype_data['new_order'] == sample_id, 'PC1':'PC10'].values[0]
            covariates_tensor = torch.tensor(covariates, dtype=torch.float32)
        else:
            covariates_tensor = torch.tensor([], dtype=torch.float32)

        mmap_file = self.file_handles[genotype_file]
        mmap_file.seek(0)
        with gzip.GzipFile(fileobj=mmap_file) as f:
            data = pd.read_csv(f, sep=r'\s+', header=None)

        chromosome_data = {}
        for chrom, snp_ids in self.chromosome_info.items():
            start_idx = self.chrom_cumulative_snps[chrom]
            end_idx = start_idx + len(snp_ids)
            chrom_data = data.iloc[start_idx:end_idx]
            chromosome_data[chrom] = torch.tensor(chrom_data.values.T, dtype=torch.float32)

        return chromosome_data, covariates_tensor, label

    def __del__(self):
        for handle in self.file_handles.values():
            handle.close()

class ChromosomeMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rate, act):
        super().__init__()
        self.pointwise_conv = nn.Conv1d(3, 1, kernel_size=1)
        
        layers = []
        layer_sizes = [input_size] + hidden_sizes
        for i in range(len(layer_sizes) - 1):
            layers.append((f'linear_{i}', nn.Linear(layer_sizes[i], layer_sizes[i+1])))
            layers.append((f'batchnorm_{i}', nn.BatchNorm1d(layer_sizes[i+1])))
            layers.append((f'activation_{i}', self.get_activation(act)))
            layers.append((f'dropout_{i}', nn.Dropout(dropout_rate)))
        
        self.mlp = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.pointwise_conv(x)
        x = x.squeeze(1)
        return self.mlp(x)
    
    def forward_lrp(self, x):
        # Skip the pointwise convolution for LRP analysis
        return self.mlp(x)

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

class GenotypeModel(nn.Module):
    def __init__(self, chromosome_sizes, hidden_sizes, dropout_rate, act, use_covariates=True, num_covariates=10):
        super().__init__()
        self.use_covariates = use_covariates
        
        self.chromosome_mlps = nn.ModuleDict({
            chrom: ChromosomeMLP(size, hidden_sizes, dropout_rate, act)
            for chrom, size in chromosome_sizes.items()
        })
        
        total_chrom_output = sum(hidden_sizes[-1] for _ in chromosome_sizes)
        # Adjust final layer input size based on covariate usage
        final_input_size = total_chrom_output + (num_covariates if use_covariates else 0)
        
        self.final_layers = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(final_input_size, hidden_sizes[-1])),
            ('batchnorm1', nn.BatchNorm1d(hidden_sizes[-1])),
            ('activation1', self.get_activation(act)),
            ('dropout1', nn.Dropout(dropout_rate)),
            ('linear2', nn.Linear(hidden_sizes[-1], 1, bias=False)),
            ('sigmoid', nn.Sigmoid())
        ]))
        print(f"GenotypeModel initialized (using covariates: {use_covariates})")

    def forward(self, chromosome_data, covariates=None):
        chrom_outputs = []
        for chrom, data in chromosome_data.items():
            if data.dim() == 2:
                data = data.unsqueeze(1)  # Add a channel dimension if it's missing
            elif data.dim() != 3:
                raise ValueError(f"Unexpected data shape for chromosome {chrom}: {data.shape}")
            chrom_outputs.append(self.chromosome_mlps[chrom](data))
        
        # Only concatenate covariates if they're being used
        if self.use_covariates and covariates is not None:
            combined_output = torch.cat(chrom_outputs + [covariates], dim=1)
        else:
            combined_output = torch.cat(chrom_outputs, dim=1)
            
        return self.final_layers(combined_output).squeeze(1)

    def forward_lrp(self, chromosome_data, covariates=None):
        chrom_outputs = []
        for chrom, data in chromosome_data.items():
            chrom_outputs.append(self.chromosome_mlps[chrom].forward_lrp(data))
        
        if self.use_covariates and covariates is not None:
            combined_output = torch.cat(chrom_outputs + [covariates], dim=1)
        else:
            combined_output = torch.cat(chrom_outputs, dim=1)
            
        return self.final_layers(combined_output).squeeze(1)


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


def custom_lrp(model, chromosome_data, covariates, target=1, epsilon=1e-9):
    model.train()  # Ensure the model is in training mode
    
    # Enable gradients for all inputs
    chromosome_data = {k: v.requires_grad_() for k, v in chromosome_data.items()}
    covariates = covariates.requires_grad_()
    
    # Forward pass
    outputs = model(chromosome_data, covariates)
    
    # Backward pass
    model.zero_grad()
    outputs.sum().backward()
    
    relevance = {}
    for chrom, data in chromosome_data.items():
        # Get the gradient for the input
        input_grad = data.grad
        
        if input_grad is None:
            print(f"Warning: No gradient for chromosome {chrom}")
            relevance[chrom] = torch.zeros_like(data[0, 0]).cpu().numpy()
        else:
            # Element-wise multiply the input with its gradient
            relevance[chrom] = (data * input_grad).sum(dim=(0, 1)).abs().detach().cpu().numpy()
        
        # Normalize the relevance
        relevance[chrom] /= (relevance[chrom].sum() + epsilon)
    
    return relevance

def perform_lrp_analysis(model, chromosome_data, covariates, target=1):
    snp_importance = custom_lrp(model, chromosome_data, covariates, target)
    return snp_importance

def get_top_snps(snp_importance, chromosome_info, top_n=1000):
    top_snps = {}
    for chrom, importance in snp_importance.items():
        snp_ids = chromosome_info[chrom]
        sorted_indices = importance.argsort()[::-1]
        #top_snps[chrom] = [(snp_ids[i], importance[i]) for i in sorted_indices[:top_n]]
        top_snps[chrom] = [(snp_ids[i], importance[i]) for i in sorted_indices]
    return top_snps

def save_snp_info_to_csv(top_snps, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for chrom, snps in top_snps.items():
        filename = os.path.join(output_dir, f"top_snps_chromosome_{chrom}.csv")
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['SNP_ID', 'Importance'])  
            for snp_id, importance in snps:
                writer.writerow([snp_id, importance])
        print(f"Saved top SNPs for Chromosome {chrom} to {filename}")

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, device='cuda'):
    # print(f"Criterion is: {criterion}\n")
    # print(f"Optimizer is: {optimizer}\n")
    # print(f"Scheduler is: {scheduler.__class__.__name__}\n")
    # print(f"num_epochs is: {num_epochs}\n")
    print(f"Training on device: {device}")
    
    scaler = GradScaler()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_val_auc = 0.0  

    history = {
        'train_loss': [], 'train_acc': [], 'train_auc': [],
        'val_loss': [], 'val_acc': [], 'val_auc': [],
        'test_loss': [], 'test_acc': [], 'test_auc': [],
        'learning_rates': []
    }

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        all_preds = {phase: [] for phase in ['train', 'val', 'test']}
        all_labels = {phase: [] for phase in ['train', 'val', 'test']}

        for phase in ['train', 'val', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            for chromosome_data, covariates, labels in dataloaders[phase]:
                # Move data to device
                chromosome_data = {chrom: data.to(device, non_blocking=True) for chrom, data in chromosome_data.items()}
                covariates = covariates.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()
                
                with autocast():
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(chromosome_data, covariates)
                        loss = criterion(outputs, labels)
                        preds = torch.round(outputs)

                        if phase == 'train':
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()

                            # Step the scheduler if it's a per-iteration scheduler
                            if isinstance(scheduler, (WarmupExponential, ExponentialDecay)):
                                old_lr = optimizer.param_groups[0]['lr']
                                scheduler.step()
                                new_lr = optimizer.param_groups[0]['lr']
                                #history['learning_rates'].append(new_lr)

                running_loss += loss.item() * labels.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += labels.size(0)

                all_labels[phase].extend(labels.cpu().numpy())
                all_preds[phase].extend(outputs.detach().cpu().numpy())

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples
            epoch_auc = roc_auc_score(all_labels[phase], all_preds[phase])

            print(f'{phase} Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f} - AUC: {epoch_auc:.4f}')

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())
            history[f'{phase}_auc'].append(epoch_auc)

            if phase == 'val' and epoch_auc > best_val_auc:
                best_val_auc = epoch_auc
                best_model_wts = copy.deepcopy(model.state_dict())

        # Step the scheduler if it's an epoch-wise scheduler
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(history['val_loss'][-1])
            elif not isinstance(scheduler, (WarmupExponential, ExponentialDecay)):
                scheduler.step()
            #history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']}")

    print(f'Best val AUC: {best_val_auc:.4f}')
    model.load_state_dict(best_model_wts)

    # Compute final metrics for each phase
    final_metrics = {}
    for phase in ['train', 'val', 'test']:
        y_true = np.array(all_labels[phase])
        y_pred = np.array(all_preds[phase])

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred.round())
        tn, fp, fn, tp = cm.ravel()

        # Compute sensitivity and specificity
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)

        final_metrics[phase] = {
            'confusion_matrix': cm,
            'sensitivity':f'{sensitivity:.5f}',
            'specificity': f'{specificity:.5f}',
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr
        }

    return model, history, final_metrics, all_preds, all_labels

def plot_all_metrics(history, final_metrics, save_dir):
    phases = ['train', 'val', 'test']
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
    chr_file = args.chr_file
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
        
    # Get input size dynamically
    first_genotype_file = glob.glob(os.path.join(genotype_dir, "sample_*.gen.gz"))[0]
    print(f"First genotype file of the directory is: {first_genotype_file}")
    input_size = get_input_size(first_genotype_file)
    print(f"Dynamically determined input size: {input_size}")


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load phenotype data
    phenotype_data = pd.read_excel(phenotype_file)
    print(f"Phenotype data loaded, shape: {phenotype_data.shape}")

    # Get list of genotype files
    file_list = glob.glob(os.path.join(genotype_dir, "sample_*.gen.gz"))
    print(f"Number of genotype files found: {len(file_list)}")

    num_samples = len(phenotype_data['new_order'].unique())
    print(f"Number of unique samples in phenotype data: {num_samples}")

    if len(file_list) != num_samples:
        raise ValueError(f"Number of files ({len(file_list)}) in {genotype_dir} does not match number of samples ({num_samples}) in phenotype data.")


    # Split data
    train_files, test_files = train_test_split(file_list, test_size=0.4, random_state=42)
    val_files, test_files = train_test_split(test_files, test_size=0.5, random_state=42)
    print(f"Data split: Train {len(train_files)}, Val {len(val_files)}, Test {len(test_files)}")

    # Load chromosome information
    chromosome_info = read_chromosome_info(chr_file)
    
    # Create datasets with covariate and label column flags
    train_dataset = GenotypeDataset(
        train_files, phenotype_data, chromosome_info,
        label_column=args.label_col,
        use_covariates=bool(args.cov),
        is_test=False 
    )
    val_dataset = GenotypeDataset(
        val_files, phenotype_data, chromosome_info,
        label_column=args.label_col,
        use_covariates=bool(args.cov),
        is_test=False 
    )
    test_dataset = GenotypeDataset(
        test_files, phenotype_data, chromosome_info,
        label_column=args.label_col,
        use_covariates=bool(args.cov),
        is_test=True 
    )

    #print information for a few items from each dataset
    print("Sampling a few items from each dataset:")
    for dataset_name, dataset in [("Train", train_dataset), ("Validation", val_dataset), ("Test", test_dataset)]:
        print(f"\n{dataset_name} dataset sample:")
        for i in range(1):  # Print info for 3 items from each dataset
            chromosome_data, covariates, label = dataset[i]
            print(f"Item {i}:")
            print(f"  Number of chromosomes: {len(chromosome_data)}")
            for chrom, data in chromosome_data.items():
                print(f"    Chromosome {chrom} shape: {data.shape}")
            print(f"  Covariates shape: {covariates.shape}")
            print(f"  Label: {label}")

    dataloaders = {
    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2, drop_last=True),
    'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2, drop_last=True),
    'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2, drop_last=True)
}

    print("DataLoaders created")

    # Calculate input sizes for each chromosome
    chromosome_sizes = {chrom: len(snps) for chrom, snps in chromosome_info.items()}
    print("Chromosome sizes:", chromosome_sizes)

    
    # Initialize model with covariate flag
    model = GenotypeModel(
        chromosome_sizes=chromosome_sizes,
        hidden_sizes=args.hidden_sizes,
        dropout_rate=dropout_rate,
        act=act,
        use_covariates=bool(args.cov),
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

    #pass scheduler to the train_model function
    model, history, final_metrics, all_preds, all_labels = train_model(
        model, dataloaders, criterion, optimizer, scheduler, num_epochs, device
    )
    
    #Plot metrics
    plot_all_metrics(history, final_metrics, experiment_dir)

    # Save the trained model
    #torch.save(model.state_dict(), os.path.join(experiment_dir, 'trained_genotype_model.pth'))
    print("Model training completed and saved")

    # Update results dictionary
    results = {
        #'Train_loss': history['train_loss'][-1],
        'train_acc': round(history['train_acc'][-1],5),
        'train_auc': round(history['train_auc'][-1],5),
        #'Val_loss': history['val_loss'][-1],
        'val_acc': round(history['val_acc'][-1],5),
        'val_auc': round(history['val_auc'][-1],5),
        #'Test_loss': history['test_loss'][-1],
        'test_acc': round(history['test_acc'][-1],5),
        'test_auc': round(history['test_auc'][-1],5)
    }

    # Add final metrics for each phase
    for phase in ['train', 'val', 'test']:
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
        'Epochs': num_epochs,
        'Start_LR': learning_rate,
        'Peak_LR': args.peak_lr,
        'Final_LR':optimizer.param_groups[0]["lr"],
        'Dropout': dropout_rate,
        'Act': act,
        'Opt': opt,
        'Sch': sch,
        'WD': wd,
        'DF': args.df,
        'Label_Column': args.label_col,
        'Use_Covariates': bool(args.cov),
        'hidden_sizes':args.hidden_sizes     
    }
    # Write results
    write_results(model, hyperparameters, results, experiment_dir)
    append_metrics_to_csv(experiment_dir, results)

    print("Performing custom LRP analysis...")
    model.train()  # Set model to train mode
    test_batch = next(iter(dataloaders['test']))
    chromosome_data, covariates, _ = test_batch
    chromosome_data = {k: v.to(device) for k, v in chromosome_data.items()}
    covariates = covariates.to(device)

    snp_importance = perform_lrp_analysis(model, chromosome_data, covariates)
    top_snps = get_top_snps(snp_importance, chromosome_info, top_n=1500)

    # Save top SNPs information to CSV files
    output_dir = os.path.join(args.exp_dir, args.ID, 'top_snps_csv')
    save_snp_info_to_csv(top_snps, output_dir)

    # Print top SNPs for each chromosome (just showing top 10 in console)
    for chrom, snps in top_snps.items():
        print(f"\nTop 10 SNPs for Chromosome {chrom}:")
        for snp_id, importance in snps[:10]:
            print(f"SNP ID: {snp_id}, Importance: {importance:.4f}")

    # Visualize SNP importance
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 10))
    for chrom, importance in snp_importance.items():
        plt.plot(range(len(importance)), importance, label=f'Chromosome {chrom}')
    plt.xlabel('SNP Index')
    plt.ylabel('Importance')
    plt.title('SNP Importance by Chromosome (Custom LRP)')
    plt.legend()
    plt.savefig(os.path.join(args.exp_dir, args.ID, 'snp_importance_custom_lrp.png'))
    plt.close()

    
if __name__ == '__main__':
    start_time = time.time()
    
    main()
    
    end_time = time.time()
    total_runtime = end_time - start_time
    
    print(f"\nTotal script runtime: {total_runtime:.2f} seconds")
    hours, rem = divmod(total_runtime, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Total runtime: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")
