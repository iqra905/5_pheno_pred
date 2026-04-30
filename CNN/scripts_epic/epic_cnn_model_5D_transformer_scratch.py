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
from sklearn.metrics import roc_auc_score
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
    parser.add_argument("-ID", type=str, default="1_1", help="ID of the experiment")
    parser.add_argument("-exp_dir", type=str, default='/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/pros/transformer/scratch/updated', help="Directory to save experiment results")
    parser.add_argument("-genotype_dir", type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/pros_can_pruned', help="Directory containing genotype files")
    parser.add_argument("-phenotype_file", type=str, default='/vol/vssp/SF_ucdatasets/gwas/data_files/disease_pheno/pros_can.xlsx', help="Path to phenotype file")

    # Model architecture parameters
    parser.add_argument("-num_transformer_layers", type=int, default=3, help="Number of transformer layers")
    parser.add_argument("-nhead", type=int, default=6, help="Number of transformer heads")
    parser.add_argument("-d_model", type=int, default=384, help="Dimension of transformer model")
    parser.add_argument("-fc_layers", type=parse_int_list, default=[64,128], help="Sizes of fully connected layers")

    # Add pooling type argument
    parser.add_argument("-pooling", type=str, default="mean", choices=["mean", "cls", "weighted"], help="Type of pooling to use in transformer")
    
    # Data parameters
    parser.add_argument("-label_col", type=str, default="pros01", help="Column name for labels")
    parser.add_argument("-use_covariates", action="store_true", help="Whether to use covariates")

    # Training parameters
    parser.add_argument("-bs", type=int, default=32, help="Batch size for training")
    parser.add_argument("-dropout", type=float, default=0.5, help="Dropout rate for the model")
    parser.add_argument("-epochs", type=int, default=2, help="Number of epochs for training")
    parser.add_argument("-act", type=str, default="tanh", choices=["tanh","relu","gelu"], help="Activation function")
    parser.add_argument("-opt", type=str, default="adamw", choices=["adam", "adamw", "sgd"], help="Optimizer to use")
    parser.add_argument("-sch", type=str, default="exponential_decay", choices=["none","plateau", "cosine", "step","multistep","explr","warmup_exponential", "exponential_decay"], help="Learning rate scheduler")
    parser.add_argument("-wd", type=float, default=0.5, help="Weight decay for optimizer")
    parser.add_argument("-df", type=float, default=0.1, help="Decay factor for custom schedulers")
    parser.add_argument("-lr", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("-peak_lr", type=float, default=1e-2, help="Peak learning rate for WarmupExponential scheduler")
    parser.add_argument("-final_lr", type=float, default=1e-5, help="Final learning rate for custom schedulers")

    parser.add_argument("-kernel_sizes", type=parse_int_list, default=[4096,1,1], help="Convolution Kernel Size")
    parser.add_argument("-stride", type=parse_int_list, default=[2048,1,1], help="Convolution Stride")
    parser.add_argument("-conv_channels", type=parse_int_list, default=[64,128,384], help="Convolution channels")

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

class GenotypeDataset(Dataset):
    def __init__(self, file_list, phenotype_data, label_col, use_covariates=True):
        self.file_list = file_list
        self.phenotype_data = phenotype_data
        self.label_col = label_col
        self.use_covariates = use_covariates
        print(f"GenotypeDataset initialized with {len(file_list)} files")
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
        
        label = self.phenotype_data.loc[self.phenotype_data['new_order'] == sample_id, self.label_col].values[0]
        
        if self.use_covariates:
            covariates = self.phenotype_data.loc[self.phenotype_data['new_order'] == sample_id, 'PC1':'PC10'].values[0]
        else:
            covariates = np.array([])  

        # Use memory-mapped file for faster access
        mmap_file = self.file_handles[genotype_file]
        mmap_file.seek(0)
        with gzip.GzipFile(fileobj=mmap_file) as f:
            data = pd.read_csv(f, sep=r'\s+', header=None)

        genotype_tensor = torch.tensor(data.values.T, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        covariates_tensor = torch.tensor(covariates, dtype=torch.float32)

        return genotype_tensor, covariates_tensor, label_tensor

    def __del__(self):
        for handle in self.file_handles.values():
            handle.close()

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
            if attention_mask is not None:
                # Masked mean
                mask = attention_mask.unsqueeze(-1).float()
                x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                # Simple mean
                x = x.mean(dim=1)
            return x
            
        elif self.pooling_type == 'weighted':
            # Compute attention weights
            weights = self.attention(x)  # (B, L, 1)
            weights = F.softmax(weights, dim=1)
            # Apply weights
            x = (x * weights).sum(dim=1)
            return x

class GenotypeModelWithTransformer(nn.Module):
    def __init__(self, input_size, kernel_sizes, stride, conv_channels, act, dropout_rate, use_covariates=True, num_covariates=10, 
                num_transformer_layers=3, d_model=384, nhead=8, fc_layers=[128], pooling_type='mean', print_dimensions=False):
        super().__init__()
        self.print_dimensions = print_dimensions
        self.has_printed_dimensions = False
        self.use_covariates = use_covariates
        self.pooling_type = pooling_type

        # Pointwise convolution to reduce channels from 3 to 1
        self.pointwise_conv = nn.Conv1d(3, 1, kernel_size=1)

        # Rest of convolutional layers
        self.input_channels = 1  
        self.conv_layers = self._create_conv_layers(conv_channels, kernel_sizes, stride, dropout_rate, act)

        self.conv_output_size = self._get_conv_output_size(input_size)
        print(f"Convolutional output size: {self.conv_output_size}")

        # Project convolutional output to d_model dimensions
        self.proj = nn.Linear(conv_channels[-1], d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout_rate)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout_rate, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # Pooling layer
        self.pooling = TransformerPooling(d_model, pooling_type)
        
        # Fully connected layers
        fc_input_size = d_model + (num_covariates if use_covariates else 0)

        # self.classifier = nn.Sequential(
        #     self._create_fc_layers(fc_input_size, fc_layers, dropout_rate, act),
        #     nn.Linear(fc_layers[-1], 1)
        # )

        self.classifier = nn.Sequential(
            #self._create_fc_layers(fc_input_size, fc_layers, dropout_rate, act),
            nn.Linear(384, 1)
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
            if i > 0:  # Apply dropout after the first layer
                layers.append(nn.Dropout(dropout_rate))
        return nn.Sequential(*layers)

    def _create_fc_layers(self, input_size, fc_sizes, dropout_rate, act):
        layers = []
        for i, fc_size in enumerate(fc_sizes):
            layers.append(nn.Linear(input_size if i == 0 else fc_sizes[i-1], fc_size))
            layers.append(self.get_activation(act))
            layers.append(nn.Dropout(dropout_rate))
        return nn.Sequential(*layers)

    def _get_conv_output_size(self, input_size):
        x = torch.randn(1, 3, input_size, dtype=torch.float32)
        x = self.pointwise_conv(x)  # First apply pointwise convolution
        x = self.conv_layers(x)
        return x.shape[2]  # Return the sequence length after convolutions

    def forward(self, x, covariates):
        if self.print_dimensions and not self.has_printed_dimensions:
            print(f"Input shape: {x.shape}")
        
        # Apply pointwise convolution first
        x = self.pointwise_conv(x)
        if self.print_dimensions and not self.has_printed_dimensions:
            print(f"After pointwise conv: {x.shape}")
        
        x = self.conv_layers(x)
        if self.print_dimensions and not self.has_printed_dimensions:
            print(f"After conv layers: {x.shape}")
        
        # Convert (B, C, L) to (B, L, C)
        x = x.permute(0, 2, 1)
        
        # Project to d_model dimensions
        x = self.proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer
        x = self.transformer(x)
        
        if self.print_dimensions and not self.has_printed_dimensions:
            print(f"After transformer layers: {x.shape}")
        
        # Apply pooling
        x = self.pooling(x)
        
        if self.print_dimensions and not self.has_printed_dimensions:
            print(f"After pooling: {x.shape}")
            if self.use_covariates:
                print(f"Covariates shape: {covariates.shape}")
        
        # Concatenate with covariates if using them
        if self.use_covariates:
            x = torch.cat([x, covariates], dim=1)
            if self.print_dimensions and not self.has_printed_dimensions:
                print(f"After concatenating covariates: {x.shape}")
        
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

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, device='cuda'):
    print(f"Training on device: {device}")
    
    scaler = GradScaler()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

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
                print(f"No data in {phase} dataloader")
                continue

            for i in range(len(dataloaders[phase])):
                # Asynchronously prefetch the next batch
                if i + 1 < len(dataloaders[phase]):
                    try:
                        with torch.cuda.stream(stream):
                            next_inputs, next_covariates, next_labels = next(batch_iter)
                            next_inputs = next_inputs.to(device, non_blocking=True)
                            next_covariates = next_covariates.to(device, non_blocking=True)
                            next_labels = next_labels.to(device, non_blocking=True)
                    except StopIteration:
                        next_inputs = None
                else:
                    next_inputs = None

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
                                scheduler.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)

                all_labels[phase].extend(labels.cpu().numpy())
                all_preds[phase].extend(outputs.detach().cpu().numpy())

                # Prepare for the next iteration
                if next_inputs is not None:
                    inputs, covariates, labels = next_inputs, next_covariates, next_labels

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples
            epoch_auc = roc_auc_score(all_labels[phase], all_preds[phase])

            print(f'{phase} Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f} - AUC: {epoch_auc:.4f}')

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())
            history[f'{phase}_auc'].append(epoch_auc)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(history['val_loss'][-1])
            elif not isinstance(scheduler, (WarmupExponential, ExponentialDecay)):
                scheduler.step()
            print_lr(optimizer)
        
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])

    print(f'Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)

    # Compute final metrics for each phase
    final_metrics = {}
    for phase in ['train', 'val', 'test']:
        y_true = np.array(all_labels[phase])
        y_pred_proba = 1 / (1 + np.exp(-np.array(all_preds[phase])))  # Apply sigmoid to logits
        y_pred_binary = (y_pred_proba > 0.5).astype(int)

        # Compute confusion matrix with binary predictions
        cm = confusion_matrix(y_true, y_pred_binary)
        
        # Handle case where confusion matrix might not be 2x2
        if cm.size == 4:  # If it's a 2x2 matrix
            tn, fp, fn, tp = cm.ravel()
        else:  # If it's not a 2x2 matrix, calculate metrics differently
            tp = np.sum((y_true == 1) & (y_pred_binary == 1))
            tn = np.sum((y_true == 0) & (y_pred_binary == 0))
            fp = np.sum((y_true == 0) & (y_pred_binary == 1))
            fn = np.sum((y_true == 1) & (y_pred_binary == 0))

        # Compute sensitivity and specificity
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Compute ROC curve and AUC using probabilities
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        final_metrics[phase] = {
            'confusion_matrix': np.array([[tn, fp], [fn, tp]]),
            'sensitivity': f'{sensitivity:.5f}',
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
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Learning Rate')
    axs[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_plot.png'))
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
    print(f"Steps per epoch: {steps_per_epoch}")
    total_steps = steps_per_epoch * args.epochs
    print(f"Total Steps: {total_steps}")
    warmup_percentage = 0.1
    wsteps = int(total_steps * warmup_percentage)
    print(f"Warmup Steps: {wsteps}")

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
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
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
        raise ValueError(f"Number of files ({len(file_list)}) does not match number of samples ({num_samples}) in phenotype data.")

    # Split data
    train_files, test_files = train_test_split(file_list, test_size=0.4, random_state=42)
    val_files, test_files = train_test_split(test_files, test_size=0.5, random_state=42)
    print(f"Data split: Train {len(train_files)}, Val {len(val_files)}, Test {len(test_files)}")

    # Create datasets
    train_dataset = GenotypeDataset(train_files, phenotype_data, args.label_col, args.use_covariates)
    val_dataset = GenotypeDataset(val_files, phenotype_data, args.label_col, args.use_covariates)
    test_dataset = GenotypeDataset(test_files, phenotype_data, args.label_col, args.use_covariates)

    # Create dataloaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True, prefetch_factor=2),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True, prefetch_factor=2),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True, prefetch_factor=2)
    }

    print("DataLoaders created")

    # Create model
    model = GenotypeModelWithTransformer(
        input_size=input_size,
        kernel_sizes=args.kernel_sizes,
        stride=args.stride,
        conv_channels=args.conv_channels,
        act=args.act,
        dropout_rate=args.dropout,
        use_covariates=args.use_covariates,
        num_covariates=10 if args.use_covariates else 0,
        num_transformer_layers=args.num_transformer_layers,
        d_model=args.d_model,
        nhead=args.nhead,
        fc_layers=args.fc_layers,
        pooling_type=args.pooling, 
        print_dimensions=True
    )

    model = model.to(device)

    with open(os.path.join(experiment_dir, 'model_architecture.txt'), 'w') as file:
        file.write(str(model))
        print(model)

    print("Model created and moved to device")

    # Set up loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = {
        "adadelta": optim.Adadelta(model.parameters(), lr=args.lr),
        "adagrad": optim.Adagrad(model.parameters(), lr=args.lr),
        "adamw": optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd),
        "rmsprop": optim.RMSprop(model.parameters(), lr=args.lr),
        "sgd": optim.SGD(model.parameters(), lr=args.lr),
        "adam": optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    }.get(args.opt.lower())

    if optimizer is None:
        raise NotImplementedError("Optimizer not implemented.")

    scheduler = get_scheduler(args.sch, optimizer, args, train_files)

    # Train model
    model, history, final_metrics, all_preds, all_labels = train_model(
        model, dataloaders, criterion, optimizer, scheduler, args.epochs, device
    )
    
    # Plot metrics
    plot_all_metrics(history, final_metrics, experiment_dir)

    # Update results dictionary
    results = {
        'train_auc': round(history['train_auc'][-1], 4),
        'train_acc': round(history['train_acc'][-1], 4),
        'val_auc': round(history['val_auc'][-1], 4),
        'val_acc': round(history['val_acc'][-1], 4),
        'test_auc': round(history['test_auc'][-1], 4),
        'test_acc': round(history['test_acc'][-1], 4)
    }

    # Add final metrics for each phase
    for phase in ['train', 'val', 'test']:
        results.update({
            f'{phase}_sens': final_metrics[phase]['sensitivity'],
            f'{phase}_spec': final_metrics[phase]['specificity'],
            f'{phase}_CM': final_metrics[phase]['confusion_matrix'],
        })

    # Update hyperparameters dictionary
    hyperparameters = {
        'Exp_ID': id,
        'BS': batch_size,
        'Epochs': args.epochs,
        'Start_LR': args.lr,
        'Peak_LR': args.peak_lr,
        'Final_LR': optimizer.param_groups[0]["lr"],
        'Dropout': dropout_rate,
        'Act': args.act,
        'Opt': args.opt,
        'Sch': args.sch,
        'WD': args.wd,
        'DF': args.df,
        'Kernel_sizes': args.kernel_sizes,
        'Stride': args.stride,
        'Conv_channels': args.conv_channels,
        'FC_layers': args.fc_layers,
        'Num_transformer_layers': args.num_transformer_layers,
        'd_model': args.d_model,
        'nhead': args.nhead,
        'Label_col': args.label_col,
        'Use_covariates': args.use_covariates
    }

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
