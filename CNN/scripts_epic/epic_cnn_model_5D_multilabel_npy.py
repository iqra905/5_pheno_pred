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
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
import argparse
import csv
import time

def parse_int_list(s):
    return [int(x) for x in s.split(',')]

def parse_str_list(s):
    return [x.strip() for x in s.split(',')]

def parse_args():
    parser = argparse.ArgumentParser(description="Multilabel Genotype Model Training")
    parser.add_argument("-ID", type=str, default="000", help="ID of the experiment")
    parser.add_argument("-exp_dir", type=str, default='/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/5d_multilabel', help="Directory to save experiment results")
    parser.add_argument("-genotype_dir", type=str, default='/vol/research/ucdatasets/gwas/gwas_mono_rm/sampled_data_5M_unq_npy', help="Directory containing genotype files")
    parser.add_argument("-phenotype_file", type=str, default='/vol/vssp/SF_ucdatasets/gwas/data_files/merged_v8_pcs_chip_added_Iqra_1_cleaned.xlsx', help="Path to phenotype file")

    parser.add_argument("-bs", type=int, default=4, help="Batch size for training")
    parser.add_argument("-dropout", type=float, default=0.5, help="Dropout rate for the model")
    parser.add_argument("-epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("-lr", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("-act", type=str, default="gelu", choices=["tanh","relu","gelu"], help="Activation function for the model")
    parser.add_argument("-opt", type=str, default="adamw", choices=["adam", "adamw", "sgd"], help="Optimizer to use")
    parser.add_argument("-sch", type=str, default="exponential_decay", choices=["none","plateau", "cosine", "step","multistep","explr","warmup_exponential", "exponential_decay"], help="Learning rate scheduler")
    parser.add_argument("-peak_lr", type=float, default=1e-2, help="Peak learning rate for WarmupExponential scheduler")
    parser.add_argument("-final_lr", type=float, default=1e-5, help="Final learning rate for custom schedulers")
    parser.add_argument("-wd", type=float, default=0.5, help="Weight decay for optimizer")
    parser.add_argument("-df", type=float, default=1, help="Decay factor for custom schedulers")

    # Model architecture
    parser.add_argument("-kernel_sizes", type=parse_int_list, default=[127,31,7], help="Convolution Kernel Size")
    parser.add_argument("-stride", type=parse_int_list, default=[64,16,4], help="Convolution Stride")
    parser.add_argument("-conv_channels", type=parse_int_list, default=[2,4,8], help="Convolution channels")
    parser.add_argument("-fc_layers", type=parse_int_list, default=[128,64], help="Fully connected layers")

    # Data-specific parameters
    parser.add_argument("-cov", type=int, default=1, choices=[0, 1], help="Whether to include covariates in the model (0: no, 1: yes)")
    parser.add_argument("-use_age", type=int, default=1, choices=[0, 1], help="Whether to include age in covariates in the model (0: no, 1: yes)")
    parser.add_argument("-use_gender", type=int, default=1, choices=[0, 1], help="Whether to include gender in covariates in the model (0: no, 1: yes)")
    
    # Early stopping parameters
    parser.add_argument("-patience", type=int, default=15, help="Patience for early stopping")
    parser.add_argument("-min_delta", type=float, default=1e-4, help="Minimum change for early stopping")

    # Normalization-related arguments
    parser.add_argument("-norm_age", type=str, default="none", choices=["none", "standard", "minmax", "robust", "quantile", "power"], help="Normalization method for age")
    parser.add_argument("-norm_pcs", type=str, default="none", choices=["none", "standard", "minmax", "robust", "quantile", "power"], help="Normalization method for PCs")
    parser.add_argument("-norm_gender", type=str, default="none", choices=["none", "minmax"], help="Normalization method for gender (usually keep as none)")


    parser.add_argument("-disease_labels", type=parse_str_list, default="pros01,panca,crc,breacancer,t2dm", help="Comma-separated list of disease column names in phenotype file")
    
    parser.add_argument("-pool_size", type=int, default=64, help="Size of the adaptive pooling output (smaller values = more aggressive pooling)")
   
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

class MultilabelGenotypeDataset(Dataset):
    def __init__(self, file_list, phenotype_data, disease_labels, use_covariates=True, use_age=True, 
                 use_gender=True, norm_age="standard", norm_pcs="standard", norm_gender="none", 
                 fit_normalizers=True, normalizers=None):
        self.file_list = file_list
        self.phenotype_data = phenotype_data
        self.disease_labels = disease_labels
        self.use_covariates = use_covariates
        self.use_age = use_age
        self.use_gender = use_gender

        # Verify that all disease label columns exist in the phenotype data
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
        
        
        # Log initialization
        print(f"\nDataset Initialization:")
        print(f"- Number of files: {len(file_list)}")
        print(f"- Disease labels: {', '.join(disease_labels)}")
        print(f"- Using PCs: {use_covariates} (normalization: {norm_pcs})")
        print(f"- Using age: {use_age} (normalization: {norm_age})")
        print(f"- Using gender: {use_gender} (normalization: {norm_gender})")


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
        
        sample_id_str = os.path.basename(genotype_file).replace("sample_", "").replace(".npy", "")
        
        sample_id = int(sample_id_str)
        
        # Get labels for all diseases
        labels = self.phenotype_data.loc[self.phenotype_data['new_order'] == sample_id, self.disease_labels].values[0]
        
        # Get covariates if needed
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

        # FAST LOADING: Use numpy.load for .npy files
        if '.npy' in genotype_file:
            genotype_data = np.load(genotype_file)  # Shape: (5000000, 3)
            genotype_tensor = torch.from_numpy(genotype_data).float()
        
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        return genotype_tensor, covariates_tensor, labels_tensor
            
class MultilabelGenotypeModel(nn.Module):
    def __init__(self, input_size, num_diseases, kernel_sizes, stride, conv_channels, fc_layers, act, dropout_rate, use_covariates=True, use_age=True, use_gender=True, num_covariates=10, pool_size=16):
        super(MultilabelGenotypeModel, self).__init__()
        self.input_channels = 3
        self.use_covariates = use_covariates
        self.use_age = use_age
        self.use_gender = use_gender
        self.num_diseases = num_diseases
        self.pool_size = pool_size  # Add pooling size parameter


        # Calculate total number of covariates
        self.total_covariates = 0
        if use_covariates:
            self.total_covariates += num_covariates  # PCs
        if use_age:
            self.total_covariates += 1  # Age
        if use_gender:
            self.total_covariates += 1  # Gender

        self.conv_layers = self._create_conv_layers(conv_channels, kernel_sizes, stride, dropout_rate, act)

        # Add pooling layer
        self.pool = nn.AdaptiveMaxPool1d(pool_size)  # Adaptive pooling to fixed output size
        
        self.conv_output_size = self._get_conv_output_size(input_size)
        print(f"Convolutional output size: {self.conv_output_size}")
        
        fc_layers_list = []
        in_features = self.conv_output_size + self.total_covariates
        
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
        
        # Create separate output layers for each disease
        self.disease_outputs = nn.Linear(in_features, num_diseases)
        
        print(f"MultilabelGenotypeModel initialized with {num_diseases} disease outputs "
              f"(using covariates: {use_covariates}, age: {use_age}, gender: {use_gender})")


    def _create_conv_layers(self, conv_channels, kernel_sizes, stride, dropout_rate, act):
        layers = []
        for i in range(len(conv_channels)):
            layers.append(nn.Conv1d(in_channels=self.input_channels if i == 0 else conv_channels[i-1],
                                   out_channels=conv_channels[i],  
                                   kernel_size=kernel_sizes[i],
                                   stride=stride[i]))
            layers.append(nn.BatchNorm1d(conv_channels[i]))
            layers.append(self.get_activation(act))
            # if i > 0:  # Apply dropout after the first layer
            #     layers.append(nn.Dropout(dropout_rate))
        return nn.Sequential(*layers)

    def _get_conv_output_size(self, input_size):
        x = torch.randn(1, 3, input_size, dtype=torch.float32)
        x = self.conv_layers(x)
        # Apply pooling to the test tensor
        x = self.pool(x)
        return x.numel() // x.size(0)

    def forward(self, x, covariates=None):
        x = x.permute(0, 2, 1)  # -> [batch_size, 3, n_snps]
        
        x = self.conv_layers(x)
        # Apply pooling before flattening
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        # Concatenate with covariates
        if covariates is not None and self.total_covariates > 0:
            x = torch.cat([x, covariates], dim=1)
        
        # Get shared features
        shared_features = self.fc_shared(x)
        
        # Get outputs for each disease
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

def train_multilabel_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, disease_labels, device='cuda', early_stopping=None):
    print(f"Training multilabel model on device: {device}")
    print(f"Disease labels: {disease_labels}")
    
    scaler = GradScaler()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    completed_epochs = 0 

    num_diseases = len(disease_labels)
    
    # Initialize history dict with metrics for each disease
    history = {
        'train_loss': [], 'test_loss': [],
        'learning_rates': []
    }
    
    # Add metrics for each disease
    for disease in disease_labels:
        for phase in ['train', 'test']:
                history[f'{phase}_{disease}_acc'] = []
                history[f'{phase}_{disease}_auc'] = []
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Store predictions and labels for each phase and disease
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

            # Initialize batch metrics for logging
            batch_times = []
            
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
                with autocast():
                    with torch.set_grad_enabled(phase == 'train'):
                        logits = model(inputs, covariates)
                        loss = criterion(logits, labels)

                        # Convert logits to probabilities for metrics
                        probs = torch.sigmoid(logits)
                        preds = (probs >= 0.5).float()
                        
                        if phase == 'train':
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()

                            # Step the scheduler if it's a per-iteration scheduler
                            if isinstance(scheduler, (WarmupExponential, ExponentialDecay)):
                                scheduler.step()

                batch_size = labels.size(0)
                running_loss += loss.item() * batch_size

                # Store predictions and labels for each disease
                for j, disease in enumerate(disease_labels):
                    running_corrects[disease] += torch.sum(preds[:, j] == labels[:, j])

                    # Store predictions and true labels for AUC calculation
                    all_labels[phase][disease].extend(labels[:, j].cpu().numpy())
                    all_preds[phase][disease].extend(probs[:, j].detach().cpu().numpy())
                
                total_samples += batch_size

                # Calculate batch processing time
                batch_end = time.time()
                batch_time = batch_end - batch_start
                batch_times.append(batch_time)

                # Print batch progress every 20 batches (adjust as needed)
                if (i + 1) % 20 == 0 or i == 0 or i == len(dataloaders[phase]) - 1:
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
                    
                except Exception as e:
                    print(f"Error calculating curves for {disease} in {phase}: {str(e)}")
                    fpr, tpr = np.array([]), np.array([])
                    roc_auc = pr_auc = 0.5
            except Exception as e:
                print(f"Error calculating metrics for {disease} in {phase}: {str(e)}")
                cm = np.zeros((2, 2))
                sensitivity = specificity = accuracy = precision = f1 = 0
                fpr, tpr = np.array([]), np.array([])
                roc_auc = pr_auc = 0.5

            disease_metrics = {
                'cm': cm,
                'sens': f'{sensitivity:.5f}',
                'spec': f'{specificity:.5f}',
                'acc': f'{accuracy:.5f}',
                'auc': roc_auc,
            }
            
            phase_metrics[disease] = disease_metrics
        
        final_metrics[phase] = phase_metrics
    
    return final_metrics

# def plot_multilabel_metrics(history, disease_labels, save_dir):
#     # Create directory for plots
#     plots_dir = os.path.join(save_dir, 'plots')
#     os.makedirs(plots_dir, exist_ok=True)
    
#     # Plot overall loss
#     plt.figure(figsize=(10, 6))
#     for phase in ['train', 'test']:
#         if f'{phase}_loss' in history and history[f'{phase}_loss']:
#             plt.plot(history[f'{phase}_loss'], label=f'{phase}')
#     plt.title('Model Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.savefig(os.path.join(plots_dir, 'loss_plot.png'))
#     plt.close()
    
#     # Plot learning rate
#     plt.figure(figsize=(10, 6))
#     plt.plot(history['learning_rates'])
#     plt.title('Learning Rate')
#     plt.xlabel('Epoch')
#     plt.ylabel('Learning Rate')
#     plt.yscale('log')
#     plt.savefig(os.path.join(plots_dir, 'lr_plot.png'))
#     plt.close()
    
#     # Plot metrics for each disease
#     metrics = ['acc', 'auc']
#     phases = ['train', 'test']
    
#     # Plot accuracy and AUC for each disease
#     for metric in metrics:
#         plt.figure(figsize=(15, 10))
#         for i, disease in enumerate(disease_labels):
#             plt.subplot(3, 2, i+1)
#             for phase in phases:
#                 key = f'{phase}_{disease}_{metric}'
#                 if key in history and history[key]:
#                     plt.plot(history[key], label=f'{phase} {metric}')
#             plt.title(f'{disease} {metric.upper()}')
#             plt.xlabel('Epoch')
#             plt.ylabel(metric.upper())
#             plt.ylim(0, 1)
#             plt.legend()
#         plt.tight_layout()
#         plt.savefig(os.path.join(plots_dir, f'{metric}_by_disease.png'))
#         plt.close()
    
#     # Plot average metrics across diseases by phase
#     for metric in metrics:
#         plt.figure(figsize=(10, 6))
#         for phase in phases:
#             # Calculate average metric across all diseases for each epoch
#             avg_metric = []
#             for epoch in range(len(history.get(f'train_loss', []))):
#                 epoch_values = []
#                 for disease in disease_labels:
#                     key = f'{phase}_{disease}_{metric}'
#                     if key in history and len(history[key]) > epoch:
#                         epoch_values.append(history[key][epoch])
#                 if epoch_values:
#                     avg_metric.append(np.mean(epoch_values))
#             if avg_metric:
#                 plt.plot(avg_metric, label=f'{phase} avg {metric}')
#         plt.title(f'Average {metric.upper()} Across All Diseases')
#         plt.xlabel('Epoch')
#         plt.ylabel(f'Average {metric.upper()}')
#         plt.ylim(0, 1)
#         plt.legend()
#         plt.savefig(os.path.join(plots_dir, f'avg_{metric}_plot.png'))
#         plt.close()

def plot_multilabel_metrics(history, disease_labels, save_dir):
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
                f.write(f"  Accuracy:    {metrics['acc']}\n")
                f.write(f"  Sensitivity: {metrics['sens']}\n")
                f.write(f"  Specificity: {metrics['spec']}\n")
                f.write(f"   AUC:     {metrics['auc']:.5f}\n")
                f.write(f"  Confusion Matrix:\n    {metrics['cm']}\n")
    
    # Write results to CSV
    with open(os.path.join(save_dir, 'experiment_results.csv'), 'w', newline='') as csvfile:
        # Start with hyperparameters
        fieldnames = list(hyperparameters.keys())
        
        # Add metrics fields for each disease and phase
        for phase in ['train', 'test']:
            for disease in disease_labels:
                for metric in ['acc', 'sens', 'spec', 'auc']:
                    fieldnames.append(f"{phase}_{disease}_{metric}")
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Create row with all results
        row = hyperparameters.copy()
        
        # Add metrics for each disease and phase
        for phase in ['train', 'test']:
            for disease in disease_labels:
                metrics = final_metrics[phase][disease]
                for metric in ['acc', 'sens', 'spec']:
                    row[f"{phase}_{disease}_{metric}"] = metrics[metric]
                
                row[f"{phase}_{disease}_auc"] = metrics['auc']        
        writer.writerow(row)

def gradient_based_feature_importance(model, train_loader, disease_labels, device):
    """Calculate gradient-based feature importance for each disease separately"""
    model.eval()
    num_diseases = len(disease_labels)
    
    # Initialize gradient storage for each disease
    total_gradients = {disease: None for disease in disease_labels}
    total_samples = 0
    
    for batch_input, covariates, batch_targets in train_loader:
        batch_size = batch_input.size(0)
        batch_input = batch_input.to(device)
        covariates = covariates.to(device)
        batch_input.requires_grad_(True)
        
        # Forward pass
        outputs = model(batch_input, covariates)
        
        # Process each disease separately
        for disease_idx, disease in enumerate(disease_labels):
            # Zero gradients from previous iterations
            model.zero_grad()
            if batch_input.grad is not None:
                batch_input.grad.zero_()
            
            # Select output for current disease
            disease_output = outputs[:, disease_idx]
            
            # Create target (1 for calculating importance for positive class)
            target = torch.ones_like(disease_output)
            
            # Compute loss for this disease
            loss = nn.BCELoss()(disease_output, target)
            
            # Backward pass for this disease only
            loss.backward(retain_graph=(disease_idx < num_diseases-1))
            
            # Accumulate gradients for this disease
            if batch_input.grad is not None:
                gradients = batch_input.grad.detach().cpu().numpy()
                
                if total_gradients[disease] is None:
                    total_gradients[disease] = np.zeros((train_loader.batch_size, *gradients.shape[1:]))
                
                total_gradients[disease][:batch_size] += np.abs(gradients)  # Use absolute gradients
        
        total_samples += batch_size
        
        # Clear gradients
        batch_input.grad = None
    
    # Compute average gradients and feature importance for each disease
    feature_importance = {}
    top_features = {}
    
    for disease in disease_labels:
        if total_gradients[disease] is not None:
            # Average across samples
            avg_gradients = total_gradients[disease] / total_samples
            
            # Compute feature importance (average across channels)
            importance = np.mean(avg_gradients, axis=0)
            
            # Flatten to get per-position importance
            importance_flat = importance.flatten()
            
            # Store results
            feature_importance[disease] = importance_flat
            
            # Get indices of top features
            top_features[disease] = np.argsort(importance_flat)[::-1]
            
            print(f"Calculated feature importance for {disease}: {len(top_features[disease])} features ranked")
    
    return feature_importance, top_features

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
    print("Starting multilabel disease prediction model...")
    args = parse_args()

    id = str(args.ID)
    print(f"Experiment ID: {id}")
    experiment_dir = os.path.join(args.exp_dir, id)

    # Create experiment directory
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

    # Create datasets
    train_dataset = MultilabelGenotypeDataset(
        train_files, 
        phenotype_data, 
        disease_labels, 
        use_covariates=bool(args.cov),
        use_age=bool(args.use_age),
        use_gender=bool(args.use_gender),
        norm_age=args.norm_age,
        norm_pcs=args.norm_pcs,
        norm_gender=args.norm_gender,
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
        norm_age=args.norm_age,
        norm_pcs=args.norm_pcs,
        norm_gender=args.norm_gender,
        fit_normalizers=False,  # Don't fit new normalizers
        normalizers=fitted_normalizers # Use normalizers from training se
        )

    # Create dataloaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True),
        'test': DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    }
    print("DataLoaders created")

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
        num_covariates=10,
        pool_size=args.pool_size
    )

    model = model.to(device)

    # Save model architecture
    with open(os.path.join(experiment_dir, 'model_architecture.txt'), 'w') as file:
        file.write(str(model))
    print("Model created and moved to device")
    print(model)

    # Set up loss function (BCELoss for multilabel)
    criterion = nn.BCEWithLogitsLoss()

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
    
    scheduler = get_scheduler(args.sch, optimizer, args, train_files)


    print(f"Using optimizer: {args.opt}")
    print(f"Using scheduler: {args.sch}")

    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta,
        verbose=True
    )

    # Train model
    start_time = time.time()
    model, history, final_metrics, all_preds, all_labels, completed_epochs = train_multilabel_model(
        model, dataloaders, criterion, optimizer, scheduler, 
        args.epochs, disease_labels, device, early_stopping=early_stopping
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Save model
    #torch.save(model.state_dict(), os.path.join(experiment_dir, 'trained_multilabel_model.pth'))
    #print(f"Model saved to {os.path.join(experiment_dir, 'trained_multilabel_model.pth')}")

    # Plot metrics
    plot_multilabel_metrics(history, disease_labels, experiment_dir)
    print("Metrics plotted")

    # Prepare hyperparameter dict
    hyperparameters = {
        'Exp_ID': id,
        'Batch_Size': args.bs,
        'Epochs': args.epochs,
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
        'Kernel_sizes': str(args.kernel_sizes),
        'Stride': str(args.stride),
        'Conv_channels': str(args.conv_channels),
        'FC_layers': str(args.fc_layers),
        'Num_Diseases': len(disease_labels),
        'Disease_Labels': ','.join(disease_labels),
    }

    # Write results
    write_results(model, hyperparameters, final_metrics, disease_labels, experiment_dir)
    print("Results written to file")

    # # Calculate feature importance for each disease
    # print("Calculating feature importance for each disease...")
    # feature_importance, top_features = gradient_based_feature_importance(
    #     model, dataloaders['train'], disease_labels, device
    # )

    # # Save feature importance for each disease
    # importance_dir = os.path.join(experiment_dir, 'feature_importance')
    # os.makedirs(importance_dir, exist_ok=True)

    # for disease in disease_labels:
    #     if disease in top_features:
    #         # Save all feature importance values
    #         np.save(os.path.join(importance_dir, f'{disease}_feature_importance.npy'), feature_importance[disease])
            
    #         # Save top 1000 feature indices (or fewer if we have less)
    #         top_k = min(1000, len(top_features[disease]))
    #         top_indices = top_features[disease][:top_k]
            
    #         # Create DataFrame and save to CSV
    #         df = pd.DataFrame({'SNP_Index': top_indices})
    #         df.to_csv(os.path.join(importance_dir, f'{disease}_top_{top_k}_indices.csv'), index=False)
    #         print(f"Top {top_k} feature indices for {disease} saved")

    for disease in disease_labels:
        print(f"\n{disease.upper()}:")
        print(f"Train - AUC: {final_metrics['train'][disease]['auc']:.4f}, "
              f"Accuracy: {final_metrics['train'][disease]['acc']}")
        print(f"Test  - AUC: {final_metrics['test'][disease]['auc']:.4f}, "
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