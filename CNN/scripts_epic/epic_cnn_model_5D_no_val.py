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
    parser = argparse.ArgumentParser(description="Genotype Model Training")
    parser.add_argument("-ID", type=str, default="000_exp", help="ID of the experiment")
    parser.add_argument("-exp_dir", type=str, default='/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/t2d/t2d_0.01/full/', help="Directory to save experiment results")
    parser.add_argument("-genotype_dir", type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/t2d_0.01', help="Directory containing genotype files")
    parser.add_argument("-phenotype_file", type=str, default='/vol/vssp/SF_ucdatasets/gwas/data_files/disease_pheno/t2d.xlsx', help="Path to phenotype file")

    parser.add_argument("-bs", type=int, default=16, help="Batch size for training")
    parser.add_argument("-dropout", type=float, default=0.5, help="Dropout rate for the model")
    parser.add_argument("-epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("-lr", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("-act", type=str, default="gelu", choices=["tanh","relu","gelu"], help="Activation function for the model")
    parser.add_argument("-opt", type=str, default="adamw", choices=["adam", "adamw", "sgd"], help="Optimizer to use")
    parser.add_argument("-sch", type=str, default="exponential_decay", choices=["none","plateau", "cosine", "step","multistep","explr","warmup_exponential", "exponential_decay"], help="Learning rate scheduler")
    parser.add_argument("-peak_lr", type=float, default=1e-2, help="Peak learning rate for WarmupExponential scheduler")
    parser.add_argument("-final_lr", type=float, default=1e-5, help="Final learning rate for custom schedulers")
    parser.add_argument("-wd", type=float, default=0.5, help="Weight decay for optimizer")
    parser.add_argument("-df", type=float, default=1, help="Decay factor for custom schedulers")

    parser.add_argument("-kernel_sizes", type=parse_int_list, default=[1024,1,1], help="Convolution Kernel Size")
    parser.add_argument("-stride", type=parse_int_list, default=[512,1,1], help="Convolution Stride")
    parser.add_argument("-conv_channels", type=parse_int_list, default=[2,4,8], help="Convolution channels")
    parser.add_argument("-fc_layers", type=parse_int_list, default=[64,128], help="Fully connected layers")

    parser.add_argument("-cov", type=int, default=1, choices=[0, 1], help="Whether to include covariates in the model (0: no, 1: yes)")
    parser.add_argument("-use_age", type=int, default=1, choices=[0, 1], help="Whether to include age in covariates in the model (0: no, 1: yes)")
    parser.add_argument("-use_gender", type=int, default=1, choices=[0, 1], help="Whether to include gender in covariates in the model (0: no, 1: yes)")
    
    parser.add_argument("-label_col", type=str, default="t2dm", help="Column name in phenotype file to use as label (e.g., 'crc')")
    # Early stopping parameters
    parser.add_argument("-patience", type=int, default=100, help="Patience for early stopping")
    parser.add_argument("-min_delta", type=float, default=1e-4, help="Minimum change for early stopping")

    # Normalization-related arguments
    parser.add_argument("-norm_age", type=str, default="none", choices=["none", "standard", "minmax", "robust", "quantile", "power"], help="Normalization method for age")
    parser.add_argument("-norm_pcs", type=str, default="none", choices=["none", "standard", "minmax", "robust", "quantile", "power"], help="Normalization method for PCs")
    parser.add_argument("-norm_gender", type=str, default="none", choices=["none", "minmax"], help="Normalization method for gender (usually keep as none)")
    parser.add_argument("-pool_size", type=int, default=64, 
                        help="Size of the adaptive pooling output (smaller values = more aggressive pooling)")
   
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

class GenotypeDataset(Dataset):
    def __init__(self, file_list, phenotype_data, label_column, use_covariates=True, use_age=True, 
                 use_gender=True, norm_age="standard", norm_pcs="standard", norm_gender="none", 
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

        # Use memory-mapped file for faster access
        mmap_file = self.file_handles[genotype_file]
        mmap_file.seek(0)
        with gzip.GzipFile(fileobj=mmap_file) as f:
            data = pd.read_csv(f, sep=r'\s+', header=None)

        # Raw genotype data - keep the format with 3 values per SNP
        genotype_tensor = torch.tensor(data.values, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        # After processing, explicitly clear references
        data = None
        torch.cuda.empty_cache()  # If using GPU tensors

        return genotype_tensor, covariates_tensor, label_tensor

    def __del__(self):
        for handle in self.file_handles.values():
            if handle is not None and hasattr(handle, 'close'):
                handle.close()
            
class GenotypeModel(nn.Module):
    def __init__(self, input_size, kernel_sizes, stride, conv_channels, fc_layers, act, dropout_rate, use_covariates=True, use_age=True, use_gender=True, num_covariates=10, pool_size=16):
        super(GenotypeModel, self).__init__()
        self.input_channels = 3
        self.use_covariates = use_covariates
        self.use_age = use_age
        self.use_gender = use_gender        
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
        
        
        
        self.fc = nn.Sequential(*fc_layers_list)
        # Add the final output layer
        self.output_layer = nn.Linear(in_features, 1, bias=False)
        #self.output_activation = nn.Sigmoid()
        print(f"GenotypeModel initialized (using covariates: {use_covariates}, age: {use_age}, gender: {use_gender})")


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
        
        x = self.fc(x)

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
 
def print_lr(optimizer):
    for param_group in optimizer.param_groups:
        print(f"Current Learning Rate: {param_group['lr']}")

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, device='cuda', early_stopping=None):
    # print(f"Criterion is: {criterion}\n")
    # print(f"Optimizer is: {optimizer}\n")
    # print(f"Scheduler is: {scheduler.__class__.__name__}\n")
    # print(f"num_epochs is: {num_epochs}\n")
    print(f"Training on device: {device}")
    
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
                inputs, covariates, labels = next(batch_iter)
                inputs = inputs.to(device, non_blocking=True)
                covariates = covariates.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
            except StopIteration:
                print(f"Warning: {phase} dataloader is empty!")
                continue

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
                running_corrects += torch.sum(preds == labels.data)
                total_samples += batch_size

                all_labels[phase].extend(labels.cpu().numpy())
                all_preds[phase].extend(probs.detach().cpu().numpy())

                # Prepare for the next iteration
                inputs, covariates, labels = next_inputs, next_covariates, next_labels
            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples
            try:
                epoch_auc = roc_auc_score(all_labels[phase], all_preds[phase])
                auc_str = f" - AUC: {epoch_auc:.4f}"
            except Exception as e:
                print(f"Warning: Could not calculate AUC for {phase} phase: {str(e)}")
                epoch_auc = 0.5  # Default value if AUC can't be calculated
                auc_str = " - AUC: N/A (need both classes)"

            print(f'{phase} Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}{auc_str}')            
            
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
                
            # Save best model
            if phase == 'test' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                
                # # Save the best model
                # if save_dir:
                #     torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
                #     print(f"Saved new best model with validation loss: {best_loss:.4f}")

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
    return model, history, compute_final_metrics(all_labels, all_preds), all_preds, all_labels, completed_epochs

def compute_final_metrics(all_labels, all_preds):
    final_metrics = {}
    for phase in ['train', 'test']:
        y_true = np.array(all_labels[phase])
        y_pred_proba = np.array(all_preds[phase])

        # Convert probabilities to binary predictions
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
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
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                roc_auc = auc(fpr, tpr)
            except Exception as e:
                print(f"Error calculating ROC curve for {phase}: {str(e)}")
                fpr, tpr = np.array([]), np.array([])
                roc_auc = 0.5
        except Exception as e:
            print(f"Error calculating metrics for {phase}: {str(e)}")
            cm = np.zeros((2, 2))
            sensitivity = 0
            specificity = 0
            fpr, tpr = np.array([]), np.array([])
            roc_auc = 0.5

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
        output = model(batch_input,covariates)

        # For binary classification with single output
        target = torch.full_like(output, float(target_class))

        # Compute loss
        loss = nn.BCELoss()(output, target)
        
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
    
    print("total_samples is:", total_samples)
    if total_gradients is not None:
        print("total_gradients shape is:", total_gradients.shape)
        # Compute average gradients
        average_gradients = total_gradients / total_samples
        return average_gradients
    else:
        print("No gradients were computed. Check if your model parameters require gradients.")
        return None



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

     # Get list of genotype files and sort numerically by sample number
    file_list = glob.glob(os.path.join(args.genotype_dir, "sample_*.gen.gz"))
    file_list.sort(key=lambda x: int(x.split('sample_')[1].split('.gen.gz')[0]))
    print(f"Number of genotype files found: {len(file_list)}")

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
        input_size = get_input_size(first_genotype_file)
        print(f"Dynamically determined input size: {input_size}")
    else:
        print("No matching genotype files found!")
        return

    if len(filtered_file_list) != len(phenotype_samples):
        raise ValueError(f"Number of files ({len(filtered_file_list)}) in {genotype_dir} does not match number of samples ({len(phenotype_samples)}) in phenotype data.")


   # Split data
    train_files, test_files = train_test_split(filtered_file_list, test_size=0.2, random_state=42)
    print(f"Data split: Train {len(train_files)}, Test {len(test_files)}")

    # Create datasets with both covariate and label column flags
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
        normalizers=None)
    
    # Get the fitted normalizers from training dataset
    fitted_normalizers = train_dataset.get_normalizers()

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
        normalizers=fitted_normalizers # Use normalizers from training se
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
        kernel_sizes=kernel_sizes,
        stride=stride,
        conv_channels=conv_channels,
        fc_layers=fc_layers,
        act=act,
        dropout_rate=dropout_rate,
        use_covariates=bool(args.cov),
        use_age=bool(args.use_age),
        use_gender=bool(args.use_gender),
        num_covariates=10,
        pool_size=args.pool_size
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
        model, dataloaders, criterion, optimizer, scheduler, num_epochs, device, early_stopping=early_stopping
    )
    
    #Plot metrics
    plot_all_metrics(history, final_metrics, experiment_dir)

    # Save the trained model
    torch.save(model.state_dict(), os.path.join(experiment_dir, 'trained_model.pth'))
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
        'Use_PCs': bool(args.cov),
        'norm_PCs': args.norm_pcs,
        'Use_Age': bool(args.use_age),
        'norm_Age': args.norm_age,
        'Use_Gender': bool(args.use_gender),
        'norm_Gender': args.norm_gender,
        'Kernel_sizes': str(args.kernel_sizes),
        'Stride': str(args.stride),
        'Conv_channels': str(args.conv_channels),
        'FC_layers': str(args.fc_layers)
    }
    # Write results
    write_results(model, hyperparameters, results, experiment_dir)
    append_metrics_to_csv(experiment_dir, results)

    print(f"\nTraining complete. Results saved to {experiment_dir}")
    
    print("\nFinal performance metrics:")
    print(f"Train accuracy: {results['train_acc']}, AUC: {results['train_auc']}")
    print(f"Test accuracy: {results['test_acc']}, AUC: {results['test_auc']}")
    
    print("\nConfusion matrices:")
    for phase in ['train', 'test']:
        print(f"{phase.capitalize()}:")
        print(final_metrics[phase]['confusion_matrix'])

    model.eval()
    # Select a target class
    #top = 23940
    target_class = 1  # Assuming you want to analyze importance for class 1 (change as needed)

    # Compute gradient-based feature importance
    grad_importance = gradient_based_feature_importance(model, dataloaders['train'], target_class, device)
    if grad_importance is not None:
        print(f"Shape of grad_importance: {grad_importance.shape}")

        # Compute the importance score for each feature
        feature_importance = np.mean(np.abs(grad_importance), axis=0)
        feature_importance = np.mean(np.abs(feature_importance), axis=0)
        print(f"Shape of feature_importance: {feature_importance.shape}")

        # Get indices of top features
        top_features_indices = np.argsort(feature_importance.flatten())[::-1]
        #top_features_indices = np.argsort(feature_importance.flatten())[-top:][::-1]
        print("Number of top features identified:", len(top_features_indices))
        print(top_features_indices)
    else:
        print("Could not compute feature importance.")
    # Create a DataFrame with the top feature indices
    df = pd.DataFrame({'SNP_Index': top_features_indices})

    # Save the DataFrame to a CSV file
    top_csv_path = os.path.join(experiment_dir,f'top_{len(top_features_indices)}_indices.csv')
    df.to_csv(top_csv_path, index=False)

    print(f"Top {len(top_features_indices)} indices saved to {experiment_dir} + top_{len(top_features_indices)}_indices.csv")


if __name__ == '__main__':
    start_time = time.time()
    
    main()
    
    end_time = time.time()
    total_runtime = end_time - start_time
    
    print(f"\nTotal script runtime: {total_runtime:.2f} seconds")
    hours, rem = divmod(total_runtime, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Total runtime: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")