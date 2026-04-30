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
from sklearn.model_selection import KFold
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
    parser = argparse.ArgumentParser(description="Genotype Model Training with Cross-Validation")
    parser.add_argument("-ID", type=str, default="Exp_CV_01", help="ID of the experiment")
    parser.add_argument("-exp_dir", type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/stat_test_exp_split/pros/mlp_res_full_cv/', help="Directory to save experiment results")
    parser.add_argument("-genotype_dir", type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/stat_test_exp_split/pros/geno_ml_filtered_full', help="Directory containing genotype files")
    parser.add_argument("-phenotype_file", type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/stat_test_exp_split/pros/pheno/pros_can.xlsx', help="Path to phenotype file")
    parser.add_argument("-bs", type=int, default=32, help="Batch size for training")
    parser.add_argument("-dropout", type=float, default=0.5, help="Dropout rate for the model")
    parser.add_argument("-epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("-lr", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("-act", type=str, default="gelu", choices=["tanh","relu","gelu"], help="Activation function")
    parser.add_argument("-opt", type=str, default="adamw", choices=["adam", "adamw", "sgd"], help="Optimizer to use")
    parser.add_argument("-sch", type=str, default="warmup_exponential", 
                       choices=["none","plateau", "cosine", "step","multistep","explr","warmup_exponential", "exponential_decay"], 
                       help="Learning rate scheduler")
    parser.add_argument("-peak_lr", type=float, default=1e-2, help="Peak learning rate for WarmupExponential scheduler")
    parser.add_argument("-final_lr", type=float, default=1e-5, help="Final learning rate for custom schedulers")
    parser.add_argument("-wd", type=float, default=0.5, help="Weight decay for optimizer")
    parser.add_argument("-df", type=float, default=0.1, help="Decay factor for custom schedulers")
    parser.add_argument("-hidden_sizes", type=parse_int_list, default=[128,128,128], help="Hidden layer sizes for MLP")
    parser.add_argument("-cov", type=int, default=1, choices=[0, 1], help="Whether to include covariates")
    parser.add_argument("-label_col", type=str, default="pros01", help="Column name in phenotype file to use as label")
    parser.add_argument("-n_folds", type=int, default=5, help="Number of folds for cross-validation")
    return parser.parse_args()

# def parse_args():
#     parser = argparse.ArgumentParser(description="Genotype Model Training")
#     parser.add_argument("-ID", type=str, default="Exp_CV_01", help="ID of the experiment")
#     parser.add_argument("-exp_dir", type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/stat_test_exp_split/t2d/mlp_res_full_cv/', help="Directory to save experiment results")
#     parser.add_argument("-genotype_dir", type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/stat_test_exp_split/t2d/geno_ml_filtered_full', help="Directory containing genotype files")
#     parser.add_argument("-phenotype_file", type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/stat_test_exp_split/t2d/pheno/t2d.xlsx', help="Path to phenotype file")

#     parser.add_argument("-bs", type=int, default=64, help="Batch size for training")
#     parser.add_argument("-dropout", type=float, default=0.5, help="Dropout rate for the model")

#     parser.add_argument("-epochs", type=int, default=100, help="Number of epochs for training")
#     parser.add_argument("-lr", type=float, default=0.001, help="Learning rate for optimizer")
#     parser.add_argument("-act", type=str, default="tanh", choices=["tanh","relu","gelu"], help="Dropout rate for the model")
#     parser.add_argument("-opt", type=str, default="adamw", choices=["adam", "adamw", "sgd"], help="Optimizer to use")
#     parser.add_argument("-sch", type=str, default="warmup_exponential", choices=["none","plateau", "cosine", "step","multistep","explr","warmup_exponential", "exponential_decay"], help="Learning rate scheduler")
#     parser.add_argument("-peak_lr", type=float, default=1e-2, help="Peak learning rate for WarmupExponential scheduler")
#     parser.add_argument("-final_lr", type=float, default=1e-5, help="Final learning rate for custom schedulers")
#     parser.add_argument("-wd", type=float, default=0.5, help="Weight decay for optimizer")
#     parser.add_argument("-df", type=float, default=0.1, help="Decay factor for custom schedulers")

#     parser.add_argument("-hidden_sizes", type=parse_int_list, default=[256,64,32], help="Hidden layer sizes for MLP")

#     parser.add_argument("-cov", type=int, default=1, choices=[0, 1], help="Whether to include covariates in the model (0: no, 1: yes)")
#     parser.add_argument("-label_col", type=str, default="t2dm", help="Column name in phenotype file to use as label (e.g., 'pan01', etc.)")
#     parser.add_argument("-n_folds", type=int, default=5, help="Number of folds for cross-validation")
#     return parser.parse_args()

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
    def __init__(self, file_list, phenotype_data, label_column, use_covariates=True):
        self.file_list = file_list
        self.phenotype_data = phenotype_data
        self.label_column = label_column
        self.use_covariates = use_covariates
        
        if self.label_column not in self.phenotype_data.columns:
            raise ValueError(f"Label column '{self.label_column}' not found in phenotype data. "
                           f"Available columns are: {', '.join(self.phenotype_data.columns)}")
            
        print(f"GenotypeDataset initialized with {len(file_list)} files")
        print(f"Using label column: {label_column}")
        print(f"Using covariates: {use_covariates}")
        
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
        
        label = self.phenotype_data.loc[self.phenotype_data['new_order'] == sample_id, self.label_column].values[0]
        
        if self.use_covariates:
            covariates = self.phenotype_data.loc[self.phenotype_data['new_order'] == sample_id, 'PC1':'PC10'].values[0]
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
    def __init__(self, input_size, hidden_sizes, dropout_rate, act, use_covariates=True, num_covariates=10):
        super(GenotypeModel, self).__init__()
        self.use_covariates = use_covariates
        
        self.pointwise_conv = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1)
        
        layer_sizes = [input_size + (num_covariates if use_covariates else 0)] + hidden_sizes + [1]
        print(f"MLP layer sizes: {layer_sizes}")

        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append((f'linear_{i}', nn.Linear(layer_sizes[i], layer_sizes[i+1])))
            if i < len(layer_sizes) - 2:
                layers.append((f'batchnorm_{i}', nn.BatchNorm1d(layer_sizes[i+1])))
                layers.append((f'activation_{i}', self.get_activation(act)))
                layers.append((f'dropout_{i}', nn.Dropout(dropout_rate)))
        
        self.mlp = nn.Sequential(OrderedDict(layers))
        self.output_activation = nn.Sigmoid()
        print(f"GenotypeModel initialized (using covariates: {use_covariates})")

    def forward(self, x, covariates=None):
        x = self.pointwise_conv(x).squeeze(1)
        if self.use_covariates and covariates is not None:
            x = torch.cat([x, covariates], dim=1)
        x = self.mlp(x)
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

def custom_lrp(model, data, covariates, epsilon=1e-9):
    model.train()
    data = data.requires_grad_()
    if covariates is not None:
        covariates = covariates.requires_grad_()
    
    outputs = model(data, covariates)
    model.zero_grad()
    outputs.sum().backward()
    
    input_grad = data.grad
    if input_grad is None:
        return torch.zeros_like(data[0, 0]).cpu().numpy()
    
    importance = (data * input_grad).sum(dim=(0, 1)).abs().detach().cpu().numpy()
    importance = importance / (importance.sum() + epsilon)
    
    return importance

def get_scheduler(scheduler_name, optimizer, args, train_files):
    steps_per_epoch = len(train_files) // args.bs
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * 0.1)

    schedulers = {
        "none": lambda: optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1),
        "warmup_exponential": lambda: WarmupExponential(
            optimizer, args.lr, args.peak_lr, args.final_lr, 
            warmup_steps, total_steps, args.df
        ),
        "exponential_decay": lambda: ExponentialDecay(
            optimizer, args.lr, args.final_lr, total_steps, args.df
        ),
        "plateau": lambda: ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.1),
        "cosine": lambda: CosineAnnealingLR(optimizer, T_max=args.epochs),
        "step": lambda: StepLR(optimizer, step_size=30, gamma=0.1)
    }
    return schedulers.get(scheduler_name, schedulers["none"])()

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, device):
    print(f"Training on device: {device}")
    
    scaler = GradScaler()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

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
            
            stream = torch.cuda.Stream()
            batch_iter = iter(dataloaders[phase])
            inputs, covariates, labels = next(batch_iter)
            inputs = inputs.to(device, non_blocking=True)
            covariates = covariates.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            for i in range(len(dataloaders[phase])):
                if i + 1 < len(dataloaders[phase]):
                    with torch.cuda.stream(stream):
                        next_inputs, next_covariates, next_labels = next(batch_iter)
                        next_inputs = next_inputs.to(device, non_blocking=True)
                        next_covariates = next_covariates.to(device, non_blocking=True)
                        next_labels = next_labels.to(device, non_blocking=True)

                torch.cuda.current_stream().wait_stream(stream)

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

                            if isinstance(scheduler, (WarmupExponential, ExponentialDecay)):
                                scheduler.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Store predictions and labels for metrics
                all_preds[phase].extend(outputs.detach().cpu().numpy())
                all_labels[phase].extend(labels.cpu().numpy())

                inputs, covariates, labels = next_inputs, next_covariates, next_labels

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            epoch_auc = roc_auc_score(all_labels[phase], all_preds[phase])

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} AUC: {epoch_auc:.4f}')

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())
            history[f'{phase}_auc'].append(epoch_auc)

            # if phase == 'test' and epoch_acc > best_acc:
            #     best_acc = epoch_acc
            #     best_model_wts = copy.deepcopy(model.state_dict())

        if scheduler is not None and not isinstance(scheduler, (WarmupExponential, ExponentialDecay)):
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(history['test_loss'][-1])
            else:
                scheduler.step()

        history['learning_rates'].append(optimizer.param_groups[0]['lr'])

    #model.load_state_dict(best_model_wts)
    
    # Calculate final metrics
    final_metrics = {}
    for phase in ['train', 'test']:
        y_true = np.array(all_labels[phase])
        y_pred = np.array(all_preds[phase])
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred.round())
        tn, fp, fn, tp = cm.ravel()
        
        # Compute sensitivity and specificity
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Compute ROC curve and AUC
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

    return model, history, final_metrics, all_preds, all_labels

def plot_all_metrics(history, final_metrics, save_dir):
    """Plot training metrics and ROC curves"""
    # Training metrics plot
    fig, axs = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('Model Performance Metrics', fontsize=16)
    
    metrics = ['loss', 'acc', 'auc']
    phases = ['train', 'test']
    
    for i, metric in enumerate(metrics):
        for phase in phases:
            axs[i//2, i%2].plot(history[f'{phase}_{metric}'], label=f'{phase}')
        axs[i//2, i%2].set_title(f'{metric.capitalize()}')
        axs[i//2, i%2].set_xlabel('Epoch')
        axs[i//2, i%2].set_ylabel(metric.capitalize())
        axs[i//2, i%2].legend()
    
    # Learning rate subplot
    axs[1, 1].plot(history['learning_rates'])
    axs[1, 1].set_title('Learning Rate')
    axs[1, 1].set_xlabel('Step')
    axs[1, 1].set_ylabel('Learning Rate')
    axs[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_plot.png'))
    plt.close()
    
    # # ROC curve plot
    # plt.figure(figsize=(10, 8))
    # for phase in phases:
    #     plt.plot(final_metrics[phase]['fpr'], 
    #             final_metrics[phase]['tpr'],
    #             label=f'{phase} (AUC = {final_metrics[phase]["roc_auc"]:.3f})')
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curves')
    # plt.legend()
    # plt.savefig(os.path.join(save_dir, 'roc_curves.png'))
    # plt.close()

def write_fold_results(args, metrics, history, fold_dir):
    """Write detailed results for a fold"""
    # Save configuration and results
    config = {
        'Batch Size': args.bs,
        'Learning Rate': args.lr,
        'Peak LR': args.peak_lr,
        'Final LR': args.final_lr,
        'Weight Decay': args.wd,
        'Dropout': args.dropout,
        'Activation': args.act,
        'Optimizer': args.opt,
        'Scheduler': args.sch,
        'Hidden Sizes': args.hidden_sizes,
        'Use Covariates': bool(args.cov)
    }
    
    results = {
        'Final Train Accuracy': f"{history['train_acc'][-1]:.4f}",
        'Final Train AUC': f"{history['train_auc'][-1]:.4f}",
        'Final Test Accuracy': f"{history['test_acc'][-1]:.4f}",
        'Final Test AUC': f"{history['test_auc'][-1]:.4f}",
        'Train Sensitivity': metrics['train']['sensitivity'],
        'Train Specificity': metrics['train']['specificity'],
        'Test Sensitivity': metrics['test']['sensitivity'],
        'Test Specificity': metrics['test']['specificity']
    }
    
    # Write text results
    with open(os.path.join(fold_dir, 'results.txt'), 'w') as f:
        f.write("Configuration:\n")
        f.write("-------------\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\nResults:\n")
        f.write("--------\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\nConfusion Matrices:\n")
        f.write("------------------\n")
        for phase in ['train', 'test']:
            f.write(f"\n{phase.capitalize()} Confusion Matrix:\n")
            f.write(str(metrics[phase]['confusion_matrix']))
            f.write("\n")
    
    # Write CSV results
    with open(os.path.join(fold_dir, 'results.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        for key, value in {**config, **results}.items():
            writer.writerow([key, value])

def run_cv_fold(fold, train_files, val_files, phenotype_data, args, device):
    """Run a single fold of cross-validation"""
    print(f"\nTraining Fold {fold+1}/{args.n_folds}")
    
    # Create fold directory
    fold_dir = os.path.join(args.exp_dir, args.ID, f'fold_{fold+1}')
    os.makedirs(fold_dir, exist_ok=True)
    
    # Create datasets
    train_dataset = GenotypeDataset(
        train_files, phenotype_data,
        label_column=args.label_col,
        use_covariates=bool(args.cov)
    )
    val_dataset = GenotypeDataset(
        val_files, phenotype_data,
        label_column=args.label_col,
        use_covariates=bool(args.cov)
    )
    
    # Create dataloaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=args.bs, shuffle=True, 
                          num_workers=4, pin_memory=True, prefetch_factor=2),
        'test': DataLoader(val_dataset, batch_size=args.bs, shuffle=False, 
                         num_workers=4, pin_memory=True, prefetch_factor=2)
    }
    
    # Initialize model
    model = GenotypeModel(
        input_size=args.input_size,
        hidden_sizes=args.hidden_sizes,
        dropout_rate=args.dropout,
        act=args.act,
        use_covariates=bool(args.cov)
    ).to(device)
    
    # Save model architecture
    with open(os.path.join(fold_dir, 'model_architecture.txt'), 'w') as f:
        f.write(str(model))
    
    # Initialize optimizer and scheduler
    optimizer = {
        "adamw": optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd),
        "adam": optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd),
        "sgd": optim.SGD(model.parameters(), lr=args.lr)
    }[args.opt]
    
    scheduler = get_scheduler(args.sch, optimizer, args, train_files)
    criterion = nn.BCEWithLogitsLoss()
    
    # Train model
    model, history, metrics, all_preds, all_labels = train_model(
        model, dataloaders, criterion, optimizer, scheduler, 
        args.epochs, device
    )
    
    # Save model
    #torch.save(model.state_dict(), os.path.join(fold_dir, 'model.pth'))
    
    # Plot metrics
    plot_all_metrics(history, metrics, fold_dir)
    
    # Save results
    write_fold_results(args, metrics, history, fold_dir)
    
    # Perform LRP analysis
    snp_importance = custom_lrp(model, 
                              next(iter(dataloaders['test']))[0].to(device),
                              next(iter(dataloaders['test']))[1].to(device))
    np.save(os.path.join(fold_dir, 'snp_importance.npy'), snp_importance)
    
    return {
        'metrics': metrics,
        'history': history,
        'snp_importance': snp_importance,
        'all_preds': all_preds,
        'all_labels': all_labels
    }

def aggregate_cv_results(fold_results, save_dir):
    """Aggregate results across all folds"""
    metrics = {
        'train_acc': [res['history']['train_acc'][-1] for res in fold_results],
        'train_auc': [res['history']['train_auc'][-1] for res in fold_results],
        'test_acc': [res['history']['test_acc'][-1] for res in fold_results],
        'test_auc': [res['history']['test_auc'][-1] for res in fold_results],
        'train_sensitivity': [float(res['metrics']['train']['sensitivity']) for res in fold_results],
        'train_specificity': [float(res['metrics']['train']['specificity']) for res in fold_results],
        'test_sensitivity': [float(res['metrics']['test']['sensitivity']) for res in fold_results],
        'test_specificity': [float(res['metrics']['test']['specificity']) for res in fold_results]
    }
    
    # Calculate means and standard deviations
    results = {}
    for metric, values in metrics.items():
        results[f'{metric}_mean'] = np.mean(values)
        results[f'{metric}_std'] = np.std(values)
    
    # Save aggregated results
    with open(os.path.join(save_dir, 'cv_results.txt'), 'w') as f:
        f.write("Cross-Validation Results\n")
        f.write("=======================\n\n")
        for metric, value in results.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    # Save as CSV
    with open(os.path.join(save_dir, 'cv_results.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Mean', 'Std'])
        for metric in metrics.keys():
            writer.writerow([
                metric,
                f"{results[f'{metric}_mean']:.4f}",
                f"{results[f'{metric}_std']:.4f}"
            ])
    
    return results

def main():
    args = parse_args()
    print("Starting experiment:", args.ID)
    
    # Create experiment directory
    experiment_dir = os.path.join(args.exp_dir, args.ID)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    phenotype_data = pd.read_excel(args.phenotype_file)
    file_list = glob.glob(os.path.join(args.genotype_dir, "sample_*.gen.gz"))
    file_list.sort(key=lambda x: int(x.split('sample_')[1].split('.gen.gz')[0]))
    
    print(f"Found {len(file_list)} genotype files")
    
    # Get input size from first file
    args.input_size = get_input_size(file_list[0])
    print(f"Input size: {args.input_size}")
    
    # Save experiment configuration
    with open(os.path.join(experiment_dir, 'config.txt'), 'w') as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
    
    fold_results = []
    
    # Run for each fold using predefined indices
    for fold in range(args.n_folds):
        # Load predefined train and test indices
        train_indices = np.load(os.path.join(args.exp_dir, f'fold{fold+1}/train_indices.npy'))
        test_indices = np.load(os.path.join(args.exp_dir, f'fold{fold+1}/test_indices.npy'))
        
        # Get corresponding files
        train_files = [file_list[i] for i in train_indices]
        val_files = [file_list[i] for i in test_indices]
        
        fold_result = run_cv_fold(
            fold=fold,
            train_files=train_files,
            val_files=val_files,
            phenotype_data=phenotype_data,
            args=args,
            device=device
        )
        fold_results.append(fold_result)
        
        print(f"\nCompleted fold {fold+1}/{args.n_folds}")
        print(f"Train ACC: {fold_result['history']['train_acc'][-1]:.4f}")
        print(f"Train AUC: {fold_result['history']['train_auc'][-1]:.4f}")
        print(f"Test ACC: {fold_result['history']['test_acc'][-1]:.4f}")
        print(f"Test AUC: {fold_result['history']['test_auc'][-1]:.4f}")
    
    # Aggregate results
    print("\nAggregating cross-validation results...")
    cv_results = aggregate_cv_results(fold_results, experiment_dir)
    
    # Calculate and save mean SNP importance
    all_snp_importance = np.array([res['snp_importance'] for res in fold_results])
    mean_snp_importance = np.mean(all_snp_importance, axis=0)
    std_snp_importance = np.std(all_snp_importance, axis=0)
    
    np.save(os.path.join(experiment_dir, 'mean_snp_importance.npy'), mean_snp_importance)
    np.save(os.path.join(experiment_dir, 'std_snp_importance.npy'), std_snp_importance)
    
    # Plot average SNP importance
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(mean_snp_importance)), mean_snp_importance)
    plt.fill_between(range(len(mean_snp_importance)), 
                    mean_snp_importance - std_snp_importance,
                    mean_snp_importance + std_snp_importance,
                    alpha=0.2)
    plt.title('Average SNP Importance Across Folds')
    plt.xlabel('SNP Index')
    plt.ylabel('Importance')
    plt.savefig(os.path.join(experiment_dir, 'average_snp_importance.png'))
    plt.close()
    
    # Print final results
    print("\nFinal Cross-Validation Results:")
    print(f"Average Train Accuracy: {cv_results['train_acc_mean']:.4f} ± {cv_results['train_acc_std']:.4f}")
    print(f"Average Train AUC: {cv_results['train_auc_mean']:.4f} ± {cv_results['train_auc_std']:.4f}")
    print(f"Average Test Accuracy: {cv_results['test_acc_mean']:.4f} ± {cv_results['test_acc_std']:.4f}")
    print(f"Average Test AUC: {cv_results['test_auc_mean']:.4f} ± {cv_results['test_auc_std']:.4f}")
    
    return cv_results

if __name__ == '__main__':
    start_time = time.time()
    
    try:
        results = main()
        end_time = time.time()
        runtime = end_time - start_time
        
        print("\nExperiment completed successfully!")
        print(f"Total runtime: {time.strftime('%H:%M:%S', time.gmtime(runtime))}")
        
    except Exception as e:
        print(f"\nError occurred during experiment: {str(e)}")
        raise