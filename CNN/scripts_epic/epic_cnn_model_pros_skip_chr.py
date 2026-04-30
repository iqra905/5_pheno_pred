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
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import csv
import time

def parse_int_list(s):
    return [int(x) for x in s.split(',')]

def parse_args():
    parser = argparse.ArgumentParser(description="Genotype Model Training")
    parser.add_argument("-ID", type=str, default="1_1", help="ID of the experiment")
    parser.add_argument("-exp_dir", type=str, default='/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/pros/skip_conn_chr/', help="Directory to save experiment results")
    parser.add_argument("-genotype_dir", type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/pros_can', help="Directory containing genotype files")
    parser.add_argument("-phenotype_file", type=str, default='/vol/vssp/SF_ucdatasets/gwas/data_files/disease_pheno/pros_can.xlsx', help="Path to phenotype file")
    parser.add_argument("-snp_file", type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/meta_data/first_5_columns_pros_can.gen', help="Path to SNP information file")
    parser.add_argument("-bs", type=int, default=32, help="Batch size for training")
    parser.add_argument("-dropout", type=float, default=0.5, help="Dropout rate for the model")
    parser.add_argument("-epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("-lr", type=float, default=0.0001, help="Learning rate for optimizer")
    parser.add_argument("-act", type=str, default="relu", choices=["tanh","relu","gelu"], help="Activation function for the model")
    parser.add_argument("-opt", type=str, default="adam", choices=["adam", "adamw", "sgd"], help="Optimizer to use")
    parser.add_argument("-sch", type=str, default="cosine", choices=["none","plateau", "cosine", "step"], help="Learning rate scheduler")
    parser.add_argument("-wd", type=float, default=0.01, help="Weight decay for optimizer")

    parser.add_argument("-conv_channels", type=parse_int_list, default=[64,128,256], help="Convolution channels")
    parser.add_argument("-fc_layers", type=parse_int_list, default=[128,64], help="Fully connected layers")
    
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

def print_lr(optimizer):
    for param_group in optimizer.param_groups:
        print(f"Current Learning Rate: {param_group['lr']}")
        
class GenotypeDataset(Dataset):
    def __init__(self, file_list, phenotype_data, snp_data):
        self.file_list = file_list
        self.phenotype_data = phenotype_data
        self.snp_data = snp_data
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
        
        label = self.phenotype_data.loc[self.phenotype_data['new_order'] == sample_id, 'pros01'].values[0]
        covariates = self.phenotype_data.loc[self.phenotype_data['new_order'] == sample_id, 'PC1':'PC10'].values[0]

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

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, act):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = self.get_activation(act)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv(x)
        out = self.bn(out)
        out += residual
        return self.activation(out)

    @staticmethod
    def get_activation(name):
        if name == 'tanh':
            return nn.Tanh()
        elif name == 'relu':
            return nn.ReLU()
        elif name == 'gelu':
            return nn.GELU()
        else:
            raise NotImplementedError("Activation function not implemented.")

class ChromosomeConvSequence(nn.Module):
    def __init__(self, in_channels, conv_channels, kernel_sizes, act):
        super(ChromosomeConvSequence, self).__init__()
        self.conv1 = ResidualBlock(in_channels, conv_channels[0], kernel_sizes[0], act)
        self.conv2 = ResidualBlock(conv_channels[0], conv_channels[1], kernel_sizes[1], act)
        self.conv3 = ResidualBlock(conv_channels[1], conv_channels[2], kernel_sizes[2], act)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class GenotypeModel(nn.Module):
    def __init__(self, chr_snp_counts, conv_channels, fc_layers, act, dropout_rate, num_covariates=10):
        super(GenotypeModel, self).__init__()
        self.chr_snp_counts = chr_snp_counts
        self.input_channels = 3  # Assuming 3 channels for ref, alt, and dosage

        self.chr_convs = nn.ModuleList([
            ChromosomeConvSequence(
                self.input_channels, 
                conv_channels, 
                [snp_count, conv_channels[0], conv_channels[1]],
                act
            ) for snp_count in chr_snp_counts
        ])

        self.conv_output_size = self._get_conv_output_size(conv_channels[-1])
        self.fc_layers = self._create_fc_layers(fc_layers, dropout_rate, act, num_covariates)
        self.output_activation = nn.Sigmoid()

    def _get_conv_output_size(self, last_conv_channels):
        return sum(self.chr_snp_counts) * last_conv_channels

    def _create_fc_layers(self, fc_layers, dropout_rate, act, num_covariates):
        layers = []
        in_features = self.conv_output_size + num_covariates

        for out_features in fc_layers:
            layers.extend([
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                self.get_activation(act),
                nn.Dropout(dropout_rate)
            ])
            in_features = out_features

        layers.append(nn.Linear(in_features, 1))
        return nn.Sequential(*layers)

    def forward(self, x, covariates):
        chr_outputs = []
        start = 0
        for i, snp_count in enumerate(self.chr_snp_counts):
            chr_input = x[:, :, start:start + snp_count]
            chr_output = self.chr_convs[i](chr_input)
            chr_outputs.append(chr_output)
            start += snp_count

        x = torch.cat(chr_outputs, dim=2)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, covariates], dim=1)
        x = self.fc_layers(x)
        return self.output_activation(x).squeeze(1)

    @staticmethod
    def get_activation(name):
        if name == 'tanh':
            return nn.Tanh()
        elif name == 'relu':
            return nn.ReLU()
        elif name == 'gelu':
            return nn.GELU()
        else:
            raise NotImplementedError("Activation function not implemented.")

def load_snp_data(file_path):
    df = pd.read_csv(file_path, sep=r'\s+', header=None, names=['Chr', 'SNP_ID', 'BP', 'Ref', 'Alt'])
    chr_snp_counts = df['Chr'].value_counts().sort_index().tolist()
    return df, chr_snp_counts

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, device='cuda'):
    # print(f"Criterion is: {criterion}\n")
    # print(f"Optimizer is: {optimizer}\n")
    # print(f"Scheduler is: {scheduler.__class__.__name__}\n")
    # print(f"num_epochs is: {num_epochs}\n")
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
                                old_lr = optimizer.param_groups[0]['lr']
                                scheduler.step()
                                new_lr = optimizer.param_groups[0]['lr']
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

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(history['val_loss'][-1])
                #print(f"Scheduler stepped with ReduceLROnPlateau")
            elif not isinstance(scheduler, (WarmupExponential, ExponentialDecay)):
                scheduler.step()
                #print(f"Scheduler stepped with {scheduler}")
            #history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            print_lr(optimizer)
        
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])

    print(f'Best val Acc: {best_acc:.4f}')
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

    # Set up directories and files
    experiment_dir = os.path.join(args.exp_dir, args.ID)
    os.makedirs(experiment_dir, exist_ok=True)

    # Load data
    snp_data, chr_snp_counts = load_snp_data(args.snp_file)
    phenotype_data = pd.read_excel(args.phenotype_file)
    file_list = glob.glob(os.path.join(args.genotype_dir, "sample_*.gen.gz"))

    # Split data
    train_files, test_files = train_test_split(file_list, test_size=0.2, random_state=42)
    train_files, val_files = train_test_split(train_files, test_size=0.2, random_state=42)

    # Create datasets
    train_dataset = GenotypeDataset(train_files, phenotype_data, snp_data)
    val_dataset = GenotypeDataset(val_files, phenotype_data, snp_data)
    test_dataset = GenotypeDataset(test_files, phenotype_data, snp_data)

    #print information for a few items from each dataset
    print("Sampling a few items from each dataset:")
    for dataset_name, dataset in [("Train", train_dataset), ("Validation", val_dataset), ("Test", test_dataset)]:
        print(f"\n{dataset_name} dataset sample:")
        for i in range(3):  # Print info for 3 items from each dataset
            genotype, covariates, label = dataset[i]
            print(f"Item {i}: Genotype shape: {genotype.shape}, Covariates shape: {covariates.shape}, Label: {label}")


    # Create dataloaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=4),
        'test': DataLoader(test_dataset, batch_size=args.bs, shuffle=False, num_workers=4)
    }
    print("DataLoaders created")

    # Set up device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create model
    model = GenotypeModel(
        chr_snp_counts=chr_snp_counts,
        conv_channels=args.conv_channels,
        fc_layers=args.fc_layers,
        act=args.act,
        dropout_rate=args.dropout,
        num_covariates=10
    ).to(device)

    with open(experiment_dir + '/model_architecture.txt', 'w') as file:
        file.write(str(model))
        print(model)

    print("Model created and moved to device")

    # Set up loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = {
        "adadelta": optim.Adadelta(model.parameters(), lr=args.lr),
        "adagrad": optim.Adagrad(model.parameters(), lr=args.lr),
        "adamw": optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd),
        "rmsprop": optim.RMSprop(model.parameters(), lr=args.lr),
        "sgd": optim.SGD(model.parameters(), lr=args.lr),
        "adam": optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    }.get(args.opt, None)

    if optimizer is None:
        raise NotImplementedError("Optimizer not implemented.")

    scheduler = get_scheduler(args.sch, optimizer, args, train_files)

    # Train model
    model, history, final_metrics, all_preds, all_labels= train_model(model, dataloaders, criterion, optimizer, scheduler, args.epochs, device)

    # Save model
    #torch.save(model.state_dict(), os.path.join(experiment_dir, 'final_model.pth'))

    # Update results dictionary
    results = {
        #'Train_loss': history['train_loss'][-1],
        #'Train_acc': history['train_acc'][-1],
        'train_auc': round(history['train_auc'][-1],5),
        #'Val_loss': history['val_loss'][-1],
        #'Val_acc': history['val_acc'][-1],
        'val_auc': round(history['val_auc'][-1],5),
        #'Test_loss': history['test_loss'][-1],
        #'Test_acc': history['test_acc'][-1],
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
        'BS': args.bs,
        'Epochs': args.epochs,
        'Start_LR': args.lr,
        'Peak_LR': args.peak_lr,
        'Final_LR':optimizer.param_groups[0]["lr"],
        'Dropout': args.dropout,
        'Act': args.act,
        'Opt': args.opt,
        'Sch': args.sch,
        'WD': args.wd,
        'DF': args.df,
        'conv_channels': args.conv_channels,
        'fc_layers': args.fc_layers
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
