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
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR
import argparse
import csv
import time

def parse_int_list(s):
    return [int(x) for x in s.split(',')]

def parse_args():
    parser = argparse.ArgumentParser(description="Genotype Model Training for Regression")
    parser.add_argument("-ID", type=str, default="exp_final", help="ID of the experiment")
    parser.add_argument("-exp_dir", type=str, default='/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/exp_residuals/results/pros/full/', help="Directory to save experiment results")
    parser.add_argument("-genotype_dir", type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/pros_can_pruned', help="Directory containing genotype files")
    parser.add_argument("-phenotype_file", type=str, default='/vol/vssp/SF_ucdatasets/gwas/data_files/disease_pheno/pros_can_res.xlsx', help="Path to phenotype file")

    parser.add_argument("-bs", type=int, default=32, help="Batch size for training")
    parser.add_argument("-dropout", type=float, default=0.5, help="Dropout rate for the model")

    parser.add_argument("-epochs", type=int, default=2, help="Number of epochs for training")
    parser.add_argument("-lr", type=float, default=0.0001, help="Learning rate for optimizer")
    parser.add_argument("-act", type=str, default="relu", choices=["tanh","relu","gelu"], help="Dropout rate for the model")
    parser.add_argument("-opt", type=str, default="adamw", choices=["adam", "adamw", "sgd"], help="Optimizer to use")
    parser.add_argument("-sch", type=str, default="warmup_exponential", choices=["none","plateau", "cosine", "step","multistep","explr","warmup_exponential", "exponential_decay"], help="Learning rate scheduler")
    parser.add_argument("-peak_lr", type=float, default=1e-2, help="Peak learning rate for WarmupExponential scheduler")
    parser.add_argument("-final_lr", type=float, default=1e-5, help="Final learning rate for custom schedulers")
    parser.add_argument("-wd", type=float, default=0.5, help="Weight decay for optimizer")
    parser.add_argument("-df", type=float, default=1, help="Decay factor for custom schedulers")

    parser.add_argument("-kernel_sizes", type=parse_int_list, default=[4096,1,1], help="Convolution Kernel Size")
    parser.add_argument("-stride", type=parse_int_list, default=[2048,1,1], help="Convolution Stride")
    parser.add_argument("-conv_channels", type=parse_int_list, default=[64,128,256], help="Convolution channels")
    parser.add_argument("-fc_layers", type=parse_int_list, default=[64,128], help="Fully connected layers")

    parser.add_argument("-label_col", type=str, default="pros01_res", help="Column name in phenotype file to use as label (e.g., 'pros01')")
    
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
    def __init__(self, file_list, phenotype_data, label_column):
        self.file_list = file_list
        self.phenotype_data = phenotype_data
        self.label_column = label_column

        # Verify that the label column exists in the phenotype data
        if self.label_column not in self.phenotype_data.columns:
            raise ValueError(f"Label column '{self.label_column}' not found in phenotype data. "
                           f"Available columns are: {', '.join(self.phenotype_data.columns)}")

        print(f"GenotypeDataset initialized with {len(file_list)} files")
        print(f"Using label column: {label_column}")
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

        # Use memory-mapped file for faster access
        mmap_file = self.file_handles[genotype_file]
        mmap_file.seek(0)
        with gzip.GzipFile(fileobj=mmap_file) as f:
            data = pd.read_csv(f, sep=r'\s+', header=None)

        genotype_tensor = torch.tensor(data.values.T, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return genotype_tensor, label_tensor

    def __del__(self):
        for handle in self.file_handles.values():
            handle.close()
            
class GenotypeModel(nn.Module):
    def __init__(self, input_size, kernel_sizes, stride, conv_channels, fc_layers, act, dropout_rate):
        super(GenotypeModel, self).__init__()
        self.input_channels = 3

        self.conv_layers = self._create_conv_layers(conv_channels, kernel_sizes, stride, dropout_rate, act)

        self.conv_output_size = self._get_conv_output_size(input_size)
        print(f"Convolutional output size: {self.conv_output_size}")
        
        fc_layers_list = []
        in_features = self.conv_output_size
        
        for i, out_features in enumerate(fc_layers):
            fc_layers_list.extend([
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features),
                self.get_activation(act),
                nn.Dropout(dropout_rate)
            ])
            in_features = out_features
        
        # Add the final output layer
        fc_layers_list.append(nn.Linear(in_features, 1, bias=False))
        
        self.fc = nn.Sequential(*fc_layers_list)
        print(f"GenotypeModel initialized")

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

    def _get_conv_output_size(self, input_size):
        x = torch.randn(1, 3, input_size, dtype=torch.float32)
        x = self.conv_layers(x)  
        return x.numel() // x.size(0)

    def forward(self, x):
        x = self.conv_layers(x)  
        
        x = x.view(x.size(0), -1)       
        
        x = self.fc(x)
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
    # print(f"Criterion is: {criterion}\n")
    # print(f"Optimizer is: {optimizer}\n")
    # print(f"Scheduler is: {scheduler.__class__.__name__}\n")
    # print(f"num_epochs is: {num_epochs}\n")
    print(f"Training on device: {device}")
    
    scaler = GradScaler()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_r2 = -float('inf')

    history = {
        'train_loss': [], 'train_r2': [], 'train_mse': [], 'train_mae': [],
        'val_loss': [], 'val_r2': [], 'val_mse': [], 'val_mae': [],
        'test_loss': [], 'test_r2': [], 'test_mse': [], 'test_mae': [],
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
            total_samples = 0

            # Create a CUDA stream for asynchronous data transfer
            stream = torch.cuda.Stream()

            # Prefetch the first batch
            batch_iter = iter(dataloaders[phase])
            inputs, labels = next(batch_iter)
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            for i in range(len(dataloaders[phase])):
                # Asynchronously prefetch the next batch
                if i + 1 < len(dataloaders[phase]):
                    with torch.cuda.stream(stream):
                        next_inputs, next_labels = next(batch_iter)
                        next_inputs = next_inputs.to(device, non_blocking=True)
                        next_labels = next_labels.to(device, non_blocking=True)

                # Wait for the current batch to be ready
                torch.cuda.current_stream().wait_stream(stream)

                # Process the current batch
                optimizer.zero_grad()
                with autocast():
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
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
                total_samples += inputs.size(0)

                all_labels[phase].extend(labels.cpu().numpy())
                all_preds[phase].extend(outputs.detach().cpu().numpy())

                # Prepare for the next iteration
                inputs, labels = next_inputs, next_labels

            epoch_loss = running_loss / total_samples
            epoch_r2 = r2_score(all_labels[phase], all_preds[phase])
            epoch_mse = mean_squared_error(all_labels[phase], all_preds[phase])
            epoch_mae = mean_absolute_error(all_labels[phase], all_preds[phase])

            print(f'{phase} Loss: {epoch_loss:.4f} - R2: {epoch_r2:.4f}  - MSE: {epoch_mse:.4f} - MAE: {epoch_mae:.4f}')

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_r2'].append(epoch_r2)
            history[f'{phase}_mse'].append(epoch_mse)
            history[f'{phase}_mae'].append(epoch_mae)

            if phase == 'val' and epoch_r2 > best_r2:
                best_r2 = epoch_r2
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

    print(f'Best val R2: {best_r2:.4f}')
    model.load_state_dict(best_model_wts)

    # Compute final metrics for each phase
    final_metrics = {}
    for phase in ['train', 'val', 'test']:
        y_true = np.array(all_labels[phase])
        y_pred = np.array(all_preds[phase])

        final_metrics[phase] = {
            'r2': r2_score(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'true_values': y_true,
            'predicted_values': y_pred
            }

    return model, history, final_metrics, all_preds, all_labels

def plot_all_metrics(history, final_metrics, save_dir):
    phases = ['train', 'val', 'test']
    metrics = ['loss', 'r2', 'mse','mae']
    
    fig, axs = plt.subplots(2, 3, figsize=(20, 15))
    fig.suptitle('Model Performance Metrics', fontsize=16)
    
    for i, metric in enumerate(metrics):
        for phase in phases:
            axs[i//3, i%3].plot(history[f'{phase}_{metric}'], label=f'{phase}')
        axs[i//3, i%3].set_title(f'{metric.upper()}')
        axs[i//3, i%3].set_xlabel('Epoch')
        axs[i//3, i%3].set_ylabel(metric.upper())
        axs[i//3, i%3].legend()
    
    # Add learning rate subplot
    axs[1,2].plot(history['learning_rates'])
    axs[1,2].set_title('Learning Rate')
    axs[1,2].set_xlabel('Step')
    axs[1,2].set_ylabel('Learning Rate')
    axs[1,2].set_yscale('log')  # Use log scale for better visualization
    
    # Remove unused subplots
    fig.delaxes(axs[1, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_plot.png'))
    plt.close()

    # Plot actual vs predicted values
    plt.figure(figsize=(10, 10))
    plt.scatter(final_metrics['test']['true_values'], final_metrics['test']['predicted_values'], alpha=0.5)
    plt.plot([min(final_metrics['test']['true_values']), max(final_metrics['test']['true_values'])], 
             [min(final_metrics['test']['true_values']), max(final_metrics['test']['true_values'])], 
             'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values (Test Set)')
    plt.savefig(os.path.join(save_dir, 'actual_vs_predicted.png'))
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

def gradient_based_feature_importance(model, train_loader, target_class, device):
    model.eval()
    total_gradients = None
    total_samples = 0
    
    for batch_input, batch_target in train_loader:
        batch_size = batch_input.size(0)
        batch_input = batch_input.to(device)
        batch_input.requires_grad_(True)
        
        # Forward pass
        output = model(batch_input)

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

    # Create datasets with label column flags
    train_dataset = GenotypeDataset(train_files, phenotype_data, label_column=args.label_col)
    val_dataset = GenotypeDataset(val_files, phenotype_data, label_column=args.label_col)
    test_dataset = GenotypeDataset(test_files, phenotype_data, label_column=args.label_col)

    #print information for a few items from each dataset
    print("Sampling a few items from each dataset:")
    for dataset_name, dataset in [("Train", train_dataset), ("Validation", val_dataset), ("Test", test_dataset)]:
        print(f"\n{dataset_name} dataset sample:")
        for i in range(3):  # Print info for 3 items from each dataset
            genotype, label = dataset[i]
            print(f"Item {i}: Genotype shape: {genotype.shape}, Label: {label}")

    dataloaders = {
    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2),
    'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2),
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
        dropout_rate=dropout_rate
    )
    model = model.to(device)

    with open(experiment_dir + '/model_architecture.txt', 'w') as file:
        file.write(str(model))
        print(model)

    print("Model created and moved to device")

    # Set up loss function and optimizer
    criterion = nn.MSELoss()
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
        'train_r2': round(final_metrics['train']['r2'], 5),
        'train_mse': round(final_metrics['train']['mse'], 5),
        'train_mae': round(final_metrics['train']['mae'], 5),
        'val_r2': round(final_metrics['val']['r2'], 5),
        'val_mse': round(final_metrics['val']['mse'], 5),
        'val_mae': round(final_metrics['val']['mae'], 5),
        'test_r2': round(final_metrics['test']['r2'], 5),
        'test_mse': round(final_metrics['test']['mse'], 5),
        'test_mae': round(final_metrics['test']['mae'], 5)
    }
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
        'Kernel_sizes':kernel_sizes,
        'Stride':stride,
        'conv_channels': conv_channels,
        'fc_layers': fc_layers
    }
    # Write results
    write_results(model, hyperparameters, results, experiment_dir)
    append_metrics_to_csv(experiment_dir, results)

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
