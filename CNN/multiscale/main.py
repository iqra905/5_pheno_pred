# multiscale/main.py

import os
import time
import glob
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Local imports
from config.args import parse_args, validate_hardcoded_parameters
from data.dataset import MultilabelGenotypeDataset
from data.preprocessing import get_input_size, clean_phenotype_data
from models.base_model import MultilabelGenotypeModel
from training.trainer import Trainer
from training.schedulers import get_scheduler
from training.early_stopping import EarlyStopping
from utils.checkpointing import find_latest_checkpoint, write_results, get_hyperparameters_dict
from utils.plotting import plot_multilabel_metrics
from utils.metrics import print_final_results


def setup_environment():
    """Setup random seeds and CUDA settings for reproducibility"""
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_and_prepare_data(args):
    """Load and prepare phenotype and genotype data"""
    print("Loading and preparing data...")
    
    # Load phenotype data
    phenotype_data = pd.read_excel(args.phenotype_file)
    print(f"Phenotype data loaded, shape: {phenotype_data.shape}")
    
    # Clean phenotype data
    phenotype_data = clean_phenotype_data(phenotype_data)
    
    # Load genotype file list
    file_list = glob.glob(os.path.join(args.genotype_dir, "sample_*.npy"))
    file_list.sort(key=lambda x: int(x.split('sample_')[1].split('.npy')[0]))
    print(f"Number of genotype files found: {len(file_list)}")
    
    # Filter files to match phenotype data
    phenotype_samples = set(phenotype_data['new_order'].unique())
    print(f"Number of unique samples in phenotype data: {len(phenotype_samples)}")
    
    filtered_file_list = []
    for file_path in file_list:
        sample_id = int(file_path.split('sample_')[1].split('.npy')[0])
        if sample_id in phenotype_samples:
            filtered_file_list.append(file_path)
    
    print(f"Number of filtered genotype files matching phenotype data: {len(filtered_file_list)}")
    
    if len(filtered_file_list) == 0:
        raise ValueError("No matching genotype files found!")
    
    # Get input size
    input_size = get_input_size(filtered_file_list[0])
    print(f"Dynamically determined input size: {input_size}")
    
    # Split data
    train_files, test_files = train_test_split(filtered_file_list, test_size=0.2, random_state=42)
    print(f"Data split: Train {len(train_files)}, Test {len(test_files)}")
    
    return phenotype_data, train_files, test_files, input_size


def create_datasets(train_files, test_files, phenotype_data, disease_labels, args):
    """Create training and testing datasets"""
    print("Creating datasets...")
    
    # Create training dataset with fitted normalizers
    train_dataset = MultilabelGenotypeDataset(
        train_files, 
        phenotype_data, 
        disease_labels, 
        use_covariates=bool(args.cov),
        use_age=bool(args.use_age),
        use_gender=bool(args.use_gender),
        use_bmi=bool(args.use_bmi),
        norm_age=args.norm_age,
        norm_pcs=args.norm_pcs,
        norm_gender=args.norm_gender,
        norm_bmi=args.norm_bmi,
        fit_normalizers=True,
        normalizers=None
    )
    
    # Get fitted normalizers and create test dataset
    fitted_normalizers = train_dataset.get_normalizers()
    
    test_dataset = MultilabelGenotypeDataset(
        test_files,
        phenotype_data, 
        disease_labels, 
        use_covariates=bool(args.cov),
        use_age=bool(args.use_age),
        use_gender=bool(args.use_gender),
        use_bmi=bool(args.use_bmi),
        norm_age=args.norm_age,
        norm_pcs=args.norm_pcs,
        norm_gender=args.norm_gender,
        norm_bmi=args.norm_bmi,
        fit_normalizers=False,
        normalizers=fitted_normalizers
    )
    
    # Create data loaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=args.bs, shuffle=True, 
                           num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True),
        'test': DataLoader(test_dataset, batch_size=args.bs, shuffle=False, 
                          num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    }
    
    print("DataLoaders created")
    return dataloaders


def create_model(args, input_size, disease_labels, device):
    """Create the multilabel genotype model"""
    print("Creating model...")
    
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
        use_bmi=bool(args.use_bmi),
        num_covariates=10,
        use_pooling=bool(args.use_pooling),
        pool_size=args.pool_size,
        pool_type=args.pool_type,
        use_multi_scale=bool(args.use_multi_scale),
        use_disease_attention=bool(args.use_disease_attention),
        use_separate_heads=bool(args.use_separate_heads),
        attention_heads=args.attention_heads,
        attention_dim=args.attention_dim,
        multi_scale_kernels=args.multi_scale_kernels,
        multi_scale_strides=args.multi_scale_strides,
        multi_scale_fusion=args.multi_scale_fusion,
        multi_scale_mode=args.multi_scale_mode,
        hardcoded_kernels=args.hardcoded_kernels,
        hardcoded_strides=args.hardcoded_strides,
        use_pointwise_conv=bool(args.use_pointwise_conv),
        pointwise_channels=args.pointwise_channels,
        use_transformer=bool(args.use_transformer),
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        transformer_dim=args.transformer_dim,
        transformer_ff_dim=args.transformer_ff_dim,
        transformer_dropout=args.transformer_dropout,
        use_positional_encoding=bool(args.use_positional_encoding),
        max_seq_len=args.max_seq_len,
        init_from_pretrained=bool(args.init_from_pretrained),
        pretrained_model_name=args.pretrained_model_name,
        init_layers_fraction=args.init_layers_fraction,
        layer_init_strategy=args.layer_init_strategy,
        custom_layer_indices=args.custom_layer_indices
    )
    
    model = model.to(device)
    print("Enhanced model created and moved to device")
    
    return model


def setup_training_components(model, args, train_files):
    """Setup optimizer, criterion, and scheduler"""
    # Create criterion
    criterion = nn.BCEWithLogitsLoss()
    
    # Create optimizer
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
    
    # Create scheduler
    scheduler = get_scheduler(args.sch, optimizer, args, train_files)
    
    print(f"Using optimizer: {args.opt}")
    print(f"Using scheduler: {args.sch}")
    
    return criterion, optimizer, scheduler


def handle_checkpoint_loading(experiment_dir, model, optimizer, scheduler, device, args):
    """Handle checkpoint loading and resuming"""
    start_epoch = 0
    history = None
    best_loss = float('inf')
    
    if args.resume:
        latest_checkpoint = find_latest_checkpoint(experiment_dir)
        if latest_checkpoint:
            print(f"Loading checkpoint: {latest_checkpoint}")
            try:
                checkpoint = torch.load(latest_checkpoint, map_location=device)
                
                model.load_state_dict(checkpoint['model_state_dict'])
                print("Loaded model weights from checkpoint")
                
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Loaded optimizer state from checkpoint")
                
                start_epoch = checkpoint.get('epoch', 0)

                if 'history' in checkpoint and checkpoint['history'] is not None:
                    history = checkpoint['history']
                    print(f"Loaded history from checkpoint with keys: {list(history.keys())}")
                else:
                    print("No valid history found in checkpoint, will create new history")
                    history = None

                best_loss = checkpoint.get('best_loss', float('inf'))
                
                print(f"Resuming from epoch {start_epoch} (total epochs completed so far: {start_epoch})")
                print(f"Best validation loss so far: {best_loss:.6f}")
                
                # Load scheduler state if available
                if 'scheduler_state_dict' in checkpoint:
                    try:
                        if hasattr(scheduler, 'load_state_dict'):
                            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                        print("Loaded scheduler state from checkpoint")
                    except Exception as e:
                        print(f"Warning: Failed to load scheduler state: {e}")
                
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Starting training from scratch.")
                start_epoch = 0
                history = None
                best_loss = float('inf')
        else:
            print("No checkpoints found. Starting training from scratch.")
    else:
        print("Resume flag is disabled. Starting training from scratch.")
    
    return start_epoch, history, best_loss


def main():
    """Main function to run the training pipeline"""
    print("Starting enhanced multilabel disease prediction model...")
    
    # Setup environment
    setup_environment()
    
    # Parse arguments
    args = parse_args()
    
    # Validate hardcoded parameters if using hardcoded mode
    if args.use_multi_scale and args.multi_scale_mode == "hardcoded":
        validate_hardcoded_parameters(args)

    # Setup experiment directory
    experiment_id = str(args.ID)
    print(f"Experiment ID: {experiment_id}")
    experiment_dir = os.path.join(args.exp_dir, experiment_id)
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Results will be saved to: {experiment_dir}")

    # Prepare disease labels
    disease_labels = args.disease_labels if isinstance(args.disease_labels, list) else args.disease_labels.split(',')
    disease_labels = [label.strip() for label in disease_labels]
    print(f"Using disease labels: {disease_labels}")

    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and prepare data
    phenotype_data, train_files, test_files, input_size = load_and_prepare_data(args)
    
    # Create datasets
    dataloaders = create_datasets(train_files, test_files, phenotype_data, disease_labels, args)
    
    # Create model
    model = create_model(args, input_size, disease_labels, device)
    
    # Setup training components
    criterion, optimizer, scheduler = setup_training_components(model, args, train_files)
    
    # Handle checkpoint loading
    start_epoch, history, best_loss = handle_checkpoint_loading(
        experiment_dir, model, optimizer, scheduler, device, args)
    
    # Setup early stopping
    early_stopping = EarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta,
        verbose=True
    )
    
    if args.resume and history is not None:
        # Try to load early stopping state if available
        try:
            latest_checkpoint = find_latest_checkpoint(experiment_dir)
            if latest_checkpoint:
                checkpoint = torch.load(latest_checkpoint, map_location=device)
                if 'early_stopping_state' in checkpoint:
                    early_stopping.load_state_dict(checkpoint['early_stopping_state'])
                    print("Loaded early stopping state from checkpoint")
        except Exception as e:
            print(f"Warning: Failed to load early stopping state: {e}")

    # Create trainer
    trainer = Trainer(model, criterion, optimizer, scheduler, device)
    
    # Train model
    print(f"DEBUG: History before training: {'None' if history is None else 'Present'}")
    
    start_time = time.time()
    model, history, final_metrics, all_preds, all_labels, completed_epochs = trainer.train(
        dataloaders, args.epochs, disease_labels, early_stopping=early_stopping,
        checkpoint_dir=experiment_dir, start_epoch=start_epoch,
        keep_last_n=args.keep_checkpoints, history=history, initial_best_loss=best_loss
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Plot metrics
    plot_multilabel_metrics(history, disease_labels, experiment_dir)

    # Prepare hyperparameters for saving
    hyperparameters = get_hyperparameters_dict(
        args, experiment_id, completed_epochs, 
        optimizer.param_groups[0]["lr"], disease_labels
    )

    # Write results
    write_results(model, hyperparameters, final_metrics, disease_labels, experiment_dir)

    # Print final results
    print_final_results(final_metrics, disease_labels)


if __name__ == '__main__':
    start_time = time.time()
    
    main()
    
    end_time = time.time()
    total_runtime = end_time - start_time
    
    print(f"\nTotal script runtime: {total_runtime:.2f} seconds")
    hours, rem = divmod(total_runtime, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Total runtime: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")