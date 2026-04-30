# multiscale/utils/checkpointing.py

import os
import shutil
import torch
import csv


def save_checkpoint(state, is_best, filename, best_filename):
    """Save model checkpoint"""
    print(f"Saving checkpoint to {filename}")
    torch.save(state, filename)
    if is_best:
        print(f"This is the best model so far. Saving to {best_filename}")
        shutil.copyfile(filename, best_filename)


def find_latest_checkpoint(dir_path, prefer_best=False):
    """Find the latest checkpoint in the directory"""
    
    # First try to find the best model checkpoint if preferred
    if prefer_best:
        best_models = [f for f in os.listdir(dir_path) if f.startswith('best_model_')]
        if best_models:
            best_models.sort(key=lambda x: int(x.split('best_model_')[1].split('.pt')[0]))
            latest_best = os.path.join(dir_path, best_models[-1])
            print(f"Found best model checkpoint: {latest_best}")
            return latest_best

    # Fallback: check for old naming scheme (best_model.pt)
    old_best_model_path = os.path.join(dir_path, 'best_model.pt')
    if os.path.exists(old_best_model_path):
        print(f"Found best model checkpoint (old format): {old_best_model_path}")
        return old_best_model_path
    
    checkpoints = [f for f in os.listdir(dir_path) if f.startswith('checkpoint_epoch_')]
    if checkpoints:
        checkpoints.sort(key=lambda x: int(x.split('checkpoint_epoch_')[1].split('.pt')[0]))
        latest_checkpoint = os.path.join(dir_path, checkpoints[-1])
        print(f"Using latest epoch checkpoint: {latest_checkpoint}")
        return latest_checkpoint
    
    print("No checkpoints found")
    return None


def cleanup_old_checkpoints(dir_path, keep_last_n=3):
    """Remove old checkpoints and best models, keeping only the most recent n checkpoints and 1 best model"""
    
    # Handle regular checkpoints
    checkpoints = [f for f in os.listdir(dir_path) if f.startswith('checkpoint_epoch_')]
    checkpoints.sort(key=lambda x: int(x.split('checkpoint_epoch_')[1].split('.pt')[0]))
    
    if len(checkpoints) > keep_last_n:
        for old_ckpt in checkpoints[:-keep_last_n]:
            old_path = os.path.join(dir_path, old_ckpt)
            print(f"Removing old checkpoint: {old_path}")
            os.remove(old_path)
    
    # Handle best model files
    best_models = [f for f in os.listdir(dir_path) if f.startswith('best_model_')]
    if best_models:
        best_models.sort(key=lambda x: int(x.split('best_model_')[1].split('.pt')[0]))
            
        if len(best_models) > 1:
            for old_best in best_models[:-1]:
                old_path = os.path.join(dir_path, old_best)
                print(f"Removing old best model: {old_path}")
                os.remove(old_path)


def write_results(model, hyperparameters, final_metrics, disease_labels, save_dir):
    """Write experiment results to files"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Write detailed text results
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
                f.write(f"  AUC:         {metrics['auc']:.5f}\n")
                f.write(f"  Confusion Matrix:\n    {metrics['cm']}\n")
    
    # Write CSV results
    with open(os.path.join(save_dir, 'experiment_results.csv'), 'w', newline='') as csvfile:
        fieldnames = list(hyperparameters.keys())
        
        for phase in ['train', 'test']:
            for disease in disease_labels:
                for metric in ['acc', 'sens', 'spec', 'auc']:
                    fieldnames.append(f"{phase}_{disease}_{metric}")
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        row = hyperparameters.copy()
        
        for phase in ['train', 'test']:
            for disease in disease_labels:
                metrics = final_metrics[phase][disease]
                for metric in ['acc', 'sens', 'spec']:
                    row[f"{phase}_{disease}_{metric}"] = metrics[metric]
                
                row[f"{phase}_{disease}_auc"] = metrics['auc']        
        writer.writerow(row)
    
    # Write model architecture
    with open(os.path.join(save_dir, 'model_architecture.txt'), 'w') as file:
        file.write(str(model))
    
    print("Results written to files")


def get_hyperparameters_dict(args, id, completed_epochs, final_lr, disease_labels):
    """Create hyperparameters dictionary for saving results"""
    return {
        'Exp_ID': id,
        'Batch_Size': args.bs,
        'Epochs': args.epochs,
        'Completed_Epochs': completed_epochs,
        'Start_LR': args.lr,
        'Peak_LR': args.peak_lr,
        'Final_LR': final_lr,
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
        'Use_Bmi': bool(args.use_bmi),
        'norm_Bmi': args.norm_bmi,
        'Kernel_sizes': str(args.kernel_sizes),
        'Stride': str(args.stride),
        'Conv_channels': str(args.conv_channels),
        'Use_Pooling': bool(args.use_pooling),
        'Pool_size': args.pool_size if args.use_pooling else 'N/A',
        'Pool_type': args.pool_type if args.use_pooling else 'N/A',
        'FC_layers': str(args.fc_layers),
        'Num_Diseases': len(disease_labels),
        'Disease_Labels': ','.join(disease_labels),
        'Use_Multi_Scale': bool(args.use_multi_scale),
        'Multi_Scale_Mode': args.multi_scale_mode if args.use_multi_scale else 'N/A',
        'Multi_Scale_Fusion': args.multi_scale_fusion if args.use_multi_scale else 'N/A',
        'use_pointwise_conv': bool(args.use_pointwise_conv),
        'Pointwise_Channels': args.pointwise_channels if args.use_pointwise_conv else 'N/A',
        'Use_Disease_Attention': bool(args.use_disease_attention),
        'Use_Separate_Heads': bool(args.use_separate_heads),
        'Attention_Heads': args.attention_heads if args.use_disease_attention else 'N/A',
        'Attention_Dim': args.attention_dim if args.use_disease_attention else 'N/A',
        'Multi_Scale_Kernels': str(args.multi_scale_kernels) if args.use_multi_scale and args.multi_scale_mode == 'progressive' else 'N/A',
        'Multi_Scale_Strides': str(args.multi_scale_strides) if args.use_multi_scale and args.multi_scale_mode == 'progressive' else 'N/A',
        'Hardcoded_Kernels': str(args.hardcoded_kernels) if args.use_multi_scale and args.multi_scale_mode == 'hardcoded' else 'N/A',
        'Hardcoded_Strides': str(args.hardcoded_strides) if args.use_multi_scale and args.multi_scale_mode == 'hardcoded' else 'N/A',
        'Use_Transformer': bool(args.use_transformer),
        'Transformer_Layers': args.transformer_layers if args.use_transformer else 'N/A',
        'Transformer_Heads': args.transformer_heads if args.use_transformer else 'N/A',
        'Transformer_Dim': args.transformer_dim if args.use_transformer else 'N/A',
        'Transformer_FF_Dim': args.transformer_ff_dim if args.use_transformer else 'N/A',
        'Transformer_Dropout': args.transformer_dropout if args.use_transformer else 'N/A',
        'Use_Positional_Encoding': bool(args.use_positional_encoding) if args.use_transformer else 'N/A',
        'Max_Seq_Len': args.max_seq_len if args.use_transformer else 'N/A',
        'Init_From_Pretrained': bool(args.init_from_pretrained) if args.use_transformer else 'N/A',
        'Pretrained_Model_Name': args.pretrained_model_name if (args.use_transformer and args.init_from_pretrained) else 'N/A',
        'Init_Layers_Fraction': args.init_layers_fraction if (args.use_transformer and args.init_from_pretrained) else 'N/A',
    }