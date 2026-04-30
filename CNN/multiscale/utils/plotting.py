# multiscale/utils/plotting.py

import os
import numpy as np
import matplotlib.pyplot as plt


def plot_multilabel_metrics(history, disease_labels, save_dir):
    """Plot and save training metrics"""
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Combined metrics plot
    plt.figure(figsize=(20, 16))
    
    # Loss plot
    plt.subplot(2, 2, 1)
    for phase in ['train', 'test']:
        if f'{phase}_loss' in history and history[f'{phase}_loss']:
            plt.plot(history[f'{phase}_loss'], label=f'{phase}')
    plt.title('Model Loss', fontsize=16)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Learning rate plot
    plt.subplot(2, 2, 2)
    plt.plot(history['learning_rates'])
    plt.title('Learning Rate', fontsize=16)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(alpha=0.3)
    
    # Average accuracy plot
    plt.subplot(2, 2, 3)
    for phase in ['train', 'test']:
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
    
    # Average AUC plot
    plt.subplot(2, 2, 4)
    for phase in ['train', 'test']:
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
    
    plt.suptitle('Enhanced Multilabel Disease Prediction Model Performance', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plt.savefig(os.path.join(plots_dir, 'combined_metrics_plot.png'), dpi=300)
    plt.close()
    
    # Individual disease plots
    metrics = ['acc', 'auc']
    phases = ['train', 'test']
    
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
    
    print("Metrics plotted and saved")