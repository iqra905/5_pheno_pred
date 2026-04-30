# multiscale/training/trainer.py

import copy
import time
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score

from .schedulers import WarmupExponential, ExponentialDecay, print_lr
from ..utils.metrics import compute_final_metrics
from ..utils.checkpointing import save_checkpoint, cleanup_old_checkpoints


class Trainer:
    """Main trainer class for multilabel genotype model"""
    
    def __init__(self, model, criterion, optimizer, scheduler, device='cuda'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.scaler = GradScaler()
        
    def train_epoch(self, dataloader, disease_labels):
        """Train for one epoch"""
        self.model.train()
        
        running_loss = 0.0
        running_corrects = {disease: 0 for disease in disease_labels}
        total_samples = 0
        
        phase_preds = {disease: [] for disease in disease_labels}
        phase_labels = {disease: [] for disease in disease_labels}
        
        batch_count = len(dataloader)
        print(f"\nStarting train phase: {batch_count} batches to process")
        
        stream = torch.cuda.Stream()
        batch_iter = iter(dataloader)
        
        try:
            inputs, covariates, labels = next(batch_iter)
            inputs = inputs.to(self.device, non_blocking=True)
            covariates = covariates.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
        except StopIteration:
            print(f"Warning: train dataloader is empty!")
            return 0.0, running_corrects, phase_preds, phase_labels, total_samples
        
        batch_times = []
        
        for i in range(len(dataloader)):
            batch_start = time.time()
            
            # Prefetch next batch
            try:
                if i + 1 < len(dataloader):
                    with torch.cuda.stream(stream):
                        next_inputs, next_covariates, next_labels = next(batch_iter)
                        next_inputs = next_inputs.to(self.device, non_blocking=True)
                        next_covariates = next_covariates.to(self.device, non_blocking=True)
                        next_labels = next_labels.to(self.device, non_blocking=True)
            except StopIteration:
                pass
            
            torch.cuda.current_stream().wait_stream(stream)
            
            self.optimizer.zero_grad()
            
            with autocast():
                logits = self.model(inputs, covariates)
                loss = self.criterion(logits, labels)
                
                with torch.no_grad():
                    probs = torch.sigmoid(logits)
                    preds = (probs >= 0.5).float()
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Update scheduler for custom schedulers
                if isinstance(self.scheduler, (WarmupExponential, ExponentialDecay)):
                    self.scheduler.step()
            
            # Memory cleanup
            if (i + 1) % 20 == 0:
                torch.cuda.empty_cache()
            
            # Update metrics
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            
            for j, disease in enumerate(disease_labels):
                running_corrects[disease] += torch.sum(preds[:, j] == labels[:, j])
                phase_labels[disease].extend(labels[:, j].cpu().numpy())
                phase_preds[disease].extend(probs[:, j].detach().cpu().numpy())
            
            total_samples += batch_size
            
            # Timing and progress
            batch_end = time.time()
            batch_time = batch_end - batch_start
            batch_times.append(batch_time)
            
            if (i + 1) % 20 == 0 or i == 0 or i == len(dataloader) - 1:
                avg_time = sum(batch_times) / len(batch_times)
                eta = avg_time * (len(dataloader) - i - 1)
                
                if eta > 60:
                    eta_str = f"{eta//60:.0f}m {eta%60:.0f}s"
                else:
                    eta_str = f"{eta:.1f}s"
                    
                print(f"train Batch {i+1}/{len(dataloader)} | " 
                      f"Time: {batch_time:.2f}s | "
                      f"ETA: {eta_str} | "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Prepare for next iteration
            try:
                inputs, covariates, labels = next_inputs, next_covariates, next_labels
            except:
                break
        
        epoch_loss = running_loss / total_samples
        return epoch_loss, running_corrects, phase_preds, phase_labels, total_samples
    
    def validate_epoch(self, dataloader, disease_labels):
        """Validate for one epoch"""
        self.model.eval()
        
        running_loss = 0.0
        running_corrects = {disease: 0 for disease in disease_labels}
        total_samples = 0
        
        phase_preds = {disease: [] for disease in disease_labels}
        phase_labels = {disease: [] for disease in disease_labels}
        
        batch_count = len(dataloader)
        print(f"\nStarting test phase: {batch_count} batches to process")
        
        stream = torch.cuda.Stream()
        batch_iter = iter(dataloader)
        
        try:
            inputs, covariates, labels = next(batch_iter)
            inputs = inputs.to(self.device, non_blocking=True)
            covariates = covariates.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
        except StopIteration:
            print(f"Warning: test dataloader is empty!")
            return 0.0, running_corrects, phase_preds, phase_labels, total_samples
        
        batch_times = []
        
        with torch.no_grad():
            for i in range(len(dataloader)):
                batch_start = time.time()
                
                # Prefetch next batch
                try:
                    if i + 1 < len(dataloader):
                        with torch.cuda.stream(stream):
                            next_inputs, next_covariates, next_labels = next(batch_iter)
                            next_inputs = next_inputs.to(self.device, non_blocking=True)
                            next_covariates = next_covariates.to(self.device, non_blocking=True)
                            next_labels = next_labels.to(self.device, non_blocking=True)
                except StopIteration:
                    pass
                
                torch.cuda.current_stream().wait_stream(stream)
                
                with autocast():
                    logits = self.model(inputs, covariates)
                    loss = self.criterion(logits, labels)
                    
                    probs = torch.sigmoid(logits)
                    preds = (probs >= 0.5).float()
                
                # Memory cleanup
                if (i + 1) % 20 == 0:
                    torch.cuda.empty_cache()
                
                # Update metrics
                batch_size = labels.size(0)
                running_loss += loss.item() * batch_size
                
                for j, disease in enumerate(disease_labels):
                    running_corrects[disease] += torch.sum(preds[:, j] == labels[:, j])
                    phase_labels[disease].extend(labels[:, j].cpu().numpy())
                    phase_preds[disease].extend(probs[:, j].detach().cpu().numpy())
                
                total_samples += batch_size
                
                # Timing and progress
                batch_end = time.time()
                batch_time = batch_end - batch_start
                batch_times.append(batch_time)
                
                if (i + 1) % 20 == 0 or i == 0 or i == len(dataloader) - 1:
                    avg_time = sum(batch_times) / len(batch_times)
                    eta = avg_time * (len(dataloader) - i - 1)
                    
                    if eta > 60:
                        eta_str = f"{eta//60:.0f}m {eta%60:.0f}s"
                    else:
                        eta_str = f"{eta:.1f}s"
                        
                    print(f"test Batch {i+1}/{len(dataloader)} | " 
                          f"Time: {batch_time:.2f}s | "
                          f"ETA: {eta_str} | "
                          f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
                
                # Prepare for next iteration
                try:
                    inputs, covariates, labels = next_inputs, next_covariates, next_labels
                except:
                    break
        
        epoch_loss = running_loss / total_samples
        return epoch_loss, running_corrects, phase_preds, phase_labels, total_samples
    
    def train(self, dataloaders, num_epochs, disease_labels, early_stopping=None, 
              checkpoint_dir=None, start_epoch=0, keep_last_n=2, history=None, 
              initial_best_loss=float('inf')):
        """Main training loop"""
        print(f"Training multilabel model on device: {self.device}")
        print(f"Disease labels: {disease_labels}")
        print(f"Starting with initial best loss: {initial_best_loss:.6f}")
        
        print(f"DEBUG: History at start of training: {'None' if history is None else 'Present'}")
        
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = initial_best_loss
        completed_epochs = start_epoch 
        
        num_diseases = len(disease_labels)
        
        # Initialize history if needed
        if history is None or 'train_loss' not in history or 'test_loss' not in history:
            print("Creating new history dictionary")
            history = {
                'train_loss': [], 'test_loss': [],
                'learning_rates': []
            }
            
            for disease in disease_labels:
                for phase in ['train', 'test']:
                    history[f'{phase}_{disease}_acc'] = []
                    history[f'{phase}_{disease}_auc'] = []
        
        # Ensure all required keys exist
        required_keys = ['train_loss', 'test_loss', 'learning_rates']
        for disease in disease_labels:
            for phase in ['train', 'test']:
                required_keys.extend([f'{phase}_{disease}_acc', f'{phase}_{disease}_auc'])
        
        for key in required_keys:
            if key not in history:
                print(f"Adding missing key {key} to history")
                history[key] = []
                
        print(f"History structure verified with keys: {list(history.keys())}")
        
        # Training loop
        for epoch in range(start_epoch, num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            print('-' * 10)
            
            epoch_time = time.time()
            
            # Training phase
            train_loss, train_corrects, train_preds, train_labels, train_samples = self.train_epoch(
                dataloaders['train'], disease_labels)
            
            # Validation phase
            val_loss, val_corrects, val_preds, val_labels, val_samples = self.validate_epoch(
                dataloaders['test'], disease_labels)
            
            epoch_time = time.time() - epoch_time
            
            # Update history
            history['train_loss'].append(train_loss)
            history['test_loss'].append(val_loss)
            
            print(f'train Loss: {train_loss:.4f} (Time: {epoch_time:.2f}s)')
            print(f'test Loss: {val_loss:.4f}')
            
            # Calculate and store metrics for each disease
            all_preds = {'train': train_preds, 'test': val_preds}
            all_labels = {'train': train_labels, 'test': val_labels}
            all_corrects = {'train': train_corrects, 'test': val_corrects}
            all_samples = {'train': train_samples, 'test': val_samples}
            
            for phase in ['train', 'test']:
                for i, disease in enumerate(disease_labels):
                    epoch_acc = all_corrects[phase][disease].double() / all_samples[phase]
                    history[f'{phase}_{disease}_acc'].append(epoch_acc.item())
                    
                    try:
                        epoch_auc = roc_auc_score(all_labels[phase][disease], all_preds[phase][disease])
                        history[f'{phase}_{disease}_auc'].append(epoch_auc)
                        auc_str = f"AUC: {epoch_auc:.4f}"
                    except Exception as e:
                        history[f'{phase}_{disease}_auc'].append(0.5)
                        auc_str = "AUC: N/A (need both classes)"
                        print(f"Warning: Could not calculate AUC for {disease} in {phase} phase: {str(e)}")
                    
                    print(f'  {disease}: Acc: {epoch_acc:.4f}, {auc_str}')
            
            # Check for best model
            new_best_model = False
            if val_loss < best_loss:
                print(f"New best model! Validation loss improved from {best_loss:.6f} to {val_loss:.6f}")
                best_loss = val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                new_best_model = True
                
                if checkpoint_dir:
                    best_model_path = f'{checkpoint_dir}/best_model_{epoch+1}.pt'
                    
                    best_checkpoint = {
                        'epoch': epoch + 1,
                        'model_state_dict': best_model_wts,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_model_state_dict': best_model_wts,
                        'history': history,
                        'best_loss': best_loss,
                        'completed_epochs': epoch + 1
                    }
                    
                    if hasattr(self.scheduler, 'state_dict'):
                        best_checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
                    elif isinstance(self.scheduler, (WarmupExponential, ExponentialDecay)):
                        best_checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
                        
                    if early_stopping is not None:
                        best_checkpoint['early_stopping_state'] = early_stopping.state_dict()
                    
                    print(f"Saving best model to {best_model_path}")
            else:
                print(f"Validation loss: {val_loss:.6f} (not improved from best: {best_loss:.6f})")
            
            # Early stopping check
            if early_stopping is not None:
                early_stopping(val_loss, global_best_loss=best_loss)
                
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    completed_epochs = epoch + 1
                    
                    if checkpoint_dir:
                        final_checkpoint_path = f'{checkpoint_dir}/checkpoint_epoch_{completed_epochs}.pt'
                        
                        final_checkpoint = {
                            'epoch': completed_epochs,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'best_model_state_dict': best_model_wts,
                            'history': history,
                            'best_loss': best_loss,
                            'completed_epochs': completed_epochs
                        }
                        
                        if hasattr(self.scheduler, 'state_dict'):
                            final_checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
                        elif isinstance(self.scheduler, (WarmupExponential, ExponentialDecay)):
                            final_checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
                            
                        final_checkpoint['early_stopping_state'] = early_stopping.state_dict()
                        
                        print(f"Saving final checkpoint to {final_checkpoint_path}")
                        torch.save(final_checkpoint, final_checkpoint_path)
                        
                        print("Note: Best model has already been saved separately and is NOT being updated by early stopping")
                    
                    break
            
            # Update scheduler
            if self.scheduler is not None and not isinstance(self.scheduler, (WarmupExponential, ExponentialDecay)):
                if hasattr(self.scheduler, 'step'):
                    if 'ReduceLROnPlateau' in str(type(self.scheduler)):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
            
            history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            print_lr(self.optimizer)
            completed_epochs = epoch + 1
            
            # Save regular checkpoint
            if checkpoint_dir:
                checkpoint = {
                    'epoch': completed_epochs,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_model_state_dict': best_model_wts,
                    'history': history,
                    'best_loss': best_loss,
                    'completed_epochs': completed_epochs
                }
                
                if hasattr(self.scheduler, 'state_dict'):
                    checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
                elif isinstance(self.scheduler, (WarmupExponential, ExponentialDecay)):
                    checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
                    
                if early_stopping is not None:
                    checkpoint['early_stopping_state'] = early_stopping.state_dict()
                
                regular_checkpoint_path = f'{checkpoint_dir}/checkpoint_epoch_{completed_epochs}.pt'
                print(f"Saving regular epoch checkpoint to {regular_checkpoint_path}")
                
                torch.save(checkpoint, regular_checkpoint_path)
                
                cleanup_old_checkpoints(checkpoint_dir, keep_last_n)
        
        # Load best model weights
        self.model.load_state_dict(best_model_wts)
        
        # Compute final metrics
        final_metrics = compute_final_metrics(all_labels, all_preds, disease_labels)
        
        return self.model, history, final_metrics, all_preds, all_labels, completed_epochs