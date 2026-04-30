# multiscale/utils/metrics.py

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve, auc


def compute_final_metrics(all_labels, all_preds, disease_labels):
    """Compute final metrics for all diseases and phases"""
    final_metrics = {}
    
    for phase in ['train', 'test']:
        phase_metrics = {}
        
        for disease in disease_labels:
            y_true = np.array(all_labels[phase][disease])
            y_pred_proba = np.array(all_preds[phase][disease])
            
            print(f"Computing metrics for {disease} in {phase} phase:")
            print(f"  - Number of samples: {len(y_true)}")
            print(f"  - Positive samples: {np.sum(y_true == 1)}")
            print(f"  - Negative samples: {np.sum(y_true == 0)}")

            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            disease_metrics = {}
            
            try:
                cm = confusion_matrix(y_true, y_pred)
                print(f"  - Raw confusion matrix shape: {cm.shape}")
                print(f"  - Confusion matrix:\n{cm}")
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                    accuracy = (tp + tn) / (tp + tn + fp + fn)
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
                else:
                    sensitivity = specificity = accuracy = precision = f1 = 0
                
                try:
                    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                    roc_auc = auc(fpr, tpr)
                    
                except Exception as e:
                    print(f"Error calculating curves for {disease} in {phase}: {str(e)}")
                    fpr, tpr = np.array([]), np.array([])
                    roc_auc = 0.5
            except Exception as e:
                print(f"Error calculating metrics for {disease} in {phase}: {str(e)}")
                cm = np.zeros((2, 2))
                sensitivity = specificity = accuracy = precision = f1 = 0
                fpr, tpr = np.array([]), np.array([])
                roc_auc = 0.5

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


def print_final_results(final_metrics, disease_labels):
    """Print final results summary"""
    for disease in disease_labels:
        print(f"\n{disease.upper()}:")
        print(f"Train - AUC: {final_metrics['train'][disease]['auc']:.4f}, "
              f"Accuracy: {final_metrics['train'][disease]['acc']}")
        print(f"Test  - AUC: {final_metrics['test'][disease]['auc']:.4f}, "
              f"Accuracy: {final_metrics['test'][disease]['acc']}")