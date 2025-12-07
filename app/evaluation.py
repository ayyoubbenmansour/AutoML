"""
Evaluation Module
Consolidates all evaluation functionality from the evaluation/ folder
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           confusion_matrix, classification_report, roc_auc_score,
                           roc_curve, precision_recall_curve, average_precision_score)
from typing import Dict, Any, List, Optional, Tuple
import os
from flask import current_app
import uuid
from datetime import datetime

# =============================================================================
# MODEL EVALUATOR (from evaluation/metrics.py)
# =============================================================================

class ModelEvaluator:
    """Comprehensive model evaluation with plotting and reporting"""
    
    def __init__(self):
        self.plotter = EvaluationPlotter()
    
    def calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate basic classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Add AUC for binary classification
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            try:
                if y_pred_proba.ndim == 2:
                    # Multi-class probabilities, take positive class
                    auc = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    # Binary probabilities
                    auc = roc_auc_score(y_true, y_pred_proba)
                metrics['auc'] = auc
            except Exception as e:
                print(f"Could not calculate AUC: {e}")
                metrics['auc'] = 0.0
        
        return metrics
    
    def get_classification_report_text(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Get detailed classification report as formatted text string"""
        try:
            return classification_report(y_true, y_pred, zero_division=0)
        except Exception as e:
            print(f"Error generating classification report: {e}")
            return "Classification report could not be generated."
    
    def get_classification_report_dict(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Get detailed classification report as dictionary"""
        try:
            return classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        except Exception as e:
            print(f"Error generating classification report: {e}")
            return {}
    
    def generate_detailed_report(self, model, X_test: np.ndarray, y_test: np.ndarray,
                               model_name: str, model_params: Dict = None,
                               class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate comprehensive evaluation report with plots"""
        try:
            print(f"\n[DEBUG] Starting evaluation for model: {model_name}")
            
            # Make predictions
            y_pred = model.predict(X_test)
            print(f"[DEBUG] Predictions shape: {y_pred.shape}, Test labels shape: {y_test.shape}")
            
            # Handle different prediction formats
            if y_pred.ndim > 1:
                if y_pred.shape[1] == 1:
                    # Single output, squeeze
                    y_pred = y_pred.squeeze()
                else:
                    # Multi-class, take argmax
                    y_pred = np.argmax(y_pred, axis=1)
            
            # Round predictions for binary classification if needed
            unique_classes = np.unique(y_test)
            if len(unique_classes) == 2:
                y_pred = np.round(y_pred).astype(int)
            
            print(f"[DEBUG] Unique classes in y_test: {unique_classes}")
            print(f"[DEBUG] Unique predictions: {np.unique(y_pred)}")
            
            # Get prediction probabilities if available
            y_pred_proba = None
            try:
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)
                elif hasattr(model, 'predict'):
                    # For neural networks, predict often returns probabilities
                    proba = model.predict(X_test)
                    if proba.ndim > 1 and proba.shape[1] > 1:
                        y_pred_proba = proba
                    elif proba.ndim > 1 and proba.shape[1] == 1:
                        # Binary classification with single output
                        y_pred_proba = np.column_stack([1 - proba.squeeze(), proba.squeeze()])
            except Exception as e:
                print(f"[DEBUG] Could not get prediction probabilities: {e}")
            
            # Calculate metrics
            metrics = self.calculate_basic_metrics(y_test, y_pred, y_pred_proba)
            print(f"[DEBUG] Calculated metrics: {metrics}")
            
            # Get classification report as text for display
            classification_report_text = self.get_classification_report_text(y_test, y_pred)
            print(f"[DEBUG] Classification report generated (length: {len(classification_report_text)})")
            
            # Get classification report as dict for processing
            classification_report_dict = self.get_classification_report_dict(y_test, y_pred)
            
            # Generate plots and get their paths
            plots = {}
            
            # Confusion Matrix (always generate)
            print("[DEBUG] Creating confusion matrix plot...")
            confusion_plot = self.plotter.create_confusion_matrix_plot(
                y_test, y_pred, model_name, class_names
            )
            if confusion_plot:
                plots['confusion_matrix'] = confusion_plot
                print(f"[DEBUG] Confusion matrix saved: {confusion_plot}")
            
            # ROC Curve (for binary classification)
            if y_pred_proba is not None and len(unique_classes) == 2:
                print("[DEBUG] Creating ROC curve plot...")
                roc_plot = self.plotter.create_roc_curve_plot(
                    y_test, y_pred_proba[:, 1] if y_pred_proba.ndim > 1 else y_pred_proba, 
                    model_name
                )
                if roc_plot:
                    plots['roc_curve'] = roc_plot
                    print(f"[DEBUG] ROC curve saved: {roc_plot}")
                
                # Precision-Recall Curve
                print("[DEBUG] Creating precision-recall plot...")
                pr_plot = self.plotter.create_precision_recall_plot(
                    y_test, y_pred_proba[:, 1] if y_pred_proba.ndim > 1 else y_pred_proba,
                    model_name
                )
                if pr_plot:
                    plots['precision_recall'] = pr_plot
                    print(f"[DEBUG] Precision-recall plot saved: {pr_plot}")
            
            # Prediction distribution plot
            print("[DEBUG] Creating prediction distribution plot...")
            pred_dist_plot = self.plotter.create_prediction_distribution_plot(
                y_test, y_pred, model_name
            )
            if pred_dist_plot:
                plots['prediction_distribution'] = pred_dist_plot
                print(f"[DEBUG] Prediction distribution saved: {pred_dist_plot}")
            
            # Feature importance (if available)
            if hasattr(model, 'feature_importances_') and hasattr(model, 'feature_names_in_'):
                print("[DEBUG] Creating feature importance plot...")
                importance_plot = self.plotter.create_feature_importance_plot(
                    model.feature_names_in_.tolist(), 
                    model.feature_importances_, 
                    model_name
                )
                if importance_plot:
                    plots['feature_importance'] = importance_plot
                    print(f"[DEBUG] Feature importance saved: {importance_plot}")
            
            # Training history (if available)
            if hasattr(model, 'history') and model.history:
                print("[DEBUG] Creating training history plot...")
                history_plot = self.plotter.create_training_history_plot(
                    model.history.history, model_name
                )
                if history_plot:
                    plots['training_history'] = history_plot
                    print(f"[DEBUG] Training history saved: {history_plot}")
            
            print(f"[DEBUG] Total plots generated: {len(plots)}")
            print(f"[DEBUG] Plot types: {list(plots.keys())}")
            
            # Compile complete report
            report = {
                'model_name': model_name,
                'model_params': model_params or {},
                'metrics': metrics,
                'classification_report': classification_report_text,
                'classification_report_dict': classification_report_dict,
                'plots': plots,
                'predictions': {
                    'y_true': y_test.tolist() if isinstance(y_test, np.ndarray) else y_test,
                    'y_pred': y_pred.tolist() if isinstance(y_pred, np.ndarray) else y_pred
                }
            }
            
            print("[DEBUG] Report generation complete!")
            return report
            
        except Exception as e:
            print(f"[ERROR] Error in generate_detailed_report: {e}")
            import traceback
            traceback.print_exc()
            
            # Return minimal report on error
            return {
                'model_name': model_name,
                'model_params': model_params or {},
                'metrics': {'error': str(e)},
                'classification_report': f"Error generating report: {str(e)}",
                'classification_report_dict': {},
                'plots': {},
                'predictions': {}
            }
    
    def compare_models(self, models_results: Dict[str, Dict], 
                      metric: str = 'accuracy') -> Optional[str]:
        """Compare multiple models and generate comparison plot"""
        if not models_results or len(models_results) < 2:
            return None
        
        # Extract metrics for comparison
        comparison_data = {}
        for model_name, results in models_results.items():
            if 'metrics' in results and metric in results['metrics']:
                comparison_data[model_name] = results['metrics']
        
        if len(comparison_data) < 2:
            return None
        
        return self.plotter.create_model_comparison_plot(
            comparison_data, metric, f"model_comparison_{metric}"
        )
    
    def cleanup_old_plots(self, max_files: int = 50):
        """Clean up old plot files"""
        self.plotter.cleanup_old_plots(max_files)

# =============================================================================
# EVALUATION PLOTTER (from evaluation/plots.py)
# =============================================================================

class EvaluationPlotter:
    """Handles all plotting for model evaluation results with Flask integration"""
    
    def __init__(self, style: str = 'default'):
        # Set plotting style
        plt.style.use(style)
        sns.set_palette("husl")
        
        # Default plot settings
        self.default_figsize = (10, 8)
        self.default_dpi = 150
    
    def _get_static_folder(self) -> str:
        """Get the Flask static folder path"""
        if current_app:
            return current_app.static_folder
        else:
            # Fallback for testing outside Flask context
            return 'static'
    
    def _get_plots_directory(self) -> str:
        """Get or create the plots directory"""
        static_folder = self._get_static_folder()
        plots_dir = os.path.join(static_folder, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        return plots_dir
    
    def _generate_filename(self, model_name: str, plot_type: str) -> str:
        """Generate unique filename for plots"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        # Clean model name for filename
        clean_model_name = "".join(c for c in model_name if c.isalnum() or c in ('-', '_')).rstrip()
        return f"{clean_model_name}_{plot_type}_{timestamp}_{unique_id}.png"
    
    def _save_plot(self, model_name: str, plot_type: str) -> str:
        """Save the current plot and return the relative path for templates"""
        plots_dir = self._get_plots_directory()
        filename = self._generate_filename(model_name, plot_type)
        filepath = os.path.join(plots_dir, filename)
        
        plt.savefig(filepath, dpi=self.default_dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        # Return relative path for use with url_for('static', filename=...)
        return f"plots/{filename}"
    
    def create_confusion_matrix_plot(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   model_name: str, class_names: Optional[List[str]] = None) -> str:
        """Create and save confusion matrix plot"""
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create plot
        plt.figure(figsize=self.default_figsize)
        
        # Use seaborn for better aesthetics
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names or ['Class 0', 'Class 1'],
                   yticklabels=class_names or ['Class 0', 'Class 1'],
                   cbar_kws={'label': 'Count'})
        
        plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        
        # Add accuracy information
        accuracy = np.sum(np.diag(cm)) / np.sum(cm)
        plt.text(0.5, -0.1, f'Overall Accuracy: {accuracy:.3f}', 
                ha='center', va='top', transform=plt.gca().transAxes,
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
        
        plt.tight_layout()
        
        return self._save_plot(model_name, 'confusion_matrix')
    
    def create_roc_curve_plot(self, y_true: np.ndarray, y_pred_probs: np.ndarray, 
                             model_name: str) -> Optional[str]:
        """Create and save ROC curve plot"""
        # Only for binary classification
        if len(np.unique(y_true)) != 2:
            return None
        
        try:
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
            
            # Calculate AUC
            from sklearn.metrics import auc
            roc_auc = auc(fpr, tpr)
            
            # Create plot
            plt.figure(figsize=self.default_figsize)
            
            # Plot ROC curve
            plt.plot(fpr, tpr, color='darkorange', lw=3, 
                    label=f'ROC Curve (AUC = {roc_auc:.3f})')
            
            # Plot diagonal (random classifier)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                    label='Random Classifier (AUC = 0.500)')
            
            # Customize plot
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
            plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
            plt.title(f'ROC Curve - {model_name}', fontsize=16, fontweight='bold')
            plt.legend(loc="lower right", fontsize=11)
            plt.grid(True, alpha=0.3)
            
            # Add optimal threshold point
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=8,
                    label=f'Optimal Threshold: {optimal_threshold:.3f}')
            plt.legend(loc="lower right", fontsize=11)
            
            plt.tight_layout()
            
            return self._save_plot(model_name, 'roc_curve')
            
        except Exception as e:
            print(f"Failed to create ROC curve: {e}")
            return None
    
    def create_precision_recall_plot(self, y_true: np.ndarray, y_pred_probs: np.ndarray, 
                                   model_name: str) -> Optional[str]:
        """Create and save Precision-Recall curve plot"""
        # Only for binary classification
        if len(np.unique(y_true)) != 2:
            return None
        
        try:
            # Calculate Precision-Recall curve
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred_probs)
            
            # Calculate average precision
            avg_precision = average_precision_score(y_true, y_pred_probs)
            
            # Create plot
            plt.figure(figsize=self.default_figsize)
            
            # Plot PR curve
            plt.plot(recall, precision, color='darkgreen', lw=3,
                    label=f'PR Curve (AP = {avg_precision:.3f})')
            
            # Plot baseline (random classifier)
            baseline = np.sum(y_true) / len(y_true)
            plt.axhline(y=baseline, color='red', linestyle='--', lw=2,
                       label=f'Random Classifier (AP = {baseline:.3f})')
            
            # Customize plot
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall', fontsize=12, fontweight='bold')
            plt.ylabel('Precision', fontsize=12, fontweight='bold')
            plt.title(f'Precision-Recall Curve - {model_name}', fontsize=16, fontweight='bold')
            plt.legend(loc="lower left", fontsize=11)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            return self._save_plot(model_name, 'precision_recall')
            
        except Exception as e:
            print(f"Failed to create Precision-Recall curve: {e}")
            return None
    
    def create_training_history_plot(self, history: Dict[str, List[float]], 
                                   model_name: str) -> Optional[str]:
        """Create and save training history plot"""
        if not history or not any(key in history for key in ['loss', 'accuracy']):
            return None
        
        # Determine subplot layout
        has_accuracy = 'accuracy' in history
        has_loss = 'loss' in history
        has_val_data = any(key.startswith('val_') for key in history.keys())
        
        if has_accuracy and has_loss:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
            ax2 = None
        
        epochs = range(1, len(history[list(history.keys())[0]]) + 1)
        
        # Plot accuracy
        if has_accuracy:
            ax1.plot(epochs, history['accuracy'], 'b-', linewidth=2, label='Training Accuracy')
            if has_val_data and 'val_accuracy' in history:
                ax1.plot(epochs, history['val_accuracy'], 'r-', linewidth=2, label='Validation Accuracy')
            ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim([0, 1])
        
        # Plot loss
        if has_loss and ax2 is not None:
            ax2.plot(epochs, history['loss'], 'b-', linewidth=2, label='Training Loss')
            if has_val_data and 'val_loss' in history:
                ax2.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
            ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        elif has_loss and ax2 is None:
            ax1.plot(epochs, history['loss'], 'b-', linewidth=2, label='Training Loss')
            if has_val_data and 'val_loss' in history:
                ax1.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
            ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        plt.suptitle(f'Training History - {model_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return self._save_plot(model_name, 'training_history')
    
    def create_prediction_distribution_plot(self, y_true: np.ndarray, y_pred: np.ndarray,
                                          model_name: str) -> str:
        """Create prediction distribution plot"""
        plt.figure(figsize=(10, 6))
        
        # Create histogram
        plt.hist(y_pred, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title(f'Prediction Distribution - {model_name}', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Values')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_pred = np.mean(y_pred)
        std_pred = np.std(y_pred)
        plt.axvline(mean_pred, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_pred:.3f}')
        plt.legend()
        
        plt.tight_layout()
        return self._save_plot(model_name, 'prediction_distribution')
    
    def create_feature_importance_plot(self, feature_names: List[str], importance_values: np.ndarray,
                                     model_name: str, top_n: int = 20) -> str:
        """Create and save feature importance plot"""
        # Sort features by importance
        indices = np.argsort(importance_values)[::-1]
        top_indices = indices[:top_n]
        
        plt.figure(figsize=(12, max(6, len(top_indices) * 0.4)))
        
        # Create horizontal bar plot
        y_pos = np.arange(len(top_indices))
        plt.barh(y_pos, importance_values[top_indices], color='skyblue', edgecolor='navy', alpha=0.7)
        
        plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
        plt.title(f'Feature Importance - {model_name} (Top {len(top_indices)})', 
                 fontsize=16, fontweight='bold')
        plt.yticks(y_pos, [feature_names[i] for i in top_indices])
        plt.gca().invert_yaxis()  # Highest importance at top
        
        # Add value labels
        for i, v in enumerate(importance_values[top_indices]):
            plt.text(v + 0.001, i, f'{v:.3f}', va='center', fontweight='bold')
        
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        return self._save_plot(model_name, 'feature_importance')
    
    def create_model_comparison_plot(self, models_results: Dict[str, Dict[str, float]], 
                                   metric_name: str = 'accuracy', save_name: str = "model_comparison") -> str:
        """Create comparison plot for multiple models"""
        if not models_results:
            return None
        
        model_names = list(models_results.keys())
        values = [models_results[model].get(metric_name, 0) for model in model_names]
        
        plt.figure(figsize=(12, 8))
        
        # Create bar plot with colors
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
        bars = plt.bar(model_names, values, color=colors, alpha=0.8, edgecolor='black')
        
        plt.title(f'Model Comparison - {metric_name.replace("_", " ").title()}', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Models', fontsize=12, fontweight='bold')
        plt.ylabel(metric_name.replace("_", " ").title(), fontsize=12, fontweight='bold')
        plt.ylim(0, 1.1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        return self._save_plot(save_name, 'comparison')
    
    @staticmethod
    def cleanup_old_plots(max_files: int = 50):
        """Clean up old plot files to prevent disk space issues"""
        try:
            if current_app:
                plots_dir = os.path.join(current_app.static_folder, 'plots')
            else:
                plots_dir = os.path.join('app', 'static', 'plots')
            
            if not os.path.exists(plots_dir):
                return
                
            plot_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
            
            if len(plot_files) > max_files:
                # Sort by creation time, remove oldest
                plot_files.sort(key=lambda x: os.path.getctime(os.path.join(plots_dir, x)))
                files_to_remove = plot_files[:len(plot_files) - max_files]
                
                for file in files_to_remove:
                    os.remove(os.path.join(plots_dir, file))
                    
        except Exception as e:
            print(f"Warning: Could not cleanup old plots: {e}")

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, 
                  model_name: str, model_params: Dict = None) -> Dict[str, Any]:
    """Convenience function for model evaluation"""
    evaluator = ModelEvaluator()
    return evaluator.generate_detailed_report(model, X_test, y_test, model_name, model_params)

def compare_models(models_results: Dict[str, Dict], metric: str = 'accuracy') -> Optional[str]:
    """Convenience function for model comparison"""
    evaluator = ModelEvaluator()
    return evaluator.compare_models(models_results, metric)

def cleanup_plots(max_files: int = 50):
    """Convenience function for plot cleanup"""
    EvaluationPlotter.cleanup_old_plots(max_files)