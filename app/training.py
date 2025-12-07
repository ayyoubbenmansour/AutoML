"""
Training Module
Provides different training strategies: Simple, Custom, and HPO
"""
import numpy as np
import optuna
from typing import Dict, Any, Tuple, Optional, List
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod
import random
import itertools

from .models import ModelRegistry
from .data_processing import DataPreprocessor
from .evaluation import ModelEvaluator

# =============================================================================
# BASE TRAINER CLASS
# =============================================================================

class BaseTrainer(ABC):
    """Abstract base class for all trainers"""
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.model = None
        self.training_history = None
    
    @abstractmethod
    def train_model(self, df, **kwargs) -> Tuple[Dict[str, Any], Any]:
        """Train a model and return results"""
        pass
    
    def prepare_data(self, df, target_col: str, preprocessing_params: Dict[str, Any]) -> Tuple:
        """Prepare data for training"""
        return self.preprocessor.preprocess_data(
            df=df,
            target_col=target_col,
            **preprocessing_params
        )
    
    def get_predictions(self, X_test: np.ndarray) -> np.ndarray:
        """Get predictions from trained model"""
        if self.model is None:
            raise ValueError("No model trained. Call train_model first.")
        return self.model.predict(X_test)

# =============================================================================
# SIMPLE TRAINER
# =============================================================================

class SimpleTrainer(BaseTrainer):
    """Handles simple/default model training with predefined parameters"""
    
    def train_model(self, df, model_type: str = 'bi_lstm',
                   target_col: str = None, **preprocessing_params) -> Tuple[Dict[str, Any], Any]:
        """Train a model with default parameters"""
        
        # Use last column as target if not specified
        target_col = target_col or df.columns[-1]
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(df, target_col, preprocessing_params)
        
        # Split for validation
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model = ModelRegistry.get_model(model_type)
        training_metrics = self.model.train(X_train_split, y_train_split, X_val, y_val)
        
        # Evaluate on test set
        test_metrics = self.model.evaluate(X_test, y_test)
        
        results = {
            'model_type': model_type,
            'training_metrics': training_metrics,
            'test_metrics': test_metrics,
            'data_shape': {
                'train_shape': X_train.shape,
                'test_shape': X_test.shape,
                'features': X_train.shape[-1]
            }
        }
        
        return results, self.model

# =============================================================================
# CUSTOM TRAINER
# =============================================================================

class CustomTrainer(BaseTrainer):
    """Handles manual model training with custom hyperparameters"""
    
    def train_model(self, df, model_type: str, target_col: str,
                   model_params: Dict[str, Any],
                   preprocessing_params: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Any]:
        """Train a model with custom hyperparameters"""
        
        preprocessing_params = preprocessing_params or self._get_default_preprocessing_params()
        
        # Validate parameters
        validated_params = self._validate_model_params(model_params, model_type)
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(df, target_col, preprocessing_params)
        
        # Split for validation
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model = ModelRegistry.get_model(model_type)
        training_metrics = self.model.train(
            X_train_split, y_train_split, X_val, y_val, params=validated_params
        )
        
        # Evaluate on test set
        test_metrics = self.model.evaluate(X_test, y_test)
        
        results = {
            'model_type': model_type,
            'model_params': validated_params,
            'training_metrics': training_metrics,
            'test_metrics': test_metrics,
            'data_shape': {
                'train_shape': X_train.shape,
                'test_shape': X_test.shape,
                'features': X_train.shape[-1]
            }
        }
        
        return results, self.model
    
    def _get_default_preprocessing_params(self) -> Dict[str, Any]:
        """Get default preprocessing parameters"""
        return {
            'seq_length': 1,
            'missing_handling': 'remove',
            'encoding': 'label',
            'scaling': 'standard'
        }
    
    def _validate_model_params(self, params: Dict[str, Any], model_type: str) -> Dict[str, Any]:
        """Validate and clean model parameters"""
        model = ModelRegistry.get_model(model_type)
        default_params = model.get_default_params()
        param_ranges = model.get_param_ranges()
        
        validated = {}
        for param_name, default_value in default_params.items():
            if param_name in params:
                value = self._validate_single_param(
                    params[param_name], 
                    default_value,
                    param_ranges.get(param_name)
                )
                validated[param_name] = value
            else:
                validated[param_name] = default_value
        
        return validated
    
    def _validate_single_param(self, value: Any, default: Any, param_range: Any) -> Any:
        """Validate a single parameter"""
        # Type conversion
        if isinstance(default, int):
            try:
                value = int(value)
            except (ValueError, TypeError):
                return default
        elif isinstance(default, float):
            try:
                value = float(value)
            except (ValueError, TypeError):
                return default
        elif isinstance(default, str):
            value = str(value).lower()
        
        # Range validation
        if param_range:
            if isinstance(param_range, tuple) and len(param_range) == 2:
                # Numeric range
                min_val, max_val = param_range
                if isinstance(value, (int, float)):
                    value = max(min_val, min(max_val, value))
            elif isinstance(param_range, list):
                # Choice list
                if value not in param_range:
                    value = default
        
        return value

# =============================================================================
# HPO TRAINER
# =============================================================================

import random
import itertools
import optuna
from typing import Dict, Any, Tuple, Optional, List
import numpy as np

from .models import ModelRegistry
from .evaluation import ModelEvaluator

class HPOTrainer:
    def __init__(self):
        self.evaluator = ModelEvaluator()

    def optimize_model(
        self,
        df,
        model_type: str = 'bi_lstm',
        target_col: str = None,
        optimization_method: str = 'bayesian',
        hpo_params: Optional[Dict[str, Any]] = None,
        preprocessing_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Any]:
        X_train, X_test, y_train, y_test = self.prepare_data(df, target_col, preprocessing_params)
        if optimization_method == 'bayesian':
            return self._bayesian_optimization(X_train, X_test, y_train, y_test, model_type, hpo_params)
        elif optimization_method == 'grid':
            return self._grid_search_optimization(X_train, X_test, y_train, y_test, model_type, hpo_params)
        elif optimization_method == 'random':
            return self._random_search_optimization(X_train, X_test, y_train, y_test, model_type, hpo_params)
        else:
            raise ValueError(f"Unknown optimization method: {optimization_method}")

    def prepare_data(self, df, target_col, preprocessing_params):
        """
        Preprocess and split the data for training and evaluation.

        Args:
            df: pandas DataFrame with all data.
            target_col: Name of the target column.
            preprocessing_params: Dict with preprocessing options (seq_length, missing_handling, encoding, scaling, test_size, etc.)

        Returns:
            X_train, X_test, y_train, y_test: numpy arrays ready for model input.
        """
        from .data_processing import DataPreprocessor

        # Set defaults if not provided
        params = preprocessing_params or {}
        seq_length = params.get('seq_length', 1)
        missing_handling = params.get('missing_handling', 'remove')
        encoding = params.get('encoding', 'label')
        scaling = params.get('scaling', 'standard')
        test_size = params.get('test_size', 0.2)
        random_state = params.get('random_state', 42)

        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.preprocess_data(
            df,
            target_col=target_col,
            seq_length=seq_length,
            missing_handling=missing_handling,
            encoding=encoding,
            scaling=scaling,
            test_size=test_size,
            random_state=random_state
        )
        return X_train, X_test, y_train, y_test
    
        

    def _bayesian_optimization(self, X_train, X_test, y_train, y_test, model_type, hpo_params):
        """Bayesian optimization using Optuna"""
        print(f"DEBUG: Starting Bayesian optimization for {model_type}")
        print(f"DEBUG: HPO params: {hpo_params}")
        
        # Split train data for validation
        from sklearn.model_selection import train_test_split
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        def objective(trial):
            # Suggest hyperparameters
            units = trial.suggest_int('units', hpo_params['units_min'], hpo_params['units_max'])
            dropout = trial.suggest_float('dropout', hpo_params['dropout_min'], hpo_params['dropout_max'])
            lr = trial.suggest_float('learning_rate', hpo_params['lr_min'], hpo_params['lr_max'], log=True)
            batch_size = trial.suggest_categorical('batch_size', hpo_params.get('batch_sizes', [16, 32, 64, 128]))
            
            # Create model parameters
            model_params = {
                'units': units,
                'dropout': dropout,
                'recurrent_dropout': dropout * 0.7,  # Slightly lower recurrent dropout
                'learning_rate': lr,
                'batch_size': batch_size,
                'epochs': hpo_params.get('max_epochs', 100),
                'patience': hpo_params.get('patience', 10),
                'optimizer': 'adam'
            }
            
            try:
                # Get and train model
                model = ModelRegistry.get_model(model_type)
                training_metrics = model.train(
                    X_train_split, y_train_split, 
                    X_val, y_val, 
                    params=model_params
                )
                
                # Return validation accuracy as the objective value
                return training_metrics.get('final_val_accuracy', 0.0)
                
            except Exception as e:
                print(f"Trial failed: {e}")
                return 0.0

        # Create and run study
        study = optuna.create_study(direction='maximize')
        study.optimize(
            objective,
            n_trials=hpo_params.get('n_trials', 25),
            timeout=hpo_params.get('timeout', 3600)
        )
        
        # Train final model with best parameters
        best_params = study.best_params
        best_params['epochs'] = hpo_params.get('max_epochs', 100)
        best_params['patience'] = hpo_params.get('patience', 10)
        best_params['optimizer'] = 'adam'
        best_params['recurrent_dropout'] = best_params.get('dropout', 0.3) * 0.7
        
        print(f"Best parameters found: {best_params}")
        
        # Train final model on full training data
        best_model = ModelRegistry.get_model(model_type)
        best_model.train(X_train_split, y_train_split, X_val, y_val, params=best_params)
        
        # Prepare results
        results = {
            'best_params': best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials),
            'optimization_method': 'bayesian',
            'model_type': model_type
        }
        
        return results, best_model

    def _grid_search_optimization(self, X_train, X_test, y_train, y_test, model_type, hpo_params):
        """Grid search optimization"""
        print(f"DEBUG: Starting Grid Search for {model_type}")
        
        # Split for validation
        from sklearn.model_selection import train_test_split
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        best_score = -np.inf
        best_config = None
        best_model = None
        results = []
        
        configurations = hpo_params.get('configurations', [])
        print(f"Testing {len(configurations)} configurations")
        
        for i, config in enumerate(configurations):
            print(f"Testing configuration {i+1}/{len(configurations)}: {config}")
            
            try:
                # Get model
                model = ModelRegistry.get_model(config.get('model_type', model_type))
                
                # Prepare parameters
                model_params = {
                    'units': config.get('units', 64),
                    'dropout': config.get('dropout', 0.3),
                    'recurrent_dropout': config.get('dropout', 0.3) * 0.7,
                    'learning_rate': config.get('learning_rate', 0.001),
                    'batch_size': config.get('batch_size', 32),
                    'epochs': config.get('epochs', 100),
                    'patience': config.get('patience', 10),
                    'optimizer': 'adam'
                }
                # Handle stacked models - convert single units to list
                if 'stacked' in model_type.lower():
                    units_value = model_params.get('units', 64)
                    if isinstance(units_value, int):
                        # Create decreasing layer sizes: [units, units//2]
                        model_params['units'] = [units_value, max(units_value//2, 16)]
                # Train model
                training_metrics = model.train(
                    X_train_split, y_train_split,
                    X_val, y_val,
                    params=model_params
                )
                
                score = training_metrics.get('final_val_accuracy', 0.0)
                results.append({'config': config, 'score': score})
                
                if score > best_score:
                    best_score = score
                    best_config = model_params
                    best_model = model
                    
            except Exception as e:
                print(f"Configuration failed: {e}")
                results.append({'config': config, 'score': 0.0, 'error': str(e)})
        
        if best_model is None:
            raise ValueError("All configurations failed during grid search")
        
        return {
            'best_params': best_config,
            'best_score': best_score,
            'all_results': results,
            'optimization_method': 'grid',
            'model_type': model_type
        }, best_model

    def _random_search_optimization(self, X_train, X_test, y_train, y_test, model_type, hpo_params):
        """Random search optimization"""
        print(f"DEBUG: Starting Random Search for {model_type}")
        
        configs = hpo_params.get('configurations', [])
        n_trials = min(hpo_params.get('n_trials', 10), len(configs))
        
        if len(configs) == 0:
            raise ValueError("No configurations provided for random search")
        
        # Randomly sample configurations
        import random
        sampled_configs = random.sample(configs, n_trials)
        
        # Use grid search with sampled configs
        hpo_params['configurations'] = sampled_configs
        results, best_model = self._grid_search_optimization(
            X_train, X_test, y_train, y_test, model_type, hpo_params
        )
        
        results['optimization_method'] = 'random'
        results['n_trials'] = n_trials
        
        return results, best_model
