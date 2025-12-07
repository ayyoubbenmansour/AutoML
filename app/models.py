"""
RNN Models Module
Model Definitions - RNN and Traditional ML Models
Contains all model architectures for time series and tabular data
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Bidirectional, LSTM, SimpleRNN, GRU, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from typing import Dict, Tuple, Any, Optional, List
from abc import ABC, abstractmethod

# Traditional ML imports (optional - graceful degradation if not installed)
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from xgboost import XGBClassifier, XGBRegressor
    from lightgbm import LGBMClassifier, LGBMRegressor
    TRADITIONAL_ML_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Traditional ML libraries not available: {e}")
    print("Only RNN models will be available. Install with: pip install xgboost lightgbm")
    TRADITIONAL_ML_AVAILABLE = False
    # Create dummy classes to prevent errors
    RandomForestClassifier = None
    RandomForestRegressor = None
    XGBClassifier = None
    XGBRegressor = None
    LGBMClassifier = None
    LGBMRegressor = None

# Import traditional ML model wrappers (only if libraries available)
if TRADITIONAL_ML_AVAILABLE:
    try:
        from app.traditional_ml_models import (
            RandomForestClassifierModel,
            RandomForestRegressorModel,
            XGBoostClassifierModel,
            XGBoostRegressorModel
        )
    except ImportError:
        # Models will be defined in this file if import fails
        TRADITIONAL_ML_AVAILABLE = False
        print("Warning: Could not import traditional ML model wrappers")

# =============================================================================
# BASE MODEL CLASS
# =============================================================================

class BaseModel(ABC):
    """Abstract base class for all RNN models"""
    
    OPTIMIZERS = {
        'adam': Adam,
        'rmsprop': RMSprop,
        'sgd': SGD
    }
    
    def __init__(self):
        self.model = None
        self.history = None
        self.is_trained = False
        self.model_type = "base"
    
    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """Return default hyperparameters for this model"""
        pass
    
    @abstractmethod
    def get_param_ranges(self) -> Dict[str, Any]:
        """Return parameter ranges for HPO"""
        pass
    
    @abstractmethod
    def build_model(self, input_shape: Tuple[int, ...], num_classes: int, **params) -> tf.keras.Model:
        """Build the model architecture"""
        pass
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Train the model with given data and parameters"""
        
        params = params or self.get_default_params()
        
        # Determine number of classes
        num_classes = len(np.unique(y_train))
        
        # Build model - handle both 2D (time series) and 3D (images) inputs
        # For RNN: X_train.shape = (samples, timesteps, features) -> input_shape = (timesteps, features)
        # For CNN: X_train.shape = (samples, height, width, channels) -> input_shape = (height, width, channels)
        input_shape = X_train.shape[1:]  # Get all dimensions except batch dimension
        
        self.model = self.build_model(
            input_shape,
            num_classes,
            problem_type=params.get('problem_type', 'classification'),
            forecasting_steps=params.get('forecasting_steps', 1),
            **params
        )
        # Configure loss and metrics based on task type
        loss = 'binary_crossentropy' if num_classes == 2 else 'sparse_categorical_crossentropy'
        
        # Compile model
        optimizer = self._get_optimizer(
            params.get('optimizer', 'adam'),
            params.get('learning_rate', 0.001)
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
        )
        
        # Training callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=params.get('patience', 5),
                restore_best_weights=True,
                verbose=0
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=params.get('epochs', 30),
            batch_size=params.get('batch_size', 32),
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=0
        )
        
        self.is_trained = True
        
        return self._get_training_metrics()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the model on test data"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Get predictions
        y_pred_probs = self.predict(X_test)
        y_pred = (y_pred_probs > 0.5).astype(int) if y_pred_probs.shape[-1] == 1 else np.argmax(y_pred_probs, axis=1)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        return {
            'accuracy': float(accuracy_score(y_test, y_pred.squeeze())),
            'precision': float(precision_score(y_test, y_pred.squeeze(), average='weighted', zero_division=0)),
            'recall': float(recall_score(y_test, y_pred.squeeze(), average='weighted', zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred.squeeze(), average='weighted', zero_division=0))
        }
    
    def _get_optimizer(self, optimizer_name: str, learning_rate: float):
        """Get TensorFlow optimizer"""
        optimizer_class = self.OPTIMIZERS.get(optimizer_name.lower(), Adam)
        return optimizer_class(learning_rate=learning_rate)
    
    def _get_training_metrics(self) -> Dict[str, Any]:
        """Extract training metrics from history"""
        if not self.history:
            return {}
        
        return {
            'final_train_accuracy': float(self.history.history['accuracy'][-1]),
            'final_val_accuracy': float(self.history.history['val_accuracy'][-1]),
            'final_train_loss': float(self.history.history['loss'][-1]),
            'final_val_loss': float(self.history.history['val_loss'][-1]),
            'epochs_trained': len(self.history.history['accuracy'])
        }

# =============================================================================
# CONCRETE MODEL IMPLEMENTATIONS
# =============================================================================

class BiRNNModel(BaseModel):
    """Bidirectional RNN Model"""
    
    def __init__(self):
        super().__init__()
        self.model_type = "bi_rnn"
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'units': 32,
            'dropout': 0.3,
            'recurrent_dropout': 0.2,
            'learning_rate': 0.001,
            'epochs': 30,
            'batch_size': 32,
            'optimizer': 'adam',
            'patience': 5
        }
    
    def get_param_ranges(self) -> Dict[str, Any]:
        return {
            'units': [16, 32, 64, 128],
            'dropout': (0.0, 0.5),
            'recurrent_dropout': (0.0, 0.3),
            'learning_rate': (0.0001, 0.01),
            'epochs': (10, 100),
            'batch_size': [16, 32, 64, 128],
            'optimizer': ['adam', 'rmsprop', 'sgd']
        }
    
    def build_model(
        self,
        input_shape: Tuple[int, ...],
        num_classes: int,
        problem_type: str = 'classification',
        forecasting_steps: int = 1,
        **params
    ) -> tf.keras.Model:
        """Build Bidirectional RNN model with forecasting support"""
        model = Sequential([
            Input(shape=input_shape),
            Bidirectional(SimpleRNN(
                params.get('units', 32),
                return_sequences=False,
                activation='relu',
                dropout=params.get('dropout', 0.3),
                recurrent_dropout=params.get('recurrent_dropout', 0.2)
            ))
        ])
        if problem_type == 'forecasting':
            model.add(Dense(forecasting_steps, activation='linear'))
        else:
            model.add(Dense(
                1 if num_classes == 2 else num_classes,
                activation='sigmoid' if num_classes == 2 else 'softmax'
            ))
        return model

class BiLSTMModel(BaseModel):
    """Bidirectional LSTM Model"""
    
    def __init__(self):
        super().__init__()
        self.model_type = "bi_lstm"
    
    def get_default_params(self) -> Dict[str, Any]:
        return BiRNNModel().get_default_params()
    
    def get_param_ranges(self) -> Dict[str, Any]:
        return BiRNNModel().get_param_ranges()
    
    def build_model(
        self,
        input_shape: Tuple[int, ...],
        num_classes: int,
        problem_type: str = 'classification',
        forecasting_steps: int = 1,
        **params
    ) -> tf.keras.Model:
        """Build Bidirectional RNN model with forecasting support"""
        model = Sequential([
            Input(shape=input_shape),
            Bidirectional(SimpleRNN(
                params.get('units', 32),
                return_sequences=False,
                activation='relu',
                dropout=params.get('dropout', 0.3),
                recurrent_dropout=params.get('recurrent_dropout', 0.2)
            ))
        ])
        if problem_type == 'forecasting':
            model.add(Dense(forecasting_steps, activation='linear'))
        else:
            model.add(Dense(
                1 if num_classes == 2 else num_classes,
                activation='sigmoid' if num_classes == 2 else 'softmax'
            ))
        return model

class BiGRUModel(BaseModel):
    """Bidirectional GRU Model"""
    
    def __init__(self):
        super().__init__()
        self.model_type = "bi_gru"
    
    def get_default_params(self) -> Dict[str, Any]:
        return BiRNNModel().get_default_params()
    
    def get_param_ranges(self) -> Dict[str, Any]:
        return BiRNNModel().get_param_ranges()
    
    def build_model(
        self,
        input_shape: Tuple[int, ...],
        num_classes: int,
        problem_type: str = 'classification',
        forecasting_steps: int = 1,
        **params
    ) -> tf.keras.Model:
        """Build Bidirectional RNN model with forecasting support"""
        model = Sequential([
            Input(shape=input_shape),
            Bidirectional(SimpleRNN(
                params.get('units', 32),
                return_sequences=False,
                activation='relu',
                dropout=params.get('dropout', 0.3),
                recurrent_dropout=params.get('recurrent_dropout', 0.2)
            ))
        ])
        if problem_type == 'forecasting':
            model.add(Dense(forecasting_steps, activation='linear'))
        else:
            model.add(Dense(
                1 if num_classes == 2 else num_classes,
                activation='sigmoid' if num_classes == 2 else 'softmax'
            ))
        return model

class StackedLSTMModel(BaseModel):
    """Stacked LSTM Model"""
    
    def __init__(self):
        super().__init__()
        self.model_type = "stacked_lstm"
    
    def get_default_params(self) -> Dict[str, Any]:
        base_params = BiRNNModel().get_default_params()
        base_params['units'] = [64, 32]
        return base_params
    
    def get_param_ranges(self) -> Dict[str, Any]:
        base_ranges = BiRNNModel().get_param_ranges()
        base_ranges['units'] = [[32, 16], [64, 32], [128, 64], [256, 128]]
        return base_ranges
    
    def build_model(self, input_shape: Tuple[int, ...], num_classes: int, **params) -> tf.keras.Model:
        """Build Stacked LSTM model"""
        model = Sequential([Input(shape=input_shape)])
        
        units = params.get('units', [64, 32])
        dropout = params.get('dropout', 0.3)
        recurrent_dropout = params.get('recurrent_dropout', 0.2)
        
        # Add stacked LSTM layers
        for i, unit_count in enumerate(units):
            model.add(LSTM(
                unit_count,
                return_sequences=(i < len(units) - 1),
                activation='relu',
                dropout=dropout,
                recurrent_dropout=recurrent_dropout
            ))
        
        model.add(Dense(
            1 if num_classes == 2 else num_classes,
            activation='sigmoid' if num_classes == 2 else 'softmax'
        ))
        
        return model

class StackedGRUModel(BaseModel):
    """Stacked GRU Model"""
    
    def __init__(self):
        super().__init__()
        self.model_type = "stacked_gru"
    
    def get_default_params(self) -> Dict[str, Any]:
        return StackedLSTMModel().get_default_params()
    
    def get_param_ranges(self) -> Dict[str, Any]:
        return StackedLSTMModel().get_param_ranges()
    
    def build_model(self, input_shape: Tuple[int, ...], num_classes: int, **params) -> tf.keras.Model:
        """Build Stacked GRU model"""
        model = Sequential([Input(shape=input_shape)])
        
        units = params.get('units', [64, 32])
        dropout = params.get('dropout', 0.3)
        recurrent_dropout = params.get('recurrent_dropout', 0.2)
        
        # Add stacked GRU layers
        for i, unit_count in enumerate(units):
            model.add(GRU(
                unit_count,
                return_sequences=(i < len(units) - 1),
                activation='relu',
                dropout=dropout,
                recurrent_dropout=recurrent_dropout
            ))
        
        model.add(Dense(
            1 if num_classes == 2 else num_classes,
            activation='sigmoid' if num_classes == 2 else 'softmax'
        ))
        
        return model

# =============================================================================
# =============================================================================
# TRADITIONAL ML MODELS (Random Forest, XGBoost, LightGBM)
# =============================================================================

class RandomForestClassifierModel(BaseModel):
    """Random Forest Classifier - Ensemble tree-based classifier"""
    
    def __init__(self):
        super().__init__()
        self.model_type = "random_forest_clf"
        self.model = None
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        }
    
    def get_param_ranges(self) -> Dict[str, Any]:
        return {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
    
    def build_model(self, input_shape: Tuple[int, ...], num_classes: int, **params) -> Any:
        """Build Random Forest model"""
        return RandomForestClassifier(**params)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Train Random Forest classifier"""
        params = params or self.get_default_params()
        
        # Flatten if needed (for tabular data)
        if len(X_train.shape) > 2:
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_val = X_val.reshape(X_val.shape[0], -1)
        
        # Build and train model
        self.model = RandomForestClassifier(**params)
        self.model.fit(X_train, y_train)
        
        # Calculate metrics
        train_acc = self.model.score(X_train, y_train)
        val_acc = self.model.score(X_val, y_val)
        
        self.is_trained = True
        
        return {
            'final_train_accuracy': train_acc,
            'final_val_accuracy': val_acc,
            'epochs_trained': 1,  # Tree models train in one pass
            'best_epoch': 1
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        return self.model.predict_proba(X)


class RandomForestRegressorModel(BaseModel):
    """Random Forest Regressor - Ensemble tree-based regressor"""
    
    def __init__(self):
        super().__init__()
        self.model_type = "random_forest_reg"
        self.model = None
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        }
    
    def get_param_ranges(self) -> Dict[str, Any]:
        return {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', None]
        }
    
    def build_model(self, input_shape: Tuple[int, ...], num_classes: int, **params) -> Any:
        """Build Random Forest model"""
        return RandomForestRegressor(**params)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Train Random Forest regressor"""
        params = params or self.get_default_params()
        
        # Flatten if needed
        if len(X_train.shape) > 2:
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_val = X_val.reshape(X_val.shape[0], -1)
        
        # Build and train
        self.model = RandomForestRegressor(**params)
        self.model.fit(X_train, y_train)
        
        # Calculate R² score
        train_score = self.model.score(X_train, y_train)
        val_score = self.model.score(X_val, y_val)
        
        self.is_trained = True
        
        return {
            'final_train_accuracy': train_score,
            'final_val_accuracy': val_score,
            'epochs_trained': 1,
            'best_epoch': 1
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        return self.model.predict(X)


class XGBoostClassifierModel(BaseModel):
    """XGBoost Classifier - Gradient boosting classifier"""
    
    def __init__(self):
        super().__init__()
        self.model_type = "xgboost_clf"
        self.model = None
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
    
    def get_param_ranges(self) -> Dict[str, Any]:
        return {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 6, 9, 12],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }
    
    def build_model(self, input_shape: Tuple[int, ...], num_classes: int, **params) -> Any:
        """Build XGBoost model"""
        return XGBClassifier(**params)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Train XGBoost classifier"""
        params = params or self.get_default_params()
        
        # Flatten if needed
        if len(X_train.shape) > 2:
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_val = X_val.reshape(X_val.shape[0], -1)
        
        # Build and train
        self.model = XGBClassifier(**params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Calculate accuracy
        train_acc = self.model.score(X_train, y_train)
        val_acc = self.model.score(X_val, y_val)
        
        self.is_trained = True
        
        return {
            'final_train_accuracy': train_acc,
            'final_val_accuracy': val_acc,
            'epochs_trained': params.get('n_estimators', 100),
            'best_epoch': params.get('n_estimators', 100)
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        return self.model.predict_proba(X)


class XGBoostRegressorModel(BaseModel):
    """XGBoost Regressor - Gradient boosting regressor"""
    
    def __init__(self):
        super().__init__()
        self.model_type = "xgboost_reg"
        self.model = None
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
    
    def get_param_ranges(self) -> Dict[str, Any]:
        return {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 6, 9, 12],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }
    
    def build_model(self, input_shape: Tuple[int, ...], num_classes: int, **params) -> Any:
        """Build XGBoost model"""
        return XGBRegressor(**params)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Train XGBoost regressor"""
        params = params or self.get_default_params()
        
        # Flatten if needed
        if len(X_train.shape) > 2:
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_val = X_val.reshape(X_val.shape[0], -1)
        
        # Build and train
        self.model = XGBRegressor(**params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Calculate R² score
        train_score = self.model.score(X_train, y_train)
        val_score = self.model.score(X_val, y_val)
        
        self.is_trained = True
        
        return {
            'final_train_accuracy': train_score,
            'final_val_accuracy': val_score,
            'epochs_trained': params.get('n_estimators', 100),
            'best_epoch': params.get('n_estimators', 100)
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        return self.model.predict(X)
# MODEL REGISTRY  
# =============================================================================

class ModelRegistry:
    """Central registry for all available models"""
    
    # Build models dict dynamically based on what's available
    _models = {
        # RNN Models (Time Series) - Always available
        'bi_rnn': BiRNNModel,
        'bi_lstm': BiLSTMModel,
        'bi_gru': BiGRUModel,
        'stacked_lstm': StackedLSTMModel,
        'stacked_gru': StackedGRUModel,
    }
    
    # Add Traditional ML models only if libraries are installed
    if TRADITIONAL_ML_AVAILABLE:
        _models.update({
            # Traditional ML - Classification
            'random_forest_clf': RandomForestClassifierModel,
            'xgboost_clf': XGBoostClassifierModel,
            # Traditional ML - Regression
            'random_forest_reg': RandomForestRegressorModel,
            'xgboost_reg': XGBoostRegressorModel,
        })
    
    @classmethod
    def get_model(cls, model_name: str) -> BaseModel:
        """Get a model instance by name"""
        if model_name not in cls._models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(cls._models.keys())}")
        return cls._models[model_name]()
    
    @classmethod
    def list_models(cls) -> List[str]:
        """Get list of all available model names"""
        return list(cls._models.keys())
    
    @classmethod
    def register_model(cls, name: str, model_class: type) -> None:
        """Register a new model type"""
        if not issubclass(model_class, BaseModel):
            raise ValueError("Model class must inherit from BaseModel")
        cls._models[name] = model_class
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Dict[str, Any]:
        """Get information about a model"""
        model = cls.get_model(model_name)
        return {
            'name': model_name,
            'class': model.__class__.__name__,
            'model_type': model.model_type,
            'default_params': model.get_default_params(),
            'param_ranges': model.get_param_ranges()
        }
    
    @classmethod
    def get_models_by_category(cls) -> Dict[str, List[str]]:
        """Get models organized by category"""
        categories = {
            'RNN - Bidirectional': ['bi_rnn', 'bi_lstm', 'bi_gru'],
            'RNN - Stacked': ['stacked_lstm', 'stacked_gru'],
        }
        
        # Only add ML categories if libraries are available
        if TRADITIONAL_ML_AVAILABLE:
            categories.update({
                'Traditional ML - Classification': ['random_forest_clf', 'xgboost_clf'],
                'Traditional ML - Regression': ['random_forest_reg', 'xgboost_reg']
            })
        
        return categories
    
    @classmethod
    def get_rnn_models(cls) -> List[str]:
        """Get list of RNN model names"""
        return ['bi_rnn', 'bi_lstm', 'bi_gru', 'stacked_lstm', 'stacked_gru']
    
    @classmethod
    def get_classification_models(cls) -> List[str]:
        """Get list of classification model names"""
        return ['random_forest_clf', 'xgboost_clf']
    
    @classmethod
    def get_regression_models(cls) -> List[str]:
        """Get list of regression model names"""
        return ['random_forest_reg', 'xgboost_reg']
    
# Convenience functions for backward compatibility
def get_model(model_name: str) -> BaseModel:
    return ModelRegistry.get_model(model_name)

def list_available_models() -> List[str]:
    return ModelRegistry.list_models()

def register_model(name: str, model_class: type) -> None:
    ModelRegistry.register_model(name, model_class)

def get_model_info(model_name: str) -> Dict[str, Any]:
    return ModelRegistry.get_model_info(model_name)

def get_models_by_category() -> Dict[str, List[str]]:
    return ModelRegistry.get_models_by_category()