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
