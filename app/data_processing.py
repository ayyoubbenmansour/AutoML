"""
Data Processing Module
Handles all data loading, cleaning, preprocessing, and analysis operations
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from typing import Dict, Any, Tuple, List, Optional
import os

class DataLoader:
    """Handles file loading operations"""
    
    ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.xls', '.json'}
    
    def is_allowed_file(self, filename: str) -> bool:
        """Check if file extension is allowed"""
        return any(filename.lower().endswith(ext) for ext in self.ALLOWED_EXTENSIONS)
    
    def load_data(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Load data from various file formats"""
        ext = os.path.splitext(filepath)[1].lower()
        
        loaders = {
            '.csv': pd.read_csv,
            '.xlsx': pd.read_excel,
            '.xls': pd.read_excel,
            '.json': pd.read_json
        }
        
        if ext not in loaders:
            raise ValueError(f"Unsupported file type: {ext}")
        
        return loaders[ext](filepath, **kwargs)

class DataAnalyzer:
    """Handles data analysis and visualization"""
    
    def analyze_dataset_characteristics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze dataset characteristics"""
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        datetime_cols = df.select_dtypes(include=['datetime64', 'timedelta64']).columns

        return {
        'shape': df.shape,
        'has_missing_values': df.isnull().any().any(),
        'missing_values_count': df.isnull().sum().to_dict(),
        'numeric_columns': numeric_cols.tolist(),
        'categorical_columns': categorical_cols.tolist(),
        'datetime_columns': datetime_cols.tolist(),
        'has_categorical_columns': len(categorical_cols) > 0,
        'has_numeric_columns': len(numeric_cols) > 0,
        'has_temporal_columns': len(datetime_cols) > 0 ,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.astype(str).to_dict()
        }
    
    
    def generate_visualizations(self, df: pd.DataFrame, filename: str) -> Dict[str, str]:
        """Generate data visualizations (placeholder for actual implementation)"""
        # This would contain actual visualization logic
        return {}

class DataCleaner:
    """Handles data cleaning operations"""
    
    def handle_missing_values(self, df: pd.DataFrame, method: str = 'remove') -> pd.DataFrame:
        """Handle missing values using specified method"""
        methods = {
            'remove': lambda df: df.dropna(),
            'mean': lambda df: df.fillna(df.mean()),
            'median': lambda df: df.fillna(df.median()),
            'mode': lambda df: df.fillna(df.mode().iloc[0] if not df.mode().empty else 0),
            'forward_fill': lambda df: df.fillna(method='ffill'),
            'backward_fill': lambda df: df.fillna(method='bfill'),
            'zero': lambda df: df.fillna(0)
        }
        
        if method not in methods:
            raise ValueError(f"Unknown method: {method}. Available: {list(methods.keys())}")
        
        return methods[method](df)
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows"""
        return df.drop_duplicates()
    
    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean column names (remove spaces, special characters)"""
        df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('[^a-zA-Z0-9_]', '', regex=True)
        return df

class DataPreprocessor:
    """Handles data preprocessing for model training"""

    def __init__(self):
        self.scaler = None
        self.label_encoders = {}
        self.cleaner = DataCleaner()

    def decompose_temporal_columns(self, df: pd.DataFrame, temporal_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Decompose temporal (datetime) columns into useful features:
        Year, Month, Day, DayOfWeek, Hour, Minute, Second, IsWeekend
        """
        df_copy = df.copy()
        if temporal_cols is None:
            temporal_cols = df_copy.select_dtypes(include=['datetime64[ns]', 'datetime64']).columns.tolist()
        for col in temporal_cols:
            df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
            if df_copy[col].isnull().all():
                continue
            df_copy[f"{col}_year"] = df_copy[col].dt.year
            df_copy[f"{col}_month"] = df_copy[col].dt.month
            df_copy[f"{col}_day"] = df_copy[col].dt.day
            df_copy[f"{col}_dayofweek"] = df_copy[col].dt.dayofweek
            df_copy[f"{col}_hour"] = df_copy[col].dt.hour
            df_copy[f"{col}_minute"] = df_copy[col].dt.minute
            df_copy[f"{col}_second"] = df_copy[col].dt.second
            df_copy[f"{col}_is_weekend"] = df_copy[col].dt.dayofweek.isin([5,6]).astype(int)
        return df_copy

    def cyclical_encode_temporal_columns(self, df: pd.DataFrame, temporal_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Encode temporal columns as cyclical features (e.g., month, day, hour)
        """
        df_copy = df.copy()
        if temporal_cols is None:
            temporal_cols = df_copy.select_dtypes(include=['datetime64[ns]', 'datetime64']).columns.tolist()
        for col in temporal_cols:
            df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
            if df_copy[col].isnull().all():
                continue
            # Example: cyclical encoding for month and hour
            df_copy[f"{col}_month_sin"] = np.sin(2 * np.pi * df_copy[col].dt.month / 12)
            df_copy[f"{col}_month_cos"] = np.cos(2 * np.pi * df_copy[col].dt.month / 12)
            df_copy[f"{col}_hour_sin"] = np.sin(2 * np.pi * df_copy[col].dt.hour / 24)
            df_copy[f"{col}_hour_cos"] = np.cos(2 * np.pi * df_copy[col].dt.hour / 24)
        return df_copy

    def timestamp_temporal_columns(self, df: pd.DataFrame, temporal_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Convert temporal columns to UNIX timestamp
        """
        df_copy = df.copy()
        if temporal_cols is None:
            temporal_cols = df_copy.select_dtypes(include=['datetime64[ns]', 'datetime64']).columns.tolist()
        for col in temporal_cols:
            df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
            if df_copy[col].isnull().all():
                continue
            df_copy[f"{col}_timestamp"] = df_copy[col].astype('int64') // 10**9
        return df_copy

    def create_sequences(self, X: np.ndarray, y: np.ndarray, seq_length: int):
        Xs, ys = [], []
        for i in range(len(X) - seq_length + 1):
            Xs.append(X[i:(i + seq_length)])
            ys.append(y[i + seq_length - 1])
        return np.array(Xs), np.array(ys)

    def preprocess_data(
        self,
        df: pd.DataFrame,
        target_col: str,
        seq_length: int = 1,
        missing_handling: str = 'remove',
        encoding: str = 'label',
        scaling: str = 'standard',
        temporal_handling: str = 'decompose',
        test_size: float = 0.2,
        random_state: int = 42,
        problem_type: str = 'classification',
        forecasting_steps: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # 1. Handle missing values
        df = self.cleaner.handle_missing_values(df, missing_handling)
    
        # 2. Temporal handling
        if temporal_handling == 'decompose':
            df = self.decompose_temporal_columns(df)
        elif temporal_handling == 'cyclical':
            df = self.cyclical_encode_temporal_columns(df)
        elif temporal_handling == 'timestamp':
            df = self.timestamp_temporal_columns(df)
        # else: keep as is
    
        # 3. Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
    
        # 4. Encode target if needed
        if problem_type == 'classification' and (y.dtype == 'object' or str(y.dtype).startswith('category')):
            le = LabelEncoder()
            y = le.fit_transform(y)
    
        # 5. Encode categorical features (pass y if target encoding)
        X = self._encode_categorical(X, encoding, y)
    
        # 6. Scale features (X must be 2D here)
        X = self._scale_features(X, scaling)
    
        # 7. Split train/test (still 2D)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
        # 8. Create sequences (now X becomes 3D if seq_length > 1)
        if problem_type == 'forecasting':
            X_train, y_train = self.create_forecasting_sequences(X_train, y_train, seq_length, forecasting_steps)
            X_test, y_test = self.create_forecasting_sequences(X_test, y_test, seq_length, forecasting_steps)
        elif seq_length > 1:
            X_train, y_train = self.create_sequences(X_train, y_train, seq_length)
            X_test, y_test = self.create_sequences(X_test, y_test, seq_length)
    
        return X_train, X_test, y_train, y_test
    
    def _scale_features(self, X: pd.DataFrame, method: str) -> np.ndarray:
        """Scale features using the specified method"""
        if method == 'standard':
            scaler = StandardScaler()
            return scaler.fit_transform(X)
        elif method == 'minmax':
            scaler = MinMaxScaler()
            return scaler.fit_transform(X)
        else:
            raise ValueError(f"Unknown scaling method: {method}")

    def create_forecasting_sequences(self, X, y, seq_length, forecasting_steps):
        Xs, ys = [], []
        for i in range(len(X) - seq_length - forecasting_steps + 1):
            Xs.append(X[i:(i + seq_length)])
            ys.append(y[(i + seq_length):(i + seq_length + forecasting_steps)])
        return np.array(Xs), np.array(ys)

    def _target_encode(self, X: pd.DataFrame, y: pd.Series, columns: List[str]) -> pd.DataFrame:
        X_encoded = X.copy()
        for col in columns:
            means = y.groupby(X_encoded[col]).mean()
            X_encoded[col] = X_encoded[col].map(means)
        return X_encoded

    def _encode_categorical(self, X: pd.DataFrame, method: str, y: Optional[pd.Series] = None) -> np.ndarray:
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) == 0:
            return X.values
        if method == 'label':
            X_encoded = X.copy()
            for col in categorical_cols:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X[col].astype(str))
            return X_encoded.values
        elif method == 'onehot':
            return pd.get_dummies(X, columns=categorical_cols).values
        elif method == 'target':
            if y is None:
                raise ValueError("Target encoding requires the target variable y.")
            X_encoded = self._target_encode(X, y, categorical_cols)
            return X_encoded.values
        else:
            raise ValueError(f"Unknown encoding method: {method}")           
         

    # Convenience functions for backward compatibility
def load_data(filepath: str, **kwargs) -> pd.DataFrame:
    """Load data from file"""
    return DataLoader().load_data(filepath, **kwargs)

def analyze_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze dataset characteristics"""
    return DataAnalyzer().analyze_dataset_characteristics(df)


def generate_visualizations(df: pd.DataFrame, filename: str) -> Dict[str, str]:
    """Generate visualizations"""
    return DataAnalyzer().generate_visualizations(df, filename)

def preprocess_data(df: pd.DataFrame, target_col: str, **kwargs) -> Tuple:
    """Preprocess data for training"""
    return DataPreprocessor().preprocess_data(df, target_col, **kwargs)