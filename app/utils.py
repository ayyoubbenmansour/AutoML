"""
Utility Functions Module
Provides helper functions and utilities for the application
"""
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union
import json
import os
import sys
import platform

# =============================================================================
# DATA UTILITIES
# =============================================================================

def convert_for_json_serialization(obj: Any) -> Any:
    """
    Convert objects for JSON serialization
    Handles numpy types and other non-serializable objects
    """
    if isinstance(obj, dict):
        return {k: convert_for_json_serialization(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(convert_for_json_serialization(item) for item in obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # numpy scalars
        return obj.item()
    elif pd.isna(obj):
        return None
    else:
        return obj

def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Comprehensive data quality check
    
    Args:
        df: DataFrame to check
        
    Returns:
        Dictionary with quality metrics and recommendations
    """
    total_cells = df.size
    missing_cells = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()
    
    # Calculate quality scores
    completeness = ((total_cells - missing_cells) / total_cells * 100) if total_cells > 0 else 0
    uniqueness = ((len(df) - duplicate_rows) / len(df) * 100) if len(df) > 0 else 0
    
    # Overall score
    overall_score = (completeness * 0.7 + uniqueness * 0.3)
    
    # Determine quality level
    if overall_score >= 90:
        quality_level, quality_color = "Excellent", "success"
    elif overall_score >= 80:
        quality_level, quality_color = "Good", "info"
    elif overall_score >= 60:
        quality_level, quality_color = "Fair", "warning"
    else:
        quality_level, quality_color = "Poor", "danger"
    
    return {
        'overall_score': round(overall_score, 1),
        'completeness': round(completeness, 1),
        'uniqueness': round(uniqueness, 1),
        'quality_level': quality_level,
        'quality_color': quality_color,
        'missing_cells': int(missing_cells),
        'duplicate_rows': int(duplicate_rows),
        'total_cells': total_cells,
        'recommendations': _generate_quality_recommendations(df, completeness, uniqueness)
    }

def _generate_quality_recommendations(df: pd.DataFrame, completeness: float, uniqueness: float) -> List[str]:
    """Generate data quality improvement recommendations"""
    recommendations = []
    
    # Missing values
    if completeness < 95:
        missing_pct = 100 - completeness
        recommendations.append(f"Handle {missing_pct:.1f}% missing values using imputation or removal")
    
    # Duplicate rows
    if uniqueness < 95:
        duplicate_pct = 100 - uniqueness
        recommendations.append(f"Remove {duplicate_pct:.1f}% duplicate rows to improve data quality")
    
    # High cardinality categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > 0.9:
            recommendations.append(f"Consider encoding or grouping '{col}' (high cardinality: {unique_ratio:.1%})")
    
    # Constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        recommendations.append(f"Remove constant columns: {', '.join(constant_cols[:3])}")
    
    # Highly correlated features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr().abs()
        high_corr = np.where((corr_matrix > 0.95) & (corr_matrix < 1))
        if len(high_corr[0]) > 0:
            recommendations.append("Consider removing highly correlated features (correlation > 0.95)")
    
    return recommendations[:5]  # Return top 5 recommendations

# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def validate_preprocessing_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize preprocessing configuration
    
    Args:
        config: Raw preprocessing configuration
        
    Returns:
        Validated configuration with defaults
    """
    validated = {}
    
    # Sequence length
    seq_length = config.get('seq_length', 1)
    validated['seq_length'] = max(1, min(100, int(seq_length)))
    
    # Missing value handling
    missing_methods = ['remove', 'mean', 'median', 'mode', 'forward_fill', 'backward_fill', 'zero']
    missing_handling = config.get('missing_handling', 'remove')
    validated['missing_handling'] = missing_handling if missing_handling in missing_methods else 'remove'
    
    # Encoding method
    encoding_methods = ['label', 'onehot']
    encoding = config.get('encoding', 'label')
    validated['encoding'] = encoding if encoding in encoding_methods else 'label'
    
    # Scaling method
    scaling_methods = ['standard', 'minmax', 'robust', 'none']
    scaling = config.get('scaling', 'standard')
    validated['scaling'] = scaling if scaling in scaling_methods else 'standard'
    
    # Test size
    test_size = config.get('test_size', 0.2)
    try:
        test_size = float(test_size)
        validated['test_size'] = max(0.1, min(0.5, test_size))
    except (ValueError, TypeError):
        validated['test_size'] = 0.2
    
    return validated

def validate_model_parameters(params: Dict[str, Any], param_ranges: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate model parameters against allowed ranges
    
    Args:
        params: Parameters to validate
        param_ranges: Allowed ranges for each parameter
        
    Returns:
        Validated parameters
    """
    validated = {}
    
    for param_name, value in params.items():
        if param_name not in param_ranges:
            validated[param_name] = value
            continue
        
        param_range = param_ranges[param_name]
        
        if isinstance(param_range, tuple) and len(param_range) == 2:
            # Numeric range (min, max)
            min_val, max_val = param_range
            try:
                numeric_value = float(value)
                validated[param_name] = max(min_val, min(max_val, numeric_value))
            except (ValueError, TypeError):
                validated[param_name] = value
        
        elif isinstance(param_range, list):
            # Choice list
            validated[param_name] = value if value in param_range else param_range[0]
        
        else:
            validated[param_name] = value
    
    return validated

# =============================================================================
# FORMATTING UTILITIES
# =============================================================================

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(size_bytes)
    
    while size >= 1024.0 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1
    
    return f"{size:.1f} {size_names[i]}"

def format_training_time(seconds: float) -> str:
    """Format training time in human readable format"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.1f} minutes"
    else:
        return f"{seconds/3600:.1f} hours"

def format_number(num: Union[int, float], decimals: int = 2) -> str:
    """Format number with thousands separator and decimal places"""
    if isinstance(num, int):
        return f"{num:,}"
    else:
        return f"{num:,.{decimals}f}"

def truncate_string(text: str, max_length: int = 50, suffix: str = "...") -> str:
    """Truncate string to maximum length with suffix"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

# =============================================================================
# SYSTEM UTILITIES
# =============================================================================

def get_system_info() -> Dict[str, Any]:
    """Get system information for debugging"""
    info = {
        'python_version': sys.version,
        'platform': platform.platform(),
        'processor': platform.processor(),
        'architecture': platform.architecture()[0]
    }
    
    try:
        import psutil
        info.update({
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage_percent': psutil.disk_usage('/').percent
        })
    except ImportError:
        info['psutil_available'] = False
    
    return info

def calculate_memory_usage(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate DataFrame memory usage"""
    memory_usage = df.memory_usage(deep=True)
    total_mb = memory_usage.sum() / (1024 * 1024)
    
    return {
        'total_mb': round(total_mb, 2),
        'per_column_mb': {
            col: round(usage / (1024 * 1024), 2)
            for col, usage in memory_usage.items()
        },
        'average_row_kb': round((memory_usage.sum() / len(df)) / 1024, 2) if len(df) > 0 else 0,
        'estimated_model_memory_mb': round(total_mb * 3, 2)  # Rough estimate for model training
    }

# =============================================================================
# MODEL UTILITIES
# =============================================================================

def extract_model_metrics_summary(metrics: Dict[str, float]) -> str:
    """Extract and format key metrics for display"""
    if not metrics:
        return "No metrics available"
    
    summary_parts = []
    
    # Priority metrics
    priority_metrics = ['accuracy', 'f1_score', 'auc', 'precision', 'recall']
    
    for metric in priority_metrics:
        if metric in metrics:
            value = metrics[metric]
            formatted_value = f"{value:.3f}" if isinstance(value, float) else str(value)
            metric_name = metric.replace('_', ' ').title()
            summary_parts.append(f"{metric_name}: {formatted_value}")
    
    return " | ".join(summary_parts) if summary_parts else "Metrics calculated"

def create_model_comparison_table(models: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create comparison table for multiple models"""
    if not models:
        return pd.DataFrame()
    
    comparison_data = []
    for model in models:
        row = {
            'Model': model.get('name', 'Unknown'),
            'Accuracy': model.get('metrics', {}).get('accuracy', 0),
            'F1-Score': model.get('metrics', {}).get('f1_score', 0),
            'Precision': model.get('metrics', {}).get('precision', 0),
            'Recall': model.get('metrics', {}).get('recall', 0),
            'Training Time': model.get('training_time', 0)
        }
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('Accuracy', ascending=False)
    return df

# =============================================================================
# FILE UTILITIES
# =============================================================================

def allowed_file(filename: str, allowed_extensions: set = None) -> bool:
    """Check if file has an allowed extension"""
    if allowed_extensions is None:
        allowed_extensions = {'csv', 'xlsx', 'xls', 'json'}
    
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def get_file_info(filepath: str) -> Dict[str, Any]:
    """Get information about a file"""
    if not os.path.exists(filepath):
        return {'exists': False}
    
    stat = os.stat(filepath)
    return {
        'exists': True,
        'size': stat.st_size,
        'size_formatted': format_file_size(stat.st_size),
        'modified': stat.st_mtime,
        'extension': os.path.splitext(filepath)[1].lower()
    }

def cleanup_old_files(directory: str, max_files: int = 50, pattern: str = '*.png') -> int:
    """Clean up old files in a directory"""
    import glob
    
    if not os.path.exists(directory):
        return 0
    
    files = glob.glob(os.path.join(directory, pattern))
    
    if len(files) <= max_files:
        return 0
    
    # Sort by modification time
    files.sort(key=lambda x: os.path.getmtime(x))
    
    # Remove oldest files
    files_to_remove = files[:len(files) - max_files]
    removed_count = 0
    
    for filepath in files_to_remove:
        try:
            os.remove(filepath)
            removed_count += 1
        except Exception:
            pass
    
    return removed_count

# =============================================================================
# MATH UTILITIES
# =============================================================================

def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default

def calculate_percentage(value: float, total: float, decimals: int = 1) -> str:
    """Calculate and format percentage"""
    if total == 0:
        return "0%"
    percentage = (value / total) * 100
    return f"{percentage:.{decimals}f}%"

def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """Calculate basic statistics for a list of values"""
    if not values:
        return {}
    
    arr = np.array(values)
    return {
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'q25': float(np.percentile(arr, 25)),
        'q75': float(np.percentile(arr, 75))
    }

# =============================================================================
# HTML/DISPLAY UTILITIES
# =============================================================================

def create_progress_bar_html(value: float, max_value: float = 100, 
                            color: str = 'primary', height: str = '20px') -> str:
    """Create HTML progress bar"""
    percentage = safe_division(value, max_value, 0) * 100
    
    return f"""
    <div class="progress" style="height: {height};">
        <div class="progress-bar bg-{color}" role="progressbar" 
             style="width: {percentage}%" 
             aria-valuenow="{value}" 
             aria-valuemin="0" 
             aria-valuemax="{max_value}">
            {percentage:.1f}%
        </div>
    </div>
    """

def create_badge_html(text: str, color: str = 'primary') -> str:
    """Create HTML badge"""
    return f'<span class="badge badge-{color}">{text}</span>'

def create_alert_html(message: str, alert_type: str = 'info', dismissible: bool = True) -> str:
    """Create HTML alert"""
    dismiss_button = """
    <button type="button" class="close" data-dismiss="alert" aria-label="Close">
        <span aria-hidden="true">&times;</span>
    </button>
    """ if dismissible else ""
    
    dismissible_class = "alert-dismissible fade show" if dismissible else ""
    
    return f"""
    <div class="alert alert-{alert_type} {dismissible_class}" role="alert">
        {message}
        {dismiss_button}
    </div>
    """

# =============================================================================
# SESSION UTILITIES
# =============================================================================

def clear_session_data(session: dict, keep_keys: List[str] = None) -> None:
    """Clear session data except specified keys"""
    if keep_keys is None:
        keep_keys = []
    
    keys_to_remove = [key for key in session.keys() if key not in keep_keys]
    for key in keys_to_remove:
        session.pop(key, None)

def get_session_info(session: dict) -> Dict[str, Any]:
    """Get summary of session data"""
    return {
        'keys': list(session.keys()),
        'has_file': 'filename' in session,
        'has_target': 'target_column' in session,
        'has_config': 'preprocessing_config' in session,
        'filename': session.get('filename', 'No file'),
        'target_column': session.get('target_column', 'Not selected'),
        'data_shape': session.get('data_shape', 'Unknown')
    }

# =============================================================================
# ERROR HANDLING UTILITIES
# =============================================================================

def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """Safely parse JSON string"""
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default

def safe_float_conversion(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int_conversion(value: Any, default: int = 0) -> int:
    """Safely convert value to integer"""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

# =============================================================================
# LOGGING UTILITIES
# =============================================================================

def log_training_info(logger, model_type: str, params: Dict[str, Any], 
                     metrics: Dict[str, float]) -> None:
    """Log training information"""
    logger.info(f"Model: {model_type}")
    logger.info(f"Parameters: {json.dumps(params, indent=2)}")
    logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")

def log_error_with_context(logger, error: Exception, context: Dict[str, Any]) -> None:
    """Log error with additional context"""
    logger.error(f"Error: {str(error)}")
    logger.error(f"Context: {json.dumps(context, indent=2)}")
    logger.exception("Full traceback:")

# =============================================================================
# OPTIMIZATION UTILITIES
# =============================================================================

def save_model(model, filepath: str, metadata: Dict[str, Any] = None) -> bool:
    """
    Save Keras model with metadata
    
    Args:
        model: Trained Keras model
        filepath: Path to save model
        metadata: Optional metadata dictionary
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import tensorflow as tf
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        model.save(filepath)
        
        # Save metadata if provided
        if metadata:
            metadata_path = filepath + '.metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False

def load_model(filepath: str) -> tuple:
    """
    Load Keras model with metadata
    
    Args:
        filepath: Path to model file
        
    Returns:
        Tuple of (model, metadata)
    """
    try:
        import tensorflow as tf
        
        # Load model
        model = tf.keras.models.load_model(filepath)
        
        # Load metadata if exists
        metadata = {}
        metadata_path = filepath + '.metadata.json'
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return model, metadata
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, {}

def cache_data(data: Any, cache_key: str, cache_dir: str = 'cache') -> bool:
    """
    Cache data to disk
    
    Args:
        data: Data to cache (must be picklable)
        cache_key: Unique identifier for cached data
        cache_dir: Directory to store cache files
        
    Returns:
        True if successful
    """
    try:
        import pickle
        
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")
        
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        
        return True
    except Exception as e:
        print(f"Error caching data: {e}")
        return False

def load_cached_data(cache_key: str, cache_dir: str = 'cache') -> Optional[Any]:
    """
    Load cached data from disk
    
    Args:
        cache_key: Unique identifier for cached data
        cache_dir: Directory where cache files are stored
        
    Returns:
        Cached data or None if not found
    """
    try:
        import pickle
        
        cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")
        
        if not os.path.exists(cache_path):
            return None
        
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        
        return data
    except Exception as e:
        print(f"Error loading cached data: {e}")
        return None

def clear_cache(cache_dir: str = 'cache') -> int:
    """
    Clear all cached files
    
    Args:
        cache_dir: Directory to clear
        
    Returns:
        Number of files deleted
    """
    import glob
    
    if not os.path.exists(cache_dir):
        return 0
    
    cache_files = glob.glob(os.path.join(cache_dir, '*.pkl'))
    deleted = 0
    
    for file in cache_files:
        try:
            os.remove(file)
            deleted += 1
        except Exception:
            pass
    
    return deleted

def profile_function(func):
    """
    Decorator to profile function execution time
    
    Usage:
        @profile_function
        def my_function():
            ...
    """
    import time
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    
    return wrapper

def batch_generator(data: np.ndarray, labels: np.ndarray, batch_size: int = 32, shuffle: bool = True):
    """
    Generate batches of data for efficient processing
    
    Args:
        data: Input data array
        labels: Label array
        batch_size: Size of each batch
        shuffle: Whether to shuffle data
        
    Yields:
        Batches of (data, labels)
    """
    n_samples = len(data)
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        
        yield data[batch_indices], labels[batch_indices]

# =============================================================================
# IMAGE UTILITIES (for CNN models)
# =============================================================================

def allowed_image_file(filename: str) -> bool:
    """Check if file is an allowed image type"""
    image_extensions = {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in image_extensions

def get_image_info(filepath: str) -> Dict[str, Any]:
    """Get information about an image file"""
    try:
        from PIL import Image
        
        with Image.open(filepath) as img:
            return {
                'exists': True,
                'width': img.width,
                'height': img.height,
                'format': img.format,
                'mode': img.mode,
                'size_bytes': os.path.getsize(filepath),
                'size_formatted': format_file_size(os.path.getsize(filepath))
            }
    except Exception as e:
        return {'exists': False, 'error': str(e)}

def resize_image(image: np.ndarray, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Resize image to target size
    
    Args:
        image: Image array
        target_size: Target (width, height)
        
    Returns:
        Resized image array
    """
    try:
        import cv2
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    except ImportError:
        from PIL import Image
        img = Image.fromarray(image.astype('uint8'))
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        return np.array(img)

def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image pixel values to [0, 1]"""
    return image.astype(np.float32) / 255.0

# =============================================================================
# EXPORT ALL UTILITIES
# =============================================================================

__all__ = [
    # Data utilities
    'convert_for_json_serialization',
    'check_data_quality',
    
    # Validation utilities
    'validate_preprocessing_config',
    'validate_model_parameters',
    
    # Formatting utilities
    'format_file_size',
    'format_training_time',
    'format_number',
    'truncate_string',
    
    # System utilities
    'get_system_info',
    'calculate_memory_usage',
    
    # Model utilities
    'extract_model_metrics_summary',
    'create_model_comparison_table',
    
    # File utilities
    'allowed_file',
    'get_file_info',
    'cleanup_old_files',
    
    # Math utilities
    'safe_division',
    'calculate_percentage',
    'calculate_statistics',
    
    # HTML utilities
    'create_progress_bar_html',
    'create_badge_html',
    'create_alert_html',
    
    # Session utilities
    'clear_session_data',
    'get_session_info',
    
    # Error handling utilities
    'safe_json_loads',
    'safe_float_conversion',
    'safe_int_conversion',
    
    # Logging utilities
    'log_training_info',
    'log_error_with_context',
    
    # Optimization utilities
    'save_model',
    'load_model',
    'cache_data',
    'load_cached_data',
    'clear_cache',
    'profile_function',
    'batch_generator',
    
    # Image utilities
    'allowed_image_file',
    'get_image_info',
    'resize_image',
    'normalize_image'
]