"""
Application Configuration Module
Simplified configuration that works out of the box
"""
import os
from typing import Dict, Any, List

class Config:
    """Base configuration class"""
    
    # Application settings
    APP_NAME = 'AutoML Pro'
    VERSION = '1.0.0'
    SECRET_KEY = 'dev-secret-key-change-in-production-2024'  # Default key that always works
    
    # File upload settings
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'json'}
    
    # Data processing settings
    MAX_CATEGORICAL_UNIQUE_VALUES = 50
    DEFAULT_TEST_SIZE = 0.2
    DEFAULT_VALIDATION_SPLIT = 0.2
    RANDOM_STATE = 42
    
    # Model training settings
    DEFAULT_EPOCHS = 100
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_PATIENCE = 10
    
    # Available model types
    AVAILABLE_MODEL_TYPES = [
        # RNN Models
        'bi_rnn', 'bi_lstm', 'bi_gru',
        'stacked_lstm', 'stacked_gru',
        # Traditional ML Classification
        'random_forest_clf', 'xgboost_clf',
        # Traditional ML Regression
        'random_forest_reg', 'xgboost_reg'
    ]
    
    # Visualization settings
    PLOT_DPI = 150
    PLOT_STYLE = 'default'
    PLOT_BACKEND = 'Agg'
    MAX_PLOT_FILES = 50
    
    # HPO settings
    HPO_DEFAULT_TRIALS = 25
    HPO_DEFAULT_TIMEOUT = 3600  # 1 hour
    HPO_AVAILABLE_METHODS = ['bayesian', 'grid', 'random']
    
    # Architecture info
    ARCHITECTURE_VERSION = '1.0.0'
    ARCHITECTURE_TYPE = 'consolidated_single_files'
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        os.makedirs(cls.UPLOAD_FOLDER, exist_ok=True)
    
    @classmethod
    def get_architecture_info(cls) -> Dict[str, Any]:
        """Get information about the current architecture"""
        return {
            'version': cls.ARCHITECTURE_VERSION,
            'type': cls.ARCHITECTURE_TYPE,
            'app_name': cls.APP_NAME,
            'model_types': cls.AVAILABLE_MODEL_TYPES,
            'hpo_methods': cls.HPO_AVAILABLE_METHODS
        }
    
    @classmethod
    def validate_file_extension(cls, filename: str) -> bool:
        """Check if file extension is allowed"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in cls.ALLOWED_EXTENSIONS

class DevelopmentConfig(Config):
    """Development environment configuration"""
    DEBUG = True
    TESTING = False
    ENV = 'development'
    
    # Development-specific settings
    SEND_FILE_MAX_AGE_DEFAULT = 0  # Disable caching in development

class ProductionConfig(Config):
    """Production environment configuration"""
    DEBUG = False
    TESTING = False
    ENV = 'production'
    
    # Production optimizations
    SEND_FILE_MAX_AGE_DEFAULT = 31536000  # 1 year for static files
    
    # Use environment variable if available, otherwise use default
    SECRET_KEY = os.environ.get('SECRET_KEY', Config.SECRET_KEY)

class TestingConfig(Config):
    """Testing environment configuration"""
    TESTING = True
    DEBUG = True
    ENV = 'testing'
    WTF_CSRF_ENABLED = False
    
    # Test-specific settings
    UPLOAD_FOLDER = 'test_uploads'
    HPO_DEFAULT_TRIALS = 5
    HPO_DEFAULT_TIMEOUT = 300  # 5 minutes
    DEFAULT_EPOCHS = 10

# Simple configuration dictionary - defaults to development
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig  # Always use development by default
}