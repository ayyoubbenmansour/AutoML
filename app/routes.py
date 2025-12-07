"""
Flask Routes Module
Handles all HTTP endpoints and request processing
"""
from flask import (
    Blueprint, request, render_template, redirect, url_for, 
    flash, session, current_app, jsonify
)
from matplotlib import units
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename
import os
import logging
from typing import Dict, Any

# Import consolidated modules
from .data_processing import DataLoader, DataAnalyzer, DataPreprocessor, load_data, analyze_data
from .training import SimpleTrainer, CustomTrainer, HPOTrainer
from .evaluation import ModelEvaluator
from .models import ModelRegistry
from .utils import (
    convert_for_json_serialization, format_file_size,
    check_data_quality, validate_preprocessing_config
)

# Setup blueprint and logger
main_bp = Blueprint('main', __name__)
logger = logging.getLogger(__name__)

# =============================================================================
# MAIN ROUTES
# =============================================================================

@main_bp.route('/')
def home():
    """Home page with project overview"""
    return render_template('home_page.html')

@main_bp.route('/upload_data', methods=['GET', 'POST'])
def upload():
    """Handle file upload and initial validation"""
    if request.method == 'POST':
        # Validate file presence
        if 'dataset' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['dataset']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        # Validate file type
        loader = DataLoader()
        if not loader.is_allowed_file(file.filename):
            flash('Invalid file type. Supported formats: CSV, Excel, JSON', 'error')
            return redirect(request.url)
        
        try:
            # Save file
            filename = secure_filename(file.filename)
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            logger.info(f"File uploaded: {filename}")
            
            # Load and validate data
            df = load_data(filepath)
            
            # Check data quality
            quality_report = check_data_quality(df)
            
            # Store in session
            session['filename'] = filename
            session['filepath'] = filepath
            session['data_shape'] = df.shape
            session['columns'] = df.columns.tolist()
            session['quality_report'] = quality_report
            
            flash(
                f'File uploaded successfully! Dataset contains {df.shape[0]:,} rows and {df.shape[1]} columns.',
                'success'
            )
            
            return redirect(url_for('main.visualize'))
            
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            flash(f'Error processing file: {str(e)}', 'error')
            return redirect(request.url)
    
    return render_template('upload_data.html')

@main_bp.route('/visualization', methods=['GET', 'POST'])
def visualize():
    """Data visualization and target column selection"""
    filename = session.get('filename')
    if not filename:
        flash("No file uploaded. Please upload a dataset first.", 'warning')
        return redirect(url_for('main.upload'))
    
    try:
        filepath = session.get('filepath')
        df = load_data(filepath)
        
        # Analyze data
        analyzer = DataAnalyzer()
        characteristics = analyzer.analyze_dataset_characteristics(df)
        
        # Handle POST request for target selection
        if request.method == 'POST':
            target_col = request.form.get('target_column')
            if not target_col:
                flash('Please select a target column.', 'warning')
            elif target_col not in df.columns:
                flash('Invalid target column selected.', 'error')
            else:
                session['target_column'] = target_col
                flash(f'Target column selected: {target_col}', 'success')
                return redirect(url_for('main.preprocess_config'))
        stats = {
            'shape': df.shape,
            'dtypes': df.dtypes.astype(str).to_dict()
        }    
        viz_data = {
            'filename': filename,
            'stats': stats,
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'characteristics': characteristics,
            'sample_data': df.head(10).to_html(
                classes='table table-striped',
                table_id='sample-table',
                index=False
            ),
            'quality_report': session.get('quality_report', {})
        }
        
        return render_template('visualization_data.html', **viz_data)
        
    except Exception as e:
        logger.error(f"Error visualizing data: {e}")
        flash(f'Error visualizing data: {str(e)}', 'error')
        return redirect(url_for('main.upload'))

@main_bp.route('/preprocess_config', methods=['GET', 'POST'])
def preprocess_config():
    if not all(k in session for k in ['filename', 'target_column', 'filepath']):
        flash("Session expirée. Veuillez recharger vos données.", "error")
        return redirect(url_for('main.upload_data'))

    if request.method == 'POST':
        try:
            # 1. Extract preprocessing configuration from form
            config = {
                'seq_length': int(request.form.get('sequence_length', 1)),
                'missing_handling': request.form.get('missing_handling', 'remove'),
                'encoding': request.form.get('encoding', 'label'),
                'scaling': request.form.get('scaling', 'standard'),
                'problem_type': request.form.get('problem_type', 'classification'),
                'temporal_handling': request.form.get('temporal_handling', 'decompose'),
                'test_size': float(request.form.get('test_size', 0.2)),
                'forecasting_steps': int(request.form.get('forecasting_steps', 1)),
                'random_state': int(request.form.get('random_state', 42))
            }

            session['preprocessing_config'] = config

            # 2. Load original data
            filepath = session.get('filepath')
            df = load_data(filepath)
            target_col = session.get('target_column')

            # 3. Preprocess data immediately, including temporal handling
            preprocessor = DataPreprocessor()
            X_train, X_test, y_train, y_test = preprocessor.preprocess_data(
                df,
                target_col,
                seq_length=config['seq_length'],
                missing_handling=config['missing_handling'],
                encoding=config['encoding'],
                scaling=config['scaling'],
                temporal_handling=config['temporal_handling'],
                test_size=config['test_size'],
                random_state=config['random_state'],
                problem_type=config['problem_type'],
                forecasting_steps=config['forecasting_steps']
            )

            processed_dir = os.path.join(current_app.root_path, 'static', 'processed')
            os.makedirs(processed_dir, exist_ok=True)

            if len(X_train.shape) == 2:
                # 2D: Save as CSV
                processed_df = pd.DataFrame(X_train)
                processed_df['target'] = y_train
                processed_filename = f"processed_{session['filename']}"
                processed_path = os.path.join(processed_dir, processed_filename)
                processed_df.to_csv(processed_path, index=False)
                session['processed_data_path'] = processed_path
            else:
                # 3D: Save as .npy
                processed_filename = f"processed_{session['filename'].rsplit('.', 1)[0]}.npy"
                processed_path = os.path.join(processed_dir, processed_filename)
                np.save(processed_path, {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test})
                session['processed_data_path'] = processed_path


            flash("Prétraitement appliqué et données sauvegardées.", 'success')
            return redirect(url_for('main.processing_mode'))

        except Exception as e:
            current_app.logger.error(f"Error during preprocessing: {e}")
            flash(f"Erreur lors du prétraitement: {str(e)}", 'error')

    # GET: Render form as before
    filepath = session.get('filepath')
    df = load_data(filepath)
    characteristics = analyze_data(df)
    context = {
        'filename': session.get('filename'),
        'target_column': session.get('target_column'),
        'characteristics': characteristics
    }
    return render_template('preprocess_config.html', **context)

@main_bp.route('/processing_mode')
def processing_mode():
    if not _validate_session(['filename', 'target_column', 'preprocessing_config', 'processed_data_path']):
        return redirect(url_for('main.upload'))
    
    processed_path = session.get('processed_data_path')
    if not processed_path or not os.path.exists(processed_path):
        flash("Données prétraitées introuvables. Veuillez recommencer le prétraitement.", "error")
        return redirect(url_for('main.preprocess_config'))
    
    data = None
    df = None

    if processed_path.endswith('.npy'):
        data = np.load(processed_path, allow_pickle=True).item()
        # data is a dict: {'X_train': ..., 'y_train': ..., ...}
        # Use data['X_train'], data['y_train'], etc. as needed
    elif processed_path.endswith('.csv'):
        df = pd.read_csv(processed_path)
    else:
        flash("Format de fichier prétraité non supporté.", "error")
        return redirect(url_for('main.preprocess_config'))

    context = {
        'filename': session.get('filename'),
        'target_column': session.get('target_column'),
        'preprocessing_config': session.get('preprocessing_config'),
        'processed_data_path': processed_path,
        'data': data,
        'df': df 
    }
    return render_template('processing_mode.html', **context)

# =============================================================================
# TRAINING ROUTES
# =============================================================================

@main_bp.route('/default_processing', methods=['GET', 'POST'])
def default_processing():
    """Simple training with default parameters"""
    if not _validate_session(['filename', 'target_column', 'preprocessing_config']):
        return redirect(url_for('main.upload'))
    
    if request.method == 'POST':
        model_type = request.form.get('model_type')
        
        if not model_type or model_type not in ModelRegistry.list_models():
            flash("Please select a valid model type.", 'warning')
        else:
            return _train_model('simple', model_type=model_type)
    
    context = {
        'filename': session.get('filename'),
        'target_column': session.get('target_column'),
        'available_models': ModelRegistry.list_models(),
        'models_by_category': ModelRegistry.get_models_by_category()
    }
    
    return render_template('default_processing.html', **context)

@main_bp.route('/manual_processing', methods=['GET', 'POST'])
def manual_processing():
    """Custom training with manual hyperparameters"""
    if not _validate_session(['filename', 'target_column', 'preprocessing_config']):
        return redirect(url_for('main.upload'))
    
    if request.method == 'POST':
        try:
            model_type = request.form.get('model_type')
            units = int(request.form.get('units', 32))
            num_layers = int(request.form.get('num_layers', 1))
            
            model_params = {
                'num_layers': int(request.form.get('num_layers', 1)),
                'dropout': float(request.form.get('dropout', 0.3)),
                'recurrent_dropout': float(request.form.get('recurrent_dropout', 0.2)),
                'learning_rate': float(request.form.get('learning_rate', 0.001)),
                'batch_size': int(request.form.get('batch_size', 32)),
                'epochs': int(request.form.get('epochs', 30)),
                'optimizer': request.form.get('optimizer', 'adam'),
                'patience': int(request.form.get('patience', 5))
            }
            
            if 'stacked' in model_type:
                model_params['units'] = [max(units // (2 ** i), 8) for i in range(num_layers)]
            else:
                model_params['units'] = units
            
            return _train_model('custom', model_type=model_type, model_params=model_params)
            
        except ValueError as e:
            flash(f"Invalid parameter value: {str(e)}", 'error')
        except Exception as e:
            logger.error(f"Error in manual processing: {e}")
            flash(f"Error during training: {str(e)}", 'error')
    
    context = {
        'filename': session.get('filename'),
        'target_column': session.get('target_column'),
        'available_models': ModelRegistry.list_models(),
        'models_by_category': ModelRegistry.get_models_by_category()
    }
    
    return render_template('manual_processing.html', **context)

@main_bp.route('/hpo_processing', methods=['GET', 'POST'])
def hpo_processing():
    """Hyperparameter optimization training"""
    if not _validate_session(['filename', 'target_column', 'preprocessing_config']):
        return redirect(url_for('main.upload'))
    
    if request.method == 'POST':
        try:
            # Extract HPO configuration
            model_type = request.form.get('model_type')
            optimization_method = request.form.get('optimization_method', 'bayesian')
            num_layers = int(request.form.get('num_layers', 1))

            if optimization_method == 'bayesian':
                # Extract Bayesian-specific parameters
                hpo_params = {
                    'num_layers': int(request.form.get('num_layers', 1)),
                    'n_trials': int(request.form.get('n_trials', 50)),
                    'timeout': int(request.form.get('timeout_minutes', 60)) * 60,
                    'units_min': int(request.form.get('units_min', 32)),
                    'units_max': int(request.form.get('units_max', 256)),
                    'dropout_min': float(request.form.get('dropout_min', 0.1)),
                    'dropout_max': float(request.form.get('dropout_max', 0.4)),
                    'lr_min': float(request.form.get('lr_min', 0.0001)),
                    'lr_max': float(request.form.get('lr_max', 0.01)),
                    'batch_sizes': [16, 32, 64, 128],
                    'max_epochs': int(request.form.get('max_epochs', 100)),
                    'patience': int(request.form.get('patience', 10))
                }
            elif optimization_method in ['grid', 'random']:
                hpo_params = {
                    'n_trials': int(request.form.get('n_random_trials', 10)) if optimization_method == 'random' else None,
                    'timeout': 3600
                }
                grid_configs = request.form.get('grid_configurations')
                if grid_configs:
                    import json
                    hpo_params['configurations'] = json.loads(grid_configs)
            else:
                hpo_params = {
                    'n_trials': 25,
                    'timeout': 1800
                }
            
            
            # Add grid search configurations if provided
            if optimization_method == 'grid':
                # Parse custom grid configurations if provided
                grid_configs = request.form.get('grid_configurations')
                if grid_configs:
                    import json
                    hpo_params['configurations'] = json.loads(grid_configs)
            
            return _train_model(
                'hpo',
                model_type=model_type,
                optimization_method=optimization_method,
                hpo_params=hpo_params
            )
            
        except ValueError as e:
            flash(f"Invalid parameter value: {str(e)}", 'error')
        except Exception as e:
            logger.error(f"Error in HPO processing: {e}")
            flash(f"Error during optimization: {str(e)}", 'error')
    
    context = {
        'filename': session.get('filename'),
        'target_column': session.get('target_column'),
        'available_models': ModelRegistry.list_models(),
        'models_by_category': ModelRegistry.get_models_by_category(),
        'optimization_methods': current_app.config['HPO_AVAILABLE_METHODS']
    }
    
    return render_template('hpo_processing.html', **context)

# =============================================================================
# API ROUTES
# =============================================================================

@main_bp.route('/api/models')
def api_models():
    """API endpoint to get available models"""
    models_info = {}
    for model_name in ModelRegistry.list_models():
        try:
            models_info[model_name] = ModelRegistry.get_model_info(model_name)
        except Exception as e:
            models_info[model_name] = {'error': str(e)}
    
    return jsonify({
        'status': 'success',
        'available_models': models_info,
        'categories': ModelRegistry.get_models_by_category()
    })

@main_bp.route('/api/model/<model_name>')
def api_model_info(model_name):
    """API endpoint to get specific model information"""
    try:
        info = ModelRegistry.get_model_info(model_name)
        return jsonify({'status': 'success', **info})
    except ValueError as e:
        return jsonify({'status': 'error', 'message': str(e)}), 404

@main_bp.route('/api/preprocessing/options')
def api_preprocessing_options():
    """API endpoint to get preprocessing options"""
    return jsonify({
        'status': 'success',
        'options': {
            'missing_handling': ['remove', 'mean', 'median', 'mode', 'forward_fill', 'backward_fill', 'zero'],
            'encoding': ['label', 'onehot'],
            'scaling': ['standard', 'minmax', 'robust', 'none'],
            'sequence_length': {'min': 1, 'max': 100, 'default': 1},
            'test_size': {'min': 0.1, 'max': 0.5, 'default': 0.2}
        }
    })

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _validate_session(required_keys: list) -> bool:
    """Validate that required keys exist in session"""
    for key in required_keys:
        if key not in session:
            flash(f"Session expired or incomplete. Please start over.", 'warning')
            logger.warning(f"Missing session key: {key}")
            return False
    return True

def _train_model(mode: str, **kwargs) -> Any:
    """Common training logic for all modes"""
    try:
        # Load data
        filepath = session.get('filepath')
        df = load_data(filepath)
        
        # Get configurations
        target_col = session.get('target_column')
        preprocessing_config = session.get('preprocessing_config')
        
        # Select trainer based on mode
        trainers = {
            'simple': SimpleTrainer,
            'custom': CustomTrainer,
            'hpo': HPOTrainer
        }
        
        trainer_class = trainers.get(mode)
        if not trainer_class:
            raise ValueError(f"Invalid training mode: {mode}")
        
        trainer = trainer_class()
        
        # Prepare training arguments
        train_args = {
            'df': df,
            'target_col': target_col
        }
        
        # Add mode-specific arguments
        if mode == 'simple':
            train_args['model_type'] = kwargs.get('model_type', 'bi_lstm')
            train_args.update(preprocessing_config)
        elif mode == 'custom':
            train_args['model_type'] = kwargs.get('model_type')
            train_args['model_params'] = kwargs.get('model_params')
            train_args['preprocessing_params'] = preprocessing_config
        elif mode == 'hpo':
            train_args['model_type'] = kwargs.get('model_type')
            train_args['optimization_method'] = kwargs.get('optimization_method')
            train_args['hpo_params'] = kwargs.get('hpo_params')
            train_args['preprocessing_params'] = preprocessing_config
        
        # Train model
        logger.info(f"Starting {mode} training with model type: {train_args.get('model_type')}")
        
        if mode == 'hpo':
            results, trained_model = trainer.optimize_model(**train_args)
        else:
            results, trained_model = trainer.train_model(**train_args)
        
        # Generate evaluation report
        from .data_processing import DataPreprocessor
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.preprocess_data(df, target_col, **preprocessing_config)
        
        evaluator = ModelEvaluator()
        report_data = evaluator.generate_detailed_report(
            model=trained_model,
            X_test=X_test,
            y_test=y_test,
            model_name=f"{mode}_{kwargs.get('model_type', 'model')}",
            model_params=kwargs.get('model_params') or results.get('best_params', {})
        )
        
        # Prepare context for template
        context = {
            'report': report_data['classification_report'],
            'plot_image': report_data['plots'].get('confusion_matrix'),
            'model_type': kwargs.get('model_type', 'model').replace('_', '-').upper(),
            'metrics': report_data['metrics'],
            'additional_plots': report_data['plots'],
            'training_mode': mode.title(),
            'training_results': results
        }
        
        # Add HPO-specific results
        if mode == 'hpo':
            context['hpo_results'] = {
                'best_params': results.get('best_params'),
                'best_score': results.get('best_score'),
                'n_trials': results.get('n_trials'),
                'optimization_method': kwargs.get('optimization_method')
            }
        if mode == 'simple':
            # Use the model's default params
            model_type = kwargs.get('model_type', 'bi_lstm')
            model = ModelRegistry.get_model(model_type)
            params = model.get_default_params()
        else:
            params = kwargs.get('model_params') or results.get('best_params', {})
        
        # Add n_layers info for display
        if 'units' in params:
            if isinstance(params['units'], list):
                params['n_layers'] = len(params['units'])
            else:
                params['n_layers'] = 1
        else:
            params['n_layers'] = 1
        
        context['model_params'] = params


        logger.info(f"Training completed successfully. Metrics: {report_data['metrics']}")
        
        print("DEBUG model_params:", context['model_params'])
        
        context['preprocessing_config'] = session.get('preprocessing_config')
        return render_template('result.html', **context)
        
    except Exception as e:
        logger.error(f"Error during {mode} training: {e}", exc_info=True)
        flash(f"Error during training: {str(e)}", 'error')
        
        # Redirect based on mode
        redirect_map = {
            'simple': 'main.default_processing',
            'custom': 'main.manual_processing',
            'hpo': 'main.hpo_processing'
        }
        return redirect(url_for(redirect_map.get(mode, 'main.processing_mode')))