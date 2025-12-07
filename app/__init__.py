"""
Flask Application Factory
Simplified version that works out of the box with python main.py
"""
from flask import Flask, render_template, url_for
import os
import logging

def create_app(config_name=None):
    """
    Create and configure Flask application
    Simplified to always work without configuration
    """
    
    # Always use development if not specified
    if config_name is None:
        config_name = 'development'
    
    # Import config here to avoid circular imports
    from .config import config
    
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Get absolute paths
    app_dir = os.path.dirname(os.path.abspath(__file__))
    static_folder = os.path.join(app_dir, 'static')
    template_folder = os.path.join(app_dir, 'templates')
    
    logger.info(f"Starting Flask Application in {config_name} mode")
    
    # Create Flask app
    app = Flask(
        __name__,
        static_folder=static_folder,
        template_folder=template_folder,
        static_url_path='/static'
    )
    
    # Load configuration
    app.config.from_object(config[config_name])
    
    # Create necessary directories
    _create_directories(app, logger)
    
    # Register blueprints
    try:
        from .routes import main_bp
        app.register_blueprint(main_bp)
        logger.info("Routes registered successfully")
    except ImportError as e:
        logger.error(f"Could not import routes: {e}")
        # Create a minimal route for testing
        @app.route('/')
        def index():
            return "<h1>AutoML Platform</h1><p>Routes not fully configured yet.</p>"
    
    # Register error handlers
    _register_error_handlers(app)
    
    # Add context processors
    @app.context_processor
    def inject_config():
        return {
            'app_name': app.config.get('APP_NAME', 'AutoML'),
            'version': app.config.get('VERSION', '2.0.0'),
            'debug': app.config.get('DEBUG', False)
        }
    
    logger.info("Flask application created successfully!")
    return app

def _create_directories(app, logger):
    """Create necessary directories for the application"""
    
    # Create upload directory
    upload_folder = app.config.get('UPLOAD_FOLDER', 'uploads')
    os.makedirs(upload_folder, exist_ok=True)
    logger.info(f"Created/verified upload directory: {upload_folder}")
    
    # Create static subdirectories
    if app.static_folder:
        directories = [
            app.static_folder,
            os.path.join(app.static_folder, 'plots'),
            os.path.join(app.static_folder, 'css'),
            os.path.join(app.static_folder, 'js')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Created/verified directory: {directory}")

def _register_error_handlers(app):
    """Register error handlers for common HTTP errors"""
    
    @app.errorhandler(404)
    def not_found_error(error):
        """Handle 404 errors"""
        try:
            return render_template('errors/404.html'), 404
        except:
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>404 - Page Not Found</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        text-align: center;
                        padding: 50px;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                    }
                    h1 { font-size: 48px; margin-bottom: 10px; }
                    p { font-size: 18px; margin-bottom: 30px; }
                    a {
                        color: white;
                        background: rgba(255,255,255,0.2);
                        padding: 10px 20px;
                        text-decoration: none;
                        border-radius: 5px;
                        display: inline-block;
                    }
                    a:hover { background: rgba(255,255,255,0.3); }
                </style>
            </head>
            <body>
                <h1>404</h1>
                <p>Page Not Found</p>
                <a href="/">← Back to Home</a>
            </body>
            </html>
            """, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors"""
        app.logger.error(f"Internal error: {error}")
        try:
            return render_template('errors/500.html'), 500
        except:
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>500 - Internal Server Error</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        text-align: center;
                        padding: 50px;
                        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                        color: white;
                    }
                    h1 { font-size: 48px; margin-bottom: 10px; }
                    p { font-size: 18px; margin-bottom: 30px; }
                    a {
                        color: white;
                        background: rgba(255,255,255,0.2);
                        padding: 10px 20px;
                        text-decoration: none;
                        border-radius: 5px;
                        display: inline-block;
                    }
                    a:hover { background: rgba(255,255,255,0.3); }
                </style>
            </head>
            <body>
                <h1>500</h1>
                <p>Something went wrong</p>
                <a href="/">← Back to Home</a>
            </body>
            </html>
            """, 500
    
    @app.errorhandler(413)
    def too_large_error(error):
        """Handle file too large errors"""
        from flask import flash, redirect, url_for
        flash(f'File too large (max {app.config.get("MAX_CONTENT_LENGTH", 16*1024*1024) // (1024*1024)}MB)')
        try:
            return redirect(url_for('main.upload'))
        except:
            return redirect('/')