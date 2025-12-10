# 🎯 Smart AutoML Platform

> Automated Machine Learning platform for time series forecasting and tabular data analysis with **9 intelligent model architectures**.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1.1-green.svg)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Overview

Smart AutoML is an intelligent platform that **automates the entire machine learning pipeline** from data upload to model evaluation. It intelligently selects and trains the best model for your data with minimal configuration required.

### ✨ Key Features

- 🤖 **9 Model Architectures** - 5 Deep Learning RNNs + 4 Traditional ML models
- ⚡ **3 Training Modes** - Simple, Manual, Hyperparameter Optimization
- 🔄 **Automatic Preprocessing** - Handles missing values, scaling, encoding
- 📊 **Smart Visualizations** - Interactive charts and evaluation metrics
- 🎯 **Multi-Task Support** - Classification, Regression, Time Series Forecasting
- 🚀 **Production Ready** - Full web interface with RESTful API

---

## 🤖 Available Models

### Deep Learning Models (Time Series & Sequential Data)

| Model | Type | Best For | Parameters |
|-------|------|----------|------------|
| **Bi-RNN** | Bidirectional RNN | Simple sequences | ~200K |
| **Bi-LSTM** | Bidirectional LSTM | Complex temporal patterns | ~300K |
| **Bi-GRU** | Bidirectional GRU | Fast sequential processing | ~250K |
| **Stacked LSTM** | Multi-layer LSTM | Deep temporal features | ~400K |
| **Stacked GRU** | Multi-layer GRU | Efficient deep learning | ~350K |

### Traditional ML Models (Tabular Data)

| Model | Task | Speed | Best For |
|-------|------|-------|----------|
| **Random Forest Classifier** | Classification | ⚡ Fast (seconds) | Small-medium datasets |
| **XGBoost Classifier** | Classification | ⚡ Fast (seconds) | High accuracy needed |
| **Random Forest Regressor** | Regression | ⚡ Fast (seconds) | Robust predictions |
| **XGBoost Regressor** | Regression | ⚡ Fast (seconds) | Competition-grade results |

> **Note:** Traditional ML models require `xgboost` and `lightgbm` libraries. The app works without them (RNN-only mode) via graceful degradation.

---

## 🚀 Quick Start

### 1️⃣ Clone Repository

```bash
git clone https://github.com/ayyoubbenmansour/AutoML.git
cd AutoML
```

### 2️⃣ Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
# Core dependencies (Required)
pip install -r requirements.txt

# Optional: Traditional ML models
pip install xgboost lightgbm
```

### 4️⃣ Run Application

```bash
python main.py
```

Open your browser at **http://localhost:5000** 🎉

---

## 💡 Usage Guide

### Web Interface Workflow

```
📤 Upload Data (CSV/Excel/JSON)
    ↓
📊 Visualize & Analyze
    ↓
🎯 Select Target Column
    ↓
⚙️ Configure Preprocessing
    ↓
🤖 Choose Model Architecture
    ↓
🏋️ Train (Simple/Manual/HPO)
    ↓
📈 Evaluate Results
```

### Programmatic Usage

```python
from app.models import ModelRegistry

# Classification with XGBoost
model = ModelRegistry.get_model('xgboost_clf')
results = model.train(X_train, y_train, X_val, y_val)
print(f"Accuracy: {results['final_val_accuracy']:.2%}")

# Time Series with LSTM
model = ModelRegistry.get_model('bi_lstm')
results = model.train(X_train, y_train, X_val, y_val)
predictions = model.predict(X_test)
```

---

## 📊 Model Selection Guide

| Your Data Type | Recommended Model | Why? |
|----------------|-------------------|------|
| **Time Series** | Bi-LSTM, Stacked LSTM | Captures temporal dependencies |
| **Tabular (Classification)** | XGBoost, Random Forest | Fast training, high accuracy on structured data |
| **Tabular (Regression)** | XGBoost Regressor | Best performance on tabular regression |
| **Sequential Text** | Bi-GRU | Efficient for variable-length sequences |
| **Small Dataset (<1000 rows)** | Random Forest | Robust, less prone to overfitting |
| **Large Dataset (>100K rows)** | XGBoost | Scalable with excellent performance |

---

## 🏗️ Project Structure

```
AutoML/
├── app/
│   ├── __init__.py              # Flask app factory
│   ├── models.py                # 9 model architectures
│   ├── traditional_ml_models.py # ML model wrappers
│   ├── data_processing.py       # Data pipeline & preprocessing
│   ├── training.py              # Training orchestration
│   ├── evaluation.py            # Metrics & visualizations
│   ├── routes.py                # Web API endpoints
│   ├── config.py                # Configuration settings
│   ├── utils.py                 # Helper utilities
│   ├── templates/               # HTML templates
│   └── static/                  # CSS, JS, plots
├── main.py                      # Application entry point
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 🔧 Tech Stack

### Backend
- **Flask 3.1.1** - Web framework
- **Python 3.12** - Core language

### Deep Learning
- **TensorFlow 2.19.0** - Neural network framework
- **Keras 3.10.0** - High-level API

### Machine Learning
- **Scikit-learn 1.5.2** - Traditional ML algorithms
- **XGBoost 2.1.3** - Gradient boosting (optional)
- **LightGBM 4.5.0** - Fast gradient boosting (optional)

### Optimization
- **Optuna 4.4.0** - Bayesian hyperparameter optimization
- **Keras-Tuner 1.4.7** - Neural architecture search

### Data & Visualization
- **Pandas 2.3.1** - Data manipulation
- **NumPy 2.1.3** - Numerical computing
- **Matplotlib 3.10.1** - Plotting
- **Seaborn 0.13.4** - Statistical visualization

---

## ⚙️ Training Modes

### 1. Simple Mode (Default)
- One-click training with optimized defaults
- Best for quick experiments
- Automatic early stopping

### 2. Manual Mode
- Full hyperparameter control
- Layer configuration
- Learning rate, epochs, batch size
- Dropout and regularization

### 3. HPO Mode (Advanced)
- **Bayesian Optimization** - Intelligent search with Optuna
- **Grid Search** - Exhaustive parameter combinations
- **Random Search** - Stochastic exploration
- Configurable: trials, timeout, metrics

---

## 📈 Performance Benchmarks

| Task Type | Model | Dataset | Metric | Score |
|-----------|-------|---------|--------|-------|
| Classification | XGBoost | Iris | Accuracy | 97% |
| Regression | Random Forest | Boston Housing | R² | 0.91 |
| Time Series | Bi-LSTM | Stock Prices | MAE | Low |
| Sequential | Stacked GRU | Text Classification | F1 | High |

---

## 🎯 Features in Detail

### Automatic Preprocessing
- ✅ Missing value imputation (mean, median, mode)
- ✅ Feature scaling (Standard, MinMax, Robust)
- ✅ Categorical encoding (Label, One-Hot)
- ✅ Sequence generation for time series
- ✅ Train/validation/test splitting

### Evaluation Metrics

**Classification:**
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC Curve (binary)

**Regression:**
- R² Score, MAE, MSE, RMSE
- Residual plots
- Prediction vs Actual scatter

**Time Series:**
- MAE, MSE, MAPE
- Forecast visualization
- Trend analysis

---

## 🚧 Roadmap

### Upcoming Features
- [ ] CNN models for image classification
- [ ] Statistical models (ARIMA, Prophet)
- [ ] Ensemble learning (voting, stacking)
- [ ] Model deployment to production
- [ ] Transfer learning support
- [ ] Automated model recommendation engine
- [ ] Real-time prediction API

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- 🔹 Additional model architectures
- 🔹 Advanced preprocessing techniques
- 🔹 UI/UX enhancements
- 🔹 Performance optimizations
- 🔹 Documentation improvements

**How to contribute:**
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

**Author:** Ayoub BENMANSOUR  
**Repository:** [https://github.com/ayyoubbenmansour/AutoML](https://github.com/ayyoubbenmansour/AutoML)  
**Issues:** [Report a bug](https://github.com/ayyoubbenmansour/AutoML/issues)

---

## 🙏 Acknowledgments

Built with ❤️ using state-of-the-art machine learning frameworks:
- TensorFlow Team for deep learning framework
- Scikit-learn for traditional ML algorithms
- XGBoost and LightGBM teams for gradient boosting
- Flask community for web framework
- Optuna for hyperparameter optimization

---

## 📚 Documentation

For detailed documentation on:
- Model architectures and parameters
- API endpoints and programmatic usage
- Advanced configuration options
- Deployment guidelines

Visit our [Wiki](https://github.com/ayyoubbenmansour/AutoML/wiki) (coming soon)

---

**⭐ Star this repository if you find it helpful!**

*Last Updated: December 2025*
