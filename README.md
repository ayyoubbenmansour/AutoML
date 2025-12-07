# 🎯 Smart AutoML - Automated Deep Learning Platform

An intelligent AutoML platform for automated machine learning with **9 model architectures** supporting time series forecasting and tabular data classification/regression.

---

## ✨ Features

### 🤖 **9 Model Architectures**

#### **RNN Models** (Time Series & Sequential Data)
- **Bi-RNN, Bi-LSTM, Bi-GRU** - Bidirectional models capturing forward/backward dependencies
- **Stacked LSTM, Stacked GRU** - Deep architectures for complex temporal patterns

#### **Traditional ML** (Tabular Classification & Regression)
- **Random Forest** - Fast, robust ensemble learning (Classification & Regression)
- **XGBoost** - State-of-the-art gradient boosting (Classification & Regression)

### 🎯 **Three Training Modes**
1. **Simple Mode** - Quick training with sensible defaults
2. **Manual Mode** - Full hyperparameter control
3. **HPO Mode** - Automated optimization (Bayesian, Grid, Random search)

### �️ **Smart Capabilities**
- ✅ Automatic data preprocessing (scaling, encoding, missing values)
- ✅ Intelligent model recommendation
- ✅ Real-time training visualization
- ✅ Comprehensive evaluation metrics
- ✅ Model comparison tools
- ✅ Export trained models

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/ayyoubbenmansour/AutoML.git
cd AutoML
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Core dependencies (required)
pip install -r requirements.txt

# Optional: Traditional ML models (if network allows)
pip install xgboost lightgbm
```

### 3. Run Application

```bash
python main.py
```

Open browser at **http://localhost:5000**

---

## 📊 Model Selection Guide

| Use Case | Recommended Models | Training Speed |
|----------|-------------------|----------------|
| **Time Series Forecasting** | Bi-LSTM, Stacked LSTM | Medium (minutes) |
| **Tabular Classification** | XGBoost, Random Forest | **Fast (seconds)** |
| **Tabular Regression** | XGBoost, Random Forest | **Fast (seconds)** |
| **Sequential Data** | Bi-GRU, Stacked GRU | Medium (minutes) |
| **Small Datasets (<1000 rows)** | Random Forest | Fast |
| **Large Datasets (>100k rows)** | XGBoost, LightGBM | Fast |

---

## � Usage Workflow

### Web Interface
1. **Upload Data** → CSV/Excel/JSON files
2. **Visualize** → Automatic EDA and data quality analysis
3. **Select Target** → Choose prediction column
4. **Preprocess** → Configure scaling, encoding, sequence length
5. **Choose Model** → Select from 9 architectures
6. **Train** → Simple/Manual/HPO mode
7. **Evaluate** → Metrics, confusion matrix, visualizations

### Programmatic API

```python
from app.models import ModelRegistry

# For Classification (XGBoost)
model = ModelRegistry.get_model('xgboost_clf')
results = model.train(X_train, y_train, X_val, y_val)
print(f"Accuracy: {results['final_val_accuracy']:.2%}")

# For Time Series (LSTM)
model = ModelRegistry.get_model('bi_lstm')
results = model.train(X_train, y_train, X_val, y_val)
```

---

## 🏗️ Architecture

```
Smart_AutoML/
├── app/
│   ├── __init__.py              # Flask app factory
│   ├── models.py                # 9 model architectures
│   ├── traditional_ml_models.py # ML model wrappers
│   ├── data_processing.py       # Data pipeline
│   ├── training.py              # Training logic
│   ├── evaluation.py            # Metrics & visualization
│   ├── routes.py                # Web endpoints
│   ├── config.py                # Configuration
│   ├── utils.py                 # Helper functions
│   ├── templates/               # HTML pages
│   └── static/                  # CSS, JS, plots
├── main.py                      # Entry point
├── requirements.txt             # Dependencies
└── uploads/                     # User data
```

---

## 🔧 Tech Stack

- **Backend**: Flask 3.1.1
- **Deep Learning**: TensorFlow 2.19.0, Keras 3.10.0
- **Traditional ML**: Scikit-learn 1.5.2, XGBoost 2.1.3, LightGBM 4.5.0
- **Optimization**: Optuna 4.4.0, Keras-Tuner 1.4.7
- **Data**: Pandas 2.3.1, NumPy 2.1.3
- **Visualization**: Matplotlib 3.10.1, Seaborn 0.13.4

---

## 🎓 Model Details

### RNN Models (Deep Learning)
**Best for:** Time series, sequences, temporal data
- Bi-directional processing
- Long-term dependency handling
- GPU acceleration support

**Parameters per model:** ~200K - 400K
**Training time:** Minutes (depends on data size)

### Traditional ML Models  
**Best for:** Tabular data, classification, regression
- No GPU required
- Fast training (seconds)
- Often outperform deep learning on structured data

**Training time:** Seconds to minutes

---

## � Performance

| Dataset Type | Model | Typical Performance | Speed |
|--------------|-------|-------------------|-------|
| Tabular (Classification) | XGBoost | 85-95% accuracy | ⚡ Fast |
| Tabular (Regression) | Random Forest | R²: 0.8-0.95 | ⚡ Fast |
| Time Series | Bi-LSTM | MAE: Low | 🐢 Medium |
| Sequential | Stacked GRU | High accuracy | 🐢 Medium |

---

## ⚙️ Configuration

All settings in `app/config.py`:

```python
# Training
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32
DEFAULT_PATIENCE = 10

# Models
AVAILABLE_MODEL_TYPES = [
    'bi_rnn', 'bi_lstm', 'bi_gru',
    'stacked_lstm', 'stacked_gru',
    'random_forest_clf', 'xgboost_clf',
    'random_forest_reg', 'xgboost_reg'
]

# HPO
HPO_DEFAULT_TRIALS = 25
HPO_AVAILABLE_METHODS = ['bayesian', 'grid', 'random']
```

---

## 🔍 Key Features

### Graceful Degradation
- App works with or without traditional ML libraries
- Automatic model availability detection
- Clear warnings if libraries missing

### Hyperparameter Optimization
- Bayesian optimization with Optuna
- Grid search and random search
- Automatic best parameters selection

### Data Processing
- Automatic missing value handling
- Feature scaling (Standard, MinMax, Robust)
- Categorical encoding (Label, One-Hot)
- Sequence generation for time series

### Evaluation
- Classification: Accuracy, Precision, Recall, F1, Confusion Matrix
- Regression: R², MAE, RMSE
- Visual plots and comparisons

---

## 🚧 Roadmap

### Planned Features
- [ ] CNN models for image data
- [ ] Statistical time series models (ARIMA, Prophet)
- [ ] Ensemble methods
- [ ] Model deployment tools
- [ ] Transfer learning
- [ ] AutoML recommendation engine

---

## 📝 License

Educational and research purposes.

---

## 🤝 Contributing

Contributions welcome! Focus areas:
- Additional model architectures
- Advanced preprocessing
- UI/UX improvements
- Performance optimization
- Documentation

---

## 📧 Contact

**Repository**: https://github.com/ayyoubbenmansour/AutoML  
**Author**: Ayoub Ben Mansour

---

## 🙏 Acknowledgments

Built with modern ML frameworks and best practices for automated machine learning.

**Last Updated**: December 2025
