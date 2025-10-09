# AutoML 🚀 

This project is a **Flask-based web application** for **data preprocessing**, **model training**, **hyperparameter optimization (HPO)**, and **result visualization**.  
It provides an interactive and modular platform for machine learning workflows — from uploading datasets to training models and visualizing the results — all accessible via a web interface.

---

## 🚀 Overview

The goal of this project is to make **end-to-end machine learning pipelines** accessible through a simple web UI.  
Users can:

- Upload raw datasets (`.csv` or `.xlsx`)
- Choose a processing mode (default, manual, or hyperparameter optimization)
- Configure preprocessing and model training parameters
- Run training and evaluation jobs
- Visualize the resulting model performance

The backend is built using **Flask** (for routing, templating, and request handling), while **Python scripts** handle data preprocessing, model training, and evaluation.

---

## 🏗️ Project Architecture

FLASK_PROJECT/
│
├── app/
│ ├── templates/
│ │ ├── base.html # Layout file (navbar, footer, etc.)
│ │ ├── home_page.html # Landing page
│ │ ├── upload_data.html # Dataset upload form
│ │ ├── preprocess_config.html # Preprocessing configuration
│ │ ├── processing_mode.html # Mode selection (default/manual/HPO)
│ │ ├── default_processing.html # Default automatic pipeline
│ │ ├── manual_processing.html # Manual pipeline configuration
│ │ ├── hpo_processing.html # Hyperparameter optimization interface
│ │ ├── visualization_data.html # Visualize data or model results
│ │ ├── result.html # Display training/evaluation results
│ │ └── errors/
│ │ └── 404.html # Error page
│ │
│ ├── init.py # Initializes Flask app and routes
│ ├── config.py # App configuration (paths, settings)
│ ├── routes.py # Defines Flask routes (view functions)
│ ├── data_processing.py # Data loading, cleaning, feature engineering
│ ├── training.py # Model training pipeline
│ ├── evaluation.py # Model evaluation metrics and reporting
│ ├── models.py # ML model definitions (e.g., sklearn)
│ └── utils.py # Helper functions (logging, validation, etc.)
│
├── uploads/ # Temporary uploaded datasets
├── architecture/ # ML model architectures or references
├── venv/ # Virtual environment (not versioned)
└── main.py # App entry point

## ⚙️ Features Breakdown

### 🔹 1. Data Upload & Validation
- Accepts CSV/XLSX uploads.
- Validates file format and structure.
- Stores uploaded data in the `/uploads` directory.

### 🔹 2. Data Preprocessing
- Missing value imputation
- Categorical encoding
- Feature scaling and normalization
- Splitting data into train/test sets

### 🔹 3. Processing Modes
The user can select between three workflow modes:

| Mode | Description |
|------|--------------|
| **Default** | Runs an automated preprocessing + model training pipeline using preset configurations |
| **Manual** | Allows the user to define their own parameters for preprocessing, model selection, and training |
| **HPO (Hyperparameter Optimization)** | Automatically tunes model parameters using grid/random search |

### 🔹 4. Model Training
- Supports scikit-learn models (e.g., RandomForest, LogisticRegression)
- Configurable hyperparameters
- Stores trained model artifacts

### 🔹 5. Evaluation & Visualization
- Generates model performance metrics: Accuracy, Precision, Recall, F1-score
- Visualizes confusion matrices and learning curves
- Interactive plots using Matplotlib or Plotly

---

## 🧰 Tech Stack

| Layer | Technology |
|:------|:------------|
| **Framework** | Flask |
| **Frontend** | HTML5, CSS3, Bootstrap, Jinja2 |
| **Backend** | Python |
| **ML/DS** | scikit-learn, NumPy, Pandas |
| **Visualization** | Matplotlib, Plotly |
| **Environment** | venv or Conda |

---

## ⚙️ Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

### 2. Create a Virtual Environment
python -m venv venv


Activate it:

### Windows
```bash
venv\Scripts\activate


### macOS/Linux
```bash
source venv/bin/activate

### 3. Install Dependencies

Install dependencies from the requirements.txt file:
```bash
pip install -r requirements.txt

### 4. Run the Application
```bash
python main.py
Then open in your browser :

http://127.0.0.1:5000/
