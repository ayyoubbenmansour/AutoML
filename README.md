# Smart AutoML 

Lightweight Flask app to upload tabular/time-series data, configure preprocessing, and train RNN models (classification, regression, forecasting) with manual, default and HPO flows.

## Features
- Web UI for data upload, preprocessing and model configuration
- Immediate preprocessing after configuration; saves processed data to `app/static/processed/`
- Supports classification, regression and multi-step forecasting
- Categorical encoding: label, one-hot, target encoding
- Temporal handling: decomposition, cyclical encoding, timestamp
- Sequence creation for RNNs (LSTM/GRU/RNN) and stacked architectures
- Training modes: simple (defaults), manual, HPO (Bayesian)
- Evaluation: accuracy / regression metrics, plots (confusion matrix, ROC, PR, history)
- Extensible model registry and trainer classes

## Repo layout (important files)
- `app/`
  - `routes.py` — Flask endpoints and control flow
  - `data_processing.py` — DataLoader, DataCleaner, DataPreprocessor
  - `models.py` — Model classes (BiRNN, BiLSTM, stacked variants)
  - `training.py` — Trainer classes (SimpleTrainer, CustomTrainer, HPOTrainer)
  - `evaluation.py` — Evaluation and plot generation
  - `templates/` — Jinja2 templates: upload, preprocess_config, manual_processing, hpo_processing, result
  - `static/processed/` — saved processed datasets (.csv for 2D, .npy for sequence data)
- `venv/` — virtualenv (not tracked)
- `README.md` — this file

## Quick start (Linux)
1. Clone and enter project:
   ```bash
   git clone <repo>
   cd flask_project
   ```
2. Create venv and install:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Create processed dir if missing:
   ```bash
   mkdir -p app/static/processed
   ```
4. Run the app:
   ```bash
   export FLASK_APP=app
   export FLASK_ENV=development
   flask run
   ```
5. Open `http://127.0.0.1:5000` in browser.

## Usage summary
1. Upload CSV dataset (via Upload page). Select target column.
2. Preprocess configuration:
   - Choose problem type: `classification`, `regression`, or `forecasting`.
   - If `forecasting`, specify `forecasting_steps` (multi-step horizon).
   - Set missing value strategy, encoding (label/onehot/target), scaling, temporal handling, `seq_length`, `test_size`, `random_state`.
3. Submit form → app preprocesses immediately and saves processed dataset in `app/static/processed/`.
4. Choose processing mode:
   - Default (simple): uses model default params.
   - Manual: user-specified model params (ensure `num_layers` is converted to `units` list for stacked models).
   - HPO: Bayesian optimization (shows config UI).
5. Training runs and results page displays model params, problem type details and evaluation plots.

## Important implementation notes & tips
- Data saving:
  - 2D processed data (no sequences): saved as CSV.
  - Sequence (3D) processed data: saved as `.npy` (dict with `X_train`, `y_train`, `X_test`, `y_test`).
  - `processing_mode` route checks extension and loads with `pd.read_csv()` for `.csv` or `np.load(..., allow_pickle=True).item()` for `.npy`.
- Preprocessing order (must follow exactly):
  1. Handle missing values
  2. Temporal handling (decompose / cyclical / timestamp)
  3. Separate X, y
  4. Encode target (classification only) → make sure target is numeric BEFORE target encoding
  5. Encode categorical features (pass `y` for target encoding)
  6. Scale features (2D only)
  7. Train/test split (still 2D)
  8. Create sequences (only after split; sequence creation produces 3D arrays)
- Target encoding requires numeric target. If `encoding == 'target'`, convert `y` to numeric first (LabelEncoder) before calling target encoding.
- Do not apply scalers (StandardScaler / MinMaxScaler) on 3D arrays. Apply scaling prior to sequence creation.
- Ensure `random_state` is present in preprocessing config (default e.g. `42`) and passed to `train_test_split` for reproducibility.
- Model output for forecasting: pass `problem_type='forecasting'` and `forecasting_steps` into model `build_model()`; output Dense should have `forecasting_steps` units with `linear` activation.
- For stacked models, convert `num_layers` and base `units` into a `units` list before training:
  ```python
  if stacked:
      params['units'] = [base_units // (2**i) for i in range(num_layers)]
  else:
      params['units'] = base_units
  ```
- Template variables: `_train_model` must include `preprocessing_config = session.get('preprocessing_config')` in the context passed to `result.html` to avoid Jinja undefined errors.

## Common Troubleshooting
- UnicodeDecodeError when reading processed file: confirm extension — `.npy` files must be loaded with `np.load(..., allow_pickle=True)`, not `pd.read_csv`.
- "Found array with dim 3" from scaler: you applied scaler after sequence creation — reorder as above.
- "agg function failed [how->mean,dtype->object]": target encoding attempted on object dtype target — convert target to numeric before target encoding.
- Jinja `preprocessing_config` undefined: add it to the template context from session before rendering result page.
- "'_encode_categorical' attribute missing": ensure `_encode_categorical` (and `_target_encode`) methods are defined inside `DataPreprocessor`.

## Extending models & training
- Models live in `app/models.py`. Add support for forecasting and regression by accepting `problem_type` and `forecasting_steps` in `build_model()` and adjusting output layer accordingly.
- Trainers call `preprocessor.preprocess_data(..., problem_type=..., forecasting_steps=..., random_state=...)` and then `model.build_model(input_shape, num_classes, problem_type=..., forecasting_steps=..., **params)`.

## Development notes
- Use the debug logs to inspect shapes and params. Helpful debug statements:
  - shapes of `X_train`, `y_train` after preprocessing
  - dtype of target column before encoding
  - `model_params` contents before training
- Plots are saved under `plots/` with time-stamped filenames.

## Tests
- Add unit tests under `tests/` for:
  - `DataPreprocessor.preprocess_data()` with classification/regression/forecasting scenarios
  - `_encode_categorical()` including target encoding
  - sequence creation and shape expectations

## Contributing
- Fork, create a feature branch, run tests, open a PR.
- Keep commits small and descriptive.

## License
Specify your license here (e.g., MIT).  
