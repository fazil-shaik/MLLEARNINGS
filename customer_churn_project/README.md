# Customer Churn Prediction Project

A comprehensive machine learning project for predicting customer churn in telecom industry using Python and Streamlit.

## Project Structure

```
customer_churn_project/
├── data/
│   └── telco_churn.csv          # Raw dataset
├── models/
│   └── churn_pipeline.pkl       # Trained model pipeline
├── src/
│   ├── __init__.py              # Package initialization
│   ├── data_collector.py        # Data collection module (with caching)
│   ├── preprocess.py            # Data preprocessing
│   ├── train.py                 # Model training
│   ├── evaluate.py              # Model evaluation
│   └── predict.py               # Inference service (with caching)
├── app.py                       # Streamlit deployment app
├── requirements.txt             # Project dependencies
└── README.md                    # This file
```

## Features

- **Data Collection**: Load and cache telco churn dataset
- **Preprocessing**: Data cleaning, feature engineering, and scaling
- **Model Training**: Multiple algorithms (Logistic Regression, Random Forest, Gradient Boosting)
- **Hyperparameter Tuning**: GridSearchCV for optimal parameters
- **Model Evaluation**: Comprehensive metrics and visualizations
- **Inference Service**: Cached predictions for efficiency
- **Streamlit App**: Interactive web interface for predictions and analysis

## Installation

1. Clone the repository:
```bash
cd customer_churn_project
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Collection
```python
from src.data_collector import DataCollector

collector = DataCollector('data/telco_churn.csv')
df = collector.data
info = collector.get_basic_info()
```

### 2. Data Preprocessing
```python
from src.preprocess import DataPreprocessor

preprocessor = DataPreprocessor()
X, y = preprocessor.fit_transform(df, target_column='Churn')
X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
```

### 3. Model Training
```python
from src.train import ModelTrainer

trainer = ModelTrainer()
trainer.train_random_forest(X_train, y_train)
trainer.save_model('models/churn_pipeline.pkl')
```

### 4. Model Evaluation
```python
from src.evaluate import ModelEvaluator

evaluator = ModelEvaluator()
y_pred = trainer.model.predict(X_test)
y_pred_proba = trainer.model.predict_proba(X_test)
metrics = evaluator.evaluate(y_test, y_pred, y_pred_proba[:, 1])
evaluator.print_report(y_test, y_pred)
```

### 5. Make Predictions
```python
from src.predict import PredictionService

predictor = PredictionService('models/churn_pipeline.pkl')
predictions = predictor.predict(X_test)
proba_predictions = predictor.predict_with_confidence(X_test)
```

### 6. Streamlit App
```bash
streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501`

## Key Modules

### data_collector.py
- `DataCollector`: Loads and caches telco churn data
- Lazy loading with caching mechanism
- Basic dataset information retrieval

### preprocess.py
- `DataPreprocessor`: Handles all preprocessing tasks
- Categorical encoding, missing value handling, feature scaling
- Train-test split with stratification

### train.py
- `ModelTrainer`: Trains multiple classifier models
- Hyperparameter tuning with GridSearchCV
- Model persistence (save/load)

### evaluate.py
- `ModelEvaluator`: Comprehensive model evaluation
- Multiple evaluation metrics (accuracy, precision, recall, F1, ROC-AUC)
- Visualization functions (confusion matrix, ROC curve)

### predict.py
- `PredictionService`: Inference with caching
- Confidence scores for predictions
- Cache management and statistics

## Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **streamlit**: Web app framework
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization
- **plotly**: Interactive visualizations

## Workflow Example

```python
import pandas as pd
from src.data_collector import DataCollector
from src.preprocess import DataPreprocessor
from src.train import ModelTrainer
from src.evaluate import ModelEvaluator
from src.predict import PredictionService

# 1. Load data
collector = DataCollector('data/telco_churn.csv')
df = collector.data

# 2. Preprocess
preprocessor = DataPreprocessor()
X, y = preprocessor.fit_transform(df)
X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)

# 3. Train
trainer = ModelTrainer()
trainer.tune_hyperparameters(X_train, y_train, 'random_forest')
trainer.save_model('models/churn_pipeline.pkl')

# 4. Evaluate
evaluator = ModelEvaluator()
y_pred = trainer.model.predict(X_test)
metrics = evaluator.evaluate(y_test, y_pred)
evaluator.print_report(y_test, y_pred)

# 5. Predict
predictor = PredictionService('models/churn_pipeline.pkl')
new_predictions = predictor.predict_with_confidence(X_test)
print(new_predictions)
```

## Performance Metrics

The model evaluates performance using:
- **Accuracy**: Overall correctness
- **Precision**: True positive rate among predicted positives
- **Recall**: True positive rate among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve

## Caching Features

- **Data Collector**: Caches loaded dataset to avoid reloading
- **Prediction Service**: Caches predictions using MD5 hashing of input data

## Future Improvements

- [ ] Feature importance analysis
- [ ] SHAP values for model explainability
- [ ] A/B testing framework
- [ ] Model versioning
- [ ] API endpoints (FastAPI)
- [ ] Real-time monitoring dashboard
- [ ] Automated retraining pipeline

## License

This project is open source and available under the MIT License.

## Contact

For questions or suggestions, please reach out or create an issue.

---

**Last Updated**: 2024
