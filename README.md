# Credit Card Fraud Detection using Machine Learning

An end-to-end machine learning project for detecting fraudulent credit
card transactions using classification models, imbalance-aware
evaluation, model serialization, and a Streamlit prediction interface.

**Author:** Ramavath Raju

------------------------------------------------------------------------

## Project Overview

Credit card fraud detection is a highly imbalanced binary classification
problem because fraudulent transactions represent only a small fraction
of all transactions.

The main goal of this project is to build a practical machine learning
workflow that can distinguish fraudulent transactions from legitimate
transactions while evaluating the model with metrics that are more
informative than accuracy alone.

The project covers:

-   Exploratory data analysis
-   Data preprocessing
-   Train/test splitting
-   Classification model training
-   Model comparison
-   Precision, Recall and F1-score evaluation
-   ROC-AUC and PR-AUC evaluation
-   Confusion-matrix analysis
-   Feature-importance analysis
-   Model serialization
-   Prediction pipeline
-   Streamlit prediction interface

------------------------------------------------------------------------

## Dataset

The project uses the publicly available **Credit Card Fraud Detection**
dataset.

The dataset contains transactions made by European cardholders over a
two-day period.

### Main Features

-   `Time` --- seconds elapsed between transactions
-   `V1` to `V28` --- anonymized principal components obtained using PCA
-   `Amount` --- transaction amount
-   `Class` --- target variable

### Target

``` text
0 → Legitimate transaction
1 → Fraudulent transaction
```

The original dataset is intentionally excluded from this repository
because of its size.

Place the downloaded dataset at:

``` text
data/creditcard.csv
```

------------------------------------------------------------------------

## Machine Learning Workflow

``` text
Raw Dataset
    ↓
Data Inspection
    ↓
Exploratory Data Analysis
    ↓
Train/Test Split
    ↓
Feature Preprocessing
    ↓
Class-Imbalance Handling
    ↓
Model Training
    ↓
Model Evaluation
    ↓
Model Selection
    ↓
Model Serialization
    ↓
Prediction Pipeline
    ↓
Streamlit Application
```

------------------------------------------------------------------------

## Models

The project compares multiple classification algorithms:

### 1. Logistic Regression

Used as an interpretable baseline classification model.

### 2. Decision Tree

A tree-based classifier capable of learning nonlinear decision
boundaries.

### 3. Random Forest

An ensemble of decision trees that can capture nonlinear relationships
and feature interactions.

### 4. XGBoost

A gradient-boosting classifier used for the final model comparison.

------------------------------------------------------------------------

## Model Evaluation

Because fraud detection is highly imbalanced, accuracy alone is not a
sufficient evaluation metric.

The project evaluates models using:

-   **Precision** --- among transactions predicted as fraud, how many
    were actually fraudulent.
-   **Recall** --- among actual fraudulent transactions, how many were
    detected.
-   **F1-score** --- harmonic mean of precision and recall.
-   **ROC-AUC** --- measures discrimination between fraudulent and
    legitimate transactions across classification thresholds.
-   **PR-AUC** --- evaluates the precision-recall trade-off and is
    particularly useful for imbalanced classification.
-   **Confusion Matrix** --- shows true positives, true negatives, false
    positives and false negatives.

------------------------------------------------------------------------

## Model Results

The current evaluation reports the following results for the
best-performing XGBoost model:

  Metric         Score
  ----------- --------
  Precision     87.23%
  Recall        83.67%
  F1-score      85.42%
  ROC-AUC       97.75%
  PR-AUC        87.60%

These metrics should be interpreted in the context of the highly
imbalanced dataset rather than relying on accuracy alone.

------------------------------------------------------------------------

## Visualizations

The repository includes model-analysis visualizations:

### Confusion Matrix

Shows the classification outcomes and helps identify false positives and
false negatives.

### Feature Importance

Shows the relative importance of features for the tree-based model.

### Precision-Recall and ROC Curves

Used to evaluate classification performance across different decision
thresholds.

Visualizations are available in:

``` text
images/
├── confusion_matrices.png
├── feature_importance.png
└── pr_roc_curves.png
```

------------------------------------------------------------------------

## Streamlit Application

A Streamlit application is included to demonstrate model inference.

The application supports:

-   CSV-based prediction
-   Individual transaction prediction
-   Fraud-probability output

The application uses the saved model and preprocessing artifacts rather
than retraining the model during prediction.

------------------------------------------------------------------------

## Project Structure

``` text
credit-card-fraud-detection-ml/
│
├── app/
│   └── app.py
│
├── data/
│   └── .gitkeep
│
├── images/
│   ├── confusion_matrices.png
│   ├── feature_importance.png
│   └── pr_roc_curves.png
│
├── models/
│   ├── best_model.pkl
│   ├── feature_names.pkl
│   ├── metadata.json
│   └── scaler.pkl
│
├── notebooks/
│   └── EDA_and_Modeling.ipynb
│
├── src/
│   ├── predict.py
│   └── train.py
│
├── .gitignore
├── README.md
├── requirements.txt
└── results.csv
```

------------------------------------------------------------------------

## Installation

### 1. Clone the repository

``` bash
git clone https://github.com/ramavathr360-jpg/credit-card-fraud-detection-ml.git
cd credit-card-fraud-detection-ml
```

### 2. Create a virtual environment

``` bash
python -m venv venv
```

### 3. Activate the environment

**Windows:**

``` bash
venv\Scripts\activate
```

**macOS/Linux:**

``` bash
source venv/bin/activate
```

### 4. Install dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Dataset Setup

Download the Credit Card Fraud Detection dataset and place the dataset
file here:

``` text
data/creditcard.csv
```

The dataset is intentionally excluded from GitHub.

------------------------------------------------------------------------

## Train the Model

From the project root:

``` bash
python src/train.py
```

The training pipeline produces the trained model and preprocessing
artifacts used by the prediction pipeline.

------------------------------------------------------------------------

## Make Predictions

Run:

``` bash
python src/predict.py
```

The prediction pipeline loads the saved model and preprocessing
components and produces fraud predictions.

------------------------------------------------------------------------

## Run the Streamlit Application

From the project root:

``` bash
streamlit run app/app.py
```

The application will open in your browser.

------------------------------------------------------------------------

## Results and Reproducibility

The repository contains:

-   Saved model artifacts in `models/`
-   Evaluation results in `results.csv`
-   Model-analysis visualizations in `images/`
-   The modeling notebook in `notebooks/EDA_and_Modeling.ipynb`

The original dataset is not stored in the repository.

------------------------------------------------------------------------

## Limitations

This project is a machine learning portfolio and experimentation project
rather than a production banking fraud-detection system.

Important limitations include:

-   The dataset contains anonymized PCA-derived features.
-   The dataset represents transactions from a limited time period.
-   Real-world fraud detection would require richer transaction and
    customer history.
-   Fraud patterns can change over time.
-   Model performance can degrade because of data drift.
-   False positives can affect legitimate customers.
-   False negatives can allow fraudulent transactions to pass through.
-   Production systems require monitoring, retraining, security controls
    and cost-sensitive decision-making.

------------------------------------------------------------------------

## Future Improvements

Potential extensions include:

-   Validation-based probability-threshold optimization
-   Cost-sensitive fraud detection
-   Model calibration
-   More systematic hyperparameter optimization
-   Explainability using SHAP
-   Temporal validation
-   Fraud-cost optimization
-   Model-drift monitoring
-   Real-time prediction API
-   Production deployment and monitoring

These are **future improvements**, not claims about the current
implementation.

------------------------------------------------------------------------

## Key Learning Outcomes

Through this project, I worked with:

-   Exploratory Data Analysis
-   Data preprocessing
-   Imbalanced classification
-   Logistic Regression
-   Decision Trees
-   Random Forest
-   XGBoost
-   Precision/Recall analysis
-   F1-score
-   ROC-AUC
-   PR-AUC
-   Confusion matrices
-   Feature-importance analysis
-   Model serialization
-   Prediction pipelines
-   Streamlit

------------------------------------------------------------------------

## Project Focus

The primary focus of this project is not simply achieving a high
accuracy score.

The project emphasizes a more realistic fraud-detection workflow where:

**class imbalance + precision + recall + PR-AUC + ROC-AUC + model
evaluation + practical prediction**

are considered together.

------------------------------------------------------------------------

## Author

**Ramavath Raju**

Machine Learning / Data Analytics Portfolio Project
