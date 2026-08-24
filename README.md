Credit Card Fraud Detection using Machine
Learning
An end-to-end machine learning project for detecting fraudulent credit card transactions using classification
models, imbalanced-data handling, model evaluation, and a Streamlit prediction interface.
Author: Ramavath Raju
📌 Project Overview
Credit card fraud detection is a highly imbalanced binary classification problem where fraudulent
transactions represent only a very small fraction of all transactions.
The goal of this project is to build a machine learning system that can identify fraudulent transactions while
maintaining a practical balance between:
Precision
Recall
F1-score
ROC-AUC
PR-AUC
Unlike a simple accuracy-based approach, this project focuses on metrics that are more meaningful for
highly imbalanced fraud detection.
🎯 Objectives
Explore and understand credit card transaction data.
Analyze the severe class imbalance between legitimate and fraudulent transactions.
Build multiple machine learning classification models.
Handle class imbalance using appropriate techniques.
Compare models using fraud-focused evaluation metrics.
Identify the best-performing model.
Save the trained model for future predictions.
Provide a Streamlit interface for making fraud predictions.
Make the project reproducible and suitable for further development.
📊 Dataset
The project uses the publicly available Credit Card Fraud Detection dataset.

The dataset contains transactions made by European cardholders over a two-day period.
Features
The dataset contains:
Time — seconds elapsed between transactions.
V1 to V28 — anonymized principal components obtained from PCA.
Amount — transaction amount.
Class — target variable.
Target:
0 → Legitimate transaction
1 → Fraudulent transaction
The dataset is highly imbalanced, making fraud detection significantly more difficult than a normal binary
classification problem.
The original dataset is not included in this repository because of its size. Place
creditcard.csv inside the data/ directory before running the training pipeline.
🔬 Machine Learning Workflow
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
Class Imbalance Handling
↓
Model Training
↓
Cross-Validation
↓
Model Comparison
↓
Final Model Selection

↓
Model Evaluation
↓
Model Serialization
↓
Prediction Pipeline
↓
Streamlit Application
🧹 Data Preprocessing
The preprocessing stage includes:
Checking dataset structure
Checking missing values
Checking duplicate records
Separating features and target
Stratified train/test splitting
Scaling relevant numerical variables
Preparing the data for machine learning
The train/test split preserves the original class distribution using stratification.
This is important because randomly splitting a severely imbalanced dataset can produce unreliable
evaluation results.
⚖️ Handling Class Imbalance
Fraudulent transactions represent a very small percentage of the dataset.
Instead of creating an artificially balanced dataset for the final evaluation, the project evaluates models
while preserving the realistic imbalance in the test set.
The project investigates imbalance-handling approaches including:
Class weighting
SMOTE
Undersampling where appropriate
SMOTE is applied to the training data rather than the untouched test data.
This prevents the evaluation set from becoming artificially balanced.

🤖 Machine Learning Models
The project compares multiple classification algorithms.
1. Logistic Regression
Used as a strong and interpretable baseline model.
2. Decision Tree
A nonlinear tree-based classifier useful for understanding decision boundaries.
3. Random Forest
An ensemble of decision trees that can capture nonlinear relationships and interactions between features.
4. XGBoost
A gradient-boosting algorithm capable of modelling complex nonlinear relationships and interactions.
📈 Model Evaluation
Because this is a highly imbalanced classification problem, accuracy alone is not sufficient.
The following metrics are used:
Precision
Of all transactions predicted as fraud, how many were actually fraudulent?
Recall
Of all actual fraudulent transactions, how many were detected?
F1-score
The harmonic mean of precision and recall.
ROC-AUC
Measures the model's ability to distinguish between fraudulent and legitimate transactions across
classification thresholds.

PR-AUC
Measures performance using the precision-recall relationship and is particularly useful for highly
imbalanced datasets.
🏆 Model Results
The current evaluation produced the following result for the best-performing XGBoost model:
Metric Score
Precision 87.23%
Recall 83.67%
F1-score 85.42%
ROC-AUC 97.75%
PR-AUC 87.60%
These results should be interpreted in the context of the highly imbalanced dataset rather than relying on
accuracy alone.
📊 Visualizations
The project includes visual analysis of:
Class distribution
Transaction amount distribution
Fraud vs legitimate transactions
Feature relationships
Correlation matrix
Confusion matrix
ROC curve
Precision-recall curve
Feature importance
Model performance comparison
Visualizations are stored in:
reports/figures/

🔎 Feature Importance
Tree-based models are used to analyze which features contribute most strongly to the model's predictions.
The feature-importance analysis helps identify the variables that the trained model relies on when
distinguishing fraudulent transactions from legitimate transactions.
Feature importance indicates model reliance and should not be interpreted as proof that a
feature causes fraud.
💻 Streamlit Application
A Streamlit application is included to demonstrate the trained model.
The application supports:
CSV Prediction
Upload transaction data and obtain model predictions.
Individual Prediction
Enter transaction feature values and receive:
Fraud probability
Predicted class
Classification result
Example:
Fraud Probability: 87.4%
Prediction:
🚨 FRAUD
The application also allows the classification threshold to be explored interactively.

🗂️ Project Structure
credit-card-fraud-detection/
│
├── app/
│   └── app.py
│
├── data/
│   └── creditcard.csv
│
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── feature_names.pkl
│   └── metadata.json
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_explainability.ipynb
│
├── reports/
│   └── figures/
│
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── preprocessing.py
│   └── evaluate.py
│
├── .gitignore
├── README.md
└── requirements.txt
⚙️ Installation
Clone the repository:
gitclonehttps://github.com/YOUR_USERNAME/credit-card-fraud-detection-ml.git

Move into the project directory:
cdcredit-card-fraud-detection-ml
Create a virtual environment:
python-mvenvvenv
Activate it on Windows:
venv\Scripts\activate
Install dependencies:
pipinstall-rrequirements.txt
📥 Dataset Setup
Download the Credit Card Fraud Detection dataset from Kaggle.
Place the dataset at:
data/creditcard.csv
The dataset is intentionally excluded from the GitHub repository.
🚀 Train the Model
From the project root:
pythonsrc/train.py
The training process will:
Load the dataset.1.

Perform preprocessing.
Handle class imbalance.
Train multiple classification models.
Compare model performance.
Select the best-performing model.
Save the trained model and preprocessing artifacts.
🔮 Make Predictions
Use the prediction script:
pythonsrc/predict.py
The prediction pipeline loads the trained model and preprocessing components and returns the predicted
class and fraud probability.
🌐 Run the Streamlit Application
Start the application:
streamlitrunapp/app.py
The application will open in your browser .
🧪 Reproducibility
The project uses fixed random seeds where appropriate to make experiments reproducible.
The same preprocessing and trained model artifacts are used during prediction to maintain consistency
between training and inference.
⚠️ Important Limitations
This project is a machine learning experimentation and portfolio project rather than a production banking
fraud-detection system.
2.
3.
4.
5.
6.
7.

Important limitations include:
The dataset contains anonymized PCA features.
The dataset represents transactions from a limited time period.
Real-world fraud detection requires much richer transaction and customer history.
Fraud patterns can change over time.
Model performance can degrade because of data drift.
False positives can negatively affect legitimate customers.
False negatives can allow fraudulent transactions to pass through.
Production systems require monitoring, retraining, security controls, and cost-sensitive decision-
making.
🔮 Future Improvements
Potential improvements include:
Proper validation-based probability threshold optimization
SHAP-based local and global model explainability
Cost-sensitive learning
Temporal validation
Model calibration
Fraud-cost optimization
Model drift monitoring
Real-time prediction API
Automated model retraining
Experiment tracking
Production deployment
💡 Key Learning Outcomes
Through this project, I worked with:
Exploratory Data Analysis
Data preprocessing
Imbalanced classification
SMOTE
Logistic Regression
Decision Trees
Random Forest
XGBoost
Cross-validation
Hyperparameter tuning
Precision/Recall analysis

PR-AUC
ROC-AUC
Confusion matrices
Feature importance
Model serialization
Prediction pipelines
Streamlit
👨‍💻 Author
Ramavath Raju
Machine Learning / Data Analytics Portfolio Project
⭐ Project Focus
The primary focus of this project is not achieving a high accuracy score.
The goal is to build a more realistic fraud-detection workflow where fraud recall, precision, PR-AUC, class
imbalance, model evaluation, and practical prediction are considered together.
