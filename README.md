Loan Default Prediction Pipeline
This repository contains an end-to-end machine learning project focused on predicting the likelihood of loan defaults. The project utilizes a comprehensive dataset of ~148,000 records and 34 features to identify high-risk applicants, assisting financial institutions in proactive risk management.

Project Overview
The objective of this project is to build a robust classification model that identifies potential loan defaulters. By analyzing applicant demographics, credit scores, and property details, the model provides a probability of default (Status), enabling data-driven decision-making in the lending process.

Tech Stack
Language: Python

Libraries: Pandas, NumPy, Scikit-Learn, XGBoost

Visualization: Matplotlib, Seaborn

Workflow: Scikit-Learn Pipelines, GridSearchCV for Hyperparameter Tuning

Deployment Tools: Joblib (Model Serialization)

Data Insights
The dataset contains a mix of categorical and numerical features including:

Target Variable: Status (0: Non-default, 1: Default).

Key Features: Credit_Score, income, loan_amount, property_value, LTV (Loan-to-Value ratio), and rate_of_interest.

Data Challenges: High missing value counts in features like rate_of_interest and property_value, which required strategic imputation.

Machine Learning Pipeline
To ensure "production-grade" code, the project implements a modular Scikit-Learn Pipeline.

1. Data Preprocessing
Numerical Handling: Missing values are handled via SimpleImputer, followed by StandardScaler to normalize distributions.

Categorical Handling: Categorical variables are processed using OneHotEncoder to convert them into a machine-readable format.

Transformation: A ColumnTransformer integrates these steps into a single preprocessing object.

2. Model Selection & Training
I evaluated multiple classifiers to find the best balance between precision and recall:

Logistic Regression: Baseline model.

Random Forest: For capturing non-linear relationships.

XGBoost: The primary model, optimized using GridSearchCV to maximize the AUC-ROC score.

Evaluation
The models are evaluated primarily on the Area Under the ROC Curve (AUC-ROC) to ensure performance across different classification thresholds.

Repository Structure

├── data/
│   └── raw/Loan_Default.csv    # Raw dataset
├── notebooks/
│   └── Loan_Default_Project.ipynb  # Exploratory Data Analysis & Modeling
├── models/
│   └── best_model.joblib       # Serialized production-ready model
└── README.md
 How to Use
Clone the repository.

Install dependencies: pip install -r requirements.txt.

Run the notebook to see the full EDA and training lifecycle.

Load the pre-trained model using joblib.load() for inference on new data.