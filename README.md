# Machine Learning Assignment - Income Classification

## Problem Statement
The objective of this project is to build and compare multiple machine learning classification models to predict whether a person's annual income exceeds $50,000 based on census data. This is a binary classification problem where we predict one of two classes:
* **Class 0:** Income ≤ $50K per year
* **Class 1:** Income > $50K per year

The project implements six different classification algorithms, evaluates them using standard metrics, and identifies the best performing model for this prediction task.

---

## Dataset Description
**Dataset Name:** Adult Census Income Dataset  
**Source:** UCI Machine Learning Repository  
**URL:** [https://archive.ics.uci.edu/dataset/2/adult](https://archive.ics.uci.edu/dataset/2/adult)

### About the Dataset
This dataset was extracted from the 1994 US Census database. It contains demographic information about individuals and is commonly used for income prediction tasks.

### Dataset Statistics
* **Total Instances:** ~48,842 records
* **Number of Features:** 14 attributes
* **Target Variable:** Income (binary: ≤50K or >50K)
* **Missing Values:** Present in workclass, occupation, and native_country columns

### Feature Description
| Feature | Type | Description |
|:---|:---|:---|
| age | Numerical | Age of the individual |
| workclass | Categorical | Type of employer (Private, Self-emp, Govt, etc.) |
| fnlwgt | Numerical | Census weight |
| education | Categorical | Highest education level achieved |
| education_num | Numerical | Education level as a number |
| marital_status | Categorical | Marital status (Married, Single, Divorced, etc.) |
| occupation | Categorical | Type of occupation |
| relationship | Categorical | Relationship status in household |
| race | Categorical | Race of the individual |
| sex | Categorical | Gender (Male/Female) |
| capital_gain | Numerical | Capital gains from investments |
| capital_loss | Numerical | Capital losses from investments |
| hours_per_week | Numerical | Average hours worked per week |
| native_country | Categorical | Country of origin |
| **income** | **Target** | **Income level: ≤50K or >50K** |

---

## Models Used
Six classification models were implemented and evaluated on this dataset:
1. **Logistic Regression** - A linear model for binary classification
2. **Decision Tree Classifier** - A tree-based model that makes decisions based on feature values
3. **K-Nearest Neighbors (kNN)** - Instance-based learning using neighbor voting
4. **Naive Bayes (Gaussian)** - Probabilistic classifier based on Bayes' theorem
5. **Random Forest (Ensemble)** - Ensemble of decision trees using bagging
6. **XGBoost (Ensemble)** - Gradient boosting ensemble method

---

## Comparison Table - Evaluation Metrics
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.8200 | 0.8478 | 0.7174 | 0.4518 | 0.5545 | 0.4677 |
| Decision Tree | 0.8496 | 0.8942 | 0.7582 | 0.5776 | 0.6557 | 0.5703 |
| kNN | 0.8284 | 0.8537 | 0.6749 | 0.5937 | 0.6317 | 0.5223 |
| Naive Bayes | 0.7958 | 0.8501 | 0.6858 | 0.3252 | 0.4411 | 0.3701 |
| Random Forest (Ensemble) | 0.8561 | 0.9144 | 0.7967 | 0.5629 | 0.6597 | 0.5857 |
| XGBoost (Ensemble) | 0.8687 | 0.9251 | 0.7921 | 0.6374 | 0.7064 | 0.6292 |

*\*Note: Values are from actual model training on the Adult Income dataset (45,222 records after preprocessing).*

---

## Model Performance Observations
| ML Model Name | Observation about model performance |
|:---|:---|
| **Logistic Regression** | Achieved 82% accuracy; high precision (0.72) but low recall (0.45). Best for interpretability. |
| **Decision Tree** | 84.96% accuracy with good AUC (0.89). Captures non-linear patterns effectively. |
| **kNN** | 82.84% accuracy. F1 score (0.63) shows a reasonable balance. Scaling is essential here. |
| **Naive Bayes** | Lowest performer (79.58%). Struggled due to feature correlation (e.g., education/occupation). |
| **Random Forest** | 85.61% accuracy. High precision (0.80) makes it very reliable for positive predictions. |
| **XGBoost** | **Best performer.** Highest accuracy (86.87%) and AUC (0.93). |

---

## Project Structure
```text
project-folder/
│
├── app.py                        # Streamlit web application
├── sample_test_data.csv          # Sample test data for Streamlit app
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
└── model/                        # Saved model files (.pkl) and main implementation
    ├── adult_income_prediction.py    # Main ML implementation file
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    ├── xgboost.pkl
    ├── scaler.pkl
    ├── label_encoders.pkl
    └── confusion_matrix_*.png    # Confusion matrix visualizations
```

---

## Streamlit Web Application Features

The Streamlit app (`app.py`) includes the following features:

| Feature | Description |
|---------|-------------|
| **Dataset Upload (CSV)** | Upload test data in CSV format for predictions |
| **Model Selection Dropdown** | Choose from 6 trained classification models |
| **Evaluation Metrics Display** | Shows Accuracy, AUC, Precision, Recall, F1, MCC |
| **Confusion Matrix** | Shown after predictions (not on home page) |
| **Classification Report** | Detailed precision, recall, f1-score per class (after predictions) |

**How to use the Streamlit app:**

- Select the **'Upload & Predict'** radio button in the sidebar to make predictions and view evaluation metrics.
- You must upload a test data CSV file (with the same columns as the Adult Income dataset).
- After uploading your test data, select the classification model from the **Choose a classification model** dropdown and hit the **Make Predictions** button to get the prediction results.
- The app will display prediction summary, metrics, confusion matrix, and classification report for the selected model (these are only shown after predictions, not on the home page).

### Sample Test Data
A sample test data file (`sample_test_data.csv`) is provided for testing the Streamlit app. 

---

## How to Run

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the ML Model Training
```bash
python model/adult_income_prediction.py
```

### Step 3: Run the Streamlit App (optional)
```bash
streamlit run app.py
```
---

## Python Version Requirement

**Important:** This project requires Python 3.12 or lower for Streamlit compatibility.
