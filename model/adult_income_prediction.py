import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')
import pickle
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# ================================================================================
# LOAD THE DATASET
# ================================================================================

def download_and_load_adult_data():
    print("\nDataset URL: https://archive.ics.uci.edu/dataset/2/adult")

    train_file_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    test_file_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

    feature_names = [
        'age',              # How old the person is
        'workclass',        # Type of employer (Private, Government, etc.)
        'fnlwgt',           # Census weight
        'education',        # Highest education level
        'education_num',    # Education level as a number
        'marital_status',   # Married, Single, Divorced, etc.
        'occupation',       # Type of job
        'relationship',     # Relationship status in family
        'race',             # Race of the person
        'sex',              # Male or Female
        'capital_gain',     # Money gained from investments
        'capital_loss',     # Money lost from investments
        'hours_per_week',   # How many hours worked per week
        'native_country',   # Country of origin
        'income'            # Target: <=50K or >50K (what we want to predict)
    ]


    print("\nDownloading training data file...")
    train_data = pd.read_csv(
        train_file_url,
        header=None,           # file doesn't have column names
        names=feature_names,   # we provide our own column names
        sep=r',\s*',
        engine='python',
        na_values='?'
    )
    print(f"Training data loaded: {len(train_data)} rows")

    print("Downloading test data file...")
    test_data = pd.read_csv(
        test_file_url,
        header=None,
        names=feature_names,
        sep=r',\s*',
        engine='python',
        na_values='?',
        skiprows=1             # Skipping first row
    )
    print(f"Test data loaded: {len(test_data)} rows")

    # Combine both files into one big dataset
    full_data = pd.concat([train_data, test_data], ignore_index=True)

    print(f"\nTotal dataset size: {len(full_data)} rows")
    print(f"Number of features: {len(feature_names) - 1}")  # Minus 1 for target
    print("\nDataset loaded successfully!")

    return full_data


# ================================================================================
# FUNCTION TO EXPLORE THE DATA
# ================================================================================

def explore_the_data(my_data):

    print("\n" + "=" * 70)
    print("EXPLORING THE DATASET")
    print("=" * 70)
    print("\n>>> First 5 rows of the dataset:")
    print(my_data.head())

    print(f"\n>>> Dataset shape:")
    print(f"    Number of people (rows): {my_data.shape[0]}")
    print(f"    Number of features (columns): {my_data.shape[1]}")
    print("\n>>> Data types of each column:")
    for column_name in my_data.columns:
        print(f"    {column_name}: {my_data[column_name].dtype}")

    print("\n>>> Missing values in each column:")
    missing_values = my_data.isnull().sum()
    columns_with_missing = missing_values[missing_values > 0]
    if len(columns_with_missing) > 0:
        for col, count in columns_with_missing.items():
            print(f"    {col}: {count} missing values")
    else:
        print("    No missing values found!")

    print("\n>>> Target variable (income) distribution:")
    income_counts = my_data['income'].value_counts()
    for income_level, count in income_counts.items():
        percentage = (count / len(my_data)) * 100
        print(f"    {income_level}: {count} people ({percentage:.1f}%)")

    print("\n>>> Statistics for numerical columns:")
    print(my_data.describe())

    return my_data


# ================================================================================
# PREPARE THE DATA
# ================================================================================

def prepare_data_for_ml(my_data):

    print("\n" + "=" * 70)
    print("PREPARING DATA FOR MACHINE LEARNING")
    print("=" * 70)
    clean_data = my_data.copy()

    print("\n>>> Cleaning the income column...")
    clean_data['income'] = clean_data['income'].astype(str)
    clean_data['income'] = clean_data['income'].str.replace('.', '', regex=False)
    clean_data['income'] = clean_data['income'].str.strip()

    rows_before = len(clean_data)
    clean_data = clean_data.dropna()
    rows_after = len(clean_data)
    rows_removed = rows_before - rows_after
    print(f">>> Removed {rows_removed} rows with missing values")
    print(f">>> Remaining rows: {rows_after}")

    text_columns = []
    number_columns = []

    for column_name in clean_data.columns:
        if clean_data[column_name].dtype == 'object':
            text_columns.append(column_name)
        else:
            number_columns.append(column_name)

    print(f"\n>>> Text (categorical) columns: {text_columns}")
    print(f">>> Number (numerical) columns: {number_columns}")
    print("\n>>> Converting text columns to numbers...")

    encoders = {}

    for col in text_columns:
        my_encoder = LabelEncoder()
        clean_data[col] = my_encoder.fit_transform(clean_data[col])
        encoders[col] = my_encoder
        unique_values = len(my_encoder.classes_)
        print(f"    {col}: converted ({unique_values} unique values)")

    print("\n>>> Data preparation complete!")

    return clean_data, encoders


# ================================================================================
#  FUNCTION TO SPLIT DATA INTO TRAINING AND TESTING SETS
# ================================================================================

def split_into_train_test(clean_data):

    print("\n" + "=" * 70)
    print("SPLITTING DATA INTO TRAINING AND TESTING SETS")
    print("=" * 70)

    # Features(X) = all columns except 'income'
    # Target(Y) = the 'income' column
    X = clean_data.drop('income', axis=1)
    y = clean_data['income']

    print(f"\n>>> Features shape: {X.shape}")
    print(f">>> Target shape: {y.shape}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"\n>>> Training set size: {len(X_train)} rows")
    print(f">>> Testing set size: {len(X_test)} rows")
    print("\n>>> Scaling features using StandardScaler...")
    my_scaler = StandardScaler()
    X_train = my_scaler.fit_transform(X_train)
    X_test = my_scaler.transform(X_test)
    print(">>> Features scaled successfully!")

    return X_train, X_test, y_train, y_test, my_scaler


# ================================================================================
# CALCULATE ALL EVALUATION METRICS
# ================================================================================

def calculate_metrics(y_actual, y_predicted, y_probability):
    # Accuracy
    acc = accuracy_score(y_actual, y_predicted)

    #AUC Score
    auc = roc_auc_score(y_actual, y_probability)

    #Precision
    prec = precision_score(y_actual, y_predicted)

    #Recall
    rec = recall_score(y_actual, y_predicted)

    #F1 Score
    f1 = f1_score(y_actual, y_predicted)

    # Matthews Correlation Coefficient
    mcc = matthews_corrcoef(y_actual, y_predicted)

    all_metrics = {
        'Accuracy': round(acc, 4),
        'AUC': round(auc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1': round(f1, 4),
        'MCC': round(mcc, 4)
    }

    return all_metrics


# ================================================================================
# FUNCTIONS TO TRAIN EACH MODEL
# ================================================================================

def train_model_logistic_regression(X_train, X_test, y_train, y_test):

    print("\n" + "-" * 50)
    print("Training Model 1: LOGISTIC REGRESSION")
    print("-" * 50)

    #logistic regression
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability of positive class

    metrics = calculate_metrics(y_test, y_pred, y_prob)

    print(f"Training complete! Accuracy: {metrics['Accuracy']}")

    return model, metrics, y_pred


def train_model_decision_tree(X_train, X_test, y_train, y_test):

    print("\n" + "-" * 50)
    print("Training Model 2: DECISION TREE CLASSIFIER")
    print("-" * 50)

    # decision tree model
    model = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = calculate_metrics(y_test, y_pred, y_prob)
    print(f"Training complete! Accuracy: {metrics['Accuracy']}")

    return model, metrics, y_pred


def train_model_knn(X_train, X_test, y_train, y_test):

    print("\n" + "-" * 50)
    print("Training Model 3: K-NEAREST NEIGHBORS (KNN)")
    print("-" * 50)

    # KNN model
    model = KNeighborsClassifier(
        n_neighbors=5,
        weights='uniform',
        metric='minkowski'
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = calculate_metrics(y_test, y_pred, y_prob)

    print(f"Training complete! Accuracy: {metrics['Accuracy']}")

    return model, metrics, y_pred


def train_model_naive_bayes(X_train, X_test, y_train, y_test):

    print("\n" + "-" * 50)
    print("Training Model 4: NAIVE BAYES (GAUSSIAN)")
    print("-" * 50)

    # GaussianNaive Bayes
    model = GaussianNB()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = calculate_metrics(y_test, y_pred, y_prob)

    print(f"Training complete! Accuracy: {metrics['Accuracy']}")

    return model, metrics, y_pred


def train_model_random_forest(X_train, X_test, y_train, y_test):

    print("\n" + "-" * 50)
    print("Training Model 5: RANDOM FOREST (ENSEMBLE)")
    print("-" * 50)

    # Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = calculate_metrics(y_test, y_pred, y_prob)

    print(f"Training complete! Accuracy: {metrics['Accuracy']}")

    return model, metrics, y_pred


def train_model_xgboost(X_train, X_test, y_train, y_test):

    print("\n" + "-" * 50)
    print("Training Model 6: XGBOOST (ENSEMBLE)")
    print("-" * 50)

    # XGBoost
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = calculate_metrics(y_test, y_pred, y_prob)

    print(f"Training complete! Accuracy: {metrics['Accuracy']}")

    return model, metrics, y_pred


# ================================================================================
# COMPARISON TABLE
# ================================================================================

def create_comparison_table(results_list):

    print("\n" + "=" * 80)
    print("MODEL COMPARISON TABLE - ALL EVALUATION METRICS")
    print("=" * 80)

    table = pd.DataFrame(results_list)
    table = table.set_index('Model')

    print("\n")
    print(table.to_string())
    print("\n")

    return table


# ================================================================================
# CONFUSION MATRICES
# ================================================================================

def save_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='YlGnBu',
        xticklabels=['<=50K', '>50K'],
        yticklabels=['<=50K', '>50K']
    )
    plt.title(f'Confusion Matrix: {model_name}', fontsize=12)
    plt.xlabel('Predicted Income Level')
    plt.ylabel('Actual Income Level')
    plt.tight_layout()
    # Save directly in the same directory as this script
    base_dir = os.path.dirname(__file__)
    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
    file_path = os.path.join(base_dir, f'confusion_matrix_{safe_name}.png')
    plt.savefig(file_path, dpi=100)
    plt.close()
    print(f"Saved: {file_path}")


# ================================================================================
# SAVE TRAINED MODELS
# ================================================================================

def save_all_models(models_dict, scaler_object, encoders_dict):
    base_dir = os.path.dirname(__file__)
    for name, model in models_dict.items():
        safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').lower()
        file_path = os.path.join(base_dir, f'{safe_name}.pkl')
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to: {file_path}")
    scaler_path = os.path.join(base_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler_object, f)
    print(f"Scaler saved to: {scaler_path}")
    encoders_path = os.path.join(base_dir, 'label_encoders.pkl')
    with open(encoders_path, 'wb') as f:
        pickle.dump(encoders_dict, f)
    print(f"Encoders saved to: {encoders_path}")


# ================================================================================
# PART K: MAIN FUNCTION
# ================================================================================

def run_full_analysis():
    """
    This is the main function that runs the complete machine learning pipeline:
    1. Load data
    2. Explore data
    3. Prepare data
    4. Split data
    5. Train all models
    6. Evaluate and compare
    7. Save results
    """

    print("\n")
    print("*" * 80)
    print("*" + " " * 78 + "*")
    print("*" + "    MACHINE LEARNING CLASSIFICATION - INCOME PREDICTION    ".center(78) + "*")
    print("*" + "    Adult Census Dataset from UCI Repository    ".center(78) + "*")
    print("*" + " " * 78 + "*")
    print("*" * 80)

    raw_data = download_and_load_adult_data()

    explore_the_data(raw_data)

    clean_data, encoders = prepare_data_for_ml(raw_data)

    X_train, X_test, y_train, y_test, scaler = split_into_train_test(clean_data)

    print("\n" + "=" * 70)
    print("TRAINING ALL 6 CLASSIFICATION MODELS")
    print("=" * 70)

    all_models = {}

    all_results = []

    all_predictions = {}

    # ----- MODEL 1: Logistic Regression -----
    lr_model, lr_metrics, lr_preds = train_model_logistic_regression(
        X_train, X_test, y_train, y_test
    )
    all_models['logistic_regression'] = lr_model
    all_results.append({'Model': 'Logistic Regression', **lr_metrics})
    all_predictions['Logistic Regression'] = lr_preds

    # ----- MODEL 2: Decision Tree -----
    dt_model, dt_metrics, dt_preds = train_model_decision_tree(
        X_train, X_test, y_train, y_test
    )
    all_models['decision_tree'] = dt_model
    all_results.append({'Model': 'Decision Tree', **dt_metrics})
    all_predictions['Decision Tree'] = dt_preds

    # ----- MODEL 3: KNN -----
    knn_model, knn_metrics, knn_preds = train_model_knn(
        X_train, X_test, y_train, y_test
    )
    all_models['knn'] = knn_model
    all_results.append({'Model': 'KNN', **knn_metrics})
    all_predictions['KNN'] = knn_preds

    # ----- MODEL 4: Naive Bayes -----
    nb_model, nb_metrics, nb_preds = train_model_naive_bayes(
        X_train, X_test, y_train, y_test
    )
    all_models['naive_bayes'] = nb_model
    all_results.append({'Model': 'Naive Bayes', **nb_metrics})
    all_predictions['Naive Bayes'] = nb_preds

    # ----- MODEL 5: Random Forest -----
    rf_model, rf_metrics, rf_preds = train_model_random_forest(
        X_train, X_test, y_train, y_test
    )
    all_models['random_forest'] = rf_model
    all_results.append({'Model': 'Random Forest', **rf_metrics})
    all_predictions['Random Forest'] = rf_preds

    # ----- MODEL 6: XGBoost -----
    xgb_model, xgb_metrics, xgb_preds = train_model_xgboost(
        X_train, X_test, y_train, y_test
    )
    all_models['xgboost'] = xgb_model
    all_results.append({'Model': 'XGBoost', **xgb_metrics})
    all_predictions['XGBoost'] = xgb_preds


    comparison_table = create_comparison_table(all_results)


    print("\n" + "=" * 70)
    print("CREATING CONFUSION MATRICES FOR ALL MODELS")
    print("=" * 70)

    for model_name, predictions in all_predictions.items():
        save_confusion_matrix(y_test, predictions, model_name)


    print("\n" + "=" * 80)
    print("DETAILED CLASSIFICATION REPORTS FOR EACH MODEL")
    print("=" * 80)

    for model_name, predictions in all_predictions.items():
        print(f"\n----- {model_name} -----")
        report = classification_report(
            y_test,
            predictions,
            target_names=['Income <=50K', 'Income >50K']
        )
        print(report)


    save_all_models(all_models, scaler, encoders)


    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)


    best_accuracy_idx = comparison_table['Accuracy'].idxmax()
    best_accuracy_val = comparison_table.loc[best_accuracy_idx, 'Accuracy']


    best_auc_idx = comparison_table['AUC'].idxmax()
    best_auc_val = comparison_table.loc[best_auc_idx, 'AUC']

    print(f"\n>>> Dataset: Adult Income Dataset (UCI Repository)")
    print(f">>> Total samples: {len(clean_data)}")
    print(f">>> Features: 14")
    print(f">>> Models trained: 6")
    print(f"\n>>> Best model by Accuracy: {best_accuracy_idx} ({best_accuracy_val})")
    print(f">>> Best model by AUC: {best_auc_idx} ({best_auc_val})")
    print(f"\n>>> Models saved in: model/")
    print(f">>> Plots saved in: model/")

    return comparison_table, all_models, all_predictions


# ================================================================================
# PART L: RUN THE PROGRAM
# ================================================================================

if __name__ == "__main__":
    print("======================================================================")
    print("MACHINE LEARNING ASSIGNMENT - INCOME CLASSIFICATION")
    print("======================================================================")
    print("\nLoading Adult Income Dataset...")
    print("Source: UCI Machine Learning Repository")
    print("Downloading dataset...")
    raw_data = download_and_load_adult_data()
    explore_the_data(raw_data)
    processed_data, encoders = prepare_data_for_ml(raw_data)
    X_train, X_test, y_train, y_test, scaler = split_into_train_test(processed_data)
    print("\n======================================================================")
    print("MODEL TRAINING AND EVALUATION")
    print("======================================================================")
    all_models = {}
    all_results = []
    all_predictions = {}
    # Logistic Regression
    lr_model, lr_metrics, lr_preds = train_model_logistic_regression(X_train, X_test, y_train, y_test)
    all_models['logistic_regression'] = lr_model
    all_results.append({'Model': 'Logistic Regression', **lr_metrics})
    all_predictions['Logistic Regression'] = lr_preds
    print("Logistic Regression Results:")
    for k, v in lr_metrics.items():
        print(f"  {k}: {v}")
    # Decision Tree
    dt_model, dt_metrics, dt_preds = train_model_decision_tree(X_train, X_test, y_train, y_test)
    all_models['decision_tree'] = dt_model
    all_results.append({'Model': 'Decision Tree', **dt_metrics})
    all_predictions['Decision Tree'] = dt_preds
    print("Decision Tree Results:")
    for k, v in dt_metrics.items():
        print(f"  {k}: {v}")
    # KNN
    knn_model, knn_metrics, knn_preds = train_model_knn(X_train, X_test, y_train, y_test)
    all_models['knn'] = knn_model
    all_results.append({'Model': 'KNN', **knn_metrics})
    all_predictions['KNN'] = knn_preds
    print("K-Nearest Neighbors Results:")
    for k, v in knn_metrics.items():
        print(f"  {k}: {v}")
    # Naive Bayes
    nb_model, nb_metrics, nb_preds = train_model_naive_bayes(X_train, X_test, y_train, y_test)
    all_models['naive_bayes'] = nb_model
    all_results.append({'Model': 'Naive Bayes', **nb_metrics})
    all_predictions['Naive Bayes'] = nb_preds
    print("Naive Bayes Results:")
    for k, v in nb_metrics.items():
        print(f"  {k}: {v}")
    # Random Forest
    rf_model, rf_metrics, rf_preds = train_model_random_forest(X_train, X_test, y_train, y_test)
    all_models['random_forest'] = rf_model
    all_results.append({'Model': 'Random Forest', **rf_metrics})
    all_predictions['Random Forest'] = rf_preds
    print("Random Forest Results:")
    for k, v in rf_metrics.items():
        print(f"  {k}: {v}")
    # XGBoost
    xgb_model, xgb_metrics, xgb_preds = train_model_xgboost(X_train, X_test, y_train, y_test)
    all_models['xgboost'] = xgb_model
    all_results.append({'Model': 'XGBoost', **xgb_metrics})
    all_predictions['XGBoost'] = xgb_preds
    print("XGBoost Results:")
    for k, v in xgb_metrics.items():
        print(f"  {k}: {v}")
    # Save models
    print("\n======================================================================")
    print("SAVING MODELS")
    print("======================================================================")
    save_all_models(all_models, scaler, encoders)
    # Save confusion matrices
    for model_name, preds in all_predictions.items():
        save_confusion_matrix(y_test, preds, model_name)
    # Print comparison table
    comparison_table = create_comparison_table(all_results)
    # Best model analysis
    print("\n======================================================================")
    print("DETAILED CLASSIFICATION REPORTS")
    print("======================================================================")
    for model_name, preds in all_predictions.items():
        print(f"--------------------------------------------------")
        print(f"Classification Report: {model_name}")
        print(f"--------------------------------------------------")
        print(classification_report(y_test, preds, target_names=['Income <=50K', 'Income >50K']))