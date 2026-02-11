# =============================================================================
# Adult Income Classification - Streamlit Web Application
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Income Classification App",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Custom CSS Styling
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Load all pre-trained models and preprocessing artifacts
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models():

    model_dir = "model"
    models = {}

    model_files = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Decision Tree': 'decision_tree.pkl',
        'K-Nearest Neighbors': 'knn.pkl',
        'Naive Bayes': 'naive_bayes.pkl',
        'Random Forest': 'random_forest.pkl',
        'XGBoost': 'xgboost.pkl'
    }

    for name, filename in model_files.items():
        filepath = os.path.join(model_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                models[name] = pickle.load(f)

    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    encoders_path = os.path.join(model_dir, 'label_encoders.pkl')

    scaler = None
    label_encoders = None

    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

    if os.path.exists(encoders_path):
        with open(encoders_path, 'rb') as f:
            label_encoders = pickle.load(f)

    return models, scaler, label_encoders

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def preprocess_data(df, label_encoders, scaler):
    df_processed = df.copy()

    if 'income' in df_processed.columns:
        # Clean income column
        df_processed['income'] = df_processed['income'].astype(str)
        df_processed['income'] = df_processed['income'].str.replace('.', '', regex=False)
        df_processed['income'] = df_processed['income'].str.strip()

        if label_encoders and 'income' in label_encoders:
            y = label_encoders['income'].transform(df_processed['income'])
        else:
            y = df_processed['income'].map({'<=50K': 0, '>50K': 1})
        df_processed = df_processed.drop('income', axis=1)
    else:
        y = None

    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()

    for col in categorical_cols:
        if label_encoders and col in label_encoders:
            le = label_encoders[col]
            # Handle unseen labels
            df_processed[col] = df_processed[col].apply(
                lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1
            )

    if scaler:
        X_scaled = scaler.transform(df_processed)
    else:
        X_scaled = df_processed.values

    return X_scaled, y

def calculate_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }

    if y_prob is not None:
        try:
            if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
                metrics['AUC'] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                metrics['AUC'] = roc_auc_score(y_true, y_prob)
        except:
            metrics['AUC'] = 0.0
    else:
        metrics['AUC'] = 0.0

    return metrics

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['<=50K', '>50K'],
                yticklabels=['<=50K', '>50K'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    return fig

# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------
def main():
    st.markdown('<p class="main-header">💰 Adult Income Classification</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict whether annual income exceeds $50,000</p>', unsafe_allow_html=True)

    try:
        models, scaler, label_encoders = load_models()
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.info("Please ensure the model files are present in the 'model/' directory.")
        return

    st.sidebar.title("⚙️ Settings")
    st.sidebar.markdown("---")

    st.sidebar.subheader("🤖 Select Model")
    available_models = list(models.keys())
    if not available_models:
        st.error("No models found. Please train the models first using adult_income_prediction.py")
        return

    selected_model = st.sidebar.selectbox(
        "Choose a classification model:",
        available_models,
        index=len(available_models) - 1  # Default to XGBoost
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Navigation")
    page = st.sidebar.radio(
        "Select Page:",
        ["🏠 Home", "📤 Upload & Predict"]
    )

    # ---------------------------------------------------------------------
    # HOME PAGE
    # ---------------------------------------------------------------------
    if page == "🏠 Home":
        st.header("Welcome to the Income Classifier")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📋 About the Dataset")
            st.markdown("""
            The **Adult Census Income Dataset** contains data extracted from the 1994 
            US Census database. The classification goal is to predict whether a person 
            earns more than $50,000 per year.
            
            **Dataset Characteristics:**
            - **Instances:** 48,842
            - **Features:** 14
            - **Target:** Binary (<=50K / >50K)
            - **Source:** UCI Machine Learning Repository
            """)

        with col2:
            st.subheader("🎯 Available Models")
            st.markdown("""
            This application includes 6 trained classification models:
            
            1. **Logistic Regression** - Linear classifier
            2. **Decision Tree** - Tree-based classifier
            3. **K-Nearest Neighbors** - Instance-based learning
            4. **Naive Bayes** - Probabilistic classifier
            5. **Random Forest** - Ensemble of decision trees
            6. **XGBoost** - Gradient boosting ensemble
            """)

        st.markdown("---")

        # Display pre-computed metrics
        st.subheader("📊 Model Performance Overview")

        # Create comparison table from stored results
        comparison_data = {
            'Model': ['Logistic Regression', 'Decision Tree', 'KNN', 'Naive Bayes', 'Random Forest', 'XGBoost'],
            'Accuracy': [0.8200, 0.8496, 0.8284, 0.7958, 0.8561, 0.8687],
            'AUC': [0.8478, 0.8942, 0.8537, 0.8501, 0.9144, 0.9251],
            'Precision': [0.7174, 0.7582, 0.6749, 0.6858, 0.7967, 0.7921],
            'Recall': [0.4518, 0.5776, 0.5937, 0.3252, 0.5629, 0.6374],
            'F1 Score': [0.5545, 0.6557, 0.6317, 0.4411, 0.6597, 0.7064],
            'MCC': [0.4677, 0.5703, 0.5223, 0.3701, 0.5857, 0.6292]
        }
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.set_index('Model')

        styled_df = comparison_df.style.highlight_max(axis=0, color='lightgreen').format("{:.4f}")
        st.dataframe(styled_df, use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Confusion Matrices")

        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]

        cm_files = [
            ('Logistic Regression', 'confusion_matrix_Logistic_Regression.png'),
            ('Decision Tree', 'confusion_matrix_Decision_Tree.png'),
            ('KNN', 'confusion_matrix_KNN.png'),
            ('Naive Bayes', 'confusion_matrix_Naive_Bayes.png'),
            ('Random Forest', 'confusion_matrix_Random_Forest.png'),
            ('XGBoost', 'confusion_matrix_XGBoost.png')
        ]

        for idx, (name, filename) in enumerate(cm_files):
            cm_path = os.path.join("model", filename)
            if os.path.exists(cm_path):
                col_index = idx % len(cols)
                with cols[col_index]:
                    st.image(cm_path, caption=name, use_container_width=True)

        st.markdown("---")
        st.subheader("📉 Metrics Visualization")

        metrics_to_plot = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        colors = sns.color_palette("husl", len(comparison_df))

        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx // 3, idx % 3]
            if metric in comparison_df.columns:
                bars = ax.bar(comparison_df.index, comparison_df[metric], color=colors)
                ax.set_title(metric, fontsize=12, fontweight='bold')
                ax.set_xticklabels(comparison_df.index, rotation=45, ha='right', fontsize=8)
                ax.set_ylim(0, 1.1)

                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("---")

    # ---------------------------------------------------------------------
    # UPLOAD & PREDICT PAGE
    # ---------------------------------------------------------------------
    elif page == "📤 Upload & Predict":
        st.header("📤 Upload Data & Make Predictions")

        st.markdown("""
        Upload a CSV file containing test data to make predictions using the selected model.
        
        **Note:** The CSV file should have the same structure as the Adult Income dataset.
        
        **Need a sample?** [Download sample_test_data.csv](https://github.com/kavyakolluru/ML/raw/main/sample_test_data.csv)
        """)

        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with the same format as the Adult Income dataset"
        )

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)

                st.success(f"✅ File uploaded successfully! Shape: {df.shape}")

                with st.expander("📋 Data Preview", expanded=True):
                    st.dataframe(df.head(10), use_container_width=True)

                has_target = 'income' in df.columns

                if has_target:
                    st.info("✅ Target column 'income' detected. Will calculate evaluation metrics.")
                else:
                    st.warning("⚠️ No target column 'income' found. Only predictions will be shown.")

                if st.button("🚀 Make Predictions", type="primary"):
                    with st.spinner("Processing data and making predictions..."):
                        X_processed, y_true = preprocess_data(df, label_encoders, scaler)

                        if selected_model is not None:
                            model = models[selected_model]
                            y_pred = model.predict(X_processed)
                            try:
                                y_prob = model.predict_proba(X_processed)
                            except:
                                y_prob = None

                            st.markdown("---")
                            st.subheader(f"🎯 Results for {selected_model}")

                            col1, col2 = st.columns(2)

                            with col1:
                                st.markdown("**Prediction Summary:**")
                                pred_df = pd.DataFrame({
                                    'Prediction': ['<=50K (Low Income)', '>50K (High Income)'],
                                    'Count': [np.sum(y_pred == 0), np.sum(y_pred == 1)]
                                })
                                st.dataframe(pred_df, use_container_width=True)

                            with col2:
                                fig, ax = plt.subplots(figsize=(4, 4))
                                ax.pie([np.sum(y_pred == 0), np.sum(y_pred == 1)],
                                       labels=['<=50K', '>50K'],
                                       autopct='%1.1f%%',
                                       colors=['#ff6b6b', '#51cf66'])
                                ax.set_title('Prediction Distribution')
                                st.pyplot(fig)

                            if has_target and y_true is not None:
                                st.markdown("---")
                                st.subheader("📈 Evaluation Metrics")

                                metrics = calculate_metrics(y_true, y_pred, y_prob)

                                col1, col2, col3 = st.columns(3)
                                col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                                col2.metric("AUC Score", f"{metrics['AUC']:.4f}")
                                col3.metric("F1 Score", f"{metrics['F1 Score']:.4f}")

                                col4, col5, col6 = st.columns(3)
                                col4.metric("Precision", f"{metrics['Precision']:.4f}")
                                col5.metric("Recall", f"{metrics['Recall']:.4f}")
                                col6.metric("MCC", f"{metrics['MCC']:.4f}")

                                st.markdown("---")

                                col1, col2 = st.columns(2)

                                with col1:
                                    st.subheader("📊 Confusion Matrix")
                                    fig = plot_confusion_matrix(y_true, y_pred, f"Confusion Matrix - {selected_model}")
                                    st.pyplot(fig)

                                with col2:
                                    st.subheader("📋 Classification Report")
                                    report = classification_report(y_true, y_pred, target_names=['<=50K', '>50K'], output_dict=True)
                                    report_df = pd.DataFrame(report).transpose()
                                    st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.info("Please ensure your CSV file has the correct format.")

    st.sidebar.markdown("---")

# -----------------------------------------------------------------------------
# Run Application
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()

