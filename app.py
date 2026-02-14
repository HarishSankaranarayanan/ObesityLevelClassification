"""
Obesity Level Classification - Streamlit Web Application
=========================================================
An interactive web application to demonstrate multiple ML classification 
models for predicting obesity levels based on eating habits and physical condition.

Features:
- Dataset upload option (CSV)
- Model selection dropdown (6 models)
- Display of evaluation metrics
- Confusion matrix visualization
- Classification report
- Individual prediction capability

Author: S Harish Sankaranarayanan 
Date: February 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                            recall_score, f1_score, matthews_corrcoef,
                            classification_report, confusion_matrix)
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Page configuration
st.set_page_config(
    page_title="Obesity Level Classification",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load all saved models and preprocessors."""
    models = {}
    model_files = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Decision Tree': 'decision_tree.pkl',
        'K-Nearest Neighbors': 'k_nearest_neighbors.pkl',
        'Naive Bayes': 'naive_bayes.pkl',
        'Random Forest': 'random_forest.pkl',
        'XGBoost': 'xgboost.pkl'
    }
    
    for name, filename in model_files.items():
        filepath = os.path.join('saved_models', filename)
        if os.path.exists(filepath):
            models[name] = joblib.load(filepath)
    
    # Load preprocessors
    scaler = joblib.load('saved_models/scaler.pkl')
    label_encoders = joblib.load('saved_models/label_encoders.pkl')
    target_encoder = joblib.load('saved_models/target_encoder.pkl')
    
    return models, scaler, label_encoders, target_encoder


def preprocess_data(df, label_encoders, scaler):
    """Preprocess the uploaded data."""
    df_processed = df.copy()
    
    # Define column types
    categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 
                       'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
    numerical_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    
    # Encode categorical features
    for col in categorical_cols:
        if col in df_processed.columns and col in label_encoders:
            le = label_encoders[col]
            # Handle unseen labels
            df_processed[col] = df_processed[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    
    # Scale numerical features
    if all(col in df_processed.columns for col in numerical_cols):
        df_processed[numerical_cols] = scaler.transform(df_processed[numerical_cols])
    
    return df_processed


def calculate_metrics(y_true, y_pred, y_prob, num_classes):
    """Calculate evaluation metrics."""
    metrics = {}
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    
    try:
        if num_classes == 2:
            metrics['AUC'] = roc_auc_score(y_true, y_prob[:, 1])
        else:
            metrics['AUC'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
    except:
        metrics['AUC'] = 0.0
    
    metrics['Precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['Recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['F1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['MCC'] = matthews_corrcoef(y_true, y_pred)
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, classes):
    """Create confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    return fig


def plot_metrics_comparison(results_df):
    """Create bar chart comparing model metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
    colors = ['#1E88E5', '#43A047', '#FB8C00', '#E53935', '#8E24AA', '#00ACC1']
    
    for idx, (metric, color) in enumerate(zip(metrics, colors)):
        ax = axes[idx // 3, idx % 3]
        bars = ax.bar(results_df['Model'], results_df[metric], color=color, alpha=0.8)
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=8)
        
        # Add value labels on bars
        for bar, val in zip(bars, results_df[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    return fig


def main():
    # Header
    st.markdown('<p class="main-header">Obesity Level Classification</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Multi-class Classification using Machine Learning Models</p>', unsafe_allow_html=True)
    
    # Load models
    try:
        models, scaler, label_encoders, target_encoder = load_models()
        st.sidebar.success("Models loaded successfully!")
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Model selection dropdown
    model_names = list(models.keys())
    selected_model = st.sidebar.selectbox(
        "Select ML Model",
        model_names,
        index=model_names.index('XGBoost') if 'XGBoost' in model_names else 0
    )
    
    st.sidebar.markdown("---")
    
    # Dataset information
    st.sidebar.header("Dataset Info")
    st.sidebar.markdown("""
    **Dataset:** Obesity Levels Based on Eating Habits and Physical Condition
    
    **Source:** Kaggle
    
    **Features:** 16
    
    **Classes:** 7
    - Insufficient Weight
    - Normal Weight
    - Overweight Level I
    - Overweight Level II
    - Obesity Type I
    - Obesity Type II
    - Obesity Type III
    """)
    
    # Main content with tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Upload & Predict", 
        "Model Metrics", 
        "Individual Prediction",
        " About"
    ])
    

    with tab1:
        st.header(" Upload Test Data & Generate Predictions")
        
        st.markdown("""
        Upload a CSV file containing test data to evaluate the selected model. 
        The CSV should contain all feature columns and optionally the target column ('NObeyesdad').
        """)
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload test data in CSV format"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"File uploaded successfully! Shape: {df.shape}")
                
                # Show data preview
                with st.expander("Data Preview", expanded=True):
                    st.dataframe(df.head(10), use_container_width=True)
                
                # Check if target column exists
                has_target = 'NObeyesdad' in df.columns
                
                if has_target:
                    y_true = target_encoder.transform(df['NObeyesdad'])
                    X = df.drop('NObeyesdad', axis=1)
                else:
                    X = df.copy()
                
                # Preprocess data
                X_processed = preprocess_data(X, label_encoders, scaler)
                
                # Get selected model
                model = models[selected_model]
                
                # Make predictions
                y_pred = model.predict(X_processed)
                y_prob = model.predict_proba(X_processed)
                
                # Decode predictions
                y_pred_labels = target_encoder.inverse_transform(y_pred)
                
                # Show predictions
                st.subheader("Predictions")
                results_df = X.copy()
                results_df['Predicted_Obesity_Level'] = y_pred_labels
                
                if has_target:
                    results_df['Actual_Obesity_Level'] = df['NObeyesdad']
                    results_df['Correct'] = results_df['Predicted_Obesity_Level'] == results_df['Actual_Obesity_Level']
                
                st.dataframe(results_df, use_container_width=True)
                
                # Download predictions
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label=" Download Predictions as CSV",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )
                
                # If target column exists, show evaluation metrics
                if has_target:
                    st.subheader(f" Evaluation Metrics for {selected_model}")
                    
                    metrics = calculate_metrics(y_true, y_pred, y_prob, len(target_encoder.classes_))
                    
                    # Display metrics in columns
                    col1, col2, col3, col4, col5, col6 = st.columns(6)
                    
                    with col1:
                        st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                    with col2:
                        st.metric("AUC Score", f"{metrics['AUC']:.4f}")
                    with col3:
                        st.metric("Precision", f"{metrics['Precision']:.4f}")
                    with col4:
                        st.metric("Recall", f"{metrics['Recall']:.4f}")
                    with col5:
                        st.metric("F1 Score", f"{metrics['F1']:.4f}")
                    with col6:
                        st.metric("MCC", f"{metrics['MCC']:.4f}")
                    
                    # Confusion Matrix
                    st.subheader("Confusion Matrix")
                    fig = plot_confusion_matrix(y_true, y_pred, target_encoder.classes_)
                    st.pyplot(fig)
                    
                    # Classification Report
                    st.subheader(" Classification Report")
                    report = classification_report(y_true, y_pred, 
                                                  target_names=target_encoder.classes_,
                                                  output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error processing file: {e}")
    

    with tab2:
        st.header("Model Performance Comparison")
        
        # Load pre-computed results
        try:
            results_df = pd.read_csv('model_results.csv')
            
            # Display metrics table
            st.subheader("Evaluation Metrics Comparison Table")
            
            # Format the dataframe for display
            styled_df = results_df.style.format({
                'Accuracy': '{:.4f}',
                'AUC': '{:.4f}',
                'Precision': '{:.4f}',
                'Recall': '{:.4f}',
                'F1': '{:.4f}',
                'MCC': '{:.4f}'
            }).background_gradient(subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'], 
                                   cmap='RdYlGn')
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Best model highlight
            best_idx = results_df['Accuracy'].idxmax()
            best_model = results_df.loc[best_idx, 'Model']
            best_accuracy = results_df.loc[best_idx, 'Accuracy']
            
            st.success(f"**Best Model:** {best_model} with Accuracy: {best_accuracy:.4f}")
            
            # Metrics visualization
            st.subheader(" Metrics Visualization")
            fig = plot_metrics_comparison(results_df)
            st.pyplot(fig)
            
            # Individual model details
            st.subheader("Selected Model Details")
            model_metrics = results_df[results_df['Model'] == selected_model].iloc[0]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{model_metrics['Accuracy']:.4f}")
                st.metric("AUC", f"{model_metrics['AUC']:.4f}")
            with col2:
                st.metric("Precision", f"{model_metrics['Precision']:.4f}")
                st.metric("Recall", f"{model_metrics['Recall']:.4f}")
            with col3:
                st.metric("F1 Score", f"{model_metrics['F1']:.4f}")
                st.metric("MCC", f"{model_metrics['MCC']:.4f}")
                
        except Exception as e:
            st.warning("Pre-computed results not found. Please upload test data in Tab 1 to see metrics.")
    

    with tab3:
        st.header(" Make Individual Prediction")
        st.markdown("Enter the values below to predict the obesity level for an individual.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.slider("Age", 14, 61, 25)
            height = st.slider("Height (m)", 1.45, 1.98, 1.70, 0.01)
            weight = st.slider("Weight (kg)", 39.0, 173.0, 70.0, 0.5)
            family_history = st.selectbox("Family History with Overweight", ["yes", "no"])
            favc = st.selectbox("Frequent High Caloric Food (FAVC)", ["yes", "no"])
        
        with col2:
            fcvc = st.slider("Vegetable Consumption (FCVC)", 1.0, 3.0, 2.0, 0.1)
            ncp = st.slider("Number of Main Meals (NCP)", 1.0, 4.0, 3.0, 0.1)
            caec = st.selectbox("Food Between Meals (CAEC)", 
                               ["no", "Sometimes", "Frequently", "Always"])
            smoke = st.selectbox("Smoking (SMOKE)", ["yes", "no"])
            ch2o = st.slider("Water Intake (CH2O)", 1.0, 3.0, 2.0, 0.1)
            scc = st.selectbox("Calorie Monitoring (SCC)", ["yes", "no"])
        
        with col3:
            faf = st.slider("Physical Activity Frequency (FAF)", 0.0, 3.0, 1.0, 0.1)
            tue = st.slider("Technology Use Time (TUE)", 0.0, 2.0, 1.0, 0.1)
            calc = st.selectbox("Alcohol Consumption (CALC)", 
                               ["no", "Sometimes", "Frequently", "Always"])
            mtrans = st.selectbox("Transportation (MTRANS)", 
                                 ["Automobile", "Motorbike", "Bike", 
                                  "Public_Transportation", "Walking"])
        
        if st.button("Predict Obesity Level", type="primary"):
            # Create input dataframe
            input_data = pd.DataFrame({
                'Gender': [gender],
                'Age': [age],
                'Height': [height],
                'Weight': [weight],
                'family_history_with_overweight': [family_history],
                'FAVC': [favc],
                'FCVC': [fcvc],
                'NCP': [ncp],
                'CAEC': [caec],
                'SMOKE': [smoke],
                'CH2O': [ch2o],
                'SCC': [scc],
                'FAF': [faf],
                'TUE': [tue],
                'CALC': [calc],
                'MTRANS': [mtrans]
            })
            
            # Preprocess
            input_processed = preprocess_data(input_data, label_encoders, scaler)
            
            # Get model and predict
            model = models[selected_model]
            prediction = model.predict(input_processed)
            probabilities = model.predict_proba(input_processed)[0]
            
            # Decode prediction
            predicted_class = target_encoder.inverse_transform(prediction)[0]
            
            # Display result
            st.markdown("---")
            st.subheader(" Prediction Result")
            
            # Color coding based on obesity level
            color_map = {
                'Insufficient_Weight': '🔵',
                'Normal_Weight': '🟢',
                'Overweight_Level_I': '🟡',
                'Overweight_Level_II': '🟠',
                'Obesity_Type_I': '🟠',
                'Obesity_Type_II': '🔴',
                'Obesity_Type_III': '🔴'
            }
            
            emoji = color_map.get(predicted_class, '⚪')
            st.markdown(f"### {emoji} Predicted Obesity Level: **{predicted_class.replace('_', ' ')}**")
            
            # Show probability distribution
            st.subheader(" Prediction Probabilities")
            prob_df = pd.DataFrame({
                'Obesity Level': target_encoder.classes_,
                'Probability': probabilities
            }).sort_values('Probability', ascending=False)
            
            # Bar chart for probabilities
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = ['#2ecc71' if c == predicted_class else '#3498db' for c in prob_df['Obesity Level']]
            bars = ax.barh(prob_df['Obesity Level'], prob_df['Probability'], color=colors)
            ax.set_xlabel('Probability')
            ax.set_title(f'Prediction Probabilities ({selected_model})')
            ax.set_xlim(0, 1)
            
            # Add percentage labels
            for bar, prob in zip(bars, prob_df['Probability']):
                ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                       f'{prob:.1%}', va='center', fontsize=10)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # BMI calculation
            bmi = weight / (height ** 2)
            st.info(f" **Calculated BMI:** {bmi:.2f}")
    

    with tab4:
        st.header("About This Application")
        
        st.markdown("""
        ### Project Overview
        
        This application demonstrates the implementation of **6 different Machine Learning 
        classification models** for predicting obesity levels based on eating habits and 
        physical condition.
        
        ###  Dataset Information
        
        - **Name:** Estimation of Obesity Levels Based on Eating Habits and Physical Condition
        - **Source:** Kaggle
        - **Instances:** 2,111
        - **Features:** 16 (8 categorical + 8 numerical)
        - **Target Classes:** 7 obesity levels
        
        ### Implemented Models
        
        1. **Logistic Regression** - Linear model for classification
        2. **Decision Tree Classifier** - Tree-based model
        3. **K-Nearest Neighbors (KNN)** - Instance-based learning
        4. **Naive Bayes (Gaussian)** - Probabilistic classifier
        5. **Random Forest** - Ensemble of decision trees
        6. **XGBoost** - Gradient boosting ensemble
        
        ###  Evaluation Metrics
        
        - **Accuracy** - Overall correctness of predictions
        - **AUC Score** - Area Under the ROC Curve
        - **Precision** - Positive predictive value
        - **Recall** - True positive rate
        - **F1 Score** - Harmonic mean of precision and recall
        - **MCC** - Matthews Correlation Coefficient
        
        ### Technologies Used
        
        - **Python 3.x**
        - **Scikit-learn** - ML models and metrics
        - **XGBoost** - Gradient boosting
        - **Streamlit** - Web application framework
        - **Pandas & NumPy** - Data manipulation
        - **Matplotlib & Seaborn** - Visualization
        
        ### Author
        
        **S Harish Sankaranarayanan**  
    
    
        """)
        
        # Feature descriptions
        st.subheader("Feature Descriptions")
        
        feature_info = {
            'Gender': 'Male or Female',
            'Age': 'Age in years (14-61)',
            'Height': 'Height in meters',
            'Weight': 'Weight in kilograms',
            'family_history_with_overweight': 'Family history of overweight (yes/no)',
            'FAVC': 'Frequent consumption of high caloric food (yes/no)',
            'FCVC': 'Frequency of vegetable consumption (1-3)',
            'NCP': 'Number of main meals per day (1-4)',
            'CAEC': 'Consumption of food between meals',
            'SMOKE': 'Smoking habit (yes/no)',
            'CH2O': 'Daily water consumption (1-3 liters)',
            'SCC': 'Calories consumption monitoring (yes/no)',
            'FAF': 'Physical activity frequency (0-3 days/week)',
            'TUE': 'Time using technology devices (0-2 hours)',
            'CALC': 'Consumption of alcohol',
            'MTRANS': 'Transportation used'
        }
        
        feature_df = pd.DataFrame({
            'Feature': feature_info.keys(),
            'Description': feature_info.values()
        })
        st.dataframe(feature_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
