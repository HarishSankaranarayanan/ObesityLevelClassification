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

# Page configuration
st.set_page_config(
    page_title="Obesity Level Classification",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal clean styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #0066cc;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .download-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #cce5ff;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-left: 4px solid #0066cc;
        border-radius: 4px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# UTILITY FUNCTIONS
# ============================================

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
    
    scaler = joblib.load('saved_models/scaler.pkl')
    label_encoders = joblib.load('saved_models/label_encoders.pkl')
    target_encoder = joblib.load('saved_models/target_encoder.pkl')
    
    return models, scaler, label_encoders, target_encoder


def preprocess_data(df, label_encoders, scaler):
    """Preprocess the uploaded data."""
    df_processed = df.copy()
    
    categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 
                       'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
    numerical_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    
    for col in categorical_cols:
        if col in df_processed.columns and col in label_encoders:
            le = label_encoders[col]
            df_processed[col] = df_processed[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
    
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
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for idx, (metric, color) in enumerate(zip(metrics, colors)):
        ax = axes[idx // 3, idx % 3]
        bars = ax.bar(results_df['Model'], results_df[metric], color=color, alpha=0.7)
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, results_df[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    return fig


# ============================================
# MAIN APPLICATION
# ============================================

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
    
    model_names = list(models.keys())
    selected_model = st.sidebar.selectbox(
        "Select ML Model",
        model_names,
        help="Choose which model to use for predictions"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("Dataset Info")
    st.sidebar.markdown("""
    **Dataset:** Obesity Levels Based on Eating Habits and Physical Condition
    
    **Source:** Kaggle  
    ([Link](https://www.kaggle.com/datasets/aravindpcoder/obesity-or-cvd-risk-classifyregressorcluster/data))
    
    **Features:** 16
    
    **Classes:** 7
    """)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Upload & Predict", 
        "Model Comparison",
        "Individual Prediction",
        "About"
    ])
    
    # ============================================
    # TAB 1: UPLOAD & PREDICT
    # ============================================
    with tab1:
        st.header("Upload Test Data & Generate Predictions")
        
        # Download test data section
        st.markdown('<div class="download-box">', unsafe_allow_html=True)
        st.subheader("Download Test Data")
        st.write("Download sample test data to evaluate the model or use as template:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if os.path.exists('test_data.csv'):
                with open('test_data.csv', 'rb') as f:
                    st.download_button(
                        label="Download Full Test Data (423 rows)",
                        data=f,
                        file_name="test_data.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        
        with col2:
            if os.path.exists('test_data_sample.csv'):
                with open('test_data_sample.csv', 'rb') as f:
                    st.download_button(
                        label="Download Sample Data (50 rows)",
                        data=f,
                        file_name="test_data_sample.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        
        with col3:
            if os.path.exists('train_data.csv'):
                with open('train_data.csv', 'rb') as f:
                    st.download_button(
                        label="Download Training Data (1688 rows)",
                        data=f,
                        file_name="train_data.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Upload section
        st.markdown("""
        <div class="info-box">
        <strong>File Requirements:</strong>
        <ul>
            <li>CSV format with all 16 feature columns</li>
            <li>Optionally include 'NObeyesdad' column for evaluation</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload test data for model evaluation"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"File uploaded successfully! Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                
                # Show data preview
                with st.expander("Data Preview", expanded=True):
                    st.dataframe(df.head(10), use_container_width=True)
                
                # Check if target column exists
                has_target = 'NObeyesdad' in df.columns
                
                if has_target:
                    st.info("Target column 'NObeyesdad' found - Will show evaluation metrics")
                    y_true = target_encoder.transform(df['NObeyesdad'])
                    X = df.drop('NObeyesdad', axis=1)
                else:
                    st.warning("No target column found - Will only show predictions")
                    X = df.copy()
                
                # Preprocess data
                X_processed = preprocess_data(X, label_encoders, scaler)
                
                # Get model and predict
                model = models[selected_model]
                y_pred = model.predict(X_processed)
                y_prob = model.predict_proba(X_processed)
                
                # Decode predictions
                predictions = target_encoder.inverse_transform(y_pred)
                
                # Display results
                st.markdown("---")
                st.subheader("Prediction Results")
                
                # Add predictions to dataframe
                results_df = df.copy()
                results_df['Predicted_Obesity_Level'] = predictions
                results_df['Prediction_Confidence'] = y_prob.max(axis=1).round(3)
                
                st.dataframe(results_df, use_container_width=True)
                
                # Download predictions
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name=f"predictions_{selected_model.lower().replace(' ', '_')}.csv",
                    mime="text/csv"
                )
                
                # Show metrics if target exists
                if has_target:
                    st.markdown("---")
                    st.subheader("Model Performance Metrics")
                    
                    num_classes = len(target_encoder.classes_)
                    metrics = calculate_metrics(y_true, y_pred, y_prob, num_classes)
                    
                    # Display metrics in columns
                    col1, col2, col3, col4, col5, col6 = st.columns(6)
                    
                    col1.metric("Accuracy", f"{metrics['Accuracy']:.2%}")
                    col2.metric("AUC", f"{metrics['AUC']:.3f}")
                    col3.metric("Precision", f"{metrics['Precision']:.3f}")
                    col4.metric("Recall", f"{metrics['Recall']:.3f}")
                    col5.metric("F1 Score", f"{metrics['F1']:.3f}")
                    col6.metric("MCC", f"{metrics['MCC']:.3f}")
                    
                    # Confusion Matrix
                    st.markdown("---")
                    st.subheader("Confusion Matrix")
                    
                    fig = plot_confusion_matrix(y_true, y_pred, target_encoder.classes_)
                    st.pyplot(fig)
                    
                    # Classification Report
                    with st.expander("Detailed Classification Report"):
                        report = classification_report(y_true, y_pred, 
                                                      target_names=target_encoder.classes_,
                                                      output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    # ============================================
    # TAB 2: MODEL COMPARISON
    # ============================================
    with tab2:
        st.header("Model Performance Comparison")
        
        st.markdown("""
        <div class="info-box">
        <strong>About These Metrics:</strong><br>
        These performance metrics were calculated during model training on the test dataset (423 samples, 20% of data).
        All models were trained on the same data split to ensure fair comparison.
        <br><br>
        <strong>Data Split:</strong>
        <ul>
            <li>Training Set: 1,688 samples (80%)</li>
            <li>Test Set: 423 samples (20%)</li>
            <li>Method: Stratified split (maintains class distribution)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Load pre-computed results
        if os.path.exists('model_results.csv'):
            results_df = pd.read_csv('model_results.csv')
            
            st.subheader("Performance Metrics Table")
            
            # Highlight best model
            best_idx = results_df['Accuracy'].idxmax()
            
            def highlight_best(row):
                if row.name == best_idx:
                    return ['background-color: #d4edda; font-weight: bold'] * len(row)
                return [''] * len(row)
            
            styled_df = results_df.style.apply(highlight_best, axis=1)\
                                       .format({
                                           'Accuracy': '{:.4f}',
                                           'AUC': '{:.4f}',
                                           'Precision': '{:.4f}',
                                           'Recall': '{:.4f}',
                                           'F1': '{:.4f}',
                                           'MCC': '{:.4f}'
                                       })
            
            st.dataframe(styled_df, use_container_width=True)
            
            best_model = results_df.loc[best_idx, 'Model']
            best_accuracy = results_df.loc[best_idx, 'Accuracy']
            
            st.success(f"Best Performing Model: {best_model} with {best_accuracy:.2%} accuracy")
            
            # Visual comparison
            st.markdown("---")
            st.subheader("Visual Performance Comparison")
            
            fig = plot_metrics_comparison(results_df)
            st.pyplot(fig)
            
            # Model Rankings
            st.markdown("---")
            st.subheader("Model Rankings by Metric")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Top 3 by Accuracy:**")
                top_acc = results_df.nlargest(3, 'Accuracy')[['Model', 'Accuracy']]
                for idx, row in top_acc.iterrows():
                    st.write(f"{idx+1}. {row['Model']}: {row['Accuracy']:.2%}")
                
                st.write("")
                st.write("**Top 3 by AUC:**")
                top_auc = results_df.nlargest(3, 'AUC')[['Model', 'AUC']]
                for idx, row in top_auc.iterrows():
                    st.write(f"{idx+1}. {row['Model']}: {row['AUC']:.4f}")
            
            with col2:
                st.write("**Top 3 by F1 Score:**")
                top_f1 = results_df.nlargest(3, 'F1')[['Model', 'F1']]
                for idx, row in top_f1.iterrows():
                    st.write(f"{idx+1}. {row['Model']}: {row['F1']:.4f}")
                
                st.write("")
                st.write("**Top 3 by MCC:**")
                top_mcc = results_df.nlargest(3, 'MCC')[['Model', 'MCC']]
                for idx, row in top_mcc.iterrows():
                    st.write(f"{idx+1}. {row['Model']}: {row['MCC']:.4f}")
            
        else:
            st.warning("Model results file not found. Please ensure 'model_results.csv' exists.")
    
    # ============================================
    # TAB 3: INDIVIDUAL PREDICTION
    # ============================================
    with tab3:
        st.header("Individual Prediction")
        st.write("Enter patient information below to predict obesity level:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Demographics")
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.slider("Age (years)", 14, 61, 25)
            height = st.slider("Height (meters)", 1.45, 1.98, 1.70, 0.01)
            weight = st.slider("Weight (kg)", 39.0, 173.0, 70.0, 0.5)
            family_history = st.selectbox("Family History with Overweight", ["yes", "no"])
            favc = st.selectbox("Frequent High Caloric Food (FAVC)", ["yes", "no"])
        
        with col2:
            st.subheader("Eating Habits")
            fcvc = st.slider("Vegetable Consumption (FCVC)", 1.0, 3.0, 2.0, 0.1)
            ncp = st.slider("Number of Main Meals (NCP)", 1.0, 4.0, 3.0, 0.1)
            caec = st.selectbox("Food Between Meals (CAEC)", 
                               ["no", "Sometimes", "Frequently", "Always"])
            smoke = st.selectbox("Smoking (SMOKE)", ["yes", "no"])
            ch2o = st.slider("Water Intake (CH2O)", 1.0, 3.0, 2.0, 0.1)
            scc = st.selectbox("Calorie Monitoring (SCC)", ["yes", "no"])
        
        with col3:
            st.subheader("Physical Activity & Lifestyle")
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
            st.subheader("Prediction Result")
            
            st.success(f"**Predicted Obesity Level:** {predicted_class.replace('_', ' ')}")
            st.info(f"**Confidence:** {probabilities.max():.1%}")
            
            # Show probability distribution
            st.subheader("Prediction Probabilities")
            prob_df = pd.DataFrame({
                'Obesity Level': target_encoder.classes_,
                'Probability': probabilities
            }).sort_values('Probability', ascending=False)
            
            # Bar chart for probabilities
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#2ecc71' if c == predicted_class else '#3498db' for c in prob_df['Obesity Level']]
            bars = ax.barh(prob_df['Obesity Level'], prob_df['Probability'], color=colors)
            ax.set_xlabel('Probability')
            ax.set_title(f'Prediction Probabilities ({selected_model})')
            ax.set_xlim(0, 1)
            
            # Add percentage labels
            for bar, prob in zip(bars, prob_df['Probability']):
                ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                       f'{prob:.1%}', va='center')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # BMI calculation
            bmi = weight / (height ** 2)
            st.info(f"**Calculated BMI:** {bmi:.2f}")
    
    # ============================================
    # TAB 4: ABOUT
    # ============================================
    with tab4:
        st.header("About This Application")
        
        st.markdown("""
        ### Project Overview
        
        This application demonstrates the implementation of **6 different Machine Learning 
        classification models** for predicting obesity levels based on eating habits and 
        physical condition.
        
        ### Dataset Information
        
        - **Name:** Estimation of Obesity Levels Based on Eating Habits and Physical Condition
        - **Source:** Kaggle (UCI Machine Learning Repository, ID: 544)
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
        
        ### Evaluation Metrics
        
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
        M.Tech (AIML/DSE)  
        BITS Pilani - Work Integrated Learning Programme
        
        ### Assignment Details
        
        - **Course:** Machine Learning
        - **Assignment:** Assignment 2
        - **Submission Deadline:** 15-Feb-2026
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