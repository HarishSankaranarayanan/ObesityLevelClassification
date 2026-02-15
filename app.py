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
    page_title="Obesity Classification",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 1rem 0;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    .download-section {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 2rem 0;
    }
    
    .info-box {
        background-color: #e7f3ff;
        border-left: 5px solid #2196F3;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        padding: 0 24px;
        background-color: white;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
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
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', 
                xticklabels=classes, yticklabels=classes, ax=ax,
                cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    return fig


def plot_metrics_comparison(results_df):
    """Create bar chart comparing model metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
    colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4ecdc4', '#45b7d1']
    
    for idx, (metric, color) in enumerate(zip(metrics, colors)):
        ax = axes[idx // 3, idx % 3]
        bars = ax.bar(results_df['Model'], results_df[metric], color=color, alpha=0.8)
        ax.set_title(metric, fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, results_df[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    return fig


# ============================================
# MAIN APPLICATION
# ============================================

def main():
    # Header
    st.markdown('<p class="main-header"> Obesity Level Classification System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Multi-Class Classification | 6 Machine Learning Models</p>', unsafe_allow_html=True)
    
    # Load models
    try:
        models, scaler, label_encoders, target_encoder = load_models()
        st.sidebar.success(" All models loaded successfully!")
    except Exception as e:
        st.error(f" Error loading models: {e}")
        st.info(" Make sure the 'saved_models' folder with .pkl files exists!")
        st.stop()
    
    # Sidebar
    st.sidebar.markdown("---")
    st.sidebar.header(" Model Configuration")
    
    model_names = list(models.keys())
    selected_model = st.sidebar.selectbox(
        " Select ML Model",
        model_names,
        index=0,  # Default to first model
        help="Choose which model to use for predictions"
    )
    
    # Model info
    model_info = {
        'Logistic Regression': ' Linear probabilistic classifier',
        'Decision Tree': ' Tree-based interpretable model',
        'K-Nearest Neighbors': ' Distance-based classifier',
        'Naive Bayes': ' Probabilistic Bayesian model',
        'Random Forest': ' Ensemble of decision trees',
        'XGBoost': ' Gradient boosting ensemble'
    }
    
    st.sidebar.info(f"**Current Model:**\n{model_info.get(selected_model, '')}")
    
    st.sidebar.markdown("---")
    st.sidebar.header("Dataset Information")
    st.sidebar.markdown("""
    **Dataset:** Obesity Levels  
    **Source:** UCI ML Repository
    
    **Features:** 16  
    **Instances:** 2,111  
    **Classes:** 7
    
    **Target Classes:**
    - Insufficient Weight
    - Normal Weight
    - Overweight Level I
    - Overweight Level II
    - Obesity Type I
    - Obesity Type II
    - Obesity Type III
    """)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Upload & Test", 
        "Model Performance",
        "Single Prediction",
        "About"
    ])
    
    # ============================================
    # TAB 1: UPLOAD & TEST
    # ============================================
    with tab1:
        st.header(" Test Your Model")
        
        # Download section
        st.markdown("""
        <div class="download-section">
            <h3 style="margin-top:0;"> Download Test Data</h3>
            <p style="margin-bottom:1.5rem;">Download our pre-split test dataset to evaluate model performance</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if os.path.exists('test_data.csv'):
                with open('test_data.csv', 'rb') as f:
                    st.download_button(
                        label=" Full Test Data (423 rows)",
                        data=f,
                        file_name="test_data.csv",
                        mime="text/csv",
                        help="Complete test dataset",
                        use_container_width=True
                    )
        
        with col2:
            if os.path.exists('test_data_sample.csv'):
                with open('test_data_sample.csv', 'rb') as f:
                    st.download_button(
                        label=" Sample Data (50 rows)",
                        data=f,
                        file_name="test_data_sample.csv",
                        mime="text/csv",
                        help="Quick demo sample",
                        use_container_width=True
                    )
        
        with col3:
            if os.path.exists('train_data.csv'):
                with open('train_data.csv', 'rb') as f:
                    st.download_button(
                        label=" Training Data (1688 rows)",
                        data=f,
                        file_name="train_data.csv",
                        mime="text/csv",
                        help="Training dataset",
                        use_container_width=True
                    )
        
        st.markdown("---")
        
        # Upload section
        st.subheader(" Upload Your CSV File")
        
        st.markdown("""
        <div class="info-box">
        <strong> File Requirements:</strong>
        <ul>
            <li>CSV format with all 16 feature columns</li>
            <li>Optionally include 'NObeyesdad' column for evaluation</li>
            <li>Use downloaded test data or your own dataset</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload test data for evaluation"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                st.markdown(f"""
                <div class="success-box">
                 <strong>File uploaded successfully!</strong><br>
                 Shape: {df.shape[0]} rows × {df.shape[1]} columns
                </div>
                """, unsafe_allow_html=True)
                
                # Preview
                with st.expander(" Preview Data (First 10 rows)", expanded=True):
                    st.dataframe(df.head(10), use_container_width=True)
                
                # Check target
                has_target = 'NObeyesdad' in df.columns
                
                if has_target:
                    st.info(" Target column found - Will show evaluation metrics")
                    y_true = target_encoder.transform(df['NObeyesdad'])
                    X = df.drop('NObeyesdad', axis=1)
                else:
                    st.warning(" No target column - Will only show predictions")
                    X = df.copy()
                
                # Predict
                X_processed = preprocess_data(X, label_encoders, scaler)
                model = models[selected_model]
                
                y_pred = model.predict(X_processed)
                y_prob = model.predict_proba(X_processed)
                predictions = target_encoder.inverse_transform(y_pred)
                
                # Results
                st.markdown("---")
                st.subheader(" Prediction Results")
                
                results_df = df.copy()
                results_df['Predicted_Obesity_Level'] = predictions
                results_df['Prediction_Confidence'] = y_prob.max(axis=1).round(3)
                
                st.dataframe(results_df, use_container_width=True)
                
                # Download
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label=" Download Predictions as CSV",
                    data=csv,
                    file_name=f"predictions_{selected_model.lower().replace(' ', '_')}.csv",
                    mime="text/csv"
                )
                
                # Metrics
                if has_target:
                    st.markdown("---")
                    st.subheader(" Model Performance Metrics")
                    
                    num_classes = len(target_encoder.classes_)
                    metrics = calculate_metrics(y_true, y_pred, y_prob, num_classes)
                    
                    col1, col2, col3, col4, col5, col6 = st.columns(6)
                    col1.metric("Accuracy", f"{metrics['Accuracy']:.2%}")
                    col2.metric("AUC", f"{metrics['AUC']:.3f}")
                    col3.metric("Precision", f"{metrics['Precision']:.3f}")
                    col4.metric("Recall", f"{metrics['Recall']:.3f}")
                    col5.metric("F1 Score", f"{metrics['F1']:.3f}")
                    col6.metric("MCC", f"{metrics['MCC']:.3f}")
                    
                    # Confusion Matrix
                    st.markdown("---")
                    st.subheader(" Confusion Matrix")
                    fig = plot_confusion_matrix(y_true, y_pred, target_encoder.classes_)
                    st.pyplot(fig)
                    
                    # Classification Report
                    with st.expander("Detailed Classification Report"):
                        report = classification_report(y_true, y_pred,
                                                      target_names=target_encoder.classes_,
                                                      output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df.style.background_gradient(cmap='RdYlGn'),
                                   use_container_width=True)
                
            except Exception as e:
                st.error(f" Error: {e}")
                st.info(" Please check your CSV format")
    
    # ============================================
    # TAB 2: MODEL PERFORMANCE
    # ============================================
    with tab2:
        st.header("Model Performance Comparison")
        
        st.markdown("""
        <div class="info-box">
        <strong> About These Metrics:</strong><br>
        These metrics were calculated during model training on the test dataset (423 samples, 20% of data).
        They represent each model's accuracy and performance on unseen data.
        <br><br>
        <strong>Data Split:</strong>
        <ul>
            <li>Training Set: 1,688 samples (80%)</li>
            <li>Test Set: 423 samples (20%)</li>
            <li>Method: Stratified split (maintains class distribution)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if os.path.exists('model_results.csv'):
            results_df = pd.read_csv('model_results.csv')
            
            st.subheader("Model Comparison Table")
            
            # Highlight best
            best_idx = results_df['Accuracy'].idxmax()
            
            def highlight_best(row):
                if row.name == best_idx:
                    return ['background-color: #d4edda; font-weight: bold'] * len(row)
                return [''] * len(row)
            
            styled_df = results_df.style.apply(highlight_best, axis=1)\
                                       .format({
                                           'Accuracy': '{:.2%}',
                                           'AUC': '{:.4f}',
                                           'Precision': '{:.4f}',
                                           'Recall': '{:.4f}',
                                           'F1': '{:.4f}',
                                           'MCC': '{:.4f}'
                                       })
            
            st.dataframe(styled_df, use_container_width=True)
            
            best_model = results_df.loc[best_idx, 'Model']
            best_acc = results_df.loc[best_idx, 'Accuracy']
            
            st.success(f"**Best Model:** {best_model} with {best_acc:.2%} accuracy")
            
            # Visual comparison
            st.markdown("---")
            st.subheader("Visual Performance Comparison")
            
            fig = plot_metrics_comparison(results_df)
            st.pyplot(fig)
        else:
            st.warning("Model results file not found")
    
    # ============================================
    # TAB 3: SINGLE PREDICTION
    # ============================================
    with tab3:
        st.header(" Individual Prediction")
        st.markdown("Enter patient information to predict obesity level")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Demographics")
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.slider("Age", 14, 61, 25)
            height = st.slider("Height (m)", 1.45, 1.98, 1.70, 0.01)
            weight = st.slider("Weight (kg)", 39.0, 173.0, 70.0, 0.5)
            
            st.markdown("#### Eating Habits")
            family_history = st.selectbox("Family History", ["yes", "no"])
            favc = st.selectbox("High Caloric Food", ["yes", "no"])
        
        with col2:
            st.markdown("#### Diet")
            fcvc = st.slider("Vegetable Consumption", 1.0, 3.0, 2.0, 0.1)
            ncp = st.slider("Main Meals/Day", 1.0, 4.0, 3.0, 0.1)
            caec = st.selectbox("Snacking", ["no", "Sometimes", "Frequently", "Always"])
            ch2o = st.slider("Water Intake (L)", 1.0, 3.0, 2.0, 0.1)
        
        with col3:
            st.markdown("#### 🏃 Activity & Lifestyle")
            faf = st.slider("Exercise (days/week)", 0.0, 3.0, 1.0, 0.1)
            tue = st.slider("Screen Time (hrs)", 0.0, 2.0, 1.0, 0.1)
            smoke = st.selectbox("Smoking", ["yes", "no"])
            scc = st.selectbox("Calorie Tracking", ["yes", "no"])
            calc = st.selectbox("Alcohol", ["no", "Sometimes", "Frequently", "Always"])
            mtrans = st.selectbox("Transport", ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"])
        
        if st.button("🔮 Predict Obesity Level", type="primary", use_container_width=True):
            input_data = pd.DataFrame({
                'Gender': [gender], 'Age': [age], 'Height': [height], 'Weight': [weight],
                'family_history_with_overweight': [family_history], 'FAVC': [favc],
                'FCVC': [fcvc], 'NCP': [ncp], 'CAEC': [caec], 'SMOKE': [smoke],
                'CH2O': [ch2o], 'SCC': [scc], 'FAF': [faf], 'TUE': [tue],
                'CALC': [calc], 'MTRANS': [mtrans]
            })
            
            input_processed = preprocess_data(input_data, label_encoders, scaler)
            model = models[selected_model]
            prediction = model.predict(input_processed)
            probabilities = model.predict_proba(input_processed)[0]
            predicted_class = target_encoder.inverse_transform(prediction)[0]
            
            st.markdown("---")
            
            # Color map
            color_map = {
                'Insufficient_Weight': ('🔵', '#2196F3'),
                'Normal_Weight': ('🟢', '#4caf50'),
                'Overweight_Level_I': ('🟡', '#ffeb3b'),
                'Overweight_Level_II': ('🟠', '#ff9800'),
                'Obesity_Type_I': ('🟠', '#ff5722'),
                'Obesity_Type_II': ('🔴', '#f44336'),
                'Obesity_Type_III': ('🔴', '#b71c1c')
            }
            
            emoji, color = color_map.get(predicted_class, ('⚪', '#999'))
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {color}22 0%, {color}44 100%); 
                        padding: 2rem; border-radius: 15px; text-align: center; 
                        border: 3px solid {color};">
                <h2 style="margin:0; color: {color};">{emoji} {predicted_class.replace('_', ' ')}</h2>
                <p style="margin-top: 0.5rem; font-size: 1.2rem;">
                Confidence: {probabilities.max():.1%}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # BMI
            bmi = weight / (height ** 2)
            bmi_cat = "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"
            st.info(f"📏 **BMI:** {bmi:.2f} ({bmi_cat})")
            
            # Probability chart
            st.markdown("---")
            st.subheader("Prediction Probabilities")
            
            prob_df = pd.DataFrame({
                'Level': target_encoder.classes_,
                'Probability': probabilities
            }).sort_values('Probability', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors_bar = [color if c == predicted_class else '#b0bec5' for c in prob_df['Level']]
            bars = ax.barh(prob_df['Level'], prob_df['Probability'], color=colors_bar, alpha=0.8)
            ax.set_xlabel('Probability', fontsize=12, fontweight='bold')
            ax.set_title(f'Probabilities - {selected_model}', fontsize=14, fontweight='bold')
            ax.set_xlim(0, 1)
            
            for bar, prob in zip(bars, prob_df['Probability']):
                ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                       f'{prob:.1%}', va='center', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
    
    # ============================================
    # TAB 4: ABOUT
    # ============================================
    with tab4:
        st.header("About This Application")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Project Overview
            
            This application demonstrates **6 Machine Learning models** for predicting
            obesity levels based on eating habits and physical condition.
            
            ### Objective
            
            Early identification of obesity risk enables timely intervention and
            lifestyle modifications.
            
            ### Dataset
            
            - **Name:** Obesity Levels Dataset
            - **Source:** UCI ML Repository (ID: 544)
            - **Origin:** Mexico, Peru, Colombia
            - **Instances:** 2,111
            - **Features:** 16
            - **Classes:** 7
            
            ### Models
            
            1. Logistic Regression
            2. Decision Tree
            3. K-Nearest Neighbors
            4. Naive Bayes
            5. Random Forest
            6. XGBoost
            """)
        
        with col2:
            st.markdown("""
            ### Evaluation Metrics
            
            - **Accuracy** - Overall correctness
            - **AUC** - ROC curve area
            - **Precision** - Positive predictive value
            - **Recall** - True positive rate
            - **F1 Score** - Harmonic mean
            - **MCC** - Matthews correlation
            
            ### Technologies
            
            - Python 3.x
            - Scikit-learn
            - XGBoost
            - Streamlit
            - Pandas & NumPy
            - Matplotlib & Seaborn
            
            ### Author
            
            **[S Harish Sankaranarayanan]**  
           
            
            **Assignment:** ML Assignment 2  
            **Date:** February 2026
            """)
        
        st.markdown("---")
        st.subheader("Feature Descriptions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Demographics (4):**
            - Gender, Age, Height, Weight
            
            **Eating Habits (6):**
            - Family history
            - High caloric food
            - Vegetable consumption
            - Main meals
            - Snacking
            - Water intake
            """)
        
        with col2:
            st.markdown("""
            **Physical Activity (4):**
            - Smoking
            - Calorie tracking
            - Exercise frequency
            - Screen time
            
            **Lifestyle (2):**
            - Alcohol consumption
            - Transportation
            """)


if __name__ == "__main__":
    main()
