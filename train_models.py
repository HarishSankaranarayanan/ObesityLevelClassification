"""
Obesity Level Classification - ML Model Training
================================================

Dataset: Estimation of Obesity Levels Based on Eating Habits and Physical Condition
Source: Kaggle
Features: 16 | Instances: 2111 | Classes: 7

Name: Harish Sankaranarayanan S
Date: February 2026
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                            recall_score, f1_score, matthews_corrcoef)

# ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import os

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*60)
print("OBESITY CLASSIFICATION - MODEL TRAINING")
print("="*60)

# 1. Load Data
print("\n1. Loading dataset...")
df = pd.read_csv('obesity_data.csv')
print(f"   Dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# 2. Preprocess
print("\n2. Preprocessing data...")
X = df.drop('NObeyesdad', axis=1)
y = df['NObeyesdad']

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Encode categorical features
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

# 3. Split data
print("\n3. Splitting data (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
)

# Scale features
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# 4. Define models
print("\n4. Training models...")
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=10),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5, weights='distance'),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, max_depth=15),
    'XGBoost': GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE, max_depth=6)
}

# 5. Train and save
os.makedirs('saved_models', exist_ok=True)
results = []

for name, model in models.items():
    print(f"\n   Training: {name}")
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
    except:
        auc = 0.0
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'AUC': auc,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'MCC': mcc
    })
    
    # Save as .pkl
    filename = name.lower().replace(' ', '_').replace('-', '_') + '.pkl'
    filepath = os.path.join('saved_models', filename)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved: {filename}")

# 6. Save preprocessors as .pkl
print("\n5. Saving preprocessors...")
with open('saved_models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Saved: scaler.pkl")

with open('saved_models/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print("Saved: label_encoders.pkl")

with open('saved_models/target_encoder.pkl', 'wb') as f:
    pickle.dump(target_encoder, f)
print("Saved: target_encoder.pkl")

# 7. Save results
results_df = pd.DataFrame(results)
results_df = results_df[['Model', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']]
results_df.to_csv('model_results.csv', index=False)

print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print("\n" + results_df.to_string(index=False))

best_model = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
best_accuracy = results_df['Accuracy'].max()

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"\nBest Model: {best_model}")
print(f"   Accuracy: {best_accuracy:.4f}")
print(f"\nAll models saved as .pkl files in saved_models/")
print("Ready to deploy on Streamlit!")
