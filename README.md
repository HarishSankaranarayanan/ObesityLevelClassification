# Obesity Level Classification - Machine Learning Assignment 2

## Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset Description](#dataset-description)
- [Models Used](#models-used)
- [Model Performance Comparison](#model-performance-comparison)
- [Observations](#observations)
- [Streamlit App Features](#streamlit-app-features)

---

## Problem Statement

Obesity is a growing global health concern and a major risk factor for cardiovascular diseases (CVD), diabetes, and other chronic conditions. Early identification of obesity risk levels can enable timely intervention and lifestyle modifications.

**Objective:** Develop a multi-class classification system that predicts an individual's obesity level based on their eating habits and physical condition. The system implements and compares **6 different machine learning models** to identify the most effective approach for obesity risk classification.

**Classification Task:** Predict one of 7 obesity levels:
1. Insufficient Weight
2. Normal Weight  
3. Overweight Level I
4. Overweight Level II
5. Obesity Type I
6. Obesity Type II
7. Obesity Type III

---

## Dataset Description

| Attribute | Value |
|-----------|-------|
| **Dataset Name** | Estimation of Obesity Levels Based on Eating Habits and Physical Condition |
| **Source** | Kaggle |
| **Original Source** | Survey data from Mexico, Peru, and Colombia |
| **Total Instances** | 2,111 |
| **Number of Features** | 16 |
| **Target Variable** | NObeyesdad (Obesity Level) |
| **Number of Classes** | 7 |
| **Missing Values** | None |

### Feature Categories

**Demographic Features (4):**
| Feature | Description | Type |
|---------|-------------|------|
| Gender | Male/Female | Categorical |
| Age | Age in years (14-61) | Numerical |
| Height | Height in meters | Numerical |
| Weight | Weight in kilograms | Numerical |

**Eating Habits Features (6):**
| Feature | Description | Type |
|---------|-------------|------|
| family_history_with_overweight | Family history of overweight | Categorical (yes/no) |
| FAVC | Frequent consumption of high caloric food | Categorical (yes/no) |
| FCVC | Frequency of vegetable consumption | Numerical (1-3) |
| NCP | Number of main meals | Numerical (1-4) |
| CAEC | Consumption of food between meals | Categorical |
| CH2O | Daily water consumption | Numerical (1-3 liters) |

**Physical Condition Features (5):**
| Feature | Description | Type |
|---------|-------------|------|
| SMOKE | Smoking habit | Categorical (yes/no) |
| SCC | Calories consumption monitoring | Categorical (yes/no) |
| FAF | Physical activity frequency | Numerical (0-3 days/week) |
| TUE | Time using technology devices | Numerical (0-2 hours) |
| MTRANS | Transportation used | Categorical |

**Lifestyle Feature (1):**
| Feature | Description | Type |
|---------|-------------|------|
| CALC | Consumption of alcohol | Categorical |

### Target Class Distribution

| Obesity Level | Count | Percentage |
|---------------|-------|------------|
| Obesity Type I | 424 | 20.1% |
| Normal Weight | 410 | 19.4% |
| Obesity Type III | 375 | 17.8% |
| Obesity Type II | 265 | 12.6% |
| Overweight Level II | 230 | 10.9% |
| Insufficient Weight | 207 | 9.8% |
| Overweight Level I | 200 | 9.5% |

---

## Models Used

Six machine learning classification models were implemented:

1. **Logistic Regression** - A linear model that uses the logistic function to model binary/multi-class classification problems
2. **Decision Tree Classifier** - A tree-structured model that makes decisions based on feature thresholds
3. **K-Nearest Neighbors (KNN)** - An instance-based learning algorithm that classifies based on majority voting of nearest neighbors
4. **Naive Bayes (Gaussian)** - A probabilistic classifier based on Bayes' theorem with independence assumptions
5. **Random Forest (Ensemble)** - An ensemble of decision trees using bagging for improved accuracy
6. **XGBoost (Ensemble)** - An optimized gradient boosting algorithm with regularization

---

## Model Performance Comparison

### Evaluation Metrics Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.8180 | 0.9842 | 0.8129 | 0.8180 | 0.8080 | 0.7851 |
| Decision Tree | 0.7801 | 0.8964 | 0.7745 | 0.7801 | 0.7759 | 0.7393 |
| K-Nearest Neighbors | 0.4515 | 0.7665 | 0.4606 | 0.4515 | 0.4493 | 0.3465 |
| Naive Bayes | 0.5579 | 0.8874 | 0.5499 | 0.5579 | 0.5517 | 0.4747 |
| Random Forest (Ensemble) | 0.7636 | 0.9656 | 0.7582 | 0.7636 | 0.7477 | 0.7209 |
| XGBoost (Ensemble) | **0.8251** | **0.9784** | **0.8223** | **0.8251** | **0.8229** | **0.7926** |

** Best Performing Model: XGBoost** with 82.51% accuracy

---

## Observations

### Model Performance Observations

| ML Model Name | Observation about Model Performance |
|---------------|-------------------------------------|
| **Logistic Regression** | Achieved second-best accuracy (81.80%) with excellent AUC score (0.9842). Performs exceptionally well on this dataset due to the relatively linear separability of classes when considering weight and height features. Shows strong performance across all classes except for Overweight Levels, indicating some difficulty distinguishing between borderline obesity categories. The high AUC suggests excellent probability calibration. |
| **Decision Tree** | Achieved good accuracy (78.01%) with interpretable results. The model captures non-linear relationships but tends to overfit certain patterns. Shows confusion primarily between adjacent obesity categories (Overweight Level I/II and Obesity Type I/II). The lower AUC (0.8964) compared to other models indicates less reliable probability estimates. Suitable for scenarios requiring model interpretability. |
| **K-Nearest Neighbors** | Performed poorly (45.15% accuracy) on this dataset. The high-dimensional feature space and mixed categorical/numerical features negatively impact distance-based calculations. KNN struggles with class imbalance and the varying scales of features. The algorithm's sensitivity to irrelevant features and the curse of dimensionality are evident. Not recommended for this classification task. |
| **Naive Bayes** | Moderate performance (55.79%) with relatively good AUC (0.8874). The independence assumption is violated as features like weight, height, and eating habits are correlated. Performs better for extreme classes (Insufficient Weight, Obesity Type III) but struggles with intermediate categories. The probabilistic nature provides decent class probability estimates despite lower accuracy. |
| **Random Forest (Ensemble)** | Good accuracy (76.36%) with excellent AUC (0.9656). The ensemble approach reduces overfitting compared to single decision trees. Shows consistent performance across most classes but has difficulty with Overweight categories. Feature importance analysis reveals Weight, Height, and Age as top predictors. The model benefits from bagging but may need hyperparameter tuning for optimal results. |
| **XGBoost (Ensemble)** | **Best overall performance** with 82.51% accuracy and balanced metrics. The gradient boosting approach effectively handles feature interactions and class imbalance. Shows superior performance in distinguishing between all obesity levels, including the challenging Overweight categories. The regularization prevents overfitting while maintaining high predictive power. Recommended as the production model for this classification task. |

### Key Insights

1. **Ensemble Methods Excel:** Both Random Forest and XGBoost outperform individual classifiers, demonstrating the power of ensemble learning for this multi-class problem.

2. **Linear vs Non-linear:** Logistic Regression surprisingly performs well, suggesting underlying linear relationships in the data when features are properly scaled.

3. **Distance-based Challenges:** KNN's poor performance highlights the challenges of distance metrics with mixed feature types and multiple classes.

4. **Class Boundary Difficulty:** All models show some confusion between adjacent obesity categories (e.g., Overweight Level I vs II), indicating subtle distinctions that are challenging to capture.

5. **Feature Importance:** Weight, Height, Age, and family history with overweight emerge as the most predictive features across all models.


## Author

**S Harish Sankaranarayanan**  
**2025AA05227**  

