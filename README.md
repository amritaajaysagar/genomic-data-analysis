# Genomic-Data-Analysis

## Project Overview
This project aims to predict and classify genetic disorders using machine learning techniques. Leveraging a dataset sourced from [Kaggle](https://www.kaggle.com/datasets/aryarishabh/of-genomes-and-genetics-hackerearth-ml-challenge), the project implements advanced preprocessing, feature selection, and model evaluation strategies. Key challenges, such as class imbalance, are tackled using techniques like Synthetic Minority Oversampling Technique (SMOTE) and class-weight adjustments to enhance predictive performance.

## Objectives
The primary objectives of this project are:
1. **Dataset Exploration**: Conduct exploratory data analysis to understand the structure, distribution, and patterns in the dataset.
2. **Feature Selection**: Utilize methods such as Random Forest feature importance to identify the most significant predictors for genetic disorders.
3. **Model Development**: Build and evaluate stacked ensemble and multi-classifier machine learning models to predict genetic disorders.
4. **Addressing Class Imbalance**: Apply targeted oversampling using SMOTE and class-weight adjustments to improve the performance on minority classes.
5. **Evaluation**: Analyze the performance of individual classifiers and ensemble models using metrics such as accuracy, precision, recall, and F1-score.
6. **Insights and Recommendations**: Derive actionable insights from the results to improve model robustness and applicability.

## Dataset
- **Source**: [Kaggle - Of Genomes and Genetics](https://www.kaggle.com/datasets/aryarishabh/of-genomes-and-genetics-hackerearth-ml-challenge)
- **Description**: The dataset includes genetic markers associated with different genetic disorders. The primary challenge is the imbalance across classes, which significantly impacts predictive accuracy for minority classes.

## Techniques and Technologies
### Preprocessing
- Median imputation for numerical features and mode imputation for categorical features.
- One-hot and label encoding for categorical variables.
- Targeted oversampling with SMOTE to balance the dataset.

### Feature Selection
- Random Forest feature importance to select key predictors.
- Comparison with mutual information and lasso regularization for validation.

### Modeling
- Ensemble Stacked Classifier comprising:
  - Random Forest
  - Gradient Boosting
  - XGBoost (with scale_pos_weight adjustment)
  - Support Vector Classifier (with class-weight adjustments)
- Logistic Regression as the meta-classifier.

### Evaluation Metrics
- Accuracy, precision, recall, and F1-score (class-wise, macro-average, and weighted-average).

### Tools and Libraries
- **Programming Language**: Python
- **Libraries**: 
  - Scikit-learn
  - XGBoost
  - Imbalanced-learn
  - Pandas, NumPy
  - Matplotlib, Seaborn

## Current Status
The project achieved a maximum accuracy of **72%** using a stacked ensemble model with targeted oversampling and class-weight adjustments. While performance improved across all classes, challenges persist in predicting minority class instances, prompting ongoing exploration of advanced strategies to enhance model performance further.
