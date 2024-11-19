import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from imblearn.over_sampling import ADASYN

def run_modeling(encoded_filepath):
    # Load encoded data
    train_data = pd.read_csv(encoded_filepath)
    
    # Define features and target
    X = train_data.drop(['Genetic_Disorder','Disorder_Subclass'], axis=1)
    y = train_data['Genetic_Disorder']
    # Define features and target

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Define base classifiers
    base_classifiers = [
        ('rf', RandomForestClassifier(random_state=42)),
        ('gb', GradientBoostingClassifier(random_state=42)),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')),
        ('svc', SVC(probability=True, random_state=42))
    ]
    
    # Define the meta-classifier
    meta_classifier = LogisticRegression(max_iter=1000, random_state=42)
    
    # Create the stacking classifier
    stacking_classifier = StackingClassifier(
        estimators=base_classifiers,
        final_estimator=meta_classifier,
        cv=5,
        n_jobs=-1
    )
    
    # Train the stacking classifier
    stacking_classifier.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = stacking_classifier.predict(X_test)
    
    # Define the mapping dictionary for target names
    disorder_mapping = {
        0: 'Mitochondrial genetic inheritance disorders',
        1: 'Multifactorial genetic inheritance disorders',
        2: 'Single-gene inheritance diseases'
    }
    
    # Evaluate the model
    print("=== Model Evaluation Before SMOTE ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=[
        disorder_mapping[0], disorder_mapping[1], disorder_mapping[2]
    ]))
    
    # Apply SMOTE for balancing the training data
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # Display class distribution after SMOTE
    print("\nClass distribution before SMOTE:\n", y_train.value_counts())
    print("Class distribution after SMOTE:\n", y_train_res.value_counts())
    
    # Retrain the stacking classifier on balanced data
    stacking_classifier.fit(X_train_res, y_train_res)
    
    # Make predictions on the test set
    y_pred_res = stacking_classifier.predict(X_test)
    
    # Evaluate the model after SMOTE
    print("\n=== Model Evaluation After SMOTE ===")
    print("Accuracy:", accuracy_score(y_test, y_pred_res))
    print(classification_report(y_test, y_pred_res, target_names=[
        disorder_mapping[0], disorder_mapping[1], disorder_mapping[2]
    ]))
    
    # Optionally, save the trained model
    # import joblib
    # joblib.dump(stacking_classifier, 'stacking_classifier.pkl')
    # ========== Apply ADASYN to the Training Data ==========
    adasyn = ADASYN(random_state=42)
    X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)

    # Verify the new class distribution after ADASYN
    print("\nClass distribution after ADASYN:")
    print(pd.Series(y_train_adasyn).value_counts())

    # ========== Model Evaluation After ADASYN ==========
    print("\n=== Model Evaluation After ADASYN ===")
    # Retrain the model on the ADASYN-resampled data

    stacking_classifier.fit(X_train_adasyn, y_train_adasyn)
    y_pred_adasyn = stacking_classifier.predict(X_test)

    # Calculate and print evaluation metrics for the ADASYN-trained model
    print("Accuracy:", accuracy_score(y_test, y_pred_adasyn))
    print(classification_report(y_test, y_pred_adasyn, target_names=[
            disorder_mapping[0], disorder_mapping[1], disorder_mapping[2]
        ]))

    return stacking_classifier

# Example usage
if __name__ == "__main__":
    run_modeling('encoded_data.csv')
