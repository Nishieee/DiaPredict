import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import os


# Function to load the datasets
def load_training_and_test_data(train_features_path, test_features_path, train_labels_path, test_labels_path):
    """
    Load the training and test datasets from CSV files.

    Parameters:
    - train_features_path: Path to the training features CSV.
    - test_features_path: Path to the test features CSV.
    - train_labels_path: Path to the training labels CSV.
    - test_labels_path: Path to the test labels CSV.

    Returns:
    - X_train: Training features as a DataFrame.
    - X_test: Test features as a DataFrame.
    - y_train: Training labels as a 1D array.
    - y_test: Test labels as a 1D array.
    """
    X_train = pd.read_csv(train_features_path)  # Load training features
    X_test = pd.read_csv(test_features_path)  # Load test features
    y_train = pd.read_csv(train_labels_path).values.ravel().astype('int')  # Convert training labels to integers
    y_test = pd.read_csv(test_labels_path).values.ravel().astype('int')  # Convert test labels to integers

    return X_train, X_test, y_train, y_test


# Function to train and evaluate a model
def fit_and_assess_model(X_train, X_test, y_train, y_test, model, model_identifier):
    """
    Train a model and evaluate its performance on the test dataset.

    Parameters:
    - X_train: Training features.
    - X_test: Test features.
    - y_train: Training labels.
    - y_test: Test labels.
    - model: Machine learning model to be trained and evaluated.
    - model_identifier: A name for the model used for saving and reporting.

    Outputs:
    - Prints evaluation metrics and saves the trained model.
    """
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    cm = confusion_matrix(y_test, y_pred)

    # Print evaluation results
    print(f"Model: {model_identifier}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\n")

    # Save the trained model
    os.makedirs('../models', exist_ok=True)  # Ensure the models directory exists
    joblib.dump(model, f"../models/{model_identifier}.pkl")  # Save the model
    print(f"Model saved as ../models/{model_identifier}.pkl")


# Main function to train and evaluate multiple models
def execute_model_training_pipeline(train_features_path, test_features_path, train_labels_path, test_labels_path):
    """
    Train and evaluate multiple models on the dataset.

    Parameters:
    - train_features_path: Path to the training features CSV.
    - test_features_path: Path to the test features CSV.
    - train_labels_path: Path to the training labels CSV.
    - test_labels_path: Path to the test labels CSV.
    """
    # Load the datasets
    X_train, X_test, y_train, y_test = load_training_and_test_data(
        train_features_path, test_features_path, train_labels_path, test_labels_path
    )

    # Logistic Regression
    logistic_regression = LogisticRegression(max_iter=1000)
    fit_and_assess_model(X_train, X_test, y_train, y_test, logistic_regression, "logistic_regression")

    # Random Forest
    random_forest = RandomForestClassifier(n_estimators=100)
    fit_and_assess_model(X_train, X_test, y_train, y_test, random_forest, "random_forest")

    # XGBoost
    xgboost = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    fit_and_assess_model(X_train, X_test, y_train, y_test, xgboost, "xgboost")


# Script entry point
if __name__ == "__main__":
    train_features_path = "../data/processed/X_train_engineered.csv"  # Path to training features
    test_features_path = "../data/processed/X_test_engineered.csv"  # Path to test features
    train_labels_path = "../data/processed/y_train.csv"  # Path to training labels
    test_labels_path = "../data/processed/y_test.csv"  # Path to test labels
    execute_model_training_pipeline(train_features_path, test_features_path, train_labels_path, test_labels_path)
