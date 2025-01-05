import pytest
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import os


# Function to load datasets
def load_datasets(train_features_path, test_features_path):
    """
    Load training and test datasets.

    Parameters:
    - train_features_path: Path to the training features CSV file.
    - test_features_path: Path to the test features CSV file.

    Returns:
    - X_train: Training features as a DataFrame.
    - X_test: Test features as a DataFrame.
    - y_train: Training labels as a Series.
    - y_test: Test labels as a Series.
    """
    X_train = pd.read_csv(train_features_path)
    X_test = pd.read_csv(test_features_path)
    y_train = pd.read_csv(train_features_path.replace('X_', 'y_')).values.ravel().astype('int')
    y_test = pd.read_csv(test_features_path.replace('X_', 'y_')).values.ravel().astype('int')
    return X_train, X_test, y_train, y_test


# Function to train and evaluate a model
def fit_and_evaluate_model(X_train, X_test, y_train, y_test, model, model_name):
    """
    Train a model and evaluate its performance on the test set.

    Parameters:
    - X_train: Training features.
    - X_test: Test features.
    - y_train: Training labels.
    - y_test: Test labels.
    - model: Machine learning model to be trained and evaluated.
    - model_name: Name for the model used for saving and reporting.

    Outputs:
    - Prints evaluation metrics and saves the trained model.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    cm = confusion_matrix(y_test, y_pred)

    # Print results
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\n")

    # Save the model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, f"models/{model_name}.pkl")
    print(f"Model saved as models/{model_name}.pkl")


# Pytest fixture for sample data
@pytest.fixture
def mock_training_data():
    """
    Generate sample training data for testing.

    Returns:
    - X_train: Training features as a DataFrame.
    - y_train: Training labels as a Series.
    """
    X_train = pd.DataFrame({
        'Age': [25, 35, 45, 55],
        'BMI': [22.5, 24.5, 28.0, 30.0]
    })
    y_train = pd.Series([0, 1, 0, 1])
    return X_train, y_train


# Test for fit_and_evaluate_model
def test_fit_and_evaluate_model(mock_training_data):
    """
    Test the fit_and_evaluate_model function.
    """
    X_train, y_train = mock_training_data
    X_test, y_test = X_train.copy(), y_train.copy()

    # Test with Logistic Regression
    lr = LogisticRegression()
    fit_and_evaluate_model(X_train, X_test, y_train, y_test, lr, "logistic_regression")

    # Test with Random Forest
    rf = RandomForestClassifier(n_estimators=10)
    fit_and_evaluate_model(X_train, X_test, y_train, y_test, rf, "random_forest")

    # Test with XGBoost
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    fit_and_evaluate_model(X_train, X_test, y_train, y_test, xgb, "xgboost")

    # Check if models are saved
    assert os.path.exists('models/logistic_regression.pkl')
    assert os.path.exists('models/random_forest.pkl')
    assert os.path.exists('models/xgboost.pkl')


# Test for load_datasets
def test_load_datasets(tmpdir):
    """
    Test the load_datasets function.
    """
    # Create temporary files for data
    train_file = tmpdir.join("X_train.csv")
    test_file = tmpdir.join("X_test.csv")
    train_file.write("Age,BMI\n25,22.5\n35,24.5\n45,28.0\n55,30.0")
    test_file.write("Age,BMI\n25,22.5\n35,24.5\n45,28.0\n55,30.0")

    y_train_file = tmpdir.join("y_train.csv")
    y_test_file = tmpdir.join("y_test.csv")
    y_train_file.write("Diagnosis\n0\n1\n0\n1")
    y_test_file.write("Diagnosis\n0\n1\n0\n1")

    # Load the datasets
    X_train, X_test, y_train, y_test = load_datasets(str(train_file), str(test_file))

    # Verify the datasets are loaded correctly
    assert X_train.shape == (4, 2)
    assert X_test.shape == (4, 2)
    assert y_train.shape == (4,)
    assert y_test.shape == (4,)
