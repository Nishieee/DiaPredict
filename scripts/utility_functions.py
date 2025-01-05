import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
import joblib
import os


# Function to load a dataset from a file
def load_dataset(file_path):
    """
    Load a dataset from a CSV file.

    Parameters:
    - file_path: Path to the CSV file.

    Returns:
    - DataFrame containing the dataset.
    """
    df = pd.read_csv(file_path)
    return df


# Function to load training and test datasets
def fetch_train_test_data(train_features_path, test_features_path, train_labels_path, test_labels_path):
    """
    Load training and test datasets from CSV files.

    Parameters:
    - train_features_path: Path to the training features CSV.
    - test_features_path: Path to the test features CSV.
    - train_labels_path: Path to the training labels CSV.
    - test_labels_path: Path to the test labels CSV.

    Returns:
    - X_train, X_test, y_train, y_test: Training and test features and labels.
    """
    X_train = pd.read_csv(train_features_path)
    X_test = pd.read_csv(test_features_path)
    y_train = pd.read_csv(train_labels_path).values.ravel().astype('int')
    y_test = pd.read_csv(test_labels_path).values.ravel().astype('int')
    return X_train, X_test, y_train, y_test


# Function to save a trained model
def persist_model(model, model_name):
    """
    Save a trained model to the disk.

    Parameters:
    - model: Trained machine learning model.
    - model_name: Name for the saved model file.

    Outputs:
    - The model is saved as a .pkl file in the '../models' directory.
    """
    os.makedirs('../models', exist_ok=True)
    joblib.dump(model, f"../models/{model_name}.pkl")
    print(f"Model saved as ../models/{model_name}.pkl")


# Function to load a trained model
def retrieve_model(model_file_path):
    """
    Load a trained model from a file.

    Parameters:
    - model_file_path: Path to the saved model file.

    Returns:
    - Loaded machine learning model.
    """
    model = joblib.load(model_file_path)
    return model


# Function to plot and save the ROC curve
def generate_roc_curve(model, X_test, y_test, model_identifier):
    """
    Generate and save the Receiver Operating Characteristic (ROC) curve.

    Parameters:
    - model: Trained model used for prediction.
    - X_test: Test features.
    - y_test: True labels for the test set.
    - model_identifier: Name of the model for labeling purposes.

    Outputs:
    - Saves the ROC curve as a PNG file in the 'plots' directory.
    """
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = roc_auc_score(y_test, y_pred_prob)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_identifier}')
    plt.legend(loc="lower right")
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/{model_identifier}_roc_curve.png')
    plt.show()


# Function to plot and save the confusion matrix
def generate_confusion_matrix(y_test, y_pred, model_identifier):
    """
    Generate and save the confusion matrix as a heatmap.

    Parameters:
    - y_test: True labels for the test set.
    - y_pred: Predicted labels by the model.
    - model_identifier: Name of the model for labeling purposes.

    Outputs:
    - Saves the confusion matrix heatmap as a PNG file in the 'plots' directory.
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_identifier}')
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/{model_identifier}_confusion_matrix.png')
    plt.show()


# Function to evaluate a trained model
def assess_model_performance(model, X_test, y_test, model_identifier):
    """
    Evaluate a trained model using test data.

    Parameters:
    - model: Trained machine learning model.
    - X_test: Test features.
    - y_test: True labels for the test set.
    - model_identifier: Name of the model for labeling purposes.

    Outputs:
    - Prints the classification report.
    - Generates and saves the ROC curve and confusion matrix.
    """
    y_pred = model.predict(X_test)

    # Generate and save ROC Curve
    generate_roc_curve(model, X_test, y_test, model_identifier)

    # Generate and save Confusion Matrix
    generate_confusion_matrix(y_test, y_pred, model_identifier)

    # Print Classification Report
    print(f"Classification Report for {model_identifier}:\n")
    print(classification_report(y_test, y_pred))


# Function to save datasets to CSV files
def store_datasets(X_train, X_test, y_train, y_test, output_directory):
    """
    Save training and test datasets to CSV files.

    Parameters:
    - X_train, X_test, y_train, y_test: Datasets to save.
    - output_directory: Directory where the datasets will be saved.

    Outputs:
    - Saves the datasets as CSV files in the specified directory.
    """
    os.makedirs(output_directory, exist_ok=True)
    X_train.to_csv(os.path.join(output_directory, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_directory, 'X_test.csv'), index=False)
    pd.DataFrame(y_train, columns=["Target"]).to_csv(os.path.join(output_directory, 'y_train.csv'), index=False)
    pd.DataFrame(y_test, columns=["Target"]).to_csv(os.path.join(output_directory, 'y_test.csv'), index=False)
    print(f"Datasets saved to {output_directory}")
