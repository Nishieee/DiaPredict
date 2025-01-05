import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
import joblib


# Function to load the model and test datasets
def load_trained_model_and_test_data(model_file_path, test_features_path, test_labels_path):
    """
    Load the trained model and test datasets.

    Parameters:
    - model_file_path: Path to the saved model file.
    - test_features_path: Path to the CSV file containing test features.
    - test_labels_path: Path to the CSV file containing test labels.

    Returns:
    - model: Loaded machine learning model.
    - X_test: Test feature dataset as a DataFrame.
    - y_test: Test labels as a 1D array.
    """
    model = joblib.load(model_file_path)  # Load the trained model
    X_test = pd.read_csv(test_features_path)  # Load test features
    y_test = pd.read_csv(test_labels_path).values.ravel().astype('int')  # Load and process test labels
    return model, X_test, y_test


# Function to plot the ROC curve
def create_and_save_roc_curve(model, X_test, y_test, model_name):
    """
    Plot and save the Receiver Operating Characteristic (ROC) curve.

    Parameters:
    - model: Trained model used for prediction.
    - X_test: Test features.
    - y_test: True labels for the test set.
    - model_name: Name of the model for labeling purposes.
    """
    y_pred_prob = model.predict_proba(X_test)[:, 1]  # Get predicted probabilities for the positive class
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)  # Calculate FPR and TPR for the ROC curve
    roc_auc = roc_auc_score(y_test, y_pred_prob)  # Calculate the ROC AUC score

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Add a diagonal line for random guessing
    plt.xlim([0.0, 1.0])  # Set x-axis limits
    plt.ylim([0.0, 1.05])  # Set y-axis limits
    plt.xlabel('False Positive Rate')  # Label for x-axis
    plt.ylabel('True Positive Rate')  # Label for y-axis
    plt.title(f'Receiver Operating Characteristic - {model_name}')  # Add a title to the plot
    plt.legend(loc="lower right")  # Position the legend
    plt.savefig(f'plots/{model_name}_roc_curve.png')  # Save the plot to a file
    plt.show()


# Function to create and save the confusion matrix plot
def create_and_save_confusion_matrix(y_test, y_pred, model_name):
    """
    Plot and save the confusion matrix.

    Parameters:
    - y_test: True labels for the test set.
    - y_pred: Predicted labels by the model.
    - model_name: Name of the model for labeling purposes.
    """
    cm = confusion_matrix(y_test, y_pred)  # Compute the confusion matrix
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])  # Create a heatmap
    plt.xlabel('Predicted')  # Label for x-axis
    plt.ylabel('True')  # Label for y-axis
    plt.title(f'Confusion Matrix - {model_name}')  # Add a title to the plot
    plt.savefig(f'plots/{model_name}_confusion_matrix.png')  # Save the plot to a file
    plt.show()


# Function to evaluate a model and generate performance reports
def analyze_model_performance(model_file_path, test_features_path, test_labels_path, model_name):
    """
    Evaluate a trained model using test data and generate performance metrics.

    Parameters:
    - model_file_path: Path to the saved model file.
    - test_features_path: Path to the CSV file containing test features.
    - test_labels_path: Path to the CSV file containing test labels.
    - model_name: Name of the model for labeling purposes.
    """
    model, X_test, y_test = load_trained_model_and_test_data(model_file_path, test_features_path, test_labels_path)
    y_pred = model.predict(X_test)  # Predict labels for the test set

    # Generate and save the ROC Curve
    create_and_save_roc_curve(model, X_test, y_test, model_name)

    # Generate and save the Confusion Matrix
    create_and_save_confusion_matrix(y_test, y_pred, model_name)

    # Display the Classification Report
    print(f"Classification Report for {model_name}:\n")
    print(classification_report(y_test, y_pred))  # Print the classification report


# Main function to execute the evaluation process for multiple models
def execute_evaluation_pipeline():
    """
    Evaluate multiple models on the test dataset and generate evaluation metrics and visualizations.
    """
    model_file_paths = ["models/logistic_regression.pkl", "models/random_forest.pkl", "models/xgboost.pkl"]  # Paths to models
    model_names = ["Logistic Regression", "Random Forest", "XGBoost"]  # Corresponding model names
    test_features_path = "../data/processed/X_test_engineered.csv"  # Path to test features
    test_labels_path = "../data/processed/y_test.csv"  # Path to test labels

    # Evaluate each model and generate performance reports
    for model_file_path, model_name in zip(model_file_paths, model_names):
        analyze_model_performance(model_file_path, test_features_path, test_labels_path, model_name)


# Script entry point
if __name__ == "__main__":
    execute_evaluation_pipeline()
