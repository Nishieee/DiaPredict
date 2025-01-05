import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report, accuracy_score, \
    precision_score, recall_score, f1_score
import seaborn as sns
import joblib
import os
from utility_functions import load_model, load_train_test_data, plot_roc_curve, plot_confusion_matrix

# Function to load the machine learning model and test datasets
def load_model_and_data(model_path, X_test_path, y_test_path):
    """
    Load the trained model and test datasets.
    
    Parameters:
    - model_path: Path to the saved model file.
    - X_test_path: Path to the CSV file containing test features.
    - y_test_path: Path to the CSV file containing test labels.

    Returns:
    - model: Loaded machine learning model.
    - X_test: Test feature dataset as a DataFrame.
    - y_test: Test labels as a 1D array.
    """
    model = load_model(model_path)  # Load the saved model
    X_test = pd.read_csv(X_test_path)  # Load test feature data
    y_test = pd.read_csv(y_test_path).values.ravel().astype('int')  # Load and process test labels
    return model, X_test, y_test

# Function to plot the ROC curve and save it to a file
def plot_and_save_roc_curve(model, X_test, y_test, model_name):
    """
    Plot and save the ROC curve for the given model and dataset.

    Parameters:
    - model: Trained model to generate predictions.
    - X_test: Test features.
    - y_test: True labels for the test set.
    - model_name: Name of the model for labeling purposes.

    Returns:
    - roc_auc: Area under the ROC curve (AUC) score.
    """
    y_pred_prob = model.predict_proba(X_test)[:, 1]  # Get predicted probabilities for positive class
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)  # Calculate false positive and true positive rates
    roc_auc = roc_auc_score(y_test, y_pred_prob)  # Calculate the ROC AUC score

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line for random guessing
    plt.xlim([0.0, 1.0])  # Set x-axis limits
    plt.ylim([0.0, 1.05])  # Set y-axis limits
    plt.xlabel('False Positive Rate')  # Label for x-axis
    plt.ylabel('True Positive Rate')  # Label for y-axis
    plt.title(f'Receiver Operating Characteristic - {model_name}')  # Add a title to the plot
    plt.legend(loc="lower right")  # Position the legend
    os.makedirs('plots', exist_ok=True)  # Create the 'plots' directory if it doesn't exist
    plt.savefig(f'plots/{model_name}_roc_curve.png')  # Save the plot to a file
    plt.show()
    return roc_auc

# Function to plot and save the confusion matrix
def plot_and_save_confusion_matrix(y_test, y_pred, model_name):
    """
    Plot and save the confusion matrix for the predictions.

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
    os.makedirs('plots', exist_ok=True)  # Create the 'plots' directory if it doesn't exist
    plt.savefig(f'plots/{model_name}_confusion_matrix.png')  # Save the plot to a file
    plt.show()

# Function to create a comprehensive model evaluation report
def comprehensive_model_report(model_path, X_test_path, y_test_path, model_name):
    """
    Generate and save a comprehensive evaluation report for a model.

    Parameters:
    - model_path: Path to the saved model file.
    - X_test_path: Path to the CSV file containing test features.
    - y_test_path: Path to the CSV file containing test labels.
    - model_name: Name of the model for labeling purposes.
    """
    model, X_test, y_test = load_model_and_data(model_path, X_test_path, y_test_path)  # Load model and data
    y_pred = model.predict(X_test)  # Predict labels for the test set

    # Plot the ROC Curve and calculate AUC score
    roc_auc = plot_and_save_roc_curve(model, X_test, y_test, model_name)

    # Plot and save the confusion matrix
    plot_and_save_confusion_matrix(y_test, y_pred, model_name)

    # Generate and print a classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    print(f"Classification Report for {model_name}:\n")
    print(classification_report(y_test, y_pred))

    # Calculate and store additional metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }

    # Save the results to a CSV file
    report_path = f'reports/{model_name}_report.csv'
    os.makedirs('reports', exist_ok=True)  # Create the 'reports' directory if it doesn't exist
    report_df = pd.DataFrame(class_report).transpose()
    report_df['accuracy'] = accuracy  # Add accuracy to the report
    report_df['roc_auc'] = roc_auc  # Add ROC AUC score to the report
    report_df.to_csv(report_path)  # Save the report as a CSV file

    print(f"Comprehensive report saved as {report_path}")  # Inform the user about the saved report
    print(f"Metrics: {metrics}")  # Display calculated metrics

# Main function to execute the evaluation process
def main():
    """
    Main function to evaluate multiple models on the test dataset.
    """
    model_paths = ["models/logistic_regression.pkl", "models/random_forest.pkl", "models/xgboost.pkl"]  # List of model paths
    model_names = ["Logistic Regression", "Random Forest", "XGBoost"]  # List of model names
    X_test_path = "../data/processed/X_test_engineered.csv"  # Path to test features
    y_test_path = "../data/processed/y_test.csv"  # Path to test labels

    # Loop through each model and generate reports
    for model_path, model_name in zip(model_paths, model_names):
        comprehensive_model_report(model_path, X_test_path, y_test_path, model_name)

# Entry point for the script
if __name__ == "__main__":
    main()
