import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import os


# Function to load the dataset
def read_dataset(file_path):
    """
    Load a dataset from a CSV file.

    Parameters:
    - file_path: Path to the CSV file.

    Returns:
    - DataFrame containing the dataset.
    """
    df = pd.read_csv(file_path)
    return df


# Function to explore and analyze the data
def explore_data(df, dataset_label=""):
    """
    Perform exploratory data analysis (EDA) on the dataset.

    Parameters:
    - df: DataFrame containing the dataset.
    - dataset_label: A label to identify the dataset in the output.
    """
    print(f"{dataset_label} Dataset First 5 Rows:\n", df.head())
    print(f"\n{dataset_label} Dataset Information:\n", df.info())
    print(f"\n{dataset_label} Missing Values:\n", df.isnull().sum())
    print(f"\n{dataset_label} Basic Statistics:\n", df.describe())

    # Plot distributions of categorical variables
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    for column in categorical_columns:
        plt.figure(figsize=(10, 5))
        sns.countplot(data=df, x=column)
        plt.title(f'{dataset_label} {column} Distribution')
        plt.show()

    # Plot distributions of numerical variables
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns].hist(bins=15, figsize=(20, 15))
    plt.suptitle(f'{dataset_label} Numerical Variables Distribution')
    plt.show()

    # Correlation heatmap (excluding irrelevant columns if any)
    if 'DoctorInCharge' in df.columns:
        df_corr = df.drop(columns=['DoctorInCharge'])
    else:
        df_corr = df.copy()
    plt.figure(figsize=(15, 10))
    sns.heatmap(df_corr.corr(), annot=True, cmap='coolwarm')
    plt.title(f'{dataset_label} Correlation Matrix')
    plt.show()


# Function to handle missing data
def process_missing_values(df):
    """
    Fill missing values in numerical columns using the mean strategy.

    Parameters:
    - df: DataFrame containing the dataset.

    Returns:
    - DataFrame with missing values handled.
    """
    imputer = SimpleImputer(strategy='mean')
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
    return df


# Function to encode categorical variables
def transform_categorical_data(df):
    """
    Convert categorical variables into one-hot encoded features.

    Parameters:
    - df: DataFrame containing the dataset.

    Returns:
    - DataFrame with categorical variables encoded.
    """
    categorical_columns = df.select_dtypes(include=['object']).columns
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_columns = pd.DataFrame(encoder.fit_transform(df[categorical_columns]))
    encoded_columns.columns = encoder.get_feature_names_out(categorical_columns)
    df = df.drop(categorical_columns, axis=1)
    df = pd.concat([df, encoded_columns], axis=1)
    return df


# Function to scale numerical data
def scale_numeric_data(df):
    """
    Normalize numerical variables using StandardScaler.

    Parameters:
    - df: DataFrame containing the dataset.

    Returns:
    - DataFrame with numerical variables scaled.
    """
    scaler = StandardScaler()
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df


# Function to split data into training and testing sets
def partition_data(df, target_column, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.

    Parameters:
    - df: DataFrame containing the dataset.
    - target_column: Column name for the target variable.
    - test_size: Proportion of the dataset to include in the test split.
    - random_state: Random seed for reproducibility.

    Returns:
    - X_train, X_test, y_train, y_test: Training and testing datasets.
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


# Function to save training and testing datasets
def store_datasets(X_train, X_test, y_train, y_test, output_dir):
    """
    Save training and testing datasets to CSV files.

    Parameters:
    - X_train, X_test, y_train, y_test: Datasets to save.
    - output_dir: Directory where the datasets will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    print(f"Datasets successfully saved to {output_dir}")


# Main processing function
def process_data(file_path, target_column, output_dir):
    """
    Perform data preprocessing, including loading, cleaning, encoding, scaling, and splitting.

    Parameters:
    - file_path: Path to the raw dataset.
    - target_column: Column name for the target variable.
    - output_dir: Directory where the processed datasets will be saved.

    Returns:
    - X_train, X_test, y_train, y_test: Training and testing datasets.
    """
    df = read_dataset(file_path)  # Load the dataset
    explore_data(df, dataset_label="Original Dataset")  # Perform EDA
    df = process_missing_values(df)  # Handle missing values
    df = transform_categorical_data(df)  # Encode categorical variables
    df = scale_numeric_data(df)  # Scale numerical data
    X_train, X_test, y_train, y_test = partition_data(df, target_column)  # Split the data
    store_datasets(X_train, X_test, y_train, y_test, output_dir)  # Save datasets

    # Perform EDA on training and testing datasets
    explore_data(pd.concat([X_train, y_train], axis=1), dataset_label="Training Set")
    explore_data(pd.concat([X_test, y_test], axis=1), dataset_label="Testing Set")

    return X_train, X_test, y_train, y_test


# Script entry point
if __name__ == "__main__":
    file_path = "../data/raw/diabetes_data.csv"  # Path to the dataset
    target_column = "Diagnosis"  # Target variable
    output_dir = "../data/processed"  # Directory for saving processed data
    X_train, X_test, y_train, y_test = process_data(file_path, target_column, output_dir)
    print("Data preprocessing complete, and datasets are ready for use.")
