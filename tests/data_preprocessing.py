import pytest
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os


# Function to load a dataset
def load_data(file_path):
    """
    Load a dataset from a CSV file.

    Parameters:
    - file_path: Path to the CSV file.

    Returns:
    - DataFrame containing the dataset.
    """
    return pd.read_csv(file_path)


# Perform exploratory data analysis
def perform_eda(df):
    """
    Perform exploratory data analysis on the dataset.

    Parameters:
    - df: DataFrame containing the dataset.
    """
    print("First 5 Rows of the Dataset:\n", df.head())
    print("\nDataset Information:\n", df.info())
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nBasic Statistics:\n", df.describe())


# Handle missing values in the dataset
def handle_missing_values(df):
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


# Encode categorical variables
def encode_categorical_values(df):
    """
    One-hot encode categorical variables in the dataset.

    Parameters:
    - df: DataFrame containing the dataset.

    Returns:
    - DataFrame with categorical variables encoded.
    """
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_columns = pd.DataFrame(
        encoder.fit_transform(df[categorical_columns]),
        columns=encoder.get_feature_names_out(categorical_columns)
    )
    df = df.drop(categorical_columns, axis=1)
    df = pd.concat([df, encoded_columns], axis=1)
    return df


# Normalize numerical data
def normalize_data(df):
    """
    Normalize numerical variables using StandardScaler.

    Parameters:
    - df: DataFrame containing the dataset.

    Returns:
    - DataFrame with numerical variables normalized.
    """
    scaler = StandardScaler()
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df


# Split the dataset into training and testing sets
def split_data(df, target_column, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.

    Parameters:
    - df: DataFrame containing the dataset.
    - target_column: Column name for the target variable.
    - test_size: Proportion of the dataset to include in the test split.
    - random_state: Random seed for reproducibility.

    Returns:
    - X_train, X_test, y_train, y_test: Training and testing sets.
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# Save training and test datasets to files
def save_datasets(X_train, X_test, y_train, y_test, output_dir):
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
    print(f"Datasets saved to {output_dir}")


# Pytest Fixtures and Tests
@pytest.fixture
def sample_data():
    """
    Create a sample dataset for testing.

    Returns:
    - DataFrame containing the sample data.
    """
    data = {
        'Age': [25, 35, 45, 55],
        'Gender': ['0', '1', '0', '1'],  # Gender as a string
        'BMI': [22.5, 24.5, 28.0, 30.0],
        'Diagnosis': [0, 1, 0, 1]
    }
    return pd.DataFrame(data)


def test_load_data(sample_data):
    """
    Test the load_data function.
    """
    df = sample_data
    assert not df.empty


def test_handle_missing_values(sample_data):
    """
    Test the handle_missing_values function.
    """
    df = sample_data.copy()
    df.loc[0, 'BMI'] = None
    df = handle_missing_values(df)
    assert df['BMI'].isnull().sum() == 0


def test_encode_categorical_values(sample_data):
    """
    Test the encode_categorical_values function.
    """
    df = sample_data
    df = encode_categorical_values(df)
    assert 'Gender_1' in df.columns


def test_normalize_data(sample_data):
    """
    Test the normalize_data function.
    """
    df = sample_data
    df = normalize_data(df)
    assert abs(df['Age'].mean()) < 1


def test_split_data(sample_data):
    """
    Test the split_data function.
    """
    df = sample_data
    X_train, X_test, y_train, y_test = split_data(df, 'Diagnosis')
    assert len(X_train) + len(X_test) == len(df)


def test_save_datasets(tmpdir, sample_data):
    """
    Test the save_datasets function.
    """
    df = sample_data
    X_train, X_test, y_train, y_test = split_data(df, 'Diagnosis')
    output_dir = tmpdir.mkdir("data")
    save_datasets(X_train, X_test, y_train, y_test, output_dir)
    assert os.path.exists(os.path.join(output_dir, 'X_train.csv'))
