import pytest
import pandas as pd
import numpy as np


# Function to generate combined features
def generate_combined_features(df):
    """
    Create a new feature by combining 'Age' and 'BMI'.

    Parameters:
    - df: DataFrame containing the dataset.

    Returns:
    - DataFrame with a new 'Age_BMI' feature added.
    """
    df['Age_BMI'] = df['Age'] * df['BMI']
    return df


# Function to augment the dataset with polynomial features
def generate_polynomial_features(df, degree=2):
    """
    Add polynomial features to numerical columns in the dataset.

    Parameters:
    - df: DataFrame containing the dataset.
    - degree: The degree of polynomial features to generate.

    Returns:
    - DataFrame with polynomial features added.
    """
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        for power in range(2, degree + 1):
            df[f'{column}^{power}'] = np.power(df[column], power)
    return df


# Pytest fixture for sample data
@pytest.fixture
def mock_data():
    """
    Generate sample data for testing.

    Returns:
    - DataFrame containing sample data.
    """
    data = {
        'Age': [25, 35, 45, 55],
        'BMI': [22.5, 24.5, 28.0, 30.0]
    }
    return pd.DataFrame(data)


# Test for generate_combined_features
def test_generate_combined_features(mock_data):
    """
    Test the generate_combined_features function.
    """
    df = mock_data
    df = generate_combined_features(df)
    assert 'Age_BMI' in df.columns
    assert (df['Age_BMI'] == df['Age'] * df['BMI']).all()


# Test for generate_polynomial_features
def test_generate_polynomial_features(mock_data):
    """
    Test the generate_polynomial_features function.
    """
    df = mock_data
    df = generate_polynomial_features(df, degree=2)
    assert 'Age^2' in df.columns
    assert (df['Age^2'] == df['Age'] ** 2).all()
    assert 'BMI^2' in df.columns
    assert (df['BMI^2'] == df['BMI'] ** 2).all()
