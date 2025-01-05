import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


# Function to create additional features
def generate_custom_features(df):
    """
    Create new features from the existing dataset.

    Example:
    - Add a feature that represents the product of 'Age' and 'BMI'.

    Parameters:
    - df: DataFrame containing the dataset.

    Returns:
    - DataFrame with newly created features added.
    """
    df['Age_BMI'] = df['Age'] * df['BMI']  # Create a new feature: product of Age and BMI
    return df


# Function to add polynomial features
def generate_polynomial_features(df, degree=2):
    """
    Generate polynomial features for numerical columns in the dataset.

    Parameters:
    - df: DataFrame containing the dataset.
    - degree: The degree of polynomial features to generate (default is 2).

    Returns:
    - DataFrame with polynomial features added.
    """
    poly = PolynomialFeatures(degree, include_bias=False)  # Initialize PolynomialFeatures
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns  # Select numeric columns
    poly_features = poly.fit_transform(df[numeric_columns])  # Generate polynomial features
    poly_feature_names = poly.get_feature_names_out(numeric_columns)  # Get names of the polynomial features
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)  # Create a DataFrame for the polynomial features

    # Reset index to ensure proper concatenation
    df = df.reset_index(drop=True)
    poly_df = poly_df.reset_index(drop=True)

    # Concatenate original data with the polynomial features
    df = pd.concat([df, poly_df], axis=1)
    return df


# Main function for feature engineering
def perform_feature_engineering(train_file_path, test_file_path, train_output_path, test_output_path, degree=2):
    """
    Perform feature engineering for training and test datasets.

    Includes:
    - Adding custom features.
    - Generating polynomial features.
    - Saving the processed datasets to specified file paths.

    Parameters:
    - train_file_path: Path to the training dataset.
    - test_file_path: Path to the test dataset.
    - train_output_path: Path to save the processed training dataset.
    - test_output_path: Path to save the processed test dataset.
    - degree: Degree of polynomial features to generate (default is 2).
    """
    # Process training dataset
    df_train = pd.read_csv(train_file_path)  # Load training dataset
    df_train = generate_custom_features(df_train)  # Add custom features
    df_train = generate_polynomial_features(df_train, degree)  # Add polynomial features
    df_train.to_csv(train_output_path, index=False)  # Save the processed training dataset
    print(f"Feature engineering for training data completed and saved to {train_output_path}.")

    # Process test dataset
    df_test = pd.read_csv(test_file_path)  # Load test dataset
    df_test = generate_custom_features(df_test)  # Add custom features
    df_test = generate_polynomial_features(df_test, degree)  # Add polynomial features
    df_test.to_csv(test_output_path, index=False)  # Save the processed test dataset
    print(f"Feature engineering for test data completed and saved to {test_output_path}.")


# Script entry point
if __name__ == "__main__":
    train_file_path = "../data/processed/X_train.csv"  # Path to the training dataset
    test_file_path = "../data/processed/X_test.csv"  # Path to the test dataset
    train_output_path = "../data/processed/X_train_engineered.csv"  # Path to save processed training dataset
    test_output_path = "../data/processed/X_test_engineered.csv"  # Path to save processed test dataset
    degree = 2  # Polynomial degree for feature generation
    perform_feature_engineering(train_file_path, test_file_path, train_output_path, test_output_path, degree)
