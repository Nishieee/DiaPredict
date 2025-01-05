from flask import Flask, request, render_template
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("./models/pipeline_xgboost.pkl")

# Load the feature names
X_train = pd.read_csv("../data/processed/X_train_engineered.csv")
feature_names = X_train.columns.tolist()

# Remove unnecessary features from the feature names
if 'PatientID' in feature_names:
    feature_names.remove('PatientID')

# Categorize features into groups
demographic_features = [feature for feature in feature_names if
                        'age' in feature.lower() or 'gender' in feature.lower() or 'ethnicity' in feature.lower()]
medical_history_features = [feature for feature in feature_names if
                            'history' in feature.lower() or 'diabetes' in feature.lower()]
lifestyle_features = [feature for feature in feature_names if
                      'bmi' in feature.lower() or 'smoking' in feature.lower() or 'activity' in feature.lower()]

# Initialize the Flask application
app = Flask(__name__)


@app.route('/')
def home():
    """
    Render the home page with feature groups.
    """
    return render_template(
        'index.html',
        demographic_features=demographic_features,
        medical_history_features=medical_history_features,
        lifestyle_features=lifestyle_features
    )


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests and render the result.
    """
    try:
        # Extract form data and ensure all required features are present
        features = [float(request.form.get(feature, 0)) for feature in feature_names]
        input_data = pd.DataFrame([features], columns=feature_names)

        # Make predictions using the loaded model
        prediction = model.predict(input_data)[0]
        prediction_probability = model.predict_proba(input_data)[0][1]

        return render_template(
            'index.html',
            prediction=prediction,
            probability=round(prediction_probability * 100, 2),
            demographic_features=demographic_features,
            medical_history_features=medical_history_features,
            lifestyle_features=lifestyle_features
        )
    except KeyError as e:
        return f"Missing form data for feature: {e.args[0]}", 400
    except ValueError as e:
        return f"Invalid data format: {str(e)}", 400
    except Exception as e:
        return f"An error occurred: {str(e)}", 500


if __name__ == "__main__":
    # Run the Flask app
    app.run(debug=True)
