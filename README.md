# 🔬 Diabetes Analysis & Prediction 🚀

## About the Project 🎯

The **Diabetes Risk Analysis & Prediction** project is a comprehensive end-to-end pipeline for identifying individuals at risk of diabetes using machine learning. This repository offers everything from data preprocessing and exploratory analysis to advanced model training and evaluation.

Whether you're a beginner or an expert, this project demonstrates best practices for handling health data in the context of predictive analytics.

---

## Key Features 🔑
- **🛠 Data Preparation**: Cleaning, imputing missing values, and normalizing datasets.
- **🔍 Exploratory Analysis**: Insightful visualizations to understand patterns and trends.
- **🧬 Feature Engineering**: Generating new features and creating polynomial transformations.
- **🤖 Model Development**: Training models including Logistic Regression, Random Forest, and XGBoost.
- **📊 Evaluation Metrics**: Assessing model performance using ROC AUC, F1 Score, and more.

---

## Project Workflow 🗂

1. **Data Preprocessing**
   - Clean raw health data and handle missing values.
2. **Exploratory Data Analysis (EDA)**
   - Analyze patterns, correlations, and distributions.
3. **Feature Engineering**
   - Create additional features to enhance prediction power.
4. **Model Training**
   - Build models and tune parameters.
5. **Evaluation**
   - Compare models using comprehensive metrics.
6. **Reporting**
   - Generate visual and textual summaries.

---

## Installation & Setup 🛠

### Step 1: Clone the Repository
```bash
git clone https://github.com/Nishieee/DiaPredict.git
cd DiaPredict
```

### Step 2: Create a Virtual Environment
```bash
python -m venv env
source env/bin/activate  # On Windows: `env\Scripts\activate`
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Pipeline
- **Data Preprocessing:**
  ```bash
  python scripts/data_preprocessing.py
  ```
- **Feature Engineering:**
  ```bash
  python scripts/feature_engineering.py
  ```
- **Model Training:**
  ```bash
  python scripts/model_training.py
  ```
- **Evaluation:**
  ```bash
  python scripts/model_evaluation.py
  ```

---

## Visual Insights 📈

### Model Comparisons

| Model               | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---------------------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | 78.99%   | 73.19%    | 70.63% | 71.89%   | 83.86%  |
| **Random Forest**       | 91.22%   | 94.35%    | 81.82% | 87.64%   | 97.69%  |
| **XGBoost**             | 91.76%   | 93.08%    | 84.62% | 88.64%   | 98.41%  |

---

