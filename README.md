# Telco Customer Churn Prediction

## 📌 Project Overview

This project implements an end-to-end **Telecom Customer Churn Prediction** solution using machine learning.

The solution combines:

- **Dask** for scalable data loading and processing
- **Pandas** for data manipulation
- **Scikit-learn** for preprocessing and model evaluation
- **Target Encoding** for categorical variables
- **Power Transformation (Yeo-Johnson)** for numerical variables
- **SMOTE** to address class imbalance
- **XGBoost** for binary churn classification
- **Streamlit** for an interactive business-facing application
- **Joblib** for saving and loading the trained model

The objective is to identify customers who are likely to churn and provide a probability/confidence score that can support customer-retention decisions.

---

## 🎯 Business Objective

Customer churn is a major challenge for telecom organizations because acquiring a new customer can be significantly more expensive than retaining an existing one.

This solution helps answer:

> **"Is this customer likely to churn?"**

The application can be used to:

1. Assess an individual customer's churn risk.
2. Process multiple customers through bulk upload.
3. Display churn percentage and prediction confidence.
4. Support proactive retention campaigns.
5. Help business teams prioritize high-risk customers.

---

## 🏗️ Solution Architecture

```text
                    ┌──────────────────────┐
                    │   Customer Dataset   │
                    │      Excel / CSV     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Dask Data Loading    │
                    │ + Pandas Processing  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Data Cleaning        │
                    │                      │
                    │ Column Standardizing │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Feature Preparation  │
                    │ Numeric / Categorical│
                    └──────────┬───────────┘
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
       ┌──────────────────┐        ┌──────────────────┐
       │ Numeric Features │        │ Categorical      │
       │                  │        │ Features         │
       │ Yeo-Johnson      │        │                  │
       └────────┬─────────┘        │ Target Encoding  │
                │                  └────────┬─────────┘
                └────────────┬─────────────┘
                             ▼
                    ┌──────────────────────┐
                    │ Training Data        │
                    │ 70% Train / 30% Test │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ SMOTE                │
                    │ Class Balancing      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ XGBoost Classifier   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Prediction +         │
                    │ Probability Score    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Streamlit Dashboard  │
                    │ Charts / KPIs / Risk │
                    └──────────────────────┘
```

---

## 📊 Dataset

The model is designed around telecom customer information such as:

- City
- Gender
- Senior Citizen
- Partner
- Dependents
- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Contract
- Paperless Billing
- Payment Method
- Tenure Months
- Monthly Charges

The original dataset may contain additional fields. During preprocessing, selected fields that are not required for the prediction model are removed.

### Target Variable

The target variable is:

```text
Churn_Label
```

The training code maps:

```text
Yes → 0
No  → 1
```

> **Important:** The target encoding should be kept consistent between training and application prediction logic. If `1` represents "No churn", the application should interpret the model output accordingly.

---

## 🧹 Data Preparation

The preprocessing workflow performs the following operations.

### 1. Column Standardization

Spaces in column names are replaced with underscores.

Example:

```text
Total Charges
```

becomes:

```text
Total_Charges
```

### 2. Missing Total Charges

Blank values in `Total Charges` are replaced with `0` and converted to floating-point values.

### 3. Unused Features

The training script removes fields such as:

```text
Lat_Long
Latitude
Longitude
Total_Charges
Churn_Value
CLTV
Country
State
Streaming_Movies
```

These exclusions should be reviewed if the model is retrained with a different dataset.

---

## 🔢 Feature Engineering & Preprocessing

### Numerical Features

Numerical variables are processed using:

1. Median imputation
2. Yeo-Johnson Power Transformation

The Yeo-Johnson transformation helps reduce skewness and makes numerical distributions more suitable for downstream modeling.

```python
PowerTransformer(method="yeo-johnson")
```

### Categorical Features

Categorical variables are processed using:

1. Missing-value replacement
2. Target Encoding

```python
TargetEncoder(
    random_state=42,
    cv=10
)
```

Target encoding converts categorical values into numerical representations based on their relationship with the target variable.

---

## ⚖️ Handling Class Imbalance

Customer churn datasets commonly contain fewer churned customers than retained customers.

To address this imbalance, the training process applies:

```python
SMOTE(random_state=42)
```

SMOTE creates synthetic samples for the minority class in the training dataset.

### Important Design Decision

SMOTE is applied **after preprocessing and only to the training data**.

This avoids introducing synthetic observations into the test set and helps maintain a more realistic evaluation.

---

## 🤖 Machine Learning Model

The primary classification algorithm is:

**XGBoost Classifier**

Configuration used in the training script:

```python
XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=1000,
    max_depth=3,
    learning_rate=0.01,
    subsample=1,
    colsample_bytree=1,
    n_jobs=-1,
    tree_method="hist",
    reg_alpha=0.01,
    reg_lambda=0.01
)
```

### Why XGBoost?

XGBoost is well suited to structured/tabular customer data because it:

- Handles nonlinear relationships
- Captures feature interactions
- Performs well on mixed business datasets
- Supports probability predictions
- Is generally effective for classification problems
- Provides a strong baseline for churn modeling

---

## 🧪 Train/Test Split

The dataset is divided into:

```text
70% → Training
30% → Testing
```

with:

```python
random_state=42
shuffle=True
```

This provides reproducible train/test partitioning.

---

## 📈 Model Evaluation

The training process evaluates the classifier using:

### Confusion Matrix

```python
confusion_matrix(y_test, y_pred)
```

The confusion matrix provides:

- True Positives
- True Negatives
- False Positives
- False Negatives

### Classification Metrics

A production evaluation should also report:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Churn-class recall
- Churn-class precision

For churn applications, **recall and precision for the churn class** are especially important because the business may prefer identifying as many high-risk customers as possible.

---

## 🎯 Prediction Confidence

The XGBoost model supports:

```python
predict_proba()
```

which can be used to obtain the probability associated with the prediction.

For example:

```python
probability = model.predict_proba(X)[:, 1]
```

The Streamlit application can present this as a customer-facing confidence/risk score.

### Example Business Interpretation

```text
Churn Risk: HIGH
Confidence: 87%
```

The exact wording should reflect the target-label mapping used by the trained model.

> A model probability is not automatically a perfectly calibrated real-world probability. For business deployment, probability calibration should be considered if the score will be interpreted as an actual likelihood of churn.

---

# 🖥️ Streamlit Application

The Streamlit layer provides a business-friendly interface on top of the trained model.

## Key Features

### 1. Single Customer Assessment

Users can enter customer information through the UI.

The application can display:

- Prediction result
- Churn risk
- Confidence/probability score
- Customer assessment summary
- Business-oriented recommendation

Example:

```text
Customer Assessment

Prediction: High Churn Risk
Confidence: 91%

Recommended Action:
Prioritize this customer for retention outreach.
```

### 2. Bulk Customer Prediction

Users can upload a file containing multiple customers.

The application can:

- Validate uploaded columns
- Process multiple records
- Generate predictions
- Calculate confidence scores
- Show churn distribution
- Display summary KPIs
- Allow the results to be downloaded

### 3. Dashboard Metrics

Useful dashboard KPIs include:

```text
Total Customers
        │
        ├── High Churn Risk
        ├── Low Churn Risk
        ├── Churn %
        └── Average Confidence
```

### 4. Recommended Visualizations

The Streamlit dashboard can contain:

- Churn vs Non-Churn donut/pie chart
- Churn percentage KPI
- Confidence distribution
- Prediction distribution
- Customer risk segmentation
- Bulk prediction summary

---

# ⚙️ Configuration

A configuration file can be used to keep environment-specific paths outside the Python code.

Example:

```yaml
data:
  path: "../data/Telco_Customer_Churn.xlsx"

model:
  path: "../model/Telco_customer_churn.pkl"

logging:
  path: "../logs/application.log"
```

Update these paths according to the local project structure.

---

# 📦 Installation

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install the required packages:

```bash
pip install pandas
pip install numpy
pip install dask
pip install dask-ml
pip install distributed
pip install scikit-learn
pip install imbalanced-learn
pip install xgboost
pip install openpyxl
pip install pyyaml
pip install joblib
pip install streamlit
```

Or install everything from:

```bash
pip install -r requirements.txt
```

---

# ▶️ Training the Model

Run:

```bash
python train_model.py
```

The training workflow:

1. Reads the Excel dataset.
2. Cleans the dataset.
3. Standardizes feature names.
4. Separates numerical and categorical features.
5. Splits the dataset into train/test sets.
6. Fits the preprocessing pipeline.
7. Applies SMOTE to training data.
8. Trains XGBoost.
9. Evaluates predictions.
10. Saves the trained model/preprocessing objects for application use.

---

# 🚀 Running the Streamlit Application

Run:

```bash
streamlit run streamlit_app.py
```

The application will normally become available at:

```text
http://localhost:8501
```

---

# 🔌 Optional FastAPI Prediction Service

The project can also expose the trained model through FastAPI.

Example endpoint:

```text
POST /predict
```

A customer JSON payload can be submitted to the prediction API.

Example:

```json
{
    "City": "Los Angeles",
    "Gender": "Male",
    "Senior_Citizen": "No",
    "Partner": "Yes",
    "Dependents": "No",
    "Phone_Service": "Yes",
    "Multiple_Lines": "No",
    "Internet_Service": "Fiber optic",
    "Online_Security": "No",
    "Online_Backup": "Yes",
    "Device_Protection": "No",
    "Tech_Support": "No",
    "Streaming_TV": "Yes",
    "Contract": "Month-to-month",
    "Paperless_Billing": "Yes",
    "Payment_Method": "Electronic check",
    "Tenure_Months": "5",
    "Monthly_Charges": "79.85",
    "Churn_Score": "70"
}
```

### API Response

A production API should preferably return structured JSON rather than a plain text string.

Example:

```json
{
    "prediction": "Churn",
    "confidence": 0.87,
    "message": "Customer shows a high risk of churn."
}
```

---

# ⚠️ Important Model Compatibility Rule

The model must receive data in the **same feature representation used during training**.

Do not independently convert categorical fields such as:

```text
City
Gender
Senior_Citizen
Contract
Payment_Method
```

into arbitrary numeric values in the Streamlit/FastAPI application.

For example, do not manually assume:

```text
Male = 1
Female = 0
```

unless that exact encoding was used during model training.

The safest production design is to save the complete preprocessing pipeline together with the classifier and pass the original customer values into that pipeline.

This prevents errors such as:

```text
'City' must contain a numeric value
```

or:

```text
'Senior Citizen' contains a value that cannot be interpreted as a number
```

---

# 🛡️ Validation

The application should validate:

### Required Columns

Uploaded files should contain all model-required features.

### Numeric Fields

Examples:

```text
Tenure_Months
Monthly_Charges
Churn_Score
```

must contain valid numeric values.

### Categorical Fields

Values such as:

```text
City
Gender
Contract
Payment_Method
```

should remain categorical strings and should be processed through the same preprocessing logic used during training.

### Missing Values

Missing values should be handled consistently with training.

---

# 📝 Logging

For production use, application errors should be written to a log file.

Example:

```text
logs/application.log
```

The log should record:

- Timestamp
- Log level
- Prediction operation
- Validation errors
- Model errors
- Unexpected exceptions

Example:

```text
2026-08-21 18:51:32 | ERROR |
Customer prediction failed:
invalid value received for feature City
```

Technical errors should be logged for administrators while the end user receives a clear business-friendly message.

---

# 🔐 Production Recommendations

Before deploying the application to production:

1. Save the preprocessing transformer with the model.
2. Validate all incoming features.
3. Avoid manually encoding categorical values in the UI.
4. Add structured logging.
5. Add model versioning.
6. Monitor prediction distributions.
7. Monitor data drift.
8. Monitor model performance after deployment.
9. Calibrate probabilities if confidence is presented as a true probability.
10. Protect API endpoints with authentication if exposed publicly.
11. Do not expose internal stack traces to customers.
12. Keep secrets and credentials outside source code.

---

# 📊 Business Use Cases

This solution can support:

### Customer Retention

Identify customers with high churn risk and prioritize retention offers.

### Customer Segmentation

Group customers into:

- High Risk
- Medium Risk
- Low Risk

### Campaign Prioritization

Marketing teams can focus resources on customers with the highest predicted risk.

### Management Reporting

Executives can monitor:

- Overall churn rate
- High-risk customer count
- Prediction confidence
- Churn distribution
- Customer risk trends

---

# 🔭 Future Enhancements

Potential improvements include:

- SHAP-based explainability
- Feature importance dashboard
- Probability calibration
- Automated model retraining
- MLflow experiment tracking
- Model version management
- Data drift monitoring
- REST API deployment
- Docker containerization
- CI/CD pipeline
- Cloud deployment
- Role-based access
- Automated retention recommendations

---

# 🧰 Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application and ML development |
| Dask | Scalable data processing |
| Pandas | Data manipulation |
| Scikit-learn | Preprocessing and evaluation |
| imbalanced-learn | SMOTE class balancing |
| XGBoost | Classification model |
| Streamlit | Interactive dashboard |
| FastAPI | Optional prediction API |
| Joblib | Model persistence |
| PyYAML | Configuration management |
| OpenPyXL | Excel file handling |

---

# 📌 Project Highlights

### Machine Learning

- Binary classification
- XGBoost
- SMOTE
- Target encoding
- Yeo-Johnson transformation
- Train/test evaluation

### Engineering

- Config-driven paths
- Reusable preprocessing
- Model persistence
- Validation
- Logging
- API-ready architecture

### Business Application

- Single customer assessment
- Bulk prediction
- Churn percentage
- Confidence score
- Interactive charts
- Business-oriented messaging

---

# 👨‍💻 Author

**Telecom Customer Churn Prediction — Machine Learning & Streamlit Project**

This project demonstrates how a machine-learning model can be taken from data preparation and model training to an interactive business application.

---

# ⭐ Conclusion

The project demonstrates an end-to-end approach to customer churn analytics:

```text
Raw Customer Data
       ↓
Data Cleaning
       ↓
Feature Preparation
       ↓
Preprocessing
       ↓
SMOTE
       ↓
XGBoost
       ↓
Model Evaluation
       ↓
Prediction Probability
       ↓
Streamlit Dashboard
       ↓
Business Decision
```

The key objective is not only to predict churn, but to convert model predictions into **actionable customer-retention insights**.
