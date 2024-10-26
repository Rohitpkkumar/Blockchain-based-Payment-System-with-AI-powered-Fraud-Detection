import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load your dataset
df = pd.read_csv('/Users/rohitkumar/Desktop/brainwave/synthetic_blockchain_fraud_dataset.csv')  # Path to your dataset

# Split the dataset into features and target variable
X = df.drop(columns=['label'])  # Exclude the target 'label' column
y = df['label']


# Define numerical and categorical columns
numeric_features = ['amount', 'gas_fee']
categorical_features = ['sender_wallet', 'recipient_wallet', 'transaction_type']

# Preprocessing pipeline for numeric and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Apply preprocessing to X before applying SMOTE
X_preprocessed = preprocessor.fit_transform(X)

# Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to balance the classes on training data only
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train the Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)  # Probability predictions

# Evaluate the model
print("\nClassification Report for Random Forest:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix for Random Forest:")
print(confusion_matrix(y_test, y_pred))

# Calculate ROC-AUC Score for all classes using 'ovr' (one-vs-rest)
roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
print(f"ROC-AUC Score for Random Forest: {roc_auc:.4f}")

scaler = StandardScaler()


# Save the trained model with preprocessing steps
joblib.dump((preprocessor, rf_model), 'random_forest_fraud_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model saved as random_forest_fraud_model.pkl")
