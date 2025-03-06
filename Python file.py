import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

# Load datasets
df_class = pd.read_csv("classification_dataset.csv")
df_reg = pd.read_csv("regression_dataset.csv")

# One-Hot Encoding for categorical features
categorical_cols = ["gender", "job_role", "city"]
df_class = pd.get_dummies(df_class, columns=categorical_cols, drop_first=True)
df_reg = pd.get_dummies(df_reg, columns=categorical_cols, drop_first=True)

# Split classification dataset
X_class = df_class.drop("classification_target", axis=1)
y_class = df_class["classification_target"]
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

# Split regression dataset
X_reg = df_reg.drop("regression_target", axis=1)
y_reg = df_reg["regression_target"]
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Classification Model: Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_class, y_train_class)
y_pred_class = clf.predict(X_test_class)

print("Classification Model Performance:")
print("Accuracy:", accuracy_score(y_test_class, y_pred_class))
print("Classification Report:\n", classification_report(y_test_class, y_pred_class))

# Regression Model: Random Forest Regressor
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train_reg, y_train_reg)
y_pred_reg = reg.predict(X_test_reg)

print("\nRegression Model Performance:")
print("Mean Squared Error:", mean_squared_error(y_test_reg, y_pred_reg))
print("R2 Score:", r2_score(y_test_reg, y_pred_reg))
