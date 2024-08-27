"""
This script is adapted from a Jupyter Notebook available at:
https://www.kaggle.com/code/zabihullah18/diabetes-prediction

Original code and methodology are used under the Apache License, Version 2.0.
For details, see: http://www.apache.org/licenses/LICENSE-2.0

Please refer to the original notebook for detailed methodology and results.

Note: The dataset used in this script does not have a license and therefore cannot be distributed.
Before running this script, please download the dataset locally from:
https://www.kaggle.com/datasets/imtkaggleteam/diabetes
"""

import pickle

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load data
df = pd.read_csv("../data/diabetes.csv")

# Outlier removal for numeric columns
numeric_columns = [
    "Insulin",
    "DiabetesPedigreeFunction",
]

for column_name in numeric_columns:
    Q1 = np.percentile(df[column_name], 25)
    Q3 = np.percentile(df[column_name], 75)
    IQR = Q3 - Q1
    low_lim = Q1 - 1.5 * IQR
    up_lim = Q3 + 1.5 * IQR
    df[column_name] = np.clip(df[column_name], low_lim, up_lim)

# Prepare features and target variable
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Train the Decision Tree model
model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    min_samples_leaf=5,
    min_samples_split=20,
    random_state=42,
)
model.fit(X_train, y_train)

# Evaluate the model
val_accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Test Accuracy: {val_accuracy:.4f}")

# Save the trained model
joblib.dump(model, "../models/diabetes_decision_tree_py.sav")
with open("../models/diabetes_decision_tree_py.pkl", "wb") as f:
    pickle.dump(model, f)
