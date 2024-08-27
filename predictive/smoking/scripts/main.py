"""
This script is adapted from a Jupyter Notebook available at:
https://www.kaggle.com/code/hardikgarg03/smoker-status-signal-80-accuracy

Original code and methodology are used under the Apache License, Version 2.0.
For details, see: http://www.apache.org/licenses/LICENSE-2.0

Please refer to the original notebook for detailed methodology and results.

Note: The dataset used in this script is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
For details, see: https://creativecommons.org/licenses/by/4.0/

The dataset is included with this script and can be freely shared and used in accordance with its license.
It is also available at: https://www.kaggle.com/competitions/playground-series-s3e24/data
"""

import pickle

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load data
train = pd.read_csv("../data/train.csv")

# Prepare features and target
train = train.drop("id", axis=1)
X = train.drop("smoking", axis=1)
y = train["smoking"]

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
lg = LGBMClassifier(n_estimators=100, force_col_wise=True, random_state=42)
lg.fit(X_train, y_train)

# Predict and evaluate on validation set
y_pred_lg = lg.predict(X_val)
accuracy = accuracy_score(y_val, y_pred_lg)

# Print validation accuracy
print(f"Test Accuracy: {accuracy:.4f}")

# Save the trained model
joblib.dump(lg, "../models/smoking_lightgbm_py.sav")
with open("../models/smoking_lightgbm_py.pkl", "wb") as f:
    pickle.dump(lg, f)
