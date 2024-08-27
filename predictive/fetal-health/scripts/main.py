"""
This script is adapted from a Jupyter Notebook available at:
https://www.kaggle.com/code/karnikakapoor/fetal-health-classification

Original code and methodology are used under the Apache License, Version 2.0.
For details, see: http://www.apache.org/licenses/LICENSE-2.0

Please refer to the original notebook for detailed methodology and results.

Note: The dataset used in this script does not have a license and therefore cannot be distributed.
Before running this script, please download the dataset locally from:
https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification
"""

import pickle

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

np.random.seed(0)

# Load data
data = pd.read_csv("../data/fetal_health.csv")

# Assign features and target
X = data.drop(["fetal_health"], axis=1)
y = data["fetal_health"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train the model
best_params = {
    "criterion": "entropy",
    "max_depth": 12,
    "max_features": "auto",
    "n_estimators": 150,
    "n_jobs": None,
}
RF_model = RandomForestClassifier(**best_params)
RF_model.fit(X_train, y_train)

# Evaluate the model
predictions = RF_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Test Accuracy: {accuracy:.4f}")

# Save the trained model
joblib.dump(RF_model, "../models/fetal_health_random_forest_py.sav")
with open("../models/fetal_health_random_forest_py.pkl", "wb") as f:
    pickle.dump(RF_model, f)
