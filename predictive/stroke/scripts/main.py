"""
This script is adapted from a Jupyter Notebook available at:
https://www.kaggle.com/code/nimapourmoradi/healthcare-stroke

Original code and methodology are used under the Apache License, Version 2.0.
For details, see: http://www.apache.org/licenses/LICENSE-2.0

Please refer to the original notebook for detailed methodology and results.

Note: The dataset used in this script does not have a license and therefore cannot be distributed.
Before running this script, please download the dataset locally from:
https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
"""

import pickle

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load the dataset
data = pd.read_csv("../data/healthcare-dataset-stroke-data.csv")

# Drop the 'id' column
data.drop(columns="id", inplace=True)

# Drop rows with missing values
data.dropna(how="any", inplace=True)

# Convert categorical columns to numerical
data_2 = data.replace(
    {
        "gender": {"Male": 0, "Female": 1, "Other": 2},
        "ever_married": {"Yes": 0, "No": 1},
        "work_type": {
            "Private": 0,
            "Self-employed": 1,
            "Govt_job": 2,
            "children": 3,
            "Never_worked": 4,
        },
        "smoking_status": {
            "formerly smoked": 0,
            "never smoked": 1,
            "smokes": 2,
            "Unknown": 3,
        },
        "Residence_type": {"Urban": 0, "Rural": 1},
    }
)

# Prepare features and target
X = data_2.drop(columns="stroke")
y = data_2.stroke

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0, test_size=0.25
)

# Train the model
best_params_ = {"C": 0.001, "gamma": 0.001}
svc = SVC(**best_params_)
svc.fit(X_train, y_train)

# Evaluate the model
svc_score = svc.score(X_test, y_test)
print(f"Test Accuracy: {svc_score:.4f}")

# Save the trained model
joblib.dump(svc, "../models/stroke_svc_py.sav")
with open("../models/stroke_svc_py.pkl", "wb") as f:
    pickle.dump(svc, f)
