"""
This script is adapted from a Jupyter Notebook available at:
https://www.kaggle.com/code/tumpanjawat/medcost-eda-k-cluster-gradient-boost-full

Original code and methodology are used under the Apache License, Version 2.0.
For details, see: http://www.apache.org/licenses/LICENSE-2.0

Please refer to the original notebook for detailed methodology and results.

Note: The dataset used in this script is licensed under the Open Data Commons Database License (ODbL) 1.0.
For details, see: https://opendatacommons.org/licenses/dbcl/1-0/

The dataset is included with this script and can be freely shared and used in accordance with its license.
It is also available at: https://www.kaggle.com/datasets/mirichoi0218/insurance
"""

import pickle

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load and prepare data
df = pd.read_csv("../data/insurance.csv")
df = df.drop_duplicates()

# Apply Label Encoding to the categorical columns
le = LabelEncoder()
df["sex"] = le.fit_transform(df["sex"])
df["smoker"] = le.fit_transform(df["smoker"])
df["region"] = le.fit_transform(df["region"])

# Split data into features (X) and target (y)
X = df.drop(columns=["charges"])
y = df["charges"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the Gradient Boosting Regressor model
model = GradientBoostingRegressor(learning_rate=0.01, n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")

# Save the trained model
joblib.dump(model, "../models/insurance_gbr_py.sav")
with open("../models/insurance_gbr_py.pkl", "wb") as f:
    pickle.dump(model, f)
