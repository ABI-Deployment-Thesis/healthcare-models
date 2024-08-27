# -----------------------------------------------------------------------------
# This script is adapted from a Jupyter Notebook available at:
# https://www.kaggle.com/code/ruslankl/health-care-cost-prediction-w-linear-regression

# Original code and methodology are used under the Apache License, Version 2.0.
# For details, see: http://www.apache.org/licenses/LICENSE-2.0
#
# Please refer to the original notebook for detailed methodology and results.
#
# Note: The dataset used in this script is licensed under the Open Data Commons Database License (ODbL) 1.0.
# For details, see: https://opendatacommons.org/licenses/dbcl/1-0/
#
# The dataset is included with this script and can be freely shared and used in accordance with its license.
# It is also available at: https://www.kaggle.com/datasets/mirichoi0218/insurance
# -----------------------------------------------------------------------------

set.seed(123)

# Load and prepare data
insurance <- read.csv("../data/insurance.csv")

# Split data into training and testing sets
n_train <- round(0.8 * nrow(insurance))
train_indices <- sample(1:nrow(insurance), n_train)
data_train <- insurance[train_indices, ]
data_test <- insurance[-train_indices, ]

# Define the formula
formula_0 <- as.formula("charges ~ age + sex + bmi + children + smoker + region")

# Train the linear model
model_0 <- lm(formula_0, data = data_train)

# Predict on test set
prediction_0 <- predict(model_0, newdata = data_test)

# Calculate residuals and RMSE
residuals_0 <- data_test$charges - prediction_0
rmse_0 <- sqrt(mean(residuals_0^2))

# Print RMSE
cat(sprintf("Root Mean Squared Error: %.4f\n", rmse_0))

# Save the trained model
saveRDS(model_0, file = "../models/insurance_linear_regression_r.rds")
