# -----------------------------------------------------------------------------
# This script is adapted from a Jupyter Notebook available at:
# https://www.kaggle.com/code/reminho/stroke-prediction-xgb-acc-0-98-f1-0-84

# Original code and methodology are used under the Apache License, Version 2.0.
# For details, see: http://www.apache.org/licenses/LICENSE-2.0

# Please refer to the original notebook for detailed methodology and results.

# Note: The dataset used in this script does not have a license and therefore cannot be distributed.
# Before running this script, please download the dataset locally from:
# https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
# -----------------------------------------------------------------------------

# Load necessary libraries
library(tidyverse)
library(caret)

# Set a seed for reproducibility
set.seed(88)

# Read and preprocess the data
stroke_data <- read_csv("../data/healthcare-dataset-stroke-data.csv")
stroke_data <- stroke_data %>%
  select(-id)

stroke_data_clean <- stroke_data %>%
  mutate(bmi = na_if(bmi, "N/A"),
         smoking_status = na_if(smoking_status, "Unknown"),
         bmi = as.numeric(bmi)) %>%
  mutate(bmi = ifelse(is.na(bmi), median(bmi, na.rm = TRUE), bmi)) %>%
  fill(smoking_status) %>%
  mutate(across(c(hypertension, heart_disease), factor),
         across(where(is.character), as.factor),
         across(where(is.factor), as.numeric),
         stroke = factor(ifelse(stroke == 0, "no", "yes")),
         bmi = case_when(
           bmi < 18.5 ~ "underweight",
           bmi >= 18.5 & bmi < 25 ~ "normal weight",
           bmi >= 25 & bmi < 30 ~ "overweight",
           bmi >= 30 ~ "obese"),
         bmi = factor(bmi, levels = c("underweight", "normal weight", "overweight", "obese"), order = TRUE))

# Split the data into training and testing sets
n_obs <- nrow(stroke_data_clean)
split <- round(n_obs * 0.7)
set.seed(88)
permuted_rows <- sample(n_obs)
stroke_shuffled <- stroke_data_clean[permuted_rows,]
train <- stroke_shuffled[1:split,]
test <- stroke_shuffled[(split + 1):nrow(stroke_shuffled),]

# Define grid and control for XGBoost model
xgbGrid <- expand.grid(
  nrounds = 3500,
  max_depth = 7,
  eta = 0.01,
  gamma = 0.01,
  colsample_bytree = 0.75,
  min_child_weight = 0,
  subsample = 0.5
)

xgbControl <- trainControl(
  method = "cv",
  number = 5
)

# Train the XGBoost model
xgb_model <- train(
  stroke ~ .,
  data = train,
  method = "xgbTree",
  tuneGrid = xgbGrid,
  trControl = xgbControl
)

# Predict on the test set
test_predictions <- predict(xgb_model, newdata = test)
test_accuracy <- mean(test_predictions == test$stroke)
cat(sprintf("Test Accuracy: %.4f\n", test_accuracy))

# Save the model
saveRDS(xgb_model, file = "../models/stroke_xgboost_r.rds")
