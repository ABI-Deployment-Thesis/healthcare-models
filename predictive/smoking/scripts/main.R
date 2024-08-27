# -----------------------------------------------------------------------------
# This script is adapted from a Jupyter Notebook available at:
# https://www.kaggle.com/code/titassaha/biomedical-analysis-risks-rf-accuracy-82-56
#
# Original code and methodology are used under the Apache License, Version 2.0.
# For details, see: http://www.apache.org/licenses/LICENSE-2.0
#
# Please refer to the original notebook for detailed methodology and results.
#
# Note: The dataset used in this script is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
# For details, see: https://creativecommons.org/licenses/by/4.0/
#
# The dataset is included with this script and can be freely shared and used in accordance with its license.
# It is also available at: https://www.kaggle.com/competitions/playground-series-s3e24/data
# -----------------------------------------------------------------------------

# Load necessary libraries
library(dplyr)
library(randomForest)
library(caTools)

# Load the dataset
df <- read.csv("../data/smoking.csv")

# Data preprocessing
df$sex_num <- ifelse(df$gender == "F", 0, 1)
df$tartar <- ifelse(df$tartar == "Y", 1, 0)

# Split data into training and testing sets
set.seed(123)
df$split <- sample.split(df$smoking, SplitRatio = 0.7)
df_train <- df %>% filter(split == TRUE) %>% select(-split)
df_test <- df %>% filter(split == FALSE) %>% select(-split)

# Build the random forest model
set.seed(1234)
model1 <- randomForest(factor(smoking) ~ age + height.cm. + weight.kg. + sex_num + 
                         eyesight.left. + eyesight.right. + hearing.left. + hearing.right. +
                         systolic + relaxation + 
                         Cholesterol + triglyceride + HDL + LDL +
                         fasting.blood.sugar + hemoglobin + Urine.protein + serum.creatinine +
                         AST + ALT + Gtp +
                         dental.caries + tartar,
                       importance = TRUE,
                       ntree = 500,
                       data = df_train)

# Predict on the test set
predictions <- predict(model1, newdata = df_test)

# Calculate accuracy
accuracy <- mean(predictions == df_test$smoking)
cat(sprintf("Test Accuracy: %.4f\n", accuracy))

# Save the model
saveRDS(model1, file = "../models/smoking_random_forest_r.rds")
