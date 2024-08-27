# -----------------------------------------------------------------------------
# This script is adapted from a Jupyter Notebook available at:
# https://www.kaggle.com/code/mayankagrawalds/predict-diabetes-multiple-ml-models
#
# Original code and methodology are used under the Apache License, Version 2.0.
# For details, see: http://www.apache.org/licenses/LICENSE-2.0
#
# Please refer to the original notebook for detailed methodology and results.
#
# Note: The dataset used in this script does not have a license and therefore cannot be distributed.
# Before running this script, please download the dataset locally from:
# https://www.kaggle.com/datasets/imtkaggleteam/diabetes
# -----------------------------------------------------------------------------

library(caret) # ML Model building package
set.seed(123)

# Load data
df <- read.csv("../data/diabetes.csv", header = TRUE)
names(df) <- c("pregnant", "glucose", "pressure", "triceps", "insulin", "mass", "pedigree", "age", "diabetes")
df$diabetes <- ifelse(df$diabetes == 0, "neg", "pos")
df$diabetes <- as.factor(df$diabetes)

# Create training and testing datasets
partition <- caret::createDataPartition(y = df$diabetes, p = 0.7, list = FALSE)
train_set <- df[partition,]
test_set <- df[-partition,]

# Train the model
model_glm <- caret::train(diabetes ~., data = train_set,
                          method = "glm",
                          metric = "Accuracy",
                          tuneLength = 10,
                          trControl = trainControl(method = "cv", number = 10,
                                                   classProbs = TRUE, summaryFunction = twoClassSummary),
                          preProcess = c("center", "scale", "pca"))

# Evaluate the model
test_predictions <- predict(model_glm, newdata = test_set)
test_accuracy <- mean(test_predictions == test_set$diabetes)

# Print the test accuracy
cat(sprintf("Test Accuracy: %.4f\n", test_accuracy))

# Save the trained model
saveRDS(model_glm, file = "../models/diabetes_logistic_regression_r.rds")
