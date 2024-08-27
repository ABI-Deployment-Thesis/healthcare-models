# -----------------------------------------------------------------------------
# This script is adapted from a Jupyter Notebook available at:
# https://www.kaggle.com/code/radhikapotey/random-forest-fetal-health-classification
#
# Original code and methodology are used under the Apache License, Version 2.0.
# For details, see: http://www.apache.org/licenses/LICENSE-2.0
#
# Please refer to the original notebook for detailed methodology and results.
#
# Note: The dataset used in this script does not have a license and therefore cannot be distributed.
# Before running this script, please download the dataset locally from:
# https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification
# -----------------------------------------------------------------------------

library(randomForest)

# Load data
health <- read.csv("../data/fetal_health.csv", header = TRUE, sep = ",")

# Handle missing values and factor levels
health$fetal_health <- as.factor(health$fetal_health)
levels(health$fetal_health) <- list(Normal = "1", Suspect = "2", Pathological = "3")

# Split data into training and testing sets
set.seed(123)
ind <- sample(2, nrow(health), replace = TRUE, prob = c(0.7, 0.3))
train <- health[ind == 1, ]
test <- health[ind == 2, ]

# Train the Random Forest model
set.seed(333)
rf <- randomForest(fetal_health ~ ., data = train, ntree = 300, mtry = 8, importance = TRUE)

# Evaluate the model
test_predictions <- predict(rf, newdata = test)
test_accuracy <- mean(test_predictions == test$fetal_health)

# Print test accuracy
cat(sprintf("Test Accuracy: %.4f\n", test_accuracy))

# Save the trained model
saveRDS(rf, file = "../models/fetal_health_random_forest_r.rds")
