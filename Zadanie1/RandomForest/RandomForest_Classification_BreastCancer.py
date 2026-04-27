# Użyty dataset: https://huggingface.co/datasets/wwydmanski/wisconsin-breast-cancer

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data
local_path = "Zadanie1/RandomForest/data/"
train_df = pd.read_csv(local_path + "train.csv", index_col=0)
test_df = pd.read_csv(local_path + "test.csv", index_col=0)

# Set features and target
X_train = train_df.drop("y", axis=1)
y_train = train_df["y"]

X_test = test_df.drop("y", axis=1)
y_test = test_df["y"]

# Convert target to boolean
y_train = y_train.map({"M": 1, "B": 0})
y_test = y_test.map({"M": 1, "B": 0})

# Searching for best random parameters
random_params = {
    "n_estimators": [50, 100, 200, 300],
    "max_depth": [None, 5, 10, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 3, 5],
    "max_features": ["sqrt", "log2"],
}
random_model = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    random_params,
    n_iter=20,
    cv=5,
    random_state=42,
)
random_model.fit(X_train, y_train)
best_random = random_model.best_params_
print("\nBest parameters (random):\n", best_random)

# Searching for best parameters locally near best random parameters
grid_params = {
    "n_estimators": [
        best_random["n_estimators"] - 25,
        best_random["n_estimators"],
        best_random["n_estimators"] + 25,
    ],
    "max_depth": range(best_random["max_depth"] - 2, best_random["max_depth"] + 2),
    "min_samples_split": [
        max(2, best_random["min_samples_split"] - 1),
        best_random["min_samples_split"],
        best_random["min_samples_split"] + 1,
    ],
    "min_samples_leaf": [
        max(1, best_random["min_samples_leaf"] - 1),
        best_random["min_samples_leaf"],
        best_random["min_samples_leaf"] + 1,
    ],
}
grid_model = GridSearchCV(
    RandomForestClassifier(random_state=42),
    grid_params,
    cv=5,
    error_score="raise",
)
grid_model.fit(X_train, y_train)
best_grid = grid_model.best_params_
print("\nBest parameters (grid):\n", best_grid)

# Predicting based on grid as it should have better results
best_model = grid_model.best_estimator_
y_pred = best_model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

results_df = X_test.copy()
results_df["Malignancy"] = y_test.values
results_df["Predicted Malignancy"] = y_pred
results_df.to_csv(local_path + "results.csv", index=False)
