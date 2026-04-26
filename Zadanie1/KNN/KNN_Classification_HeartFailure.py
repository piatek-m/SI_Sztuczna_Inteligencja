# Użyty dataset: https://huggingface.co/datasets/mstz/heart_failure

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Loading data
local_path = "Zadanie1/KNN/data/"
df = pd.read_csv(local_path + "death/train.csv")

# Separating features and target
X = df.drop("is_dead", axis=1)
y = df["is_dead"]
# Separating numeric and boolean attributes in order to scale
num_cols = X.select_dtypes(include=["number"]).columns
bool_cols = X.select_dtypes(include=["bool"]).columns

# Splitting training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=None, random_state=42
)
print("\nValue counts:\n", y.value_counts())

# Scaling, so that big numeric values (e.g. platletes_concentration) don't dominate the model's learning
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
scaler = StandardScaler()
X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

# Search for best parameters
params = {
    "n_neighbors": [3, 4, 5, 6, 19, 23, 25],
    "weights": ["uniform", "distance"],
}
grid = GridSearchCV(
    KNeighborsClassifier(),
    params,
    cv=5,
)
grid.fit(X_train_scaled, y_train)
y_pred = grid.predict(X_test_scaled)
print("\nBest parameters (grid):\n", grid.best_params_)

# Train the model with best parameters found
model_with_grid = KNeighborsClassifier(
    n_neighbors=grid.best_params_["n_neighbors"],
    weights=grid.best_params_["weights"],
)
model_with_grid.fit(X_train_scaled, y_train)
y_pred = model_with_grid.predict(X_test_scaled)

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# Export results
results_df = X_test.copy()
results_df["Actual_is_dead"] = y_test.values
results_df["Predicted_is_dead"] = y_pred
results_df.to_csv(local_path + "results.csv", index=False)

print("\n============================================")
# RandomSearch for comparison
random_params = {
    "n_neighbors": range(1, 25),
    "weights": ["uniform", "distance"],
}
random = RandomizedSearchCV(
    KNeighborsClassifier(),
    random_params,
    n_iter=20,
    cv=5,
)
random.fit(X_train_scaled, y_train)
y_pred = grid.predict(X_test_scaled)
print("\nBest parameters (random):\n", random.best_params_)

# Train with best params found by random
model_with_random = KNeighborsClassifier(
    n_neighbors=random.best_params_["n_neighbors"],
    weights=random.best_params_["weights"],
)
model_with_random.fit(X_train_scaled, y_train)
y_pred = model_with_random.predict(X_test_scaled)
