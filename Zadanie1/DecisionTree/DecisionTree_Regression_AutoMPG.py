# Użyty dataset: https://huggingface.co/datasets/scikit-learn/auto-mpg

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load data
local_path = "Zadanie1/DecisionTree/data/"
df = pd.read_csv(local_path + "auto-mpg.csv")

# Drop missing data
df = df.replace("?", np.nan)
df = df.dropna()

# Car name isn't a good indicator of MPG
car_name = df["car name"]  # Preserve for exported result
X = df.drop(["mpg", "car name"], axis=1)
y = df["mpg"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=None, random_state=42
)

# Searching for best random parameters
random_params = {
    "max_depth": range(3, 8),
    "min_samples_leaf": range(3, 7),
    "min_samples_split": range(3, 10),
    "ccp_alpha": [0.0, 0.0001, 0.001, 0.1],
    "min_impurity_decrease": [
        0.0,
        0.00001,
        0.0001,
        0.001,
        0.01,
        0.03,
        0.035,
        0.04,
        0.045,
        0.06,
    ],
}
random_model = RandomizedSearchCV(
    DecisionTreeRegressor(
        criterion="squared_error",
        random_state=42,
    ),
    random_params,
    random_state=42,
    n_iter=50,
    cv=5,
)
random_model.fit(X_train, y_train)
best_random = random_model.best_params_
print("\nBest parameters (random):\n", best_random)

# Searching for best parameters locally near best random parameters
grid_params = {
    "max_depth": range(best_random["max_depth"] - 1, best_random["max_depth"] + 1),
    "min_samples_leaf": range(
        best_random["min_samples_leaf"] - 1, best_random["min_samples_leaf"] + 1
    ),
    # min_samples_split doesn't provide any gains here; other parameters deactivate it before it's used
    "min_samples_split": range(
        max(2, best_random["min_samples_split"] - 10),
        best_random["min_samples_split"] + 1,
    ),
    "ccp_alpha": [
        max(0.0, best_random["ccp_alpha"] * 0.1),
        best_random["ccp_alpha"],
        best_random["ccp_alpha"] * 10,
    ],
    "min_impurity_decrease": [
        max(0.0, best_random["min_impurity_decrease"] - 0.01),
        best_random["min_impurity_decrease"],
        best_random["min_impurity_decrease"] + 0.01,
    ],
}
grid_model = GridSearchCV(
    DecisionTreeRegressor(
        criterion="squared_error",
        random_state=42,
    ),
    grid_params,
    cv=5,
)
grid_model.fit(X_train, y_train)
best_grid = grid_model.best_params_
print("\nBest parameters (grid):\n", best_grid)

# Predicting based on grid as it should have better results
best_model = grid_model.best_estimator_
y_pred = best_model.predict(X_test)

print("\nMSE:", mean_squared_error(y_test, y_pred))
print("\nMAE:", mean_absolute_error(y_test, y_pred))
print("\nR_squared:", r2_score(y_test, y_pred))

results_df = X_test.copy()
results_df["Real MPG"] = y_test.values
results_df["Predicted MPG"] = y_pred
results_df["car name"] = car_name  # Append previously removed car name
results_df.to_csv(local_path + "results.csv", index=False)
