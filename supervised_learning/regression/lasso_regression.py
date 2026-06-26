"""
==========================================================
LASSO REGRESSION
==========================================================

Theory

• Lasso Regression is an extension of Linear Regression.
• It uses L1 Regularization.
• It reduces overfitting.
• It performs automatic feature selection.
• Some coefficients become exactly ZERO.

Formula

Cost = RSS + α(|β₁| + |β₂| + ... + |βₙ|)

Author : Alfred Joseph T
==========================================================
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================================
# Load Dataset
# ==========================================================

DATASET_PATH = "../../datasets/student_performance_dataset.csv"

data = pd.read_csv(DATASET_PATH)

print("=" * 60)
print("Student Performance Dataset")
print("=" * 60)

print(data.head())

print("\nDataset Shape :", data.shape)

# ==========================================================
# Select Numerical Features
# ==========================================================

X = data[
    [
        "Hours_Studied",
        "Attendance",
        "Previous_Grade",
        "Assignments_Completed",
        "Sleep_Hours",
        "Study_Sessions_Per_Week",
        "Internet_Usage_Hours",
        "Family_Income",
        "Sports_Hours",
        "Social_Media_Hours"
    ]
]

y = data["Final_Grade"]

# ==========================================================
# Train Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================================================
# Train Linear Regression
# ==========================================================

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_prediction = linear_model.predict(X_test)

# ==========================================================
# Train Ridge Regression
# ==========================================================

ridge_model = Ridge(alpha=1)

ridge_model.fit(X_train, y_train)

ridge_prediction = ridge_model.predict(X_test)

# ==========================================================
# Train Lasso Regression
# ==========================================================

lasso_model = Lasso(alpha=0.1)

lasso_model.fit(X_train, y_train)

lasso_prediction = lasso_model.predict(X_test)

# ==========================================================
# Print Coefficients
# ==========================================================

comparison = pd.DataFrame({

    "Feature": X.columns,

    "Linear": linear_model.coef_,

    "Ridge": ridge_model.coef_,

    "Lasso": lasso_model.coef_

})

print("\n")
print("="*70)
print("Coefficient Comparison")
print("="*70)

print(comparison)

# ==========================================================
# Evaluate Models
# ==========================================================

def evaluate(name, actual, prediction):

    mae = mean_absolute_error(actual, prediction)

    mse = mean_squared_error(actual, prediction)

    rmse = mse ** 0.5

    r2 = r2_score(actual, prediction)

    print("\n")
    print("="*60)
    print(name)
    print("="*60)

    print("MAE :", round(mae,2))
    print("MSE :", round(mse,2))
    print("RMSE:", round(rmse,2))
    print("R²  :", round(r2,4))


evaluate(
    "Linear Regression",
    y_test,
    linear_prediction
)

evaluate(
    "Ridge Regression",
    y_test,
    ridge_prediction
)

evaluate(
    "Lasso Regression",
    y_test,
    lasso_prediction
)

# ==========================================================
# Prediction Comparison
# ==========================================================

prediction_df = pd.DataFrame({

    "Actual": y_test.values,

    "Linear": linear_prediction.round(2),

    "Ridge": ridge_prediction.round(2),

    "Lasso": lasso_prediction.round(2)

})

print("\n")
print("="*60)
print("Prediction Comparison")
print("="*60)

print(prediction_df.head(10))

# ==========================================================
# Scatter Plot
# ==========================================================

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    linear_prediction,
    label="Linear"
)

plt.scatter(
    y_test,
    ridge_prediction,
    label="Ridge"
)

plt.scatter(
    y_test,
    lasso_prediction,
    label="Lasso"
)

plt.xlabel("Actual Grade")

plt.ylabel("Predicted Grade")

plt.title("Linear vs Ridge vs Lasso")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()

# ==========================================================
# Alpha Comparison
# ==========================================================

alphas = [
    0.001,
    0.01,
    0.1,
    1,
    10
]

scores = []

for alpha in alphas:

    model = Lasso(alpha=alpha)

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    score = r2_score(y_test, prediction)

    scores.append(score)

plt.figure(figsize=(8,5))

plt.plot(
    alphas,
    scores,
    marker="o",
    linewidth=2
)

plt.xlabel("Alpha")

plt.ylabel("R² Score")

plt.title("Effect of Alpha on Lasso Regression")

plt.grid(True)

plt.tight_layout()

plt.show()

# ==========================================================
# Feature Importance
# ==========================================================

plt.figure(figsize=(10,5))

plt.bar(
    X.columns,
    lasso_model.coef_
)

plt.xticks(rotation=45)

plt.title("Lasso Feature Coefficients")

plt.ylabel("Coefficient")

plt.grid(axis="y")

plt.tight_layout()

plt.show()

# ==========================================================
# Features Removed
# ==========================================================

print("\n")
print("="*60)
print("Features Removed By Lasso")
print("="*60)

removed = False

for feature, coefficient in zip(
        X.columns,
        lasso_model.coef_
):

    if abs(coefficient) < 0.0001:

        print(feature)

        removed = True

if not removed:

    print("No feature removed for current alpha.")

# ==========================================================
# Program Finished
# ==========================================================

print("\n")
print("="*60)
print("Program Finished Successfully")
print("="*60)