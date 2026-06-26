"""
==========================================================
RIDGE REGRESSION
==========================================================

Theory:
- Ridge Regression is an extension of Linear Regression.
- It adds L2 Regularization to reduce overfitting.
- It shrinks coefficients towards zero but never makes them exactly zero.

Formula:

Cost = RSS + α(β₁² + β₂² + β₃² + ...)

Author : Alfred Joseph T
==========================================================
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================================
# Load Dataset
# ==========================================================

DATASET_PATH = "../../datasets/student_scores.csv"

data = pd.read_csv(DATASET_PATH)

print("=" * 60)
print("Student Dataset")
print("=" * 60)

print(data.head())

print("\nDataset Shape :", data.shape)

print("\nColumns")

print(list(data.columns))

# ==========================================================
# Select Features and Target
# ==========================================================

X = data[
    [
        "Hours_Studied",
        "Attendance",
        "Previous_Grade"
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

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ==========================================================
# Linear Regression Model
# ==========================================================

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_prediction = linear_model.predict(X_test)

# ==========================================================
# Ridge Regression Model
# ==========================================================

alpha = 1.0

ridge_model = Ridge(alpha=alpha)

ridge_model.fit(X_train, y_train)

ridge_prediction = ridge_model.predict(X_test)

# ==========================================================
# Display Coefficients
# ==========================================================

print("\n")
print("=" * 60)
print("LINEAR REGRESSION COEFFICIENTS")
print("=" * 60)

print("Intercept :", round(linear_model.intercept_,4))

for feature, coefficient in zip(
        X.columns,
        linear_model.coef_
):
    print(f"{feature:20} : {coefficient:.4f}")

print("\n")
print("=" * 60)
print("RIDGE REGRESSION COEFFICIENTS")
print("=" * 60)

print("Intercept :", round(ridge_model.intercept_,4))

for feature, coefficient in zip(
        X.columns,
        ridge_model.coef_
):
    print(f"{feature:20} : {coefficient:.4f}")

# ==========================================================
# Evaluation Function
# ==========================================================

def evaluate_model(model_name, y_actual, y_prediction):

    mae = mean_absolute_error(y_actual, y_prediction)

    mse = mean_squared_error(y_actual, y_prediction)

    rmse = mse ** 0.5

    r2 = r2_score(y_actual, y_prediction)

    print("\n")
    print("=" * 60)
    print(model_name)
    print("=" * 60)

    print(f"MAE  : {mae:.2f}")
    print(f"MSE  : {mse:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

# ==========================================================
# Evaluate Models
# ==========================================================

evaluate_model(
    "Linear Regression",
    y_test,
    linear_prediction
)

evaluate_model(
    "Ridge Regression",
    y_test,
    ridge_prediction
)

# ==========================================================
# Compare Predictions
# ==========================================================

comparison = pd.DataFrame({

    "Actual Grade": y_test.values,

    "Linear Prediction":
        linear_prediction.round(2),

    "Ridge Prediction":
        ridge_prediction.round(2)

})

print("\n")
print("=" * 60)
print("Prediction Comparison")
print("=" * 60)

print(comparison.head(10))

# ==========================================================
# Scatter Plot
# ==========================================================

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    linear_prediction,
    label="Linear Regression"
)

plt.scatter(
    y_test,
    ridge_prediction,
    label="Ridge Regression"
)

plt.xlabel("Actual Grade")

plt.ylabel("Predicted Grade")

plt.title("Linear Regression vs Ridge Regression")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()

# ==========================================================
# Alpha Comparison
# ==========================================================

alphas = [
    0,
    0.1,
    1,
    10,
    100
]

r2_scores = []

for alpha in alphas:

    model = Ridge(alpha=alpha)

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    score = r2_score(y_test, prediction)

    r2_scores.append(score)

plt.figure(figsize=(8,5))

plt.plot(
    alphas,
    r2_scores,
    marker="o",
    linewidth=2
)

plt.xlabel("Alpha")

plt.ylabel("R² Score")

plt.title("Effect of Alpha on Ridge Regression")

plt.grid(True)

plt.tight_layout()

plt.show()

# ==========================================================
# Coefficient Comparison
# ==========================================================

comparison_df = pd.DataFrame({

    "Feature": X.columns,

    "Linear Regression": linear_model.coef_,

    "Ridge Regression": ridge_model.coef_

})

print("\n")
print("=" * 60)
print("Coefficient Comparison")
print("=" * 60)

print(comparison_df)

comparison_df.set_index("Feature").plot(
    kind="bar",
    figsize=(8,5)
)

plt.title("Linear vs Ridge Coefficients")

plt.ylabel("Coefficient Value")

plt.grid(axis="y")

plt.tight_layout()

plt.show()

print("\n")
print("=" * 60)
print("Program Finished Successfully")
print("=" * 60)