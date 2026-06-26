import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =====================================
# Load Dataset
# =====================================

DATASET_PATH = "../../datasets/student_scores.csv"

data = pd.read_csv(DATASET_PATH)

print("=" * 50)
print("Student Dataset")
print("=" * 50)

print(data.head())

print("\nDataset Shape :", data.shape)
print("Columns :", list(data.columns))

# =====================================
# Select Features and Target
# =====================================

X = data[
    [
        "Hours_Studied",
        "Attendance",
        "Previous_Grade"
    ]
]

y = data["Final_Grade"]

# =====================================
# Split Dataset
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# =====================================
# Create Model
# =====================================

model = LinearRegression()

# =====================================
# Train Model
# =====================================

model.fit(X_train, y_train)

# =====================================
# Display Model Equation
# =====================================

print("\nIntercept")
print(model.intercept_)

print("\nCoefficients")

for feature, coefficient in zip(X.columns, model.coef_):
    print(f"{feature:20} : {coefficient:.4f}")

# =====================================
# Make Predictions
# =====================================

y_pred = model.predict(X_test)

# =====================================
# Evaluation Metrics
# =====================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 50)
print("Model Evaluation")
print("=" * 50)

print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

# =====================================
# Actual vs Predicted Table
# =====================================

results = pd.DataFrame({
    "Actual Grade": y_test.values,
    "Predicted Grade": y_pred.round(2)
})

print("\nSample Predictions")

print(results.head(10))

# =====================================
# Scatter Plot
# =====================================

plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Grade")
plt.ylabel("Predicted Grade")
plt.title("Actual vs Predicted Grades")

plt.grid(True)

plt.tight_layout()

plt.show()

# =====================================
# Line Plot
# =====================================

plt.figure(figsize=(10,5))

plt.plot(
    y_test.values,
    label="Actual",
    marker="o"
)

plt.plot(
    y_pred,
    label="Predicted",
    marker="x"
)

plt.xlabel("Students")

plt.ylabel("Final Grade")

plt.title("Actual vs Predicted")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()