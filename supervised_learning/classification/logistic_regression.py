"""
==========================================================================
                         LOGISTIC REGRESSION
==========================================================================

Theory
------
Logistic Regression is a CLASSIFICATION algorithm despite its name.

It predicts the PROBABILITY that an input belongs to a particular class.

Example
    Given a student's study hours, attendance, and sleep hours:
    → Predict : Will the student PASS or FAIL?

------------------------------------------------------------------

WHY NOT LINEAR REGRESSION FOR CLASSIFICATION?

Linear Regression predicts continuous values like 2.5 or -1.3.
For classification we need values between 0 and 1 (probabilities).
This is where the SIGMOID FUNCTION comes in.

------------------------------------------------------------------

THE SIGMOID FUNCTION

                1
    σ(z)  =  --------
               1 + e⁻ᶻ

Where:
    z  =  β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ

Properties
    • Output is always between 0 and 1
    • S-shaped curve
    • σ(0) = 0.5  →  Decision boundary
    • z → +∞      →  σ(z) → 1
    • z → -∞      →  σ(z) → 0

------------------------------------------------------------------

DECISION RULE

    If P(y=1 | X) ≥ 0.5  →  Predict : PASS  (Class 1)
    If P(y=1 | X)  < 0.5  →  Predict : FAIL  (Class 0)

------------------------------------------------------------------

COST FUNCTION — LOG LOSS (Binary Cross Entropy)

    J(W) = -1/m Σ [ yᵢ log(ŷᵢ) + (1 - yᵢ) log(1 - ŷᵢ) ]

    When actual y = 1 and ŷ → 1  →  cost → 0   (correct)
    When actual y = 1 and ŷ → 0  →  cost → ∞   (wrong)

------------------------------------------------------------------

EVALUATION METRICS

    True Positive  (TP) : Predicted Pass,  Actually Pass
    True Negative  (TN) : Predicted Fail,  Actually Fail
    False Positive (FP) : Predicted Pass,  Actually Fail  ← Type I Error
    False Negative (FN) : Predicted Fail,  Actually Pass  ← Type II Error

    Accuracy  = (TP + TN) / (TP + TN + FP + FN)
    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)
    F1 Score  = 2 × (Precision × Recall) / (Precision + Recall)
    AUC-ROC   : Area Under Curve. Closer to 1.0 = better model.

Author : Alfred Joseph T
==========================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
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

print("\nColumns")
print(list(data.columns))

print("\nClass Distribution")
print(data["Pass_Fail"].value_counts())

# ==========================================================
# Label Encoding
# ==========================================================

encoder = LabelEncoder()

data["Pass_Fail"] = encoder.fit_transform(data["Pass_Fail"])

print("\nLabel Mapping")
print(dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))

# ==========================================================
# Select Features and Target
# ==========================================================

X = data[[
    "Hours_Studied",
    "Attendance",
    "Previous_Grade",
    "Assignments_Completed",
    "Sleep_Hours",
    "Study_Sessions_Per_Week",
    "Internet_Usage_Hours",
    "Sports_Hours",
    "Social_Media_Hours"
]]

y = data["Pass_Fail"]

# ==========================================================
# Train Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ==========================================================
# Standard Scaling
# ==========================================================

"""
StandardScaler transforms each feature to have:
    Mean = 0
    Std  = 1

Formula : z = (x - μ) / σ

Why scale?
    Logistic Regression uses gradient-based optimization.
    Unscaled features cause slow or unstable convergence.

Important : Fit ONLY on training data. Transform both sets.
            This prevents data leakage from the test set.
"""

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ==========================================================
# Logistic Regression Model
# ==========================================================

model = LogisticRegression(max_iter=1000, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ==========================================================
# Display Coefficients
# ==========================================================

print("\n")
print("=" * 60)
print("LOGISTIC REGRESSION COEFFICIENTS")
print("=" * 60)

print("Intercept :", round(model.intercept_[0], 4))

for feature, coefficient in zip(X.columns, model.coef_[0]):
    print(f"{feature:28} : {coefficient:.4f}")

# ==========================================================
# Evaluation
# ==========================================================

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
auc       = roc_auc_score(y_test, y_prob)

print("\n")
print("=" * 60)
print("EVALUATION METRICS")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"AUC-ROC   : {auc:.4f}")

print("\n")
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# ==========================================================
# Prediction Comparison
# ==========================================================

comparison = pd.DataFrame({
    "Actual":            encoder.inverse_transform(y_test),
    "Predicted":         encoder.inverse_transform(y_pred),
    "Pass Probability %": (y_prob * 100).round(2)
})

print("\n")
print("=" * 60)
print("Prediction Comparison")
print("=" * 60)

print(comparison.head(10))

# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(y_test, y_pred)

ConfusionMatrixDisplay(cm, display_labels=encoder.classes_).plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.tight_layout()

plt.show()

# ==========================================================
# ROC Curve
# ==========================================================

fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(7, 5))

plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.4f}")

plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()

# ==========================================================
# Actual vs Predicted (Scatter)
# ==========================================================

plt.figure(figsize=(7, 5))

plt.scatter(y_test, y_pred, alpha=0.5)

plt.xlabel("Actual")

plt.ylabel("Predicted")

plt.title("Actual vs Predicted")

plt.grid(True)

plt.tight_layout()

plt.show()

# ==========================================================
# Coefficient Bar Chart
# ==========================================================

plt.figure(figsize=(9, 5))

plt.bar(X.columns, model.coef_[0])

plt.xlabel("Feature")

plt.ylabel("Coefficient Value")

plt.title("Feature Coefficients")

plt.xticks(rotation=35, ha="right")

plt.grid(axis="y")

plt.tight_layout()

plt.show()

# ==========================================================
# Probability Distribution
# ==========================================================

labels = encoder.inverse_transform(y_test)

for cls, color in zip(encoder.classes_, ["crimson", "steelblue"]):
    plt.hist(y_prob[labels == cls], bins=25, alpha=0.6, color=color, label=f"Actual: {cls}")

plt.axvline(0.5, color="black", linestyle="--", linewidth=1.5, label="Threshold = 0.5")

plt.xlabel("Predicted Probability of PASS")

plt.ylabel("Count")

plt.title("Predicted Probability Distribution")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()

# ==========================================================
# Conclusion
# ==========================================================

print("\n")
print("=" * 60)
print("CONCLUSION")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"AUC-ROC   : {auc:.4f}")

print("""
Key Takeaways
  - Logistic Regression is simple, interpretable, and fast.
  - StandardScaler is essential for stable convergence.
  - AUC-ROC is more informative than accuracy alone.
  - Coefficients reveal which features influence predictions most.

Applications
  Medical Diagnosis  |  Spam Detection  |  Credit Risk
  Student Performance  |  Fraud Detection  |  Churn Prediction
""")

print("=" * 60)
print("Program Finished Successfully")
print("=" * 60)