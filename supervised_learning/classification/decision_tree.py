
"""
==========================================================================
                      DECISION TREE CLASSIFICATION
==========================================================================

Theory
------
A Decision Tree is a CLASSIFICATION (and regression) algorithm that
splits data into branches based on feature values, forming a tree
of yes/no decisions until it reaches a final prediction.

Example
    Is Hours_Studied > 5?
        Yes → Is Attendance > 75%?
                Yes → PASS
                No  → FAIL
        No  → FAIL

It mimics how a human would reason through a decision step by step.

------------------------------------------------------------------

HOW DOES THE TREE DECIDE WHERE TO SPLIT?

At every node, the tree tries every feature and every possible
threshold, and picks the split that makes the resulting groups as
"pure" as possible (i.e. mostly one class).

Purity is measured using ENTROPY or GINI INDEX.

------------------------------------------------------------------

ENTROPY

Entropy measures the amount of disorder / impurity in a node.

    Entropy(S) = - Σ pᵢ log₂(pᵢ)

Where:
    pᵢ = proportion of samples belonging to class i

    Entropy = 0     →  Node is pure (only one class present)
    Entropy = 1     →  Node is maximally impure (50/50 split)

Example
    10 students, 5 Pass, 5 Fail
    Entropy = -(0.5 log₂0.5 + 0.5 log₂0.5) = 1.0   (maximum impurity)

    10 students, 10 Pass, 0 Fail
    Entropy = 0.0   (completely pure)

------------------------------------------------------------------

INFORMATION GAIN

Information Gain measures how much Entropy decreases after a split.
The tree picks the split with the HIGHEST Information Gain.

    Gain(S, A) = Entropy(S) - Σ (|Sᵥ| / |S|) × Entropy(Sᵥ)

Where:
    S  = parent node before split
    Sᵥ = child node after split on feature A

Higher Gain  →  Better split  →  Tree becomes more confident faster

------------------------------------------------------------------

GINI INDEX

Gini Index is an alternative impurity measure (used by Scikit-Learn
by default). It is faster to compute than Entropy and usually gives
similar results.

    Gini(S) = 1 - Σ pᵢ²

    Gini = 0     →  Node is pure
    Gini = 0.5   →  Node is maximally impure (binary case)

Example
    10 students, 5 Pass, 5 Fail
    Gini = 1 - (0.5² + 0.5²) = 0.5   (maximum impurity)

    10 students, 10 Pass, 0 Fail
    Gini = 1 - (1² + 0²) = 0.0   (completely pure)

------------------------------------------------------------------

HOW THE TREE GROWS

    1. Start with all data at the root node.
    2. Try every feature / threshold combination.
    3. Pick the split with the lowest Gini (or highest Information Gain).
    4. Repeat for each child node.
    5. Stop when a stopping condition is met:
         - Maximum depth reached
         - Minimum samples per split reached
         - Node is already pure

------------------------------------------------------------------

ADVANTAGES

    - Easy to understand and visualize (white-box model)
    - No feature scaling required
    - Handles both numerical and categorical data
    - Captures non-linear relationships

DISADVANTAGES

    - Prone to OVERFITTING if grown too deep
    - Small changes in data can produce a very different tree
    - Can create biased trees if some classes dominate

APPLICATIONS

    Loan approval | Medical diagnosis | Customer churn |
    Fraud detection | Student performance prediction

Author : Alfred Joseph T
==========================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
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
# Decision Tree Classifier
# ==========================================================

"""
criterion        : "gini" or "entropy" — impurity measure used for splits
max_depth        : limits how deep the tree can grow (prevents overfitting)
min_samples_split: minimum samples required to split a node
"""

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    min_samples_split=5,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ==========================================================
# Evaluation
# ==========================================================

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

print("\n")
print("=" * 60)
print("EVALUATION METRICS")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\n")
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# ==========================================================
# Prediction Comparison
# ==========================================================

comparison = pd.DataFrame({
    "Actual":    encoder.inverse_transform(y_test),
    "Predicted": encoder.inverse_transform(y_pred)
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
# Tree Visualization
# ==========================================================

plt.figure(figsize=(16, 8))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=encoder.classes_,
    filled=True
)

plt.title("Decision Tree Structure")

plt.show()

# ==========================================================
# Feature Importance
# ==========================================================

plt.figure(figsize=(9, 5))

plt.bar(X.columns, model.feature_importances_)

plt.xlabel("Feature")

plt.ylabel("Importance")

plt.title("Feature Importance")

plt.xticks(rotation=35, ha="right")

plt.grid(axis="y")

plt.tight_layout()

plt.show()

# ==========================================================
# Actual vs Predicted
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

print("""
Advantages
  - Easy to understand and visualize
  - No feature scaling required
  - Handles non-linear relationships

Disadvantages
  - Prone to overfitting if too deep
  - Sensitive to small changes in data

Applications
  Loan Approval | Medical Diagnosis | Customer Churn |
  Fraud Detection | Student Performance Prediction
""")

print("=" * 60)
print("Program Finished Successfully")
print("=" * 60)
