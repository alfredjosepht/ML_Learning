"""
==========================================================================
                        BAGGING CLASSIFICATION
==========================================================================

Theory
------
Bagging (Bootstrap Aggregating) is an ENSEMBLE LEARNING technique.
Instead of relying on a single model, it trains MANY models on
different random subsets of the training data and combines their
predictions to produce a more stable, accurate result.

------------------------------------------------------------------

WHY ENSEMBLE LEARNING?

A single Decision Tree is prone to OVERFITTING — it can memorize
noise in the training data, making it unstable. A small change in
the training set can produce a completely different tree.

Ensemble methods fix this by combining many "weak" or unstable
models into one strong model. The idea: individual errors cancel
out when predictions are averaged or voted on.

------------------------------------------------------------------

BOOTSTRAP SAMPLING

"Bootstrap" means sampling the training data WITH REPLACEMENT to
create multiple new datasets of the same size as the original.

Example
    Original dataset : [1, 2, 3, 4, 5]
    Bootstrap sample 1 : [2, 2, 4, 5, 1]
    Bootstrap sample 2 : [1, 3, 3, 5, 5]
    Bootstrap sample 3 : [4, 1, 2, 2, 3]

Because sampling is with replacement, some rows appear multiple
times in a sample while others are left out entirely. On average,
each bootstrap sample contains about 63% of the unique original
rows — the remaining ~37% are called "out-of-bag" (OOB) samples,
which can be used for free validation.

------------------------------------------------------------------

HOW BAGGING WORKS (STEP BY STEP)

    1. Create N bootstrap samples from the training data.
    2. Train one model (e.g. a Decision Tree) on each sample,
       independently and in parallel.
    3. For a new prediction:
         - Each of the N models makes its own prediction.
         - CLASSIFICATION → take a MAJORITY VOTE across all models.
         - (Regression would instead average the predictions.)

Example with 5 trees predicting Pass/Fail for one student:

    Tree 1 → Pass
    Tree 2 → Pass
    Tree 3 → Fail
    Tree 4 → Pass
    Tree 5 → Pass

    Majority Vote → PASS  (4 out of 5 trees agree)

------------------------------------------------------------------

WHY DOES THIS REDUCE OVERFITTING?

A single Decision Tree has HIGH VARIANCE — it reacts strongly to
small changes in data. Averaging many trees trained on different
bootstrap samples smooths out this variance without increasing
bias much, leading to a model that generalizes better to unseen
data.

------------------------------------------------------------------

BAGGING vs A SINGLE DECISION TREE

    Single Tree  : Fast, interpretable, but unstable (high variance)
    Bagging      : Slower, less interpretable, but more stable and
                   usually more accurate

------------------------------------------------------------------

A NOTE ON ACCURACY WITH IMBALANCED DATA

If one class (e.g. "Pass") makes up the vast majority of the
dataset, a model can reach high accuracy simply by predicting that
majority class almost every time — without ever learning to detect
the minority class ("Fail").

This is why Precision, Recall, and F1 Score (especially per-class,
as shown in the Classification Report) must always be checked
alongside Accuracy. High accuracy with 0.00 precision/recall on
the minority class is a strong signal of class imbalance, not a
good model.

------------------------------------------------------------------

ADVANTAGES

    - Reduces overfitting (lower variance)
    - Improves stability and accuracy
    - Trees can be trained in parallel
    - Works well with high-variance base models like Decision Trees

DISADVANTAGES

    - Higher memory and computation cost (many models instead of one)
    - Slower to train and predict than a single model
    - Loses the easy interpretability of a single Decision Tree
    - Does not fix BIAS — only reduces VARIANCE, so a poor base
      model will still bag into a poor ensemble

APPLICATIONS

    Fraud Detection | Medical Diagnosis | Credit Scoring |
    Student Performance Prediction | Customer Churn

Author : Alfred Joseph T
==========================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
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
# Features and Target
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
# Single Decision Tree
# ==========================================================

tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=42
)

tree.fit(X_train, y_train)
tree_pred = tree.predict(X_test)

# ==========================================================
# Bagging Classifier
# ==========================================================

"""
estimator    : the base model to bag (Decision Tree here)
n_estimators : number of bootstrap samples / trees to train
"""

bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=4),
    n_estimators=50,
    random_state=42
)

bagging.fit(X_train, y_train)

y_pred = bagging.predict(X_test)

# ==========================================================
# Evaluation
# ==========================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n" + "=" * 60)
print("BAGGING RESULTS")
print("=" * 60)
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nClassification Report")
print(classification_report(
    y_test,
    y_pred,
    target_names=encoder.classes_,
    zero_division=0
))

# ==========================================================
# Comparison with Single Decision Tree
# ==========================================================

tree_accuracy = accuracy_score(y_test, tree_pred)

print("\nComparison")
print("-" * 40)
print(f"Decision Tree Accuracy : {tree_accuracy:.4f}")
print(f"Bagging Accuracy       : {accuracy:.4f}")

# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(y_test, y_pred)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=encoder.classes_
).plot(cmap="Blues")

plt.title("Bagging Confusion Matrix")
plt.tight_layout()
plt.show()

# ==========================================================
# Feature Importance (Base Tree)
# ==========================================================

plt.figure(figsize=(9, 5))

plt.bar(
    X.columns,
    tree.feature_importances_
)

plt.xticks(rotation=35, ha="right")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Decision Tree Feature Importance")
plt.grid(axis="y")
plt.tight_layout()
plt.show()

# ==========================================================
# Prediction Comparison
# ==========================================================

comparison = pd.DataFrame({
    "Actual": encoder.inverse_transform(y_test),
    "Decision Tree": encoder.inverse_transform(tree_pred),
    "Bagging": encoder.inverse_transform(y_pred)
})

print("\nPrediction Comparison")
print(comparison.head(10))

# ==========================================================
# Conclusion
# ==========================================================

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)

print(f"Decision Tree Accuracy : {tree_accuracy:.4f}")
print(f"Bagging Accuracy       : {accuracy:.4f}")

print("""
Advantages
----------
* Reduces overfitting
* Improves stability
* Better generalization
* Parallel training

Disadvantages
-------------
* Higher memory usage
* Slower than a single model
* Less interpretable
* Does not fix bias, only reduces variance

Applications
------------
* Fraud Detection
* Medical Diagnosis
* Credit Scoring
* Student Performance Prediction
* Customer Churn

Note
----
If precision/recall for the minority class (e.g. Fail) is near 0
despite high accuracy, this signals class imbalance — the model is
likely defaulting to the majority class. Check the class distribution
and consider techniques like class_weight="balanced" or resampling.
""")

print("=" * 60)
print("Program Finished Successfully")
print("=" * 60)