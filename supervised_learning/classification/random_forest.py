"""
==========================================================================
                    RANDOM FOREST CLASSIFICATION
==========================================================================

Theory
------
Random Forest is an ENSEMBLE LEARNING algorithm that builds MANY
Decision Trees and combines their predictions using MAJORITY VOTING.
It is essentially Bagging with one extra twist: RANDOM FEATURE
SELECTION at every split.

------------------------------------------------------------------

RECAP : BAGGING

Bagging trains multiple Decision Trees, each on a different
bootstrap sample (random rows, sampled with replacement) of the
training data, then combines their predictions by majority vote.

This reduces variance, but there is a problem: if one or two
features are very strong predictors (e.g. Hours_Studied), almost
EVERY tree in the ensemble will choose to split on that same
feature first. This makes the trees highly correlated with each
other, which limits how much bagging alone can reduce variance.

------------------------------------------------------------------

WHAT RANDOM FOREST ADDS : RANDOM FEATURE SELECTION

In addition to bootstrap sampling rows, Random Forest also restricts
each split to a RANDOM SUBSET of features, not all of them.

    max_features = "sqrt"
        → At each split, only √(total features) are considered,
          chosen randomly.

Example
    9 total features → √9 = 3 features considered per split

This forces trees to consider different features at different
splits, making the trees more DIVERSE (less correlated). Diverse
trees make independent errors, so averaging/voting across them
cancels out more noise than bagging alone.

------------------------------------------------------------------

HOW RANDOM FOREST WORKS (STEP BY STEP)

    1. Create N bootstrap samples from the training data.
    2. For each sample, grow a Decision Tree:
         - At every split, randomly select a subset of features.
         - Pick the best split only from that subset (Gini/Entropy).
    3. Repeat for all N trees, trained independently and in parallel.
    4. For a new prediction:
         - Each tree votes for a class.
         - CLASSIFICATION → majority vote wins.
         - (Regression would average the predictions instead.)

------------------------------------------------------------------

BAGGING vs RANDOM FOREST

    Bagging        : Random ROWS only (bootstrap sampling),
                     considers ALL features at every split
    Random Forest  : Random ROWS (bootstrap sampling) AND
                     random FEATURES at every split

This extra randomness is why Random Forest usually outperforms
plain Bagging — the trees are more diverse, so their combined
vote is more reliable.

------------------------------------------------------------------

FEATURE IMPORTANCE

Random Forest can rank how useful each feature was across all
trees, based on how much each feature reduced impurity (Gini)
on average whenever it was used for a split.

    Higher importance → feature was a stronger, more frequent
                         predictor across the forest

This is one of Random Forest's biggest practical advantages:
it explains WHICH factors (e.g. Hours_Studied, Attendance) matter
most for predicting Pass/Fail, even though individual trees are
no longer easy to visualize.

------------------------------------------------------------------

ADVANTAGES

    - Usually more accurate than a single tree or plain bagging
    - Reduces overfitting through both row and feature randomness
    - Handles non-linear relationships well
    - Provides a built-in feature importance ranking
    - Robust to noisy data and outliers

DISADVANTAGES

    - Slower to train than a single Decision Tree
    - Higher memory usage (stores many trees)
    - Loses the simple interpretability of one decision tree
    - Many hyperparameters to tune (n_estimators, max_depth,
      max_features, etc.)

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
from sklearn.ensemble import RandomForestClassifier
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
# Single Decision Tree (Comparison)
# ==========================================================

tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=42
)

tree.fit(X_train, y_train)
tree_pred = tree.predict(X_test)

# ==========================================================
# Random Forest Model
# ==========================================================

"""
n_estimators : number of trees in the forest
max_depth    : limits how deep each tree can grow
max_features : number of features considered at each split
               ("sqrt" → √(total features), adds randomness)
"""

forest = RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    max_depth=4,
    max_features="sqrt",
    random_state=42
)

forest.fit(X_train, y_train)

y_pred = forest.predict(X_test)

# ==========================================================
# Evaluation
# ==========================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n" + "=" * 60)
print("RANDOM FOREST RESULTS")
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
# Comparison
# ==========================================================

tree_acc = accuracy_score(y_test, tree_pred)

print("\nComparison")
print("-" * 40)
print(f"Decision Tree Accuracy : {tree_acc:.4f}")
print(f"Random Forest Accuracy : {accuracy:.4f}")

# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(y_test, y_pred)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=encoder.classes_
).plot(cmap="Blues")

plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.show()

# ==========================================================
# Feature Importance
# ==========================================================

importance = pd.Series(
    forest.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance")
print(importance)

plt.figure(figsize=(9, 5))
plt.bar(importance.index, importance.values)
plt.xticks(rotation=35, ha="right")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importance")
plt.grid(axis="y")
plt.tight_layout()
plt.show()

# ==========================================================
# Prediction Comparison
# ==========================================================

comparison = pd.DataFrame({
    "Actual": encoder.inverse_transform(y_test),
    "Decision Tree": encoder.inverse_transform(tree_pred),
    "Random Forest": encoder.inverse_transform(y_pred)
})

print("\nPrediction Comparison")
print(comparison.head(10))

# ==========================================================
# Conclusion
# ==========================================================

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)

print(f"Decision Tree Accuracy : {tree_acc:.4f}")
print(f"Random Forest Accuracy : {accuracy:.4f}")

print("""
Advantages
----------
* High accuracy
* Reduces overfitting
* Handles nonlinear data
* Provides feature importance
* Robust to noise

Disadvantages
-------------
* Slower than a single tree
* More memory usage
* Less interpretable

Applications
------------
* Fraud Detection
* Medical Diagnosis
* Credit Scoring
* Student Performance Prediction
* Customer Churn

Note
----
If precision/recall for the minority class is near 0 despite high
accuracy, this signals class imbalance — check the class distribution
above and consider class_weight="balanced" or resampling.
""")

print("=" * 60)
print("Program Finished Successfully")
print("=" * 60)