"""
==========================================================================
                    ADABOOST CLASSIFICATION
==========================================================================

Theory
------
AdaBoost (Adaptive Boosting) is an ENSEMBLE LEARNING algorithm that
combines many WEAK LEARNERS (usually Decision Stumps) into one
STRONG classifier. Unlike Bagging and Random Forest, which train
trees independently and in parallel, AdaBoost trains models
SEQUENTIALLY — each new model learns from the mistakes of the
previous ones.

------------------------------------------------------------------

WHAT IS A WEAK LEARNER?

A weak learner is a model that performs only slightly better than
random guessing. AdaBoost typically uses DECISION STUMPS — Decision
Trees with max_depth=1, meaning they make just ONE split.

Example
    A stump might only check: "Is Hours_Studied > 5?"
    That single rule alone is a weak predictor on its own.

AdaBoost's idea: combine MANY weak stumps, each focusing on
different parts of the problem, into one strong, accurate model.

------------------------------------------------------------------

BAGGING/RANDOM FOREST vs ADABOOST

    Bagging / Random Forest : Trees trained INDEPENDENTLY and in
                              PARALLEL on random samples; combined
                              by majority vote. Reduces VARIANCE.

    AdaBoost                : Models trained SEQUENTIALLY, each one
                              correcting the errors of the one
                              before it; combined by a WEIGHTED vote.
                              Reduces BIAS.

------------------------------------------------------------------

HOW ADABOOST WORKS (STEP BY STEP)

    1. Assign EQUAL WEIGHT to every training sample.
       (e.g. 100 samples → each starts with weight 1/100)

    2. Train a weak learner (stump) on the weighted data.

    3. Calculate the learner's ERROR RATE — how many samples it
       misclassified, weighted by their current importance.

    4. Calculate the learner's SAY (its vote weight in the final
       model) based on its accuracy:

           say = 0.5 × ln( (1 - error) / error )

           Lower error  →  Higher say  →  Trusted more in final vote
           Higher error →  Lower say   →  Trusted less in final vote

    5. UPDATE SAMPLE WEIGHTS:
         - Misclassified samples get their weight INCREASED
         - Correctly classified samples get their weight DECREASED

       This forces the NEXT weak learner to pay more attention to
       the samples that were previously gotten wrong.

    6. Repeat steps 2–5 for n_estimators rounds, building a chain
       of stumps, each specializing in the mistakes of the last.

    7. FINAL PREDICTION : combine all stumps using a WEIGHTED vote,
       where each stump's vote is scaled by its "say" value.

------------------------------------------------------------------

WORKED EXAMPLE (SIMPLIFIED)

    Round 1
        Stump 1 trained on equally-weighted data.
        Gets 3 students wrong → those 3 students' weights increase.

    Round 2
        Stump 2 trained on the reweighted data — it is now forced
        to focus on those 3 difficult students.
        Gets a different 2 students wrong → their weights increase.

    Round 3
        Stump 3 focuses on the new difficult cases...

    Final Prediction
        For a new student, every stump votes Pass/Fail.
        Each vote is weighted by that stump's "say".
        The class with the highest total weighted vote wins.

------------------------------------------------------------------

LEARNING RATE

The learning_rate parameter shrinks the contribution of each
stump's "say" before adding it to the final weighted vote.

    Higher learning rate → each stump has more influence,
                            converges faster, but risks overfitting
    Lower learning rate  → each stump has less influence,
                            needs more estimators, but is more robust

------------------------------------------------------------------

ADVANTAGES

    - Converts weak learners into a strong, accurate classifier
    - Automatically focuses on hard-to-classify samples
    - Often achieves high accuracy with simple base models
    - Less prone to overfitting than a single deep tree (in
      moderate amounts)

DISADVANTAGES

    - Sensitive to noisy data and outliers (their weights keep
      increasing if the model can never classify them correctly)
    - Sequential training cannot be parallelized like Bagging/
      Random Forest, so it is typically slower to train
    - Can overfit if n_estimators is too high

APPLICATIONS

    Fraud Detection | Medical Diagnosis | Customer Churn |
    Credit Scoring | Student Performance Prediction

Author : Alfred Joseph T
==========================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
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

print("\nClass Distribution")
print(data["Pass_Fail"].value_counts())

# ==========================================================
# Label Encoding
# ==========================================================

encoder = LabelEncoder()
data["Pass_Fail"] = encoder.fit_transform(data["Pass_Fail"])

print("\nLabel Mapping")
print(dict(zip(encoder.classes_,
               encoder.transform(encoder.classes_))))

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
# Decision Stump (Weak Learner)
# ==========================================================

stump = DecisionTreeClassifier(
    max_depth=1,
    random_state=42
)

# ==========================================================
# AdaBoost Model
# ==========================================================

"""
estimator     : the weak learner to boost (a Decision Stump here)
n_estimators  : number of sequential boosting rounds
learning_rate : shrinks each stump's contribution to the final vote
"""

model = AdaBoostClassifier(
    estimator=stump,
    n_estimators=100,
    learning_rate=1.0,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ==========================================================
# Evaluation
# ==========================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n" + "=" * 60)
print("ADABOOST RESULTS")
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
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(y_test, y_pred)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=encoder.classes_
).plot(cmap="Blues")

plt.title("AdaBoost Confusion Matrix")
plt.tight_layout()
plt.show()

# ==========================================================
# Feature Importance
# ==========================================================

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance")
print(importance)

plt.figure(figsize=(9, 5))
plt.bar(importance.index, importance.values)
plt.xticks(rotation=35, ha="right")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("AdaBoost Feature Importance")
plt.grid(axis="y")
plt.tight_layout()
plt.show()

# ==========================================================
# Prediction Comparison
# ==========================================================

comparison = pd.DataFrame({
    "Actual": encoder.inverse_transform(y_test),
    "Predicted": encoder.inverse_transform(y_pred)
})

print("\nPrediction Comparison")
print(comparison.head(10))

# ==========================================================
# Conclusion
# ==========================================================

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("""
Advantages
----------
* Improves weak learners
* Focuses on difficult samples
* High predictive accuracy
* Simple ensemble algorithm

Disadvantages
-------------
* Sensitive to noisy data
* Sequential training
* Can overfit with excessive estimators

Applications
------------
* Fraud Detection
* Medical Diagnosis
* Customer Churn
* Credit Scoring
* Student Performance Prediction

Note
----
If precision/recall for the minority class is near 0 despite high
accuracy, this signals class imbalance — check the class distribution
above and consider class_weight="balanced" on the base estimator or
resampling the data.
""")

print("=" * 60)
print("Program Finished Successfully")
print("=" * 60)