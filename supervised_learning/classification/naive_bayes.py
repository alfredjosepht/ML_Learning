"""
==========================================================================
                    NAIVE BAYES CLASSIFICATION
==========================================================================

Theory
------
Naive Bayes is a supervised classification algorithm based on
BAYES' THEOREM. It calculates the probability of each class given
the input features, and predicts whichever class has the highest
probability.

------------------------------------------------------------------

BAYES' THEOREM

    P(y | X) = [ P(X | y) × P(y) ] / P(X)

Where:
    P(y | X)  = Posterior probability — probability of class y
                given the features X (what we want to find)
    P(X | y)  = Likelihood — probability of observing features X
                given that the class is y
    P(y)      = Prior probability — how common class y is overall
    P(X)      = Evidence — overall probability of observing X
                (same for every class, so it can be ignored when
                comparing classes)

In short:

    Posterior  ∝  Likelihood × Prior

Example
    P(Pass | Hours_Studied=6, Attendance=90%)
        = P(Hours_Studied=6, Attendance=90% | Pass) × P(Pass)
          ---------------------------------------------------
                  P(Hours_Studied=6, Attendance=90%)

------------------------------------------------------------------

WHY "NAIVE"?

Naive Bayes assumes that ALL features are CONDITIONALLY INDEPENDENT
given the class. This means it assumes Hours_Studied, Attendance,
Sleep_Hours, etc. have no relationship with each other — only with
the target class.

This assumption is rarely true in real life (study hours and sleep
hours likely affect each other), which is why the algorithm is
called "naive". Despite this simplification, it performs surprisingly
well in practice.

Because of this assumption, the formula simplifies to:

    P(y | x₁, x₂, ..., xₙ)  ∝  P(y) × P(x₁|y) × P(x₂|y) × ... × P(xₙ|y)

Instead of modeling complex relationships between features, it just
multiplies each feature's individual probability together.

------------------------------------------------------------------

WHY GAUSSIAN NAIVE BAYES?

There are different versions of Naive Bayes depending on the type
of data:

    GaussianNB    : Continuous numerical features (assumes Normal
                    / Bell-curve distribution) — used here
    MultinomialNB : Count-based data (e.g. word frequencies in text)
    BernoulliNB   : Binary / yes-no features

Since our features (Hours_Studied, Attendance, Sleep_Hours, etc.)
are continuous numbers, Gaussian Naive Bayes is the right choice.

It assumes each feature follows a Normal Distribution within each
class, calculated using the Gaussian Probability Density Function:

                    1                  (x - μ)²
    P(x|y) =  ------------- × exp( - ----------- )
               √(2π σ²y)                2σ²y

Where:
    μy  = mean of the feature for class y
    σ²y = variance of the feature for class y

This means the model just needs to learn the mean and variance of
each feature, separately for each class — no complex training
required, which is why Naive Bayes is extremely fast.

------------------------------------------------------------------

HOW PREDICTION WORKS

    1. For a new student, calculate P(Pass) × P(Hours|Pass) ×
       P(Attendance|Pass) × ... for every feature.
    2. Do the same for P(Fail) × P(Hours|Fail) × ...
    3. Whichever total probability is HIGHER becomes the prediction.

------------------------------------------------------------------

ADVANTAGES

    - Very fast to train and predict, even on large datasets
    - Works well with small training data
    - Performs well despite its simplistic independence assumption
    - Naturally handles multi-class problems

DISADVANTAGES

    - Assumes features are independent, which is rarely true
    - Performance may drop when features are highly correlated
    - Gaussian assumption may not fit features that are not
      normally distributed

APPLICATIONS

    Spam Detection | Medical Diagnosis | Sentiment Analysis |
    Document Classification | Student Performance Prediction

Author : Alfred Joseph T
==========================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
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
# Feature Scaling
# ==========================================================

"""
StandardScaler is not strictly required for Naive Bayes since it
relies on per-feature mean/variance rather than distances, but
scaling still keeps features on a comparable footing and is good
practice.
"""

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================================
# Gaussian Naive Bayes Model
# ==========================================================

model = GaussianNB()

model.fit(X_train, y_train)

# ==========================================================
# Prediction
# ==========================================================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ==========================================================
# Evaluation
# ==========================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n")
print("=" * 60)
print("EVALUATION METRICS")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"AUC-ROC   : {auc:.4f}")

print("\nClassification Report")
print(classification_report(
    y_test,
    y_pred,
    target_names=encoder.classes_
))

# ==========================================================
# Prediction Comparison
# ==========================================================

comparison = pd.DataFrame({
    "Actual": encoder.inverse_transform(y_test),
    "Predicted": encoder.inverse_transform(y_pred),
    "Pass Probability (%)": (y_prob * 100).round(2)
})

print("\nPrediction Comparison")
print(comparison.head(10))

# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(y_test, y_pred)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=encoder.classes_
).plot(cmap="Blues")

plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# ==========================================================
# ROC Curve
# ==========================================================

fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(7, 5))

plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], "k--", label="Random")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ==========================================================
# Feature Probability Plot
# ==========================================================

plt.figure(figsize=(7, 5))

for cls, color in zip(encoder.classes_, ["crimson", "steelblue"]):
    plt.hist(
        y_prob[encoder.inverse_transform(y_test) == cls],
        bins=15,
        alpha=0.6,
        label=cls,
        color=color
    )

plt.axvline(0.5, color="black", linestyle="--", label="Threshold")
plt.xlabel("Predicted Probability")
plt.ylabel("Count")
plt.title("Probability Distribution")
plt.legend()
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
Advantages
----------
* Very fast training
* Simple to implement
* Works well for many classification tasks

Disadvantages
-------------
* Assumes feature independence
* Performance may drop when features are highly correlated

Applications
------------
* Spam Detection
* Medical Diagnosis
* Sentiment Analysis
* Student Performance Prediction
""")

print("=" * 60)
print("Program Finished Successfully")
print("=" * 60)