# ML-Learning (Machine Learning Playground)

A curated playground directory for learning, implementing, and visualizing fundamental Machine Learning algorithms from scratch and using Scikit-Learn.

---

## 📂 Project Directory Structure

```text
ML-Learning/
├── datasets/
│   ├── student_scores.csv
│   └── student_performance_dataset.csv
└── supervised_learning/
    ├── classification/
    │   ├── bagging.py
    │   ├── decision_tree.py
    │   ├── logistic_regression.py
    │   └── naive_bayes.py
    └── regression/
        ├── gradient_descent.py
        ├── linear_regression.py
        ├── ridge_regression.py
        └── lasso_regression.py
```

---

## 📊 Datasets

The models in this repository are trained and evaluated on student academic datasets:
*   **`student_scores.csv`**: A simple dataset mapping hours studied to exam scores.
*   **`student_performance_dataset.csv`**: A rich multi-feature dataset (500 records) including features such as `Hours_Studied`, `Attendance`, `Previous_Grade`, `Assignments_Completed`, `Sleep_Hours`, `Study_Sessions_Per_Week`, and others, mapped to continuous labels (grades) and binary outcomes (pass/fail).

---

## 🛠 Implemented Algorithms & Modules

### 📈 Regression & Optimization

#### 1. Gradient Descent Optimizer (`regression/gradient_descent.py`)
Implements optimization algorithms from scratch using NumPy to train linear models:
*   **Batch Gradient Descent (BGD)**: Stable updates utilizing the full dataset.
*   **Stochastic Gradient Descent (SGD)**: Fast, stochastic, single-sample weight updates.
*   **Mini-Batch Gradient Descent (MBGD)**: Standard minibatch partitioning balancing accuracy and speed.
*   Includes sensitivity experiments comparing convergence behavior across different learning rates.

#### 2. Linear Regression (`regression/linear_regression.py`)
Ordinary Least Squares (OLS) regression mapping student studying metrics to their continuous grades. Includes standard evaluation metrics ($R^2$, MAE, MSE, RMSE) and predictive checks.

#### 3. Ridge Regression (`regression/ridge_regression.py`)
Regularized regression incorporating an $L_2$ penalty:
$$J(W) = \text{MSE} + \alpha \sum W^2$$
Prevents overfitting by shrinking coefficients, with cross-validation experiments comparing models across different penalty values ($\alpha$).

#### 4. Lasso Regression (`regression/lasso_regression.py`)
Regularized regression incorporating an $L_1$ penalty:
$$J(W) = \text{MSE} + \alpha \sum |W|$$
Enforces sparsity on features, behaving as a built-in feature selection method. Displays features completely zeroed out by the model (e.g., `Family_Income`).

### 🎯 Classification

#### 1. Logistic Regression (`classification/logistic_regression.py`)
A classification algorithm mapping inputs through the Sigmoid function:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$
Predicts the probability of a student passing or failing. Outputs complete evaluation metrics including Accuracy, Precision, Recall, F1 Score, and AUC-ROC, along with ROC curves and probability distribution plots.

#### 2. Decision Tree Classification (`classification/decision_tree.py`)
A model that splits data recursively based on feature values to maximize purity. Supports both Gini impurity and Information Gain (Entropy) splits. Visualizes Gini vs Entropy feature importance and displays prediction probability distributions.

#### 3. Naive Bayes Classification (`classification/naive_bayes.py`)
A probabilistic classifier based on Bayes' Theorem under the conditional independence assumption. Implements Gaussian Naive Bayes for continuous features. Plots actual vs predicted distributions and displays feature probability comparisons.

#### 4. Bagging Classification (`classification/bagging.py`)
An ensemble learning technique (Bootstrap Aggregating) that trains multiple Decision Trees on random subsets of the data (with replacement) to reduce variance and prevent overfitting. Outputs voting comparisons and Out-Of-Bag (OOB) validation assessments.

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed. Install the required external libraries:
```bash
pip install numpy pandas scikit-learn matplotlib
```

### 2. Running a Script
Run any of the Python files to observe the algorithm training output, console logs, and matplotlib visualization plots:
```bash
cd supervised_learning/classification
python decision_tree.py
```
