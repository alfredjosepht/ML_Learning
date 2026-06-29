"""
==========================================================================
                        GRADIENT DESCENT FROM SCRATCH
==========================================================================

Author      : Alfred Joseph T
Description : Implements Batch, Stochastic, and Mini-Batch Gradient
              Descent from scratch using only NumPy and Matplotlib.

Algorithms  : BGD | SGD | MBGD
==========================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


# ==========================================================================
# GRADIENT DESCENT CLASS
# ==========================================================================

class GradientDescent:
    """
    Gradient Descent optimizer for Linear Regression.

    Supports three variants:
      - Batch Gradient Descent      : Uses full dataset per update
      - Stochastic Gradient Descent : Uses one sample per update
      - Mini-Batch Gradient Descent : Uses a small batch per update

    Parameters
    ----------
    learning_rate : float
        Step size for each weight update. Default: 0.01
    iterations : int
        Number of passes over the dataset. Default: 1000
    """

    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.cost_history = []

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _initialize_weights(self, n_features):
        """Zero-initialize weight vector of shape (n_features, 1)."""
        self.weights = np.zeros((n_features, 1))

    def _reset_history(self):
        """Clear cost history before a new training run."""
        self.cost_history = []

    # ------------------------------------------------------------------
    # Core Math
    # ------------------------------------------------------------------

    def predict(self, X):
        """
        Compute predictions: ŷ = X · W

        Parameters
        ----------
        X : ndarray, shape (m, n)

        Returns
        -------
        ndarray, shape (m, 1)
        """
        return X.dot(self.weights)

    def compute_cost(self, X, y):
        """
        Mean Squared Error cost: J = (1 / 2m) Σ (ŷ - y)²

        Parameters
        ----------
        X : ndarray, shape (m, n)
        y : ndarray, shape (m, 1)

        Returns
        -------
        float
        """
        m = len(y)
        error = self.predict(X) - y
        return (1 / (2 * m)) * np.sum(np.square(error))

    # ------------------------------------------------------------------
    # Training Algorithms
    # ------------------------------------------------------------------

    def batch_gradient_descent(self, X, y):
        """
        Batch Gradient Descent (BGD)

        Computes the gradient over the *entire* training set before
        updating weights. Stable but slower for large datasets.

        Update rule:
            W = W - α · (1/m) · Xᵀ(XW - y)

        Parameters
        ----------
        X : ndarray, shape (m, n)  — must include bias column
        y : ndarray, shape (m, 1)

        Returns
        -------
        weights : ndarray, shape (n, 1)
        """
        m = len(y)
        self._initialize_weights(X.shape[1])
        self._reset_history()

        for _ in range(self.iterations):
            error = self.predict(X) - y
            gradients = (1 / m) * X.T.dot(error)
            self.weights -= self.learning_rate * gradients
            self.cost_history.append(self.compute_cost(X, y))

        return self.weights

    def stochastic_gradient_descent(self, X, y):
        """
        Stochastic Gradient Descent (SGD)

        Updates weights after *each individual sample*. Faster per
        iteration but produces noisy cost curves.

        Parameters
        ----------
        X : ndarray, shape (m, n)
        y : ndarray, shape (m, 1)

        Returns
        -------
        weights : ndarray, shape (n, 1)
        """
        m = len(y)
        self._initialize_weights(X.shape[1])
        self._reset_history()

        for _ in range(self.iterations):
            indices = np.random.permutation(m)
            X_s, y_s = X[indices], y[indices]
            total_cost = 0.0

            for i in range(m):
                Xi = X_s[i:i + 1]
                yi = y_s[i:i + 1]
                error = Xi.dot(self.weights) - yi
                self.weights -= self.learning_rate * Xi.T.dot(error)
                total_cost += self.compute_cost(Xi, yi)

            self.cost_history.append(total_cost / m)

        return self.weights

    def mini_batch_gradient_descent(self, X, y, batch_size=16):
        """
        Mini-Batch Gradient Descent (MBGD)

        Splits the dataset into small batches and updates weights
        after each batch. Balances stability and speed — the most
        widely used variant in deep learning.

        Parameters
        ----------
        X          : ndarray, shape (m, n)
        y          : ndarray, shape (m, 1)
        batch_size : int, default 16

        Returns
        -------
        weights : ndarray, shape (n, 1)
        """
        m = len(y)
        self._initialize_weights(X.shape[1])
        self._reset_history()

        for _ in range(self.iterations):
            indices = np.random.permutation(m)
            X_s, y_s = X[indices], y[indices]

            for start in range(0, m, batch_size):
                Xb = X_s[start:start + batch_size]
                yb = y_s[start:start + batch_size]
                error = Xb.dot(self.weights) - yb
                gradients = (1 / len(Xb)) * Xb.T.dot(error)
                self.weights -= self.learning_rate * gradients

            self.cost_history.append(self.compute_cost(X, y))

        return self.weights

    # ------------------------------------------------------------------
    # Display Utilities
    # ------------------------------------------------------------------

    def print_weights(self):
        """Print all learned weights."""
        print("\n" + "=" * 60)
        print("MODEL WEIGHTS")
        print("=" * 60)
        for i, w in enumerate(self.weights):
            print(f"  Weight {i} : {w[0]:.6f}")

    def print_equation(self):
        """Display the learned linear equation y = β0 + β1·x."""
        print("\n" + "=" * 60)
        print("LEARNED EQUATION")
        print("=" * 60)
        print(f"  y = {self.weights[0][0]:.4f} + {self.weights[1][0]:.4f} · x")

    def print_cost(self):
        """Display the final cost after training."""
        print("\n" + "=" * 60)
        print("FINAL COST")
        print("=" * 60)
        print(f"  {self.cost_history[-1]:.6f}")

    # ------------------------------------------------------------------
    # Plotting Utilities
    # ------------------------------------------------------------------

    def plot_cost(self, title="Cost Function"):
        """Plot cost vs. iterations."""
        plt.figure(figsize=(8, 5))
        plt.plot(self.cost_history, linewidth=2, color="steelblue")
        plt.title(title, fontsize=14)
        plt.xlabel("Iterations")
        plt.ylabel("Cost (MSE)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()

    def plot_regression_line(self, X, y):
        """Scatter-plot data and overlay the learned regression line."""
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 1], y, alpha=0.7, label="Training data")

        sorted_idx = np.argsort(X[:, 1])
        plt.plot(
            X[sorted_idx, 1],
            self.predict(X)[sorted_idx],
            color="crimson",
            linewidth=2,
            label="Regression line",
        )

        plt.xlabel("Feature (x)")
        plt.ylabel("Target (y)")
        plt.title("Learned Regression Line")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()

    def plot_predictions(self, X, y):
        """Scatter plot of actual vs. predicted values."""
        predictions = self.predict(X)
        plt.figure(figsize=(6, 6))
        plt.scatter(y, predictions, alpha=0.7)
        min_val = min(y.min(), predictions.min())
        max_val = max(y.max(), predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Perfect fit")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Actual vs. Predicted")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()


# ==========================================================================
# MAIN
# ==========================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("GRADIENT DESCENT DEMONSTRATION")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Dataset
    # True relationship: y = 4 + 3x + noise
    # Goal: recover intercept ≈ 4, slope ≈ 3
    # ------------------------------------------------------------------
    np.random.seed(42)
    X_raw = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X_raw + np.random.randn(100, 1)
    X = np.c_[np.ones((100, 1)), X_raw]   # prepend bias column

    # ------------------------------------------------------------------
    # 1. Batch Gradient Descent
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BATCH GRADIENT DESCENT")
    print("=" * 70)

    bgd = GradientDescent(learning_rate=0.1, iterations=100)
    bgd.batch_gradient_descent(X, y)
    bgd.print_weights()
    bgd.print_equation()
    bgd.print_cost()
    bgd.plot_cost("Batch Gradient Descent — Cost Curve")
    bgd.plot_regression_line(X, y)
    bgd.plot_predictions(X, y)

    # ------------------------------------------------------------------
    # 2. Stochastic Gradient Descent
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STOCHASTIC GRADIENT DESCENT")
    print("=" * 70)

    sgd = GradientDescent(learning_rate=0.01, iterations=20)
    sgd.stochastic_gradient_descent(X, y)
    sgd.print_weights()
    sgd.print_equation()
    sgd.print_cost()
    sgd.plot_cost("Stochastic Gradient Descent — Cost Curve")

    # ------------------------------------------------------------------
    # 3. Mini-Batch Gradient Descent
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("MINI-BATCH GRADIENT DESCENT")
    print("=" * 70)

    mbgd = GradientDescent(learning_rate=0.05, iterations=100)
    mbgd.mini_batch_gradient_descent(X, y, batch_size=16)
    mbgd.print_weights()
    mbgd.print_equation()
    mbgd.print_cost()
    mbgd.plot_cost("Mini-Batch Gradient Descent — Cost Curve")

    # ------------------------------------------------------------------
    # 4. Compare all three cost curves
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(bgd.cost_history,  label="Batch GD",       linewidth=2)
    plt.plot(sgd.cost_history,  label="Stochastic GD",  linewidth=2)
    plt.plot(mbgd.cost_history, label="Mini-Batch GD",  linewidth=2)
    plt.title("Cost Comparison Across Variants", fontsize=14)
    plt.xlabel("Iterations")
    plt.ylabel("Cost (MSE)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # 5. Learning rate sensitivity experiment
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("LEARNING RATE EXPERIMENT")
    print("=" * 70)

    plt.figure(figsize=(10, 6))
    for lr in [0.001, 0.01, 0.1, 0.5]:
        model = GradientDescent(learning_rate=lr, iterations=100)
        model.batch_gradient_descent(X, y)
        plt.plot(model.cost_history, label=f"LR = {lr}")

    plt.title("Effect of Learning Rate on Convergence", fontsize=14)
    plt.xlabel("Iterations")
    plt.ylabel("Cost (MSE)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Batch GD     final cost : {bgd.cost_history[-1]:.6f}")
    print(f"  Stochastic GD final cost: {sgd.cost_history[-1]:.6f}")
    print(f"  Mini-Batch GD final cost: {mbgd.cost_history[-1]:.6f}")

    print("\n  Characteristics")
    print("  ├── Batch GD    : Stable, accurate — slow on large data")
    print("  ├── SGD         : Fast updates — noisy convergence")
    print("  └── Mini-Batch  : Best of both — standard in deep learning")

    print("\n  Common Applications")
    print("  Linear Regression · Logistic Regression · Neural Networks · Deep Learning")

    print("\nProgram finished successfully.")