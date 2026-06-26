import numpy as np
import matplotlib.pyplot as plt

class GradientDescent:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.cost_history = []
        self.weights = None

    def compute_cost(self, X, y):
        """Calculates Mean Squared Error (MSE)"""
        m = len(y)
        predictions = X.dot(self.weights)
        cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))
        return cost

    def batch_gradient_descent(self, X, y):
        """Updates weights using the entire dataset at once."""
        m = len(y)
        self.weights = np.zeros((X.shape[1], 1))
        self.cost_history = []

        for _ in range(self.iterations):
            # 1. Make predictions
            predictions = X.dot(self.weights)
            
            # 2. Calculate the error gradient
            gradients = (1 / m) * X.T.dot(predictions - y)
            
            # 3. Update the weights by stepping in the opposite direction of the gradient
            self.weights -= self.lr * gradients
            
            # 4. Record the cost to see if we are improving
            self.cost_history.append(self.compute_cost(X, y))
            
        return self.weights

    def stochastic_gradient_descent(self, X, y):
        """Updates weights using one single sample at a time."""
        m = len(y)
        self.weights = np.zeros((X.shape[1], 1))
        self.cost_history = []

        for _ in range(self.iterations):
            cost_accum = 0
            for i in range(m):
                # Isolate a single row of data
                X_i = X[i, :].reshape(1, -1)
                y_i = y[i].reshape(1, 1)
                
                prediction = X_i.dot(self.weights)
                gradients = X_i.T.dot(prediction - y_i)
                self.weights -= self.lr * gradients
                
                cost_accum += self.compute_cost(X_i, y_i)
            
            # Average cost for this full pass over the data
            self.cost_history.append(cost_accum / m)
            
        return self.weights

    def plot_cost(self, title="Cost Reduction Over Time"):
        """Visualizes the learning process."""
        plt.plot(range(len(self.cost_history)), self.cost_history, color='red')
        plt.title(title)
        plt.xlabel('Iterations')
        plt.ylabel('Cost (Error)')
        plt.grid(True)
        plt.show()

# ==========================================
# Execution Area - Run this file to test it
# ==========================================
# ==========================================
# Execution Area - Run this file to test it
# ==========================================
# ==========================================
# Execution Area - Run this file to test it
# ==========================================
if __name__ == "__main__":
    np.random.seed(42)
    X_data = 2 * np.random.rand(100, 1)
    y_data = 4 + 3 * X_data + np.random.randn(100, 1) 

    X_b = np.c_[np.ones((100, 1)), X_data]

    # --- 1. Test Batch Gradient Descent ---
    print("\n--- Training via Batch Gradient Descent ---")
    bgd = GradientDescent(learning_rate=0.1, iterations=100)
    bgd_weights = bgd.batch_gradient_descent(X_b, y_data)
    
    # Print the results for Batch
    print(f"BGD Final Guess (Intercept, Slope): {bgd_weights.flatten()}")
    print("Expected roughly: [4.0, 3.0]")
    bgd.plot_cost("Batch Gradient Descent - Smooth Curve")

    # --- 2. Test Stochastic Gradient Descent ---
    print("\n--- Training via Stochastic Gradient Descent ---")
    # SGD takes 100 steps per iteration (since m=100), so we only need 10 iterations here
    sgd = GradientDescent(learning_rate=0.01, iterations=10) 
    sgd_weights = sgd.stochastic_gradient_descent(X_b, y_data)
    
    # Print the results for Stochastic
    print(f"SGD Final Guess (Intercept, Slope): {sgd_weights.flatten()}")
    print("Expected roughly: [4.0, 3.0]")
    sgd.plot_cost("Stochastic Gradient Descent - Bouncy Curve")