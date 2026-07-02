import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate: float = 0.01, iterations: int = 1000, fit_intercept: bool = True, verbose: bool = False):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.theta = None

    def _sigmoid(self, z):
        # Clip z to prevent overflow/underflow in np.exp
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Fit the model parameters using Gradient Descent.
        """
        # Convert to numpy arrays if they are pandas objects
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if self.fit_intercept:
            # Prepend a column of 1s to represent the bias/intercept term
            ones = np.ones((X.shape[0], 1))
            X = np.hstack((ones, X))

        num_samples, num_features = X.shape
        self.theta = np.zeros(num_features)

        # Gradient Descent loop
        for i in range(self.iterations):
            z = np.dot(X, self.theta)
            h = self._sigmoid(z)
            
            # Print loss if verbose
            if self.verbose and i % 100 == 0:
                epsilon = 1e-15  # to prevent log(0)
                loss = -np.mean(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
                print(f"Iteration {i}: Loss = {loss:.6f}")

            # Calculate gradient
            gradient = np.dot(X.T, (h - y)) / num_samples
            
            # Update parameters
            self.theta -= self.learning_rate * gradient

    def predict_proba(self, X):
        """
        Predict probability of class 1.
        """
        X = np.asarray(X, dtype=float)
        if self.fit_intercept:
            ones = np.ones((X.shape[0], 1))
            X = np.hstack((ones, X))
        
        return self._sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold: float = 0.5):
        """
        Predict binary labels (0 or 1) based on probability threshold.
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def save_model(self, filepath: str):
        """
        Save the model weights (theta) to a file.
        """
        if self.theta is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        np.save(filepath, self.theta)

    def load_model(self, filepath: str):
        """
        Load the model weights (theta) from a file.
        """
        self.theta = np.load(filepath)

