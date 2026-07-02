import numpy as np

class KNearestNeighbors:
    def __init__(self, k: int = 5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Store the training datasets.
        """
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y, dtype=int)

    def _compute_distances(self, X):
        """
        Vectorized pairwise Euclidean distance calculation:
        dist(a, b) = sqrt(sum(a^2) + sum(b^2) - 2 * a . b)
        """
        X = np.asarray(X, dtype=float)
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        
        sum_test = np.sum(X**2, axis=1, keepdims=True)  # (num_test, 1)
        sum_train = np.sum(self.X_train**2, axis=1, keepdims=True).T  # (1, num_train)
        
        dot_product = np.dot(X, self.X_train.T)  # (num_test, num_train)
        
        # Compute distances and handle numerical precision issues via clipping at 0
        dists = np.sqrt(np.clip(sum_test + sum_train - 2 * dot_product, 0, None))
        return dists

    def predict(self, X):
        """
        Predict classes of test samples using majority vote of k-nearest neighbors.
        """
        dists = self._compute_distances(X)
        num_test = X.shape[0]
        y_pred = np.zeros(num_test, dtype=int)
        
        for i in range(num_test):
            # Find indices of the k closest neighbors
            nearest_indices = np.argsort(dists[i])[:self.k]
            nearest_labels = self.y_train[nearest_indices]
            
            # Vote by taking the majority class
            y_pred[i] = int(np.round(np.mean(nearest_labels)))
            
        return y_pred

    def save_model(self, filepath: str):
        """
        Save the model training data and k parameter to disk.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        np.savez(filepath, X_train=self.X_train, y_train=self.y_train, k=np.array([self.k]))

    def load_model(self, filepath: str):
        """
        Load the model training data and k parameter from disk.
        """
        data = np.load(filepath)
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.k = int(data['k'][0])
