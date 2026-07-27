import numpy as np

class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes implemented from scratch using NumPy.
    Used for the Customer Churn prediction project.
    """
    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None

    def fit(self, X, y):
        """
        Train the model using X features and y targets.
        """
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Initialize parameters
        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            self.priors[idx] = X_c.shape[0] / float(n_samples)

    def _pdf(self, class_idx, x):
        """
        Probability Density Function calculation.
        """
        mean = self.mean[class_idx]
        var = self.var[class_idx] + 1e-9 # Smoothing to avoid division by zero
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def predict(self, X):
        X = np.array(X)
        y_pred = [self._predict_single(x) for x in X]
        return np.array(y_pred)

    def _predict_single(self, x):
        posteriors = []
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x) + 1e-9)) # avoid log(0)
            posterior = prior + class_conditional
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]

    def predict_proba(self, X):
        """
        Returns probabilities for each class (useful for ensembling).
        """
        X = np.array(X)
        probas = []
        for x in X:
            posteriors = []
            for idx, c in enumerate(self.classes):
                prior = np.log(self.priors[idx])
                class_conditional = np.sum(np.log(self._pdf(idx, x) + 1e-9))
                posteriors.append(prior + class_conditional)
            posteriors = np.array(posteriors)
            # Softmax for probabilities
            posteriors -= np.max(posteriors) # Numerical stability
            exp_p = np.exp(posteriors)
            prob = exp_p / np.sum(exp_p)
            probas.append(prob)
        return np.array(probas)
