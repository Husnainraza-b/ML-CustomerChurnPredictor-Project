import numpy as np

class MLPClassifier:
    """
    Multi-Layer Perceptron implemented from scratch using NumPy.
    Designed for binary classification (Customer Churn).
    """
    def __init__(self, hidden_layer_sizes=(16,), learning_rate=0.01, max_iter=1000, random_state=42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.weights = []
        self.biases = []
        self.history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

    def _relu(self, Z):
        return np.maximum(0, Z)

    def _relu_derivative(self, Z):
        return (Z > 0).astype(float)

    def _sigmoid(self, Z):
        return 1 / (1 + np.exp(-np.clip(Z, -250, 250)))

    def _compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        # Cross-entropy loss
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = - (1/m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
        
    def _initialize_parameters(self, n_features):
        np.random.seed(self.random_state)
        layer_sizes = [n_features] + list(self.hidden_layer_sizes) + [1]
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            # He initialization for robust training
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2. / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(W)
            self.biases.append(b)

    def _forward(self, X):
        activations = [X]
        Z_vals = []
        
        A = X
        # Hidden layers
        for i in range(len(self.weights) - 1):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            Z_vals.append(Z)
            A = self._relu(Z)
            activations.append(A)
            
        # Output layer
        Z = np.dot(A, self.weights[-1]) + self.biases[-1]
        Z_vals.append(Z)
        A = self._sigmoid(Z)
        activations.append(A)
        
        return activations, Z_vals

    def _backward(self, y, activations, Z_vals):
        m = y.shape[0]
        gradients_W = [np.zeros_like(W) for W in self.weights]
        gradients_b = [np.zeros_like(b) for b in self.biases]
        
        # Output layer error
        A_final = activations[-1]
        dZ = A_final - y.reshape(-1, 1)
        
        for i in reversed(range(len(self.weights))):
            A_prev = activations[i]
            dW = (1/m) * np.dot(A_prev.T, dZ)
            db = (1/m) * np.sum(dZ, axis=0, keepdims=True)
            
            gradients_W[i] = dW
            gradients_b[i] = db
            
            if i > 0:
                dA_prev = np.dot(dZ, self.weights[i].T)
                dZ = dA_prev * self._relu_derivative(Z_vals[i-1])
                
        return gradients_W, gradients_b

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        n_samples, n_features = X_train.shape
        
        if X_val is not None and y_val is not None:
            X_val = np.array(X_val)
            y_val = np.array(y_val)
        
        self._initialize_parameters(n_features)
        self.history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
        
        for epoch in range(self.max_iter):
            # Forward pass
            activations, Z_vals = self._forward(X_train)
            
            # Compute loss
            loss = self._compute_loss(y_train, activations[-1])
            self.history['loss'].append(loss)
            
            # Accuracy
            preds = (activations[-1] > 0.5).astype(int).flatten()
            acc = np.mean(preds == y_train)
            self.history['accuracy'].append(acc)
            
            # Validation
            if X_val is not None and y_val is not None:
                val_act, _ = self._forward(X_val)
                val_loss = self._compute_loss(y_val, val_act[-1])
                self.history['val_loss'].append(val_loss)
                val_preds = (val_act[-1] > 0.5).astype(int).flatten()
                val_acc = np.mean(val_preds == y_val)
                self.history['val_accuracy'].append(val_acc)
            
            # Backward pass
            gradients_W, gradients_b = self._backward(y_train, activations, Z_vals)
            
            # Update parameters
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * gradients_W[i]
                self.biases[i] -= self.learning_rate * gradients_b[i]

    def predict_proba(self, X):
        X = np.array(X)
        activations, _ = self._forward(X)
        probs_class1 = activations[-1].flatten()
        probs_class0 = 1 - probs_class1
        return np.vstack((probs_class0, probs_class1)).T

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
