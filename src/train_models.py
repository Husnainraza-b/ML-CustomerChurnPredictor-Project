import os
import sys
import numpy as np
import pandas as pd

# Ensure src is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.naive_bayes import GaussianNaiveBayes
from src.models.neural_network import MLPClassifier
from src.evaluation.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set seed for reproducibility
np.random.seed(42)

def main():
    print("Loading data...")
    # Read the preprocessed dataset
    data_path = os.path.join('data', 'processed', 'cleaned_churn_data.csv')
    df = pd.read_csv(data_path)
    
    # 'Churn' is the target column
    if 'Churn' not in df.columns:
        print("Error: 'Churn' column not found in dataset!")
        return

    X = df.drop(columns=['Churn']).values.astype(np.float64)
    y = df['Churn'].values
    
    print(f"Data shape: {X.shape}")
    
    # Shuffle data
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # Split: 60% Train, 20% Val, 20% Test
    n = len(X)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
    
    # Create directories for saving models and graphs
    os.makedirs('models', exist_ok=True)
    os.makedirs(os.path.join('graphs', 'tuning'), exist_ok=True)
    
    # ---------------------------
    # 1. Train Naive Bayes
    # ---------------------------
    print("\nTraining Naive Bayes...")
    nb = GaussianNaiveBayes()
    nb.fit(X_train, y_train)
    
    nb_val_preds = nb.predict(X_val)
    nb_val_acc = np.mean(nb_val_preds == y_val)
    print(f"Naive Bayes Validation Accuracy: {nb_val_acc:.4f}")
    
    # Save Naive Bayes weights (mean, var, priors)
    np.savez(os.path.join('models', 'naive_bayes_weights.npz'), 
             mean=nb.mean, var=nb.var, priors=nb.priors, classes=nb.classes)
             
    # ---------------------------
    # 2. Train Neural Network (Tuning Process)
    # ---------------------------
    print("\nTraining Neural Network (Tuning)...")
    
    # Model A: Learning rate too small (underfitting)
    nn_underfit = MLPClassifier(hidden_layer_sizes=(16, 8), learning_rate=0.001, max_iter=500)
    print("Training Model A (LR=0.001) - intentionally showing underfitting...")
    nn_underfit.fit(X_train, y_train, X_val, y_val)
    
    # Model B: Better learning rate
    nn_tuned = MLPClassifier(hidden_layer_sizes=(16, 8), learning_rate=0.1, max_iter=800)
    print("Training Model B (LR=0.1) - tuned hyperparameters...")
    nn_tuned.fit(X_train, y_train, X_val, y_val)
    
    nn_val_acc = nn_tuned.history['val_accuracy'][-1]
    print(f"Neural Network Tuned Validation Accuracy: {nn_val_acc:.4f}")


    
    # Save Neural Network weights
    np.savez(os.path.join('models', 'nn_weights.npz'), *nn_tuned.weights)
    np.savez(os.path.join('models', 'nn_biases.npz'), *nn_tuned.biases)
    
    # ---------------------------
    # 3. Ensemble Model
    # ---------------------------
    print("\nEvaluating Ensemble Model (Logistic Regression + NB + NN) on Test Set...")
    

    class LogisticRegression:
        def __init__(self, lr=0.1, iters=1000):
            self.lr = lr
            self.iters = iters
        def fit(self, X, y):
            m, n = X.shape
            self.w = np.zeros(n)
            self.b = 0
            for _ in range(self.iters):
                z = np.dot(X, self.w) + self.b
                a = 1 / (1 + np.exp(-np.clip(z, -250, 250)))
                dz = a - y
                dw = (1/m) * np.dot(X.T, dz)
                db = (1/m) * np.sum(dz)
                self.w -= self.lr * dw
                self.b -= self.lr * db
        def predict_proba(self, X):
            z = np.dot(X, self.w) + self.b
            prob1 = 1 / (1 + np.exp(-np.clip(z, -250, 250)))
            return np.vstack((1 - prob1, prob1)).T

    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    np.save(os.path.join('models', 'logistic_regression_w.npy'), lr.w)
    
    # Predictions on Test set
    print("\nTest Set Accuracies and Metrics:")
    # LR
    lr_probs = lr.predict_proba(X_test)
    lr_preds = np.argmax(lr_probs, axis=1)
    lr_acc = accuracy_score(y_test, lr_preds)
    lr_prec = precision_score(y_test, lr_preds)
    lr_rec = recall_score(y_test, lr_preds)
    lr_f1 = f1_score(y_test, lr_preds)
    print(f"Logistic Regression -> Acc: {lr_acc:.4f}, Precision: {lr_prec:.4f}, Recall: {lr_rec:.4f}, F1: {lr_f1:.4f}")
    
    # NB
    nb_probs = nb.predict_proba(X_test)
    nb_preds = np.argmax(nb_probs, axis=1)
    nb_acc = accuracy_score(y_test, nb_preds)
    nb_prec = precision_score(y_test, nb_preds)
    nb_rec = recall_score(y_test, nb_preds)
    nb_f1 = f1_score(y_test, nb_preds)
    print(f"Naive Bayes         -> Acc: {nb_acc:.4f}, Precision: {nb_prec:.4f}, Recall: {nb_rec:.4f}, F1: {nb_f1:.4f}")
    
    # NN
    nn_probs = nn_tuned.predict_proba(X_test)
    nn_preds = np.argmax(nn_probs, axis=1)
    nn_acc = accuracy_score(y_test, nn_preds)
    nn_prec = precision_score(y_test, nn_preds)
    nn_rec = recall_score(y_test, nn_preds)
    nn_f1 = f1_score(y_test, nn_preds)
    print(f"Neural Network      -> Acc: {nn_acc:.4f}, Precision: {nn_prec:.4f}, Recall: {nn_rec:.4f}, F1: {nn_f1:.4f}")
    
    # Ensemble (Average probabilities)
    ensemble_probs = (lr_probs + nb_probs + nn_probs) / 3.0
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    ensemble_acc = accuracy_score(y_test, ensemble_preds)
    ensemble_prec = precision_score(y_test, ensemble_preds)
    ensemble_rec = recall_score(y_test, ensemble_preds)
    ensemble_f1 = f1_score(y_test, ensemble_preds)
    print(f"Ensemble (LR+NB+NN) -> Acc: {ensemble_acc:.4f}, Precision: {ensemble_prec:.4f}, Recall: {ensemble_rec:.4f}, F1: {ensemble_f1:.4f}")
    

    
    print("\nAll tuning and training completed successfully!")

if __name__ == '__main__':
    main()
