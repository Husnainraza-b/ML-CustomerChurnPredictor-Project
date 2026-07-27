import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.naive_bayes import GaussianNaiveBayes
from src.models.neural_network import MLPClassifier

def main():
    print("Loading Pre-Trained Ultimate Ensemble Model (Using Saved Weights)...")
    
    # 1. Load Data
    data_path = os.path.join('data', 'processed', 'cleaned_churn_data.csv')
    df = pd.read_csv(data_path)
    X = df.drop(columns=['Churn']).values.astype(np.float64)
    y = df['Churn'].values
    
    # Pick 5 random samples for the live demo
    np.random.seed(99)
    idx = np.random.choice(len(X), 5, replace=False)
    X_demo = X[idx]
    y_demo = y[idx]
    
    print("\nLoading weights from disk...")
    # 2. Load Naive Bayes
    nb = GaussianNaiveBayes()
    nb_data = np.load(os.path.join('models', 'naive_bayes_weights.npz'))
    nb.mean = nb_data['mean']
    nb.var = nb_data['var']
    nb.priors = nb_data['priors']
    nb.classes = nb_data['classes']
    nb_probs = nb.predict_proba(X_demo)
    
    # 3. Load Neural Network
    nn = MLPClassifier()
    nn_w_data = np.load(os.path.join('models', 'nn_weights.npz'))
    nn_b_data = np.load(os.path.join('models', 'nn_biases.npz'))
    # Load sorted by original order
    nn.weights = [nn_w_data[k] for k in sorted(nn_w_data.files)]
    nn.biases = [nn_b_data[k] for k in sorted(nn_b_data.files)]
    nn_probs = nn.predict_proba(X_demo)
    
    # 4. Load Logistic Regression
    class LogisticRegression:
        def predict_proba(self, X):
            z = np.dot(X, self.w) + self.b
            prob1 = 1 / (1 + np.exp(-np.clip(z, -250, 250)))
            return np.vstack((1 - prob1, prob1)).T
            
    lr = LogisticRegression()
    lr.w = np.load(os.path.join('models', 'logistic_regression_w.npy'))
    lr.b = 0 # Approximated base bias
    lr_probs = lr.predict_proba(X_demo)
    
    # 5. Ensemble Prediction
    print("-" * 50)
    print("LIVE DEMO: Predicting 5 Random Customers without Training")
    print("-" * 50)
    
    # Average the probabilities together
    ensemble_probs = (lr_probs + nb_probs + nn_probs) / 3.0
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    
    for i in range(5):
        print(f"Customer {i+1}:")
        print(f"  Actual Truth    : {'CHURN' if y_demo[i]==1 else 'NO CHURN'}")
        print(f"  Ensemble Predict: {'CHURN' if ensemble_preds[i]==1 else 'NO CHURN'} (Confidence: {ensemble_probs[i][ensemble_preds[i]]:.2%})")
        print()

if __name__ == '__main__':
    main()
