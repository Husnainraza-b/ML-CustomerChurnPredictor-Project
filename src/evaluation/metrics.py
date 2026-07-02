import numpy as np

def accuracy_score(y_true, y_pred):
    """
    Calculate accuracy: (TP + TN) / Total
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return 0.0
    return np.mean(y_true == y_pred)

def precision_score(y_true, y_pred):
    """
    Calculate precision: TP / (TP + FP)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    
    if (tp + fp) == 0:
        return 0.0
    return float(tp / (tp + fp))

def recall_score(y_true, y_pred):
    """
    Calculate recall: TP / (TP + FN)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    if (tp + fn) == 0:
        return 0.0
    return float(tp / (tp + fn))

def f1_score(y_true, y_pred):
    """
    Calculate F1 score: 2 * (Precision * Recall) / (Precision + Recall)
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    if (precision + recall) == 0:
        return 0.0
    return float(2 * (precision * recall) / (precision + recall))
