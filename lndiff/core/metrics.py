"""
Metrics calculation module.
"""

import numpy as np
from sklearn.metrics import roc_curve, auc


def calculate_metrics(y_true, y_pred, y_scores):
    """Calculate classification metrics."""
    acc = (y_pred == y_true).mean()
    
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * sens / (precision + sens) if (precision + sens) > 0 else 0.0
    
    try:
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            auc_score = auc(fpr, tpr)
        else:
            auc_score = 0.5
    except:
        auc_score = 0.5
    
    return {
        "Accuracy": round(acc, 4),
        "Sensitivity": round(sens, 4),
        "Specificity": round(spec, 4),
        "AUC": round(auc_score, 4),
        "F1-Score": round(f1, 4),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn)
    }
