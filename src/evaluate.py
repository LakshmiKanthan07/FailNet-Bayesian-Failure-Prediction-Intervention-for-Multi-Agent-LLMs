import numpy as np
from sklearn.metrics import f1_score, log_loss, brier_score_loss, classification_report
import pandas as pd

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Expected Calibration Error (ECE).
    Divides predictions into n_bins, computes average accuracy and average confidence in each bin.
    ECE is the weighted average of the absolute difference between accuracy and confidence.
    """
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    
    ece = 0.0
    for i in range(n_bins):
        mask = binids == i
        if np.any(mask):
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            bin_weight = np.sum(mask) / len(y_true)
            ece += bin_weight * np.abs(bin_acc - bin_conf)
            
    return ece

def evaluate_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    
    metrics = {
        "F1_Score": f1_score(y_true, y_pred),
        "Log_Loss": log_loss(y_true, y_prob),
        "Brier_Score": brier_score_loss(y_true, y_prob),
        "ECE": expected_calibration_error(y_true, y_prob)
    }
    
    return metrics

if __name__ == "__main__":
    # Test metrics with dummy data
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 1000)
    y_prob = np.random.rand(1000)
    
    metrics = evaluate_metrics(y_true, y_prob)
    print("Test Metrics on Random Data:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
