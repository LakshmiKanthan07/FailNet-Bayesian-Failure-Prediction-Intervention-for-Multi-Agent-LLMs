from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, log_loss
import pandas as pd
import numpy as np

def evaluate_baselines(data_path="data/processed/bn_evidence_table.csv"):
    df = pd.read_csv(data_path)
    
    # Drop columns not used as features or targets
    if "label_provenance" in df.columns:
        df = df.drop(columns=["label_provenance"])
    
    # Target is Task_Failure
    if "Failure_Severity" in df.columns:
        df = df.drop(columns=["Failure_Severity"])
        
    df["Task_Failure"] = df["Task_Failure"].map({"Yes": 1, "No": 0})
    
    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df.drop(columns=["Task_Failure"]))
    X, y = df_encoded, df["Task_Failure"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Logistic Regression
    logreg = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    lr_preds = logreg.predict(X_test)
    lr_probs = logreg.predict_proba(X_test)[:, 1]
    
    print("=== Logistic Regression ===")
    print(classification_report(y_test, lr_preds))
    print("Log Loss:", log_loss(y_test, lr_probs))
    
    # XGBoost
    xgb = XGBClassifier(eval_metric="logloss").fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    xgb_probs = xgb.predict_proba(X_test)[:, 1]
    
    print("\n=== XGBoost ===")
    print(classification_report(y_test, xgb_preds))
    print("Log Loss:", log_loss(y_test, xgb_probs))
    
    return logreg, xgb

if __name__ == "__main__":
    evaluate_baselines()
