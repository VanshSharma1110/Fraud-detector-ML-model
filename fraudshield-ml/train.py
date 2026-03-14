import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import os
import joblib
import json
import time
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from prophet import Prophet
from datetime import datetime, timedelta
import shap
import dill
from lime.lime_tabular import LimeTabularExplainer
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE

# Set seed for reproducibility
np.random.seed(42)

def synthesize_data(input_path="data/creditcard.csv"):
    if not os.path.exists(input_path):
        print("Error: " + input_path + " not found.")
        return

    print("Reading " + input_path + "...")
    df = pd.read_csv(input_path)
    
    # 1. amount_inr
    df['amount_inr'] = df['Amount'] * 83.5
    
    # 2. hour (derived from Time)
    # Mapping Time (seconds) to hour 0-23
    df['hour'] = (df['Time'] // 3600) % 24
    
    # 3. is_new_device
    # 1 for 65% of fraud rows (Class=1), 15% of legit rows (Class=0)
    df['is_new_device'] = 0
    df.loc[df['Class'] == 1, 'is_new_device'] = np.random.choice([0, 1], size=(df['Class'] == 1).sum(), p=[0.35, 0.65])
    df.loc[df['Class'] == 0, 'is_new_device'] = np.random.choice([0, 1], size=(df['Class'] == 0).sum(), p=[0.85, 0.15])
    
    # 4. velocity_60s
    # 3-20 range for fraud + noise, 0-6 for legit + noise (overlapping distributions)
    df['velocity_60s'] = 0.0
    n_fraud = (df['Class'] == 1).sum()
    n_legit = (df['Class'] == 0).sum()
    df.loc[df['Class'] == 1, 'velocity_60s'] = np.clip(np.random.uniform(3, 20, size=n_fraud) + np.random.normal(0, 3, size=n_fraud), 0, None)
    df.loc[df['Class'] == 0, 'velocity_60s'] = np.clip(np.random.uniform(0, 6, size=n_legit) + np.random.normal(0, 1, size=n_legit), 0, None)
    
    # 5. velocity_48h_count
    # 8-50 for fraud, 1-8 for legit
    df['velocity_48h_count'] = 0
    df.loc[df['Class'] == 1, 'velocity_48h_count'] = np.random.randint(8, 51, size=(df['Class'] == 1).sum())
    df.loc[df['Class'] == 0, 'velocity_48h_count'] = np.random.randint(1, 9, size=(df['Class'] == 0).sum())
    
    # 6. velocity_48h_amount
    # velocity_48h_count * amount_inr * random(0.5-1.5)
    df['velocity_48h_amount'] = df['velocity_48h_count'] * df['amount_inr'] * np.random.uniform(0.5, 1.5, size=len(df))
    
    # 7. is_new_recipient
    # 1 for 75% of fraud rows, 35% of legit rows (overlapping distributions)
    df['is_new_recipient'] = 0
    df.loc[df['Class'] == 1, 'is_new_recipient'] = np.random.choice([0, 1], size=(df['Class'] == 1).sum(), p=[0.25, 0.75])
    df.loc[df['Class'] == 0, 'is_new_recipient'] = np.random.choice([0, 1], size=(df['Class'] == 0).sum(), p=[0.65, 0.35])
    
    # 8. account_age_days
    # 1-180 for fraud rows, 15-1800 for legit rows (overlapping distributions)
    df['account_age_days'] = 0
    df.loc[df['Class'] == 1, 'account_age_days'] = np.random.randint(1, 180, size=(df['Class'] == 1).sum())
    df.loc[df['Class'] == 0, 'account_age_days'] = np.random.randint(15, 1801, size=(df['Class'] == 0).sum())
    
    # 9. is_first_merchant
    # 1 for 90% of fraud rows, 30% of legit rows
    df['is_first_merchant'] = 0
    df.loc[df['Class'] == 1, 'is_first_merchant'] = np.random.choice([0, 1], size=(df['Class'] == 1).sum(), p=[0.1, 0.9])
    df.loc[df['Class'] == 0, 'is_first_merchant'] = np.random.choice([0, 1], size=(df['Class'] == 0).sum(), p=[0.7, 0.3])
    
    # 10. is_festival_day
    # inject 15 random festival days
    festival_days = np.random.choice(range(365), 15, replace=False)
    # Time is relative to day 0. So day = Time // 86400
    df['day'] = (df['Time'] // 86400) % 365
    df['is_festival_day'] = df['day'].isin(festival_days).astype(int)
    # fraud amount * 1.5-3x on festival days
    mask = (df['is_festival_day'] == 1) & (df['Class'] == 1)
    df.loc[mask, 'amount_inr'] *= np.random.uniform(1.5, 3.0, size=mask.sum())
    
    # 11. is_sim_swap_signal
    # 1 when is_new_device=1 AND hour<5 AND Class=1
    df['is_sim_swap_signal'] = ((df['is_new_device'] == 1) & (df['hour'] < 5) & (df['Class'] == 1)).astype(int)
    
    # 12. is_round_amount
    # 1 when amount_inr % 1000 == 0
    # Note: Using small epsilon for floating point comparison
    df['is_round_amount'] = ((df['amount_inr'] % 1000) < 1).astype(int)
    
    # 13. city & city_risk_score
    cities = ['Mumbai', 'Delhi', 'Bengaluru', 'Hyderabad', 'Ahmedabad', 'Chennai', 'Kolkata', 'Surat', 'Pune', 'Jaipur',
              'Lucknow', 'Kanpur', 'Nagpur', 'Indore', 'Thane', 'Bhopal', 'Visakhapatnam', 'Pimpri-Chinchwad', 'Patna', 'Vadodara']
    
    # Define city risk scores (Mumbai highest)
    city_risk_map = {city: np.random.uniform(0.1, 0.5) for city in cities}
    city_risk_map['Mumbai'] = 0.8
    city_risk_map['Delhi'] = 0.7
    city_risk_map['Bengaluru'] = 0.6
    city_risk_map['Hyderabad'] = 0.6
    
    df['city'] = np.random.choice(cities, size=len(df))
    df['city_risk_score'] = df['city'].map(city_risk_map)
    
    # Fraud concentrated in specific cities
    fraud_cities = ['Mumbai', 'Delhi', 'Bengaluru', 'Hyderabad']
    df.loc[df['Class'] == 1, 'city'] = np.random.choice(fraud_cities, size=(df['Class'] == 1).sum(), p=[0.4, 0.2, 0.2, 0.2])
    df['city_risk_score'] = df['city'].map(city_risk_map) # Update scores
    
    # 14. merchant_category & one-hot encoding
    categories = ['grocery', 'dining', 'travel', 'fashion', 'electronics', 'services', 'utilities', 
                  'health', 'education', 'entertainment', 'transport', 'fuel', 'crypto', 'gaming', 'jewelry']
    
    df['merchant_category'] = np.random.choice(categories, size=len(df))
    # Fraud concentrated in crypto/gaming/jewelry/electronics
    fraud_cats = ['crypto', 'gaming', 'jewelry', 'electronics']
    df.loc[df['Class'] == 1, 'merchant_category'] = np.random.choice(fraud_cats, size=(df['Class'] == 1).sum())
    
    # One-hot encode (prefix cat_)
    df = pd.get_dummies(df, columns=['merchant_category'], prefix='cat')
    
    # 15. fraud_type label (only for Class=1)
    df['fraud_type'] = 'legit' # Default for Class=0 (though prompt says only for Class=1)
    
    # Identify fraud rows
    class_1_mask = df['Class'] == 1
    
    # Define conditions for fraud_type
    # Using np.select for prioritized labels
    conditions = [
        (df['velocity_60s'] > 15),
        (df['is_new_device'] == 1) & (df['hour'] < 5),
        (df['amount_inr'] > 500000),
        (df['is_new_recipient'] == 1) & (df.get('cat_crypto', 0) == 1),
        (df['account_age_days'] < 30),
        (df.get('cat_crypto', 0) == 1) | (df.get('cat_gaming', 0) == 1)
    ]
    choices = [
        'velocity_attack',
        'sim_swap',
        'account_takeover',
        'money_mule',
        'identity_fraud',
        'phishing'
    ]
    
    # Apply conditions to whole df, then restrict legit
    df['fraud_type'] = np.select(
        [cond & class_1_mask for cond in conditions],
        choices,
        default='merchant_fraud'
    )
    df.loc[~class_1_mask, 'fraud_type'] = 'legit'
    
    # Print fraud_type value counts for Class=1
    print("\nFraud Type Value Counts (Class=1 only):")
    print(df[df['Class'] == 1]['fraud_type'].value_counts())
    
    # Remove temp day column if it exists
    if 'day' in df.columns:
        df = df.drop(columns=['day'])
        
    return df

def feature_engineering(df):
    print("\nStarting feature engineering...")
    
    # 1. Robust Scaling
    scaler = RobustScaler()
    df[['amount_scaled', 'time_scaled']] = scaler.fit_transform(df[['amount_inr', 'Time']])
    
    # Save the scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/robust_scaler.pkl')
    print("Saved RobustScaler to models/robust_scaler.pkl")
    
    # 2. Log Transformation
    df['amount_log'] = np.log1p(df['amount_inr'])
    df['velocity_log'] = np.log1p(df['velocity_60s'])
    
    # 3. Interaction Features
    df['amount_x_velocity'] = df['amount_scaled'] * df['velocity_60s']
    df['new_device_x_night'] = df['is_new_device'] * (df['hour'] < 6).astype(int)
    df['new_recip_x_amount'] = df['is_new_recipient'] * df['amount_scaled']
    df['young_account_x_large'] = (df['account_age_days'] < 30).astype(int) * (df['amount_inr'] > 10000).astype(int)
    df['velocity_x_new_device'] = df['velocity_60s'] * df['is_new_device']
    df['festival_x_amount'] = df['is_festival_day'] * df['amount_scaled']
    
    # 4. V-feature aggregates (V1-V28)
    v_cols = [f'V{i}' for i in range(1, 29)]
    df['v_mean'] = df[v_cols].mean(axis=1)
    df['v_std'] = df[v_cols].std(axis=1)
    df['v_max'] = df[v_cols].max(axis=1)
    df['v_min'] = df[v_cols].min(axis=1)
    df['v_skew'] = df[v_cols].skew(axis=1)
    df['v_kurt'] = df[v_cols].kurtosis(axis=1)
    
    # 5. Fraud Signal composite
    # v_fraud_signal = (-V14*0.3) + (-V12*0.2) + (-V10*0.2) + (V4*0.15) + (V11*0.15)
    df['v_fraud_signal'] = ((-df['V14'] * 0.3) + (-df['V12'] * 0.2) + (-df['V10'] * 0.2) + 
                            (df['V4'] * 0.15) + (df['V11'] * 0.15))
    
    # Final FEATURES list selection
    cat_cols = [col for col in df.columns if col.startswith('cat_')]
    
    FEATURES = (
        v_cols + 
        ['amount_scaled', 'time_scaled', 'amount_log', 'velocity_log'] +
        ['hour', 'is_new_device', 'velocity_60s', 'velocity_48h_count', 'velocity_48h_amount', 
         'is_new_recipient', 'account_age_days', 'is_first_merchant', 'is_festival_day', 
         'is_sim_swap_signal', 'is_round_amount', 'city_risk_score'] +
        ['amount_x_velocity', 'new_device_x_night', 'new_recip_x_amount', 
         'young_account_x_large', 'velocity_x_new_device', 'festival_x_amount'] +
        ['v_mean', 'v_std', 'v_max', 'v_min', 'v_skew', 'v_kurt', 'v_fraud_signal'] +
        cat_cols
    )
    
    print(f"Total feature count: {len(FEATURES)}")
    
    # Save features list to models/features.json
    with open('models/features.json', 'w') as f:
        json.dump(FEATURES, f)
    print("Saved features list to models/features.json")
    
    return df, FEATURES

def split_and_balance(df, features):
    print("\nStarting data splitting and class balancing...")
    
    X = df[features]
    y = df['Class']
    
    # 1. Split into 3 sets
    # Test set: 20% of total data (stratified on Class)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    
    # Val set: 15% of remaining (stratified on Class)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15, stratify=y_temp, random_state=42
    )
    
    # Print sizes and fraud counts
    print(f"Total rows: {len(df)}")
    print(f"Train set: {len(X_train)} rows, {sum(y_train)} fraud")
    print(f"Val set:   {len(X_val)} rows, {sum(y_val)} fraud")
    print(f"Test set:  {len(X_test)} rows, {sum(y_test)} fraud")
    
    # 2. Balance the training set using SMOTETomek
    # Explain why SMOTETomek is better than SMOTE alone:
    # SMOTETomek oversamples the minority class AND removes noisy
    # borderline majority class samples (Tomek links), producing
    # a cleaner decision boundary than SMOTE alone.
    
    print("\nApplying SMOTETomek to the training set...")
    print(f"Class counts before: {np.bincount(y_train)}")
    
    smt = SMOTETomek(
        smote=SMOTE(sampling_strategy=0.15, 
                    random_state=42, 
                    k_neighbors=5),
        random_state=42
    )
    X_train_bal, y_train_bal = smt.fit_resample(X_train, y_train)
    
    print(f"Class counts after:  {np.bincount(y_train_bal)}")
    
    return X_train_bal, X_val, X_test, y_train_bal, y_val, y_test

def train_models(X_train, y_train, X_val, y_val, X_test, y_test):
    print("\nStarting model training...")
    os.makedirs('models', exist_ok=True)
    
    # 1. XGBoost
    print("\nTraining XGBoost...")
    start_time = time.time()
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000, max_depth=7,
        learning_rate=0.02, min_child_weight=3,
        subsample=0.8, colsample_bytree=0.8,
        colsample_bylevel=0.8,
        reg_alpha=0.1, reg_lambda=1.5, gamma=0.1,
        scale_pos_weight=15,
        eval_metric='aucpr', tree_method='hist',
        random_state=42, verbosity=0,
        early_stopping_rounds=50,
        n_jobs=-1
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    xgb_time = time.time() - start_time
    print(f"XGBoost trained in {xgb_time:.2f}s. Best iteration: {xgb_model.best_iteration}")
    
    # 2. LightGBM
    print("\nTraining LightGBM...")
    start_time = time.time()
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1000, max_depth=7,
        learning_rate=0.02, num_leaves=63,
        min_child_samples=20,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.5,
        scale_pos_weight=15,
        random_state=42, verbose=-1, n_jobs=-1
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    lgb_time = time.time() - start_time
    print(f"LightGBM trained in {lgb_time:.2f}s. Best iteration: {lgb_model.best_iteration_}")
    
    # 3. CatBoost
    print("\nTraining CatBoost...")
    start_time = time.time()
    cat_model = CatBoostClassifier(
        iterations=1000, depth=7,
        learning_rate=0.02, l2_leaf_reg=3,
        subsample=0.8, colsample_bylevel=0.8,
        scale_pos_weight=15,
        eval_metric='AUC',
        early_stopping_rounds=50,
        random_seed=42, verbose=0
    )
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
    cat_time = time.time() - start_time
    print(f"CatBoost trained in {cat_time:.2f}s. Best iteration: {cat_model.get_best_iteration()}")
    
    # Evaluation
    print("\nEvaluation on Test Set (AUC-ROC):")
    xgb_auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
    lgb_auc = roc_auc_score(y_test, lgb_model.predict_proba(X_test)[:, 1])
    cat_auc = roc_auc_score(y_test, cat_model.predict_proba(X_test)[:, 1])
    
    print(f"XGBoost AUC:  {xgb_auc:.4f}")
    print(f"LightGBM AUC: {lgb_auc:.4f}")
    print(f"CatBoost AUC: {cat_auc:.4f}")
    
    # Save Models
    joblib.dump(xgb_model, 'models/xgb_fraud_scorer.pkl')
    joblib.dump(lgb_model, 'models/lgb_fraud_scorer.pkl')
    joblib.dump(cat_model, 'models/cat_fraud_scorer.pkl')
    print("\nModels saved to models/ directory.")
    
    return xgb_model, lgb_model, cat_model

def train_stacked_ensemble(models, X_val, y_val):
    print("\nStarting stacked ensemble training...")
    xgb_model, lgb_model, cat_model = models
    
    # Step 1 — Get base model predictions on val set
    xgb_val = xgb_model.predict_proba(X_val)[:, 1]
    lgb_val = lgb_model.predict_proba(X_val)[:, 1]
    cat_val = cat_model.predict_proba(X_val)[:, 1]
    
    # Step 2 — Stack into meta-features
    meta_X_val = np.column_stack([xgb_val, lgb_val, cat_val])
    
    # Step 3 — Train Logistic Regression meta-learner
    meta_learner = LogisticRegression(C=1.0, random_state=42)
    meta_learner.fit(meta_X_val, y_val)
    
    # Step 4 — Print meta-learner weights
    print("\nMeta-learner weights:")
    print(f"XGBoost weight:  {meta_learner.coef_[0][0]:.4f}")
    print(f"LightGBM weight: {meta_learner.coef_[0][1]:.4f}")
    print(f"CatBoost weight: {meta_learner.coef_[0][2]:.4f}")
    print(f"Intercept:       {meta_learner.intercept_[0]:.4f}")
    
    # Step 5 — Create ensemble_predict_proba function
    def ensemble_predict_proba(X):
        p_xgb = xgb_model.predict_proba(X)[:, 1]
        p_lgb = lgb_model.predict_proba(X)[:, 1]
        p_cat = cat_model.predict_proba(X)[:, 1]
        meta_X = np.column_stack([p_xgb, p_lgb, p_cat])
        return meta_learner.predict_proba(meta_X)[:, 1]
    
    # Step 6 — Find optimal threshold using F2 score
    probs_val = ensemble_predict_proba(X_val)
    precisions, recalls, thresholds = precision_recall_curve(y_val, probs_val)
    
    # F2 = (5 * precision * recall) / (4 * precision + recall)
    f2 = (5 * precisions * recalls) / (4 * precisions + recalls + 1e-8)
    
    # The thresholds array has length n_thresholds, precisions/recalls have length n_thresholds + 1
    # We take the argmax of f2 (up to the last element which corresponds to threshold=1)
    idx = np.argmax(f2[:-1])
    THRESHOLD = float(thresholds[idx])
    
    print(f"\nOptimal Threshold (F2-score): {THRESHOLD:.4f}")
    
    # Save Meta-learner
    joblib.dump(meta_learner, 'models/meta_learner.pkl')
    print("Saved meta-learner to models/meta_learner.pkl")
    
    # Save Threshold
    with open('models/threshold.json', 'w') as f:
        json.dump({"threshold": THRESHOLD}, f)
    print("Saved optimal threshold to models/threshold.json")
    
    return meta_learner, THRESHOLD

def evaluate_ensemble(models, meta_learner, X_test, y_test, threshold, feature_count, total_rows):
    print("\nStarting full evaluation on held-out test set...")
    xgb_model, lgb_model, cat_model = models
    
    # helper for ensemble predictions
    def get_ensemble_probs(X):
        p_xgb = xgb_model.predict_proba(X)[:, 1]
        p_lgb = lgb_model.predict_proba(X)[:, 1]
        p_cat = cat_model.predict_proba(X)[:, 1]
        meta_X = np.column_stack([p_xgb, p_lgb, p_cat])
        return meta_learner.predict_proba(meta_X)[:, 1]
    
    # 1. Individual Model AUC-ROC
    xgb_auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
    lgb_auc = roc_auc_score(y_test, lgb_model.predict_proba(X_test)[:, 1])
    cat_auc = roc_auc_score(y_test, cat_model.predict_proba(X_test)[:, 1])
    
    ensemble_probs = get_ensemble_probs(X_test)
    ensemble_auc = roc_auc_score(y_test, ensemble_probs)
    
    # 2. PR-AUC
    pr_auc = average_precision_score(y_test, ensemble_probs)
    
    # 3. Decision threshold predictions
    y_pred = (ensemble_probs >= threshold).astype(int)
    
    # 4. Classification Report
    report_dict = classification_report(y_test, y_pred, digits=4, output_dict=True)
    report_str = classification_report(y_test, y_pred, digits=4)
    
    # 5. Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # 6. Rates
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # Print the exact report requested
    print("\n=== Individual Model AUC-ROC ===")
    print(f"XGBoost  : {xgb_auc:.5f}")
    print(f"LightGBM : {lgb_auc:.5f}")
    print(f"CatBoost : {cat_auc:.5f}")
    print(f"Ensemble : {ensemble_auc:.5f}  <-- final model")
    
    print("\n=== Ensemble Results on Test Set ===")
    print(f"PR-AUC (Average Precision): {pr_auc:.5f}")
    
    print("\nClassification report with precision/recall/f1 for Legit and Fraud classes:")
    print(report_str)
    
    print("\nConfusion matrix showing:")
    print(f"  True  Negatives : {tn:,}  (legit correctly passed)")
    print(f"  False Positives : {fp:,}  (legit flagged as fraud -- false alarms)")
    print(f"  False Negatives : {fn}        (fraud missed -- escaped)")
    print(f"  True  Positives : {tp}        (fraud caught)")
    
    print(f"\nFraud catch rate (Recall) : {recall*100:.2f}%")
    print(f"False alarm rate           : {far*100:.2f}%")
    
    # Save full report
    os.makedirs('reports', exist_ok=True)
    report_data = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_rows": total_rows,
        "feature_count": feature_count,
        "threshold": threshold,
        "metrics": {
            "individual_auc": {
                "xgboost": xgb_auc,
                "lightgbm": lgb_auc,
                "catboost": cat_auc
            },
            "ensemble_auc": ensemble_auc,
            "pr_auc": pr_auc,
            "recall": recall,
            "false_alarm_rate": far,
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp)
            },
            "classification_report": report_dict
        }
    }
    
    with open('reports/accuracy_report.json', 'w') as f:
        json.dump(report_data, f, indent=4)
    print("\nFull report saved to reports/accuracy_report.json")

def train_supporting_models(df, X_train_bal, y_train_bal, X_test, y_test, features):
    print("\n\n--- Training Supporting Models ---")
    
    # Model A — Fraud Type Classifier (RandomForest)
    print("\nTraining Model A: Fraud Type Classifier (RandomForest)...")
    
    # Filter training data to fraud rows only (from original df or provided training set)
    # The prompt implies we should re-assign labels based on a priority order
    fraud_df = df[df['Class'] == 1].copy()
    
    # Priority order for labels:
    # velocity_60s > 15           → velocity_attack
    # is_new_device=1 AND hour<5  → sim_swap
    # amount_scaled > 3           → account_takeover
    # is_new_recipient=1 AND cat_crypto=1 → money_mule
    # account_age_days < 30       → identity_fraud
    # cat_crypto=1 OR cat_gaming=1→ phishing
    # else                        → merchant_fraud
    
    def get_fraud_type(row):
        if row['velocity_60s'] > 15: return 'velocity_attack'
        if row['is_new_device'] == 1 and row['hour'] < 5: return 'sim_swap'
        if row.get('amount_scaled', 0) > 3: return 'account_takeover'
        if row['is_new_recipient'] == 1 and row.get('cat_crypto', 0) == 1: return 'money_mule'
        if row['account_age_days'] < 30: return 'identity_fraud'
        if row.get('cat_crypto', 0) == 1 or row.get('cat_gaming', 0) == 1: return 'phishing'
        return 'merchant_fraud'
    
    fraud_df['fraud_type'] = fraud_df.apply(get_fraud_type, axis=1)
    
    print("Fraud type distribution in training data:")
    print(fraud_df['fraud_type'].value_counts())
    
    X_fraud = fraud_df[features]
    y_fraud = fraud_df['fraud_type']
    
    rf_type_model = RandomForestClassifier(
        n_estimators=300, max_depth=10,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf_type_model.fit(X_fraud, y_fraud)
    joblib.dump(rf_type_model, 'models/rf_fraud_type.pkl')
    print("Saved Fraud Type Classifier to models/rf_fraud_type.pkl")
    
    # Model B — Logistic Regression (W29 live retraining)
    print("\nTraining Model B: Logistic Regression (Live Retraining)...")
    lr_live = LogisticRegression(
        C=0.5, max_iter=1000,
        class_weight='balanced', solver='saga',
        random_state=42, n_jobs=-1
    )
    lr_live.fit(X_train_bal, y_train_bal)
    
    lr_auc = roc_auc_score(y_test, lr_live.predict_proba(X_test[features])[:, 1])
    print(f"Live Retrain LR AUC-ROC on test set: {lr_auc:.5f}")
    # This model retrains in < 2 seconds when 50 analyst corrections accumulate (W29).
    # XGBoost ensemble is never retrained live.
    joblib.dump(lr_live, 'models/lr_live_retrain.pkl')
    print("Saved Live Retrain model to models/lr_live_retrain.pkl")
    
    # Model C — Prophet 7-day forecast
    print("\nTraining Model C: Prophet Fraud Forecast...")
    
    # Build daily fraud counts
    # Using 'Time' to derive day_number. Time is in seconds.
    df['day_number'] = (df['Time'] // 86400).astype(int)
    daily_fraud = df[df['Class'] == 1].groupby('day_number').size().reset_index(name='y')
    
    # Convert day numbers to real dates starting 2024-01-01
    start_date = datetime(2024, 1, 1)
    daily_fraud['ds'] = daily_fraud['day_number'].apply(lambda d: start_date + timedelta(days=d))
    
    # Indian festivals
    festivals = pd.DataFrame({
        'holiday': ['diwali', 'navratri', 'salary_day', 'ipl_final', 'holi', 'dussehra', 'eid', 'christmas', 'new_year', 'onam'],
        'ds': [start_date + timedelta(days=np.random.randint(0, 300)) for _ in range(10)], # Randomly distribute for demo
        'lower_window': 0,
        'upper_window': 1,
    })
    
    m = Prophet(
        holidays=festivals,
        weekly_seasonality=True,
        changepoint_prior_scale=0.1,
        seasonality_prior_scale=10,
        holidays_prior_scale=20,
        interval_width=0.80
    )
    m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    m.fit(daily_fraud[['ds', 'y']])
    
    future = m.make_future_dataframe(periods=7)
    forecast = m.predict(future)
    
    # Print the 7-day forecast table
    print("\n7-Day Fraud Forecast:")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7))
    
    # Save model and forecast
    joblib.dump(m, 'models/prophet_forecaster.pkl')
    
    forecast_tail = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7)
    forecast_tail['ds'] = forecast_tail['ds'].dt.strftime('%Y-%m-%d')
    forecast_json = forecast_tail.to_dict(orient='records')
    
    with open('models/forecast_7day.json', 'w') as f:
        json.dump(forecast_json, f, indent=4)
    print("Saved Prophet model and 7-day forecast results.")

def setup_explainability(xgb_model, X_train_bal, features):
    print("\n\n--- Setting up Explainability (SHAP & LIME) ---")
    
    # 1. SHAP Setup
    print("Initializing SHAP TreeExplainer...")
    explainer_shap = shap.TreeExplainer(xgb_model)
    joblib.dump(explainer_shap, 'models/shap_explainer.pkl')
    
    # SHAP 8-dimension aggregation map (PRD W05)
    # Filtered to only include features that exist in the features list
    SHAP_8D_RAW = {
        "velocity": ["velocity_60s", "velocity_48h_count", "velocity_log", "amount_x_velocity", "velocity_x_new_device"],
        "device_risk": ["is_new_device", "new_device_x_night", "is_sim_swap_signal"],
        "time_risk": ["hour", "time_scaled", "new_device_x_night"],
        "amount_risk": ["amount_scaled", "amount_log", "new_recip_x_amount", "young_account_x_large"],
        "recipient": ["is_new_recipient", "is_first_merchant", "new_recip_x_amount"],
        "network": ["v_fraud_signal", "V4", "V11", "V14", "v_mean", "v_std"],
        "identity": ["account_age_days", "is_sim_swap_signal", "young_account_x_large", "city_risk_score"],
        "location": ["city_risk_score", "is_festival_day", "festival_x_amount"]
    }
    
    SHAP_8D = {dim: [f for f in flist if f in features] for dim, flist in SHAP_8D_RAW.items()}
    
    with open('models/shap_8d_map.json', 'w') as f:
        json.dump(SHAP_8D, f, indent=4)
    print("Saved SHAP 8D aggregation map to models/shap_8d_map.json")
    
    # Reference fingerprints for radar overlays
    FRAUD_FINGERPRINTS = {
        "velocity_attack": {
            "velocity": 0.95, "device_risk": 0.4, "time_risk": 0.3, "amount_risk": 0.5,
            "recipient": 0.3, "network": 0.2, "identity": 0.2, "location": 0.1
        },
        "sim_swap": {
            "velocity": 0.2, "device_risk": 0.95, "time_risk": 0.85, "amount_risk": 0.8,
            "recipient": 0.5, "network": 0.3, "identity": 0.7, "location": 0.2
        },
        "social_engineering": {
            "velocity": 0.1, "device_risk": 0.3, "time_risk": 0.7, "amount_risk": 0.6,
            "recipient": 0.9, "network": 0.2, "identity": 0.3, "location": 0.2
        }
    }
    
    with open('models/fraud_fingerprints.json', 'w') as f:
        json.dump(FRAUD_FINGERPRINTS, f, indent=4)
    print("Saved Fraud Fingerprints to models/fraud_fingerprints.json")
    
    # 2. LIME Setup
    print("Initializing LIME TabularExplainer...")
    # IMPORTANT: Do NOT call LIME during GET /explain requests.
    # LIME must be called eagerly inside POST /score and the 
    # result cached. GET /explain just reads the cache.
    # Why? LIME takes ~200ms, which would break the sub-50ms latency budget 
    # for a single explain request if generated on-the-fly.
    
    explainer_lime = LimeTabularExplainer(
        np.array(X_train_bal),
        feature_names=features,
        class_names=['Legit', 'Fraud'],
        mode='classification',
        random_state=42
    )
    # joblib.dump(explainer_lime, 'models/lime_explainer.pkl')
    with open('models/lime_explainer.pkl', 'wb') as f:
        dill.dump(explainer_lime, f)
    print("Saved LIME explainer to models/lime_explainer.pkl")

if __name__ == "__main__":
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # 1. Synthesize Data
    df_synthesized = synthesize_data()
    
    if df_synthesized is not None:
        total_rows = len(df_synthesized)
        # Save synthesized base data
        df_synthesized.to_csv("data/synthesized.csv", index=False)
        print("Saved synthesized data to data/synthesized.csv")
        
        # 2. Feature Engineering
        df_final, features = feature_engineering(df_synthesized)
        feature_count = len(features)
        
        # Save final feature-engineered dataset for training (optional but helpful)
        df_final.to_csv("data/final_features.csv", index=False)
        print("Final feature engineering complete. Data saved to data/final_features.csv")
        
        # 3. Data Splitting and Balancing
        X_train_bal, X_val, X_test, y_train_bal, y_val, y_test = split_and_balance(df_final, features)
        
        # 4. Model Training (Base Models)
        base_models = train_models(X_train_bal, y_train_bal, X_val, y_val, X_test, y_test)
        
        # 5. Stacked Ensemble & Threshold Optimization
        meta_learner, threshold = train_stacked_ensemble(base_models, X_val, y_val)
        
        # 6. Full Evaluation
        evaluate_ensemble(base_models, meta_learner, X_test, y_test, threshold, feature_count, total_rows)
        
        # 7. Supporting Models
        train_supporting_models(df_final, X_train_bal, y_train_bal, X_test, y_test, features)
        
        # 8. Explainability Setup
        setup_explainability(base_models[0], X_train_bal, features)
        
        print("\nProject setup, model training, evaluation, supporting models, and explainability complete.")
