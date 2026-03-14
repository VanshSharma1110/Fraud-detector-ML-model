import pandas as pd
import numpy as np
import joblib
import json
import time
import os
import shap
import dill
from lime.lime_tabular import LimeTabularExplainer

# Load models and assets ONCE at module import
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

def load_json(filename):
    with open(os.path.join(MODEL_DIR, filename), 'r') as f:
        return json.load(f)

# Base Models
model_xgb = joblib.load(os.path.join(MODEL_DIR, 'xgb_fraud_scorer.pkl'))
model_lgb = joblib.load(os.path.join(MODEL_DIR, 'lgb_fraud_scorer.pkl'))
model_cat = joblib.load(os.path.join(MODEL_DIR, 'cat_fraud_scorer.pkl'))

# Meta & Supporting Models
meta_learner = joblib.load(os.path.join(MODEL_DIR, 'meta_learner.pkl'))
model_rf_type = joblib.load(os.path.join(MODEL_DIR, 'rf_fraud_type.pkl'))

# Explainers
explainer_shap = joblib.load(os.path.join(MODEL_DIR, 'shap_explainer.pkl'))
with open(os.path.join(MODEL_DIR, 'lime_explainer.pkl'), 'rb') as f:
    explainer_lime = dill.load(f)

# Transformers & Config
robust_scaler = joblib.load(os.path.join(MODEL_DIR, 'robust_scaler.pkl'))
FEATURES = load_json('features.json')
SHAP_8D = load_json('shap_8d_map.json')
THRESHOLD_DATA = load_json('threshold.json')
THRESHOLD = 0.45  # Changed from THRESHOLD_DATA['threshold'] to fixed 0.45 per requirements

def score_transaction(txn: dict) -> dict:
    t_start = time.time()
    
    # 1. Build feature row from txn dict
    # Fill missing features with 0 as requested
    row_dict = {f: txn.get(f, 0) for f in FEATURES}

    # --- Derive engineered features from raw inputs ---
    # These mirror the feature engineering done in train.py.
    # Only compute a derived feature if the caller did NOT
    # already supply it (check txn, not row_dict, to avoid
    # overwriting explicitly provided values).

    v_cols = [f'V{i}' for i in range(1, 29)]
    v_vals = [txn.get(c, 0) for c in v_cols]

    # V-aggregate features
    if 'v_mean' not in txn:
        row_dict['v_mean'] = float(np.mean(v_vals))
    if 'v_std' not in txn:
        row_dict['v_std'] = float(np.std(v_vals, ddof=1)) if len(v_vals) > 1 else 0.0
    if 'v_max' not in txn:
        row_dict['v_max'] = float(np.max(v_vals))
    if 'v_min' not in txn:
        row_dict['v_min'] = float(np.min(v_vals))
    if 'v_skew' not in txn:
        mean_v = np.mean(v_vals)
        std_v  = np.std(v_vals, ddof=1) if np.std(v_vals, ddof=1) > 0 else 1e-8
        row_dict['v_skew'] = float(
            np.mean(((np.array(v_vals) - mean_v) / std_v) ** 3)
        )
    if 'v_kurt' not in txn:
        mean_v = np.mean(v_vals)
        std_v  = np.std(v_vals, ddof=1) if np.std(v_vals, ddof=1) > 0 else 1e-8
        row_dict['v_kurt'] = float(
            np.mean(((np.array(v_vals) - mean_v) / std_v) ** 4) - 3
        )

    # Fraud signal composite
    # Formula from train.py:
    # v_fraud_signal = -V14*0.3 - V12*0.2 - V10*0.2 + V4*0.15 + V11*0.15
    if 'v_fraud_signal' not in txn:
        row_dict['v_fraud_signal'] = float(
            (-txn.get('V14', 0) * 0.3) +
            (-txn.get('V12', 0) * 0.2) +
            (-txn.get('V10', 0) * 0.2) +
            ( txn.get('V4',  0) * 0.15) +
            ( txn.get('V11', 0) * 0.15)
        )

    # Log transforms
    if 'amount_log' not in txn:
        row_dict['amount_log'] = float(
            np.log1p(abs(txn.get('amount_inr', 0)))
        )
    if 'velocity_log' not in txn:
        row_dict['velocity_log'] = float(
            np.log1p(abs(txn.get('velocity_60s', 0)))
        )

    # Interaction features
    amount_scaled  = txn.get('amount_scaled', 0)
    velocity_60s   = txn.get('velocity_60s', 0)
    is_new_device  = txn.get('is_new_device', 0)
    hour           = txn.get('hour', 12)
    is_new_recip   = txn.get('is_new_recipient', 0)
    acct_age       = txn.get('account_age_days', 999)
    amount_inr     = txn.get('amount_inr', 0)
    is_festival    = txn.get('is_festival_day', 0)

    if 'amount_x_velocity' not in txn:
        row_dict['amount_x_velocity'] = float(amount_scaled * velocity_60s)
    if 'new_device_x_night' not in txn:
        row_dict['new_device_x_night'] = float(
            is_new_device * (1 if hour < 6 else 0)
        )
    if 'new_recip_x_amount' not in txn:
        row_dict['new_recip_x_amount'] = float(is_new_recip * amount_scaled)
    if 'young_account_x_large' not in txn:
        row_dict['young_account_x_large'] = float(
            (1 if acct_age < 30 else 0) *
            (1 if amount_inr > 10000 else 0)
        )
    if 'velocity_x_new_device' not in txn:
        row_dict['velocity_x_new_device'] = float(velocity_60s * is_new_device)
    if 'festival_x_amount' not in txn:
        row_dict['festival_x_amount'] = float(is_festival * amount_scaled)

    # --- End feature derivation ---

    row = pd.DataFrame([row_dict])
    
    # 2. Run ensemble
    p_xgb = float(model_xgb.predict_proba(row)[0][1])
    p_lgb = float(model_lgb.predict_proba(row)[0][1])
    p_cat = float(model_cat.predict_proba(row)[0][1])
    
    # Use weighted average instead of meta-learner to avoid LGB driving score to 0
    score = (p_xgb * 0.5) + (p_lgb * 0.1) + (p_cat * 0.4)
    
    # 3. Apply threshold
    is_fraud = bool(score >= THRESHOLD)
    risk_level = "HIGH" if score > 0.75 else "MEDIUM" if score > 0.40 else "LOW"
    
    # 4. Fraud type (only if is_fraud=True)
    fraud_type = None
    if is_fraud:
        fraud_type = str(model_rf_type.predict(row)[0])
    
    # 5. LIME — top 4 risk-increasing reasons only
    # IMPORTANT: Do NOT call LIME during GET /explain requests.
    # LIME must be called eagerly inside POST /score and the result cached.
    # LIME takes ~200ms, would break latency budget for real-time explain.
    # Here we include it as part of the scoring result.
    lime_exp = explainer_lime.explain_instance(
        row.values[0], 
        model_xgb.predict_proba, 
        num_features=4
    )
    lime_reasons = [
        {"feature": f, "weight": round(float(w), 4), "direction": "RISK" if w > 0 else "SAFE"}
        for f, w in lime_exp.as_list() if w > 0
    ]
    
    # 6. SHAP 8D aggregation
    raw_shap = explainer_shap.shap_values(row)
    # TreeExplainer returns a list of arrays for binary classification in some versions, 
    # or a single array. Handling both.
    if isinstance(raw_shap, list):
        raw_shap = raw_shap[1][0] # Positive class, first row
    else:
        raw_shap = raw_shap[0] # Single row
        
    shap_dict = dict(zip(FEATURES, raw_shap))
    shap_8d = {
        dim: round(float(sum(shap_dict.get(f, 0) for f in feats)), 4)
        for dim, feats in SHAP_8D.items()
    }
    
    inference_ms = round((time.time() - t_start) * 1000, 1)
    
    # 7. Return
    return {
        "score": round(score * 100, 1),  # 0-100
        "raw_proba": round(score, 4),
        "is_fraud": is_fraud,
        "risk_level": risk_level,
        "fraud_type": fraud_type,
        "model_breakdown": {
            "xgb": round(p_xgb, 4), 
            "lgb": round(p_lgb, 4), 
            "cat": round(p_cat, 4)
        },
        "lime": lime_reasons,
        "shap_8d": shap_8d,
        "inference_ms": inference_ms
    }

if __name__ == "__main__":
    print(f"Loaded FraudShield ML Inference Engine. Threshold: {THRESHOLD:.4f}")
    
    # Manual Smoke Tests
    
    # Test 1 — obvious fraud (high velocity, new device, 3am)
    txn_fraud = {
        "velocity_60s": 22.5,
        "is_new_device": 1,
        "hour": 3,
        "V14": -5.5,
        "V12": -4.0,
        "v_fraud_signal": 2.5,
        "is_sim_swap_signal": 1,
        "amount_scaled": 2.5,
        "cat_crypto": 1
    }
    
    # Test 2 — obvious legit (low velocity, known device, 2pm)
    txn_legit = {
        "velocity_60s": 0.5,
        "is_new_device": 0,
        "hour": 14,
        "V14": 0.5,
        "V12": 0.8,
        "v_fraud_signal": -0.5,
        "amount_scaled": 0.2,
        "is_new_recipient": 0
    }
    
    # Test 3 — borderline (medium signals)
    txn_borderline = {
        "velocity_60s": 4.5,
        "is_new_device": 1,
        "hour": 10,
        "V14": -1.2,
        "v_fraud_signal": 0.5,
        "amount_scaled": 1.1,
        "is_new_recipient": 1
    }
    
    for i, txn in enumerate([txn_fraud, txn_legit, txn_borderline], 1):
        print(f"\n--- Smoke Test {i} ---")
        res = score_transaction(txn)
        print(f"Score   : {res['score']}")
        print(f"Is Fraud: {res['is_fraud']}")
        print(f"Level   : {res['risk_level']}")
        if res['is_fraud']:
            print(f"Type    : {res['fraud_type']}")
        print(f"Latency : {res['inference_ms']}ms")
        print(f"SHAP 8D : {res['shap_8d']}")
