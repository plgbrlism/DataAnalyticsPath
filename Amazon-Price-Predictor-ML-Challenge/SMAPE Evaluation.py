# SMAPE Evaluation

import numpy as np
test_out_path = r'c:\Users\Gerald\Documents\SQL PRACTICE\Amazon ML Challenge 2025\student_resource\test_out.csv'

test_output = (test_out_path)

# -----------------------------------------------------
    # --- STEP 3B: VALIDATE MODEL USING SMAPE (Internal) ---
    # -----------------------------------------------------

def calculate_smape(y_true, y_pred):
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
        
    # Ensure predictions are positive before SMAPE calculation
    y_pred[y_pred < 0] = 0.01 
        
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        
    return np.mean(numerator / denominator) * 100

    # 1. Predict on the internal validation set (log-transformed)
y_val_pred_log = xgb_model.predict(X_val)

    # 2. Inverse transform targets and predictions back to original scale
y_val_original = np.expm1(y_val)
y_pred_original = np.expm1(y_val_pred_log)

    # 3. Calculate SMAPE
validation_smape_score = calculate_smape(y_val_original, y_pred_original)

print(f"Validation SMAPE Score (Internal): {validation_smape_score:.4f}%")
    
    # --- REST OF THE CODE CONTINUES WITH TEST SET PREDICTION ---
