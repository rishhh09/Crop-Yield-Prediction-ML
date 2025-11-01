import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# ---------- Config ----------
DATA_PATH = os.path.join("data", "processed_dataset.csv") 
TARGET_COL = "Yield"
MODEL_PATH = os.path.join("models", "final_model.pkl")
METRICS_PATH = os.path.join("models", "final_metrics.json")
RANDOM_STATE = 42

# ---------- Helper Functions (Must match training script!) ----------

def load_data(path=DATA_PATH, target=TARGET_COL):
    """
    Loads data and applies the critical leakage fix.
    This MUST be identical to the function in model_training.py
    """
    df = pd.read_csv(path)
    
    # --- The Critical Fix ---
    leaky_columns = ['production_per_area', 'Production', 'Area']
    cols_to_drop = [col for col in leaky_columns if col in df.columns]
    
    if cols_to_drop:
        print(f"INFO: Removing leaky columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    # -------------------------------

    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in {path}")
    
    X = df.drop(columns=[target])
    y = df[target]
    
    return X, y

def evaluate_model(y_true, y_pred):
    """
    Calculates metrics.
    This MUST be identical to the function in model_training.py
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) # Using np.sqrt for compatibility
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

# ---------- Main Test Flow ----------
def main():
    print(f"--- Loading Model from {MODEL_PATH} ---")
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        return
        
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
    
    # 1. Load the "official" metrics saved by the training script
    print(f"\n--- Loading Official Metrics from {METRICS_PATH} ---")
    if not os.path.exists(METRICS_PATH):
        print(f"ERROR: Metrics file not found at {METRICS_PATH}")
        return
        
    with open(METRICS_PATH, 'r') as f:
        saved_metrics = json.load(f)
    print(f"Official metrics (from training): {saved_metrics}")

    # 2. Re-create the test set to validate
    print("\n--- Re-creating Test Set for Validation ---")
    X, y = load_data()
    # We re-split with the *same random_state* to get the exact same test set
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    print(f"Test set created with {len(X_test)} samples.")
    
    # 3. Make new predictions and calculate metrics
    print("\n--- Making New Predictions on Test Set ---")
    y_pred_new = model.predict(X_test)
    test_metrics_new = evaluate_model(y_test, y_pred_new)
    print(f"Newly calculated metrics: {test_metrics_new}")

    # 4. Compare the metrics
    print("\n--- Comparison ---")
    official_r2 = saved_metrics['R2']
    new_r2 = test_metrics_new['R2']
    
    # We round to 6 decimal places for a fair comparison
    if np.round(official_r2, 6) == np.round(new_r2, 6):
        print("✅ SUCCESS: Model test passed!")
        print("The loaded model's performance matches the official training metrics.")
    else:
        print("❌ FAILURE: Model test failed.")
        print(f"Official R2 ({official_r2}) does not match new R2 ({new_r2}).")

    # 5. Bonus: Predict on a single sample
    print("\n--- Example: Predicting a Single Sample ---")
    # Get the first row of the test set as a sample
    single_sample = X_test.iloc[[0]] 
    
    # model.predict() expects a 2D array, which .iloc[[0]] provides
    sample_prediction = model.predict(single_sample)
    
    print(f"Data for sample 0: (first 5 features) {single_sample.values[0, :5]}...")
    print(f"Predicted Yield for sample: {sample_prediction[0]}")
    print(f"Actual Yield for sample:    {y_test.iloc[0]}")

if __name__ == "__main__":
    main()