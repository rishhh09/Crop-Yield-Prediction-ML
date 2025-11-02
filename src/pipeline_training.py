import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Import the "translator" we just defined
from preprocessing import create_preprocessor

# ---------- Config ----------
# We now use the RAW data file
DATA_PATH = os.path.join("data", "uttarakhand_crop_yield.csv")
TARGET_COL = "Yield"
MODEL_DIR = "models"
# This is our new, final model file
PIPELINE_PATH = os.path.join(MODEL_DIR, "final_pipeline.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "final_metrics.json")
RANDOM_STATE = 42

os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    print(f"Loading raw data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # 1. Clean column names (a small but important step from eda.py)
    df.columns = df.columns.str.strip()

    # 2. Define features (X) and target (y)
    # X is the *entire* dataframe (raw)
    # y is just the target column
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # 3. Split the *raw* data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    print(f"Raw data split into {len(X_train)} train and {len(X_test)} test samples.")

    # 4. Get the "translator"
    preprocessor = create_preprocessor()

    # 5. Get the "brain" (our best model from the previous run)
    # We use the best params we found before!
    best_params = {
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }
    rf_model = RandomForestRegressor(**best_params)

    # 6. Create the full Pipeline (Translator + Brain)
    main_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', rf_model)
    ])

    # 7. Train the *entire* pipeline on the *raw* training data
    print("Training the full pipeline (preprocessor + model)...")
    main_pipeline.fit(X_train, y_train)
    print("Pipeline training complete.")

    # 8. Evaluate the pipeline on the *raw* test data
    y_pred = main_pipeline.predict(X_test)
    final_r2 = r2_score(y_test, y_pred)
    
    print(f"\n--- Final Pipeline Performance ---")
    print(f"R-squared (R2): {final_r2:0.4f}")
    
    # 9. Save the single pipeline file
    joblib.dump(main_pipeline, PIPELINE_PATH)
    print(f"\nSuccessfully saved the new pipeline to {PIPELINE_PATH}")

    # 10. Save the metrics
    final_metrics = {'R2': final_r2}
    with open(METRICS_PATH, "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"Saved new metrics to {METRICS_PATH}")

if __name__ == "__main__":
    main()
