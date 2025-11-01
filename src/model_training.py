# src/model_training.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
import warnings
warnings.filterwarnings("ignore")

# ---------- Config ----------
DATA_PATH = "data/processed_dataset.csv" # from Member1/2 (they saved this)
TARGET_COL = "Yield"                  # matches eda.py
MODEL_DIR = "models"
RANDOM_STATE = 42

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------- Helpers ----------
# ---------- Helpers ----------
def load_data(path=DATA_PATH, target=TARGET_COL):
    df = pd.read_csv(path)
    
    # Remove the columns that "leak" the answer
    leaky_columns = ['production_per_area', 'Production', 'Area']
    
    # Find which of these leaky columns actually exist in the dataframe
    cols_to_drop = [col for col in leaky_columns if col in df.columns]
    
    if cols_to_drop:
        print(f"Removing leaky columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    # -------------------------------

    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in {path}")
    
    X = df.drop(columns=[target])
    y = df[target]
    
    print(f"Data loaded. X shape: {X.shape}, y shape: {y.shape}")
    return X, y

def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

# ---------- Baseline training & compare ----------
def train_and_compare(X_train, y_train, X_test, y_test):
    results = {}

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        res = evaluate_model(y_test, preds)
        # add cross-val score (R2)
        cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring="r2", n_jobs=-1)
        res["CV_R2_mean"] = float(np.mean(cv_scores))
        results[name] = {"metrics": res, "model": model}
        print(f"{name} -> {res}")

    return results

# ---------- Hyperparameter tuning (simple/randomized) ----------
def tune_random_forest(X_train, y_train, n_iter=12):
    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }
    search = RandomizedSearchCV(rf, param_dist, n_iter=n_iter, cv=3, scoring="r2", random_state=RANDOM_STATE, n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)
    return search

def tune_xgboost(X_train, y_train, n_iter=12):
    xgb = XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0]
    }
    search = RandomizedSearchCV(xgb, param_dist, n_iter=n_iter, cv=3, scoring="r2", random_state=RANDOM_STATE, n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)
    return search

# ---------- Main flow ----------
def main():
    X, y = load_data()
    # simple split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # Baseline
    print("Training baseline models...")
    baseline_results = train_and_compare(X_train, y_train, X_test, y_test)

    # Choose best baseline by R2 on test
    best_name = max(baseline_results.keys(), key=lambda n: baseline_results[n]['metrics']['R2'])
    print("Best baseline:", best_name)

    # Tune only the best candidate among RF/XGB if applicable
    tuned_search = None
    if best_name == "RandomForest":
        print("Tuning RandomForest...")
        tuned_search = tune_random_forest(X_train, y_train, n_iter=12)
    elif best_name == "XGBoost":
        print("Tuning XGBoost...")
        tuned_search = tune_xgboost(X_train, y_train, n_iter=12)
    else:
        # If linear was best (rare), just keep it
        print("Linear model best — skipping heavy tuning.")
        tuned_search = None

    if tuned_search:
        print("Best params:", tuned_search.best_params_)
        best_model = tuned_search.best_estimator_
    else:
        best_model = baseline_results[best_name]['model']

    # Final evaluation
    y_pred = best_model.predict(X_test)
    final_metrics = evaluate_model(y_test, y_pred)
    print("Final model metrics:", final_metrics)

    # Save model + metrics
    model_path = os.path.join(MODEL_DIR, "final_model.pkl")
    joblib.dump(best_model, model_path)
    print("Saved model to", model_path)

    metrics_path = os.path.join(MODEL_DIR, "final_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    print("Saved metrics to", metrics_path)

if __name__ == "__main__":
    main()
