from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib

DATASET = Path("data/ml/run_level_features.csv")
MODEL_OUT = Path("models/coarsening_rf.joblib")

FEATURES = [
    "r_mean_early_mean","r_mean_early_std","r_mean_early_slope",
    "n_early_mean","n_early_slope","circ_early_mean",
    "avg_degree_early_mean","avg_degree_early_slope",
    "density_early_mean","density_early_slope",
    "gcr_early_mean","gcr_early_slope",
    "energy_proxy_early",
]

def main():
    df = pd.read_csv(DATASET)

    # Choose target: coarsening rate is always available (best for now)
    ycol = "y_coarsen_rate"
    df = df.dropna(subset=[ycol])

    X = df[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    y = df[ycol].to_numpy()
    groups = df["family"].astype(str).to_numpy()

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    Xtr, Xte = X[train_idx], X[test_idx]
    ytr, yte = y[train_idx], y[test_idx]

    model = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1,
    )
    model.fit(Xtr, ytr)

    pred = model.predict(Xte)
    mae = mean_absolute_error(yte, pred)
    rmse = mean_squared_error(yte, pred) ** 0.5
    r2 = r2_score(yte, pred)

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "target": ycol}, MODEL_OUT)

    print("✅ Model saved:", MODEL_OUT)
    print(f"Test MAE:  {mae:.6f}")
    print(f"Test RMSE: {rmse:.6f}")
    print(f"Test R2:   {r2:.3f}")

    # Show top importances for the demo/writeup
    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nTop feature importances:")
    print(importances.head(8).to_string())

if __name__ == "__main__":
    main()
