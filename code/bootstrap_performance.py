# code/bootstrap_performance.py
import pandas as pd
import numpy as np
import os

IN = "results/cv_results.csv"
OUT = "results/bootstrap_results.csv"
os.makedirs("results", exist_ok=True)

df = pd.read_csv(IN)
models = {"y_lr":"Linear","y_lasso":"LASSO","y_ridge":"Ridge","y_svr":"SVR"}
y = df["y_true"].values
n_boot = 5000
rng = np.random.default_rng(2026)

def metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return rmse, mae, r2

out_rows = []
for col,name in models.items():
    boot_metrics = []
    preds = df[col].values
    n = len(df)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        rmse, mae, r2 = metrics(y[idx], preds[idx])
        boot_metrics.append((rmse, mae, r2))
    boot_arr = np.array(boot_metrics)
    for i, metric_name in enumerate(["RMSE","MAE","R2"]):
        lo, hi = np.percentile(boot_arr[:,i], [2.5,97.5])
        mean = np.mean(boot_arr[:,i])
        out_rows.append({
            "model": name,
            "metric": metric_name,
            "mean": mean,
            "ci_lower": lo,
            "ci_upper": hi
        })

pd.DataFrame(out_rows).to_csv(OUT, index=False)
print(f"Wrote bootstrap results to {OUT}")
