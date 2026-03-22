# code/train_models.py
import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from joblib import dump

DATA_PROCESSED = "data/processed_folds"
OUT_RESULTS = "results"
HYPERPARAMS_FILE = "code/hyperparams.json"
SEED = 2026
os.makedirs(OUT_RESULTS, exist_ok=True)

# Load hyperparameter grid
with open(HYPERPARAMS_FILE, "r") as f:
    hyper = json.load(f)

loo = LeaveOneOut()
fold_files = sorted([f for f in os.listdir(DATA_PROCESSED) if f.endswith("_train.csv")])
n = len(fold_files)

rows = []
models_info = {"lasso": [], "ridge": [], "svr": []}

for i in range(1, n+1):
    train = pd.read_csv(os.path.join(DATA_PROCESSED, f"fold_{i}_train.csv"))
    test = pd.read_csv(os.path.join(DATA_PROCESSED, f"fold_{i}_test.csv"))

    X_train = train.drop(columns=["COR"])
    y_train = train["COR"].values
    X_test = test.drop(columns=["COR"])
    y_test = test["COR"].values

    # Baseline linear regression
    lr = LinearRegression().fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    # LASSO (inner LOOCV via GridSearchCV with leave-one-out)
    lasso = GridSearchCV(Lasso(max_iter=10000, random_state=SEED),
                         param_grid=hyper["lasso"],
                         cv=LeaveOneOut(), scoring="neg_mean_squared_error", n_jobs=1)
    lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)
    models_info["lasso"].append(lasso.best_params_)

    # Ridge
    ridge = GridSearchCV(Ridge(random_state=SEED),
                         param_grid=hyper["ridge"],
                         cv=LeaveOneOut(), scoring="neg_mean_squared_error", n_jobs=1)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    models_info["ridge"].append(ridge.best_params_)

    # SVR
    svr = GridSearchCV(SVR(kernel="rbf"),
                       param_grid=hyper["svr"],
                       cv=LeaveOneOut(), scoring="neg_mean_squared_error", n_jobs=1)
    svr.fit(X_train, y_train)
    y_pred_svr = svr.predict(X_test)
    models_info["svr"].append(svr.best_params_)

    # Save per-fold predictions
    rows.append({
        "fold": i,
        "y_true": float(y_test[0]),
        "y_lr": float(y_pred_lr[0]),
        "y_lasso": float(y_pred_lasso[0]),
        "y_ridge": float(y_pred_ridge[0]),
        "y_svr": float(y_pred_svr[0]),
        "best_lasso": lasso.best_params_,
        "best_ridge": ridge.best_params_,
        "best_svr": svr.best_params_
    })

# Save cv_results
cv_df = pd.DataFrame(rows)
cv_df.to_csv(os.path.join(OUT_RESULTS, "cv_results.csv"), index=False)

# Save model hyperparams summary
with open(os.path.join(OUT_RESULTS, "selected_hyperparams.json"), "w") as f:
    json.dump(models_info, f, indent=2)

# Compute aggregate metrics per model
def metrics(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred))
    }

y_true = cv_df["y_true"].values
for model_col, name in [("y_lr","Linear"),("y_lasso","LASSO"),("y_ridge","Ridge"),("y_svr","SVR")]:
    m = metrics(y_true, cv_df[model_col].values)
    pd.Series(m).to_csv(os.path.join(OUT_RESULTS, f"metrics_{name.lower()}.csv"))

print("Training complete. Results saved to results/")
