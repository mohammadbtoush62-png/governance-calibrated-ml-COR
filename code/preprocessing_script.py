# code/preprocessing_script.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

INPUT = "data/data.csv"
OUT_DIR = "data/processed_folds"
NUMERIC = ["TV", "DC", "RFIs", "DUR"]
ORDINAL = ["GRL", "CL"]
RESPONSE = "COR"
SEED = 2026

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT)

loo = LeaveOneOut()
for i, (train_idx, test_idx) in enumerate(loo.split(df), start=1):
    train = df.iloc[train_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)

    # Fit scaler on training fold only
    scaler = StandardScaler()
    train_num = scaler.fit_transform(train[NUMERIC])
    test_num = scaler.transform(test[NUMERIC])

    train_proc = pd.DataFrame(train_num, columns=[f"{c}_s" for c in NUMERIC])
    test_proc = pd.DataFrame(test_num, columns=[f"{c}_s" for c in NUMERIC])

    # Copy ordinals and response unchanged (keep original encoding)
    train_proc[ORDINAL] = train[ORDINAL].values
    test_proc[ORDINAL] = test[ORDINAL].values
    train_proc[RESPONSE] = train[RESPONSE].values
    test_proc[RESPONSE] = test[RESPONSE].values

    # Save fold files
    train_proc.to_csv(os.path.join(OUT_DIR, f"fold_{i}_train.csv"), index=False)
    test_proc.to_csv(os.path.join(OUT_DIR, f"fold_{i}_test.csv"), index=False)

    # Save scaler params for reproducibility
    params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "numeric_cols": NUMERIC
    }
    pd.Series(params).to_json(os.path.join(OUT_DIR, f"fold_{i}_scaler.json"))

print(f"Processed {i} LOOCV folds -> {OUT_DIR}")
