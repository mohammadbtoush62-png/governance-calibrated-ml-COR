Title: Supplementary Materials for "Governance Calibrated Machine Learning Pipeline for Change Order Ratio Prediction"

Contents

data/data.csv — anonymized dataset (or data_synthetic.csv)
data/data_dictionary.csv — variable definitions
code/preprocessing_script.py
code/train_models.py
code/hyperparams.json
results/cv_results.csv
results/model_coefficients.csv
results/leave_one_out_influence.csv
results/bootstrap_results.csv
results/permutation_test_results.csv
results/prediction_intervals.csv
docs/figures.pdf
requirements.txt
random_seeds.txt
provenance.txt
LICENSE
Reproduction steps

Create Python env: pip install -r requirements.txt
Run code/preprocessing_script.py to generate foldwise preprocessed data
Run code/train_models.py (nested LOOCV + hyperparameter search)
Run diagnostics and interpretability scripts (bootstrap, permutation, SHAP)
Outputs saved in results/
Environment

Python 3.11; scikit-learn 1.4; numpy 1.26; pandas; shap; matplotlib; seaborn
Random seeds

Global seed: 2026 (details in random_seeds.txt)
Contact
Corresponding author: Mohammed A. KA. Al-Btoush, Email: muhammad.albtoosh@iu.edu.jo
