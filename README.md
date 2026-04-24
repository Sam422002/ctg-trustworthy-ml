# Results Summary

This folder will be populated with CSV outputs when you run `notebooks/CTG_Paper_Pipeline.ipynb`.

## Files produced by the notebook

| File | Source cell | Description |
|------|-------------|-------------|
| `leakage_audit.csv` | §3 Leakage Audit | Test metrics under P-LEAK, P-SCALE, P-CLEAN pipelines |
| `tuned_smote.csv` | §5 Tuning | Best GridSearchCV hyperparameters + test metrics, SMOTE pipeline |
| `tuned_cw.csv` | §5 Tuning | Best hyperparameters + test metrics, Class-Weighting pipeline |
| `repeated_seeds.csv` | §6 30-Seed Evaluation | Raw per-seed metrics for each (model, strategy) pair |
| `seed_aggregate.csv` | §6 | Mean ± SD across 30 seeds |
| `nemenyi_posthoc.csv` | §6 | Pairwise p-values from Nemenyi post-hoc test |
| `calibration.csv` | §7 Calibration | ECE and Brier scores for raw / Platt / Isotonic calibration |
| `tpr_at_fixed_fpr.csv` | §8 Threshold Optimization | TPR at FPR = 5%, 10%, 15%, 20% |
| `cost_sensitive_threshold.csv` | §8 | Optimal threshold for each FN:FP cost ratio |
| `shap_pathological.csv` | §10 SHAP | Feature importances for Pathological class |

## Headline numbers

- **Best model:** LightGBM + Class Weighting
- **Canonical test accuracy:** 94.60%
- **Macro F1:** 0.8945
- **Pathological Recall at default threshold (0.5):** 91.43%
- **Pathological Recall at cost-optimal threshold (0.17):** 94.29%
- **Pathological Recall at 15% FPR operating point:** 97.14%
- **Leakage-induced inflation (P-LEAK vs P-CLEAN):** 4.69–6.27 pp accuracy, up to 16.5 pp Pathological Recall

See `../figures/` for the publication-ready visualisations.
