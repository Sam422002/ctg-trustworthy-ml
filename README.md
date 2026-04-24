# CTG-Trustworthy-ML

**A leakage-audited, calibration-aware benchmark for fetal health classification from cardiotocography data**

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-IEEE%20Access-red.svg)](#paper)

> This repository contains the complete, reproducible code and experimental results for the paper *"Beyond Accuracy: A Leakage-Audited, Calibration-Aware Benchmark for Fetal Health Classification from Cardiotocography with Threshold-Optimised Clinical Utility"* by Soumyadeep Roy, Post Graduate Department of Data Science, St. Xavier's College (Autonomous), Kolkata.

---

## What this project does

Cardiotocography (CTG) classification from the UCI fetal-health dataset has become a standard machine learning benchmark, with recent studies reporting test accuracies above 99%. This project shows that **much of that apparent progress is attributable to methodological choices in evaluation rather than genuine model improvement**, and proposes a clinically-aligned evaluation framework that reframes the benchmark.

### Key findings

| Finding | Numbers |
|---------|---------|
| **Data leakage inflation** — Pre-split SMOTE (a pattern in several published papers) inflates accuracy by 4.69–6.27 pp and Pathological Recall by up to 16.5 pp | Figure 1, Table 1 |
| **SMOTE vs Class Weighting is statistically equivalent** across all tested classifier families (Friedman χ² = 154.4, p = 1.07×10⁻²⁸; Nemenyi post-hoc shows no intra-model difference) | Figure 2, Table 3 |
| **Discrimination ≠ Calibration** — LightGBM wins Macro F1 (0.8945); CatBoost wins ECE (0.0110, 4× better than LightGBM) | Table 4 |
| **The 91.43% Pathological Recall "ceiling" is a default-threshold artefact** — Optimal thresholds achieve 94.29% at 1.28% FPR and 97.14% at 15% FPR | Table 5, Figure 4 |
| **SHAP feature importances align with FIGO 2015 guidelines** — Mean FHR, ASTV, ALTV, and prolongued decelerations dominate, matching expert clinical criteria | Figure 5 |

---

## Repository structure

```
ctg-trustworthy-ml/
├── README.md                      ← You are here
├── LICENSE                        ← MIT license
├── requirements.txt               ← Python dependencies
├── notebooks/
│   └── CTG_Paper_Pipeline.ipynb   ← Complete end-to-end experiment pipeline
├── figures/                        ← All 6 publication figures (PNG)
│   ├── fig_leakage_audit.png
│   ├── fig_cd_diagram.png
│   ├── fig_calibration.png
│   ├── fig_decision_curve.png
│   ├── fig_shap_pathological.png
│   └── fig_umap_errors.png
├── results/                        ← CSV outputs from the experiments
│   ├── leakage_audit.csv
│   ├── tuned_smote.csv
│   ├── tuned_cw.csv
│   ├── seed_aggregate.csv
│   ├── calibration.csv
│   ├── tpr_at_fixed_fpr.csv
│   ├── cost_sensitive_threshold.csv
│   └── shap_pathological.csv
└── paper/
    └── manuscript.pdf              ← Final submitted PDF (added after acceptance)
```

---

## Reproducing the results

### Option 1 — Google Colab (recommended, no local setup)

1. Click the badge: <a href="https://colab.research.google.com/github/soumyadeep-roy/ctg-trustworthy-ml/blob/main/notebooks/CTG_Paper_Pipeline.ipynb" target="_blank">Open In Colab</a>
2. Run all cells (`Runtime → Run all`)
3. Total runtime: ~45–75 minutes on Colab CPU, ~25–40 minutes on T4 GPU

The notebook will:
- Download the UCI CTG dataset automatically via `ucimlrepo`
- Run the leakage audit, benchmark, 30-seed statistical tests, calibration analysis, threshold optimization, and SHAP analysis
- Save all figures to `outputs/figures/` and tables to `outputs/tables/`

### Option 2 — Local installation

```bash
# Clone the repo
git clone https://github.com/soumyadeep-roy/ctg-trustworthy-ml.git
cd ctg-trustworthy-ml

# Create a Python environment (recommended)
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook notebooks/CTG_Paper_Pipeline.ipynb
```

---

## Dataset

UCI Cardiotocography Dataset (Campos & Bernardes, 2010)
- 2,126 CTG recordings labelled by consensus of three expert obstetricians
- 21 tabular features per record
- 3 classes: Normal (77.8%), Suspect (13.9%), Pathological (8.3%)
- Automatically downloaded by the notebook via `ucimlrepo`
- Original source: https://archive.ics.uci.edu/dataset/193/cardiotocography

---

## Why this matters

**For researchers:** When re-running high-accuracy CTG papers, always check whether the resampling is applied before or after the train/test split. Pre-split resampling inflates accuracy by 5+ percentage points.

**For clinicians:** The default 0.5 probability threshold used by standard machine learning classifiers is not clinically optimal. This repository provides an operating-point analysis that recovers 2.86–5.71 additional percentage points of Pathological sensitivity by shifting to clinically-motivated thresholds — without retraining the model.

**For students:** This is a complete, honest reproduction of a state-of-the-art ML benchmark with every methodological choice explained in the notebook comments.

---

## Paper

This code supports the paper submitted to IEEE Access:

> **Roy, S.** (2026). *Beyond Accuracy: A Leakage-Audited, Calibration-Aware Benchmark for Fetal Health Classification from Cardiotocography with Threshold-Optimised Clinical Utility.* Submitted to IEEE Access.

Once accepted, the DOI and citation will be updated here.

### Citing this work

If you use this code or dataset split in your research, please cite:

```bibtex
@article{roy2026ctg,
  title   = {Beyond Accuracy: A Leakage-Audited, Calibration-Aware Benchmark for Fetal Health Classification from Cardiotocography with Threshold-Optimised Clinical Utility},
  author  = {Roy, Soumyadeep},
  journal = {IEEE Access},
  year    = {2026},
  note    = {Submitted}
}
```

---

## Methodology notes

### Leakage-free pipeline (P-CLEAN)

All reported results use this pipeline:

1. Stratified 80/20 train/test split (seed 42) — **before** any preprocessing
2. `StandardScaler` fit only on training data
3. SMOTE (when used) applied only to training folds inside the CV loop
4. GridSearchCV with 5-fold stratified CV, optimising Macro F1-score
5. Final evaluation on the untouched held-out test set

### Statistical tests

- **Friedman rank test** across 10 conditions (5 models × 2 strategies) over 30 seeds
- **Nemenyi post-hoc** with critical difference diagram
- **Paired McNemar tests** on the canonical split

### Calibration

- Expected Calibration Error (ECE) computed with 10 equal-width bins on max-probability confidence
- Brier score averaged across the 3 classes
- Post-hoc calibration via Platt scaling, isotonic regression, and temperature scaling

### Threshold optimisation

- Cost-sensitive thresholds computed over a 99-point grid minimising `c_FN · FN + c_FP · FP`
- Fixed-FPR operating points at 5%, 10%, 15%, 20%
- Decision Curve Analysis (Vickers & Elkin 2006)

---

## License

MIT License. See [LICENSE](LICENSE) for details.

Dataset is CC BY 4.0 per UCI ML Repository terms.

---

## Contact

**Soumyadeep Roy**
Post Graduate Department of Data Science
St. Xavier's College (Autonomous), Kolkata
Email: roysam422002@gmail.com

---

## Acknowledgments

- UCI Machine Learning Repository and the SisPorto team (Ayres-de-Campos et al.) for maintaining the CTG dataset
- The `scikit-learn`, `lightgbm`, `xgboost`, `catboost`, `imbalanced-learn`, `shap`, and `umap-learn` maintainers
- The Post Graduate Department of Data Science at St. Xavier's College (Autonomous), Kolkata
- AI-assisted scaffolding via Claude (Anthropic) for manuscript drafting and code structure. All experimental design, execution, results interpretation, and scientific conclusions are the author's own responsibility.
