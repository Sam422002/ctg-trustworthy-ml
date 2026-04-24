# CTG-Trustworthy-ML

**A leakage-audited, calibration-aware benchmark for fetal health classification from cardiotocography data**

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sam422002/ctg-trustworthy-ml/blob/main/CTG_Paper_Pipeline.ipynb)

> Reproducible code and experimental results for the paper *"Beyond Accuracy: A Leakage-Audited, Calibration-Aware Benchmark for Fetal Health Classification from Cardiotocography with Threshold-Optimised Clinical Utility"* by **Soumyadeep Roy**, Post Graduate Department of Data Science, St. Xavier's College (Autonomous), Kolkata.

---

## What this project shows

Cardiotocography (CTG) classification from the UCI fetal-health dataset is a popular ML benchmark, with recent studies reporting test accuracies above 99%. This project demonstrates that **much of that apparent progress comes from methodological choices in evaluation rather than genuine model improvement**, and proposes a clinically-aligned evaluation framework.

### Five headline findings

| # | Finding | See |
|---|---------|-----|
| 1 | **Data leakage inflation** — Pre-split SMOTE inflates accuracy by 4.69–6.27 pp and Pathological Recall by up to 16.5 pp | Figure 1 |
| 2 | **SMOTE ≡ Class Weighting** (Friedman χ² = 154.4, p = 1.07×10⁻²⁸; Nemenyi post-hoc shows no intra-model difference) | Figure 2 |
| 3 | **Discrimination ≠ Calibration** — LightGBM wins F1 (0.8945); CatBoost wins ECE (0.0110, 4× better) | Figure 3 |
| 4 | **91.43% Pathological Recall "ceiling" is a default-threshold artefact** — Optimal thresholds give 94.29% at 1.28% FPR and 97.14% at 15% FPR | Figure 4 |
| 5 | **Model learned FIGO-aligned features** — SHAP top-ranked Mean, ASTV, ALTV, DP, UC match FIGO 2015 guidelines | Figures 5, 6 |

---

## Repository contents

```
ctg-trustworthy-ml/
├── README.md                      ← You are here
├── LICENSE                        ← MIT license
├── requirements.txt               ← Python dependencies
├── CTG_Paper_Pipeline.ipynb       ← Complete end-to-end experiment notebook
├── fig_leakage_audit.png          ← Figure 1: Leakage audit
├── fig_cd_diagram.png             ← Figure 2: Nemenyi CD diagram
├── fig_calibration.png            ← Figure 3: Reliability diagrams
├── fig_decision_curve.png         ← Figure 4: Decision Curve Analysis
├── fig_shap_pathological.png      ← Figure 5: SHAP feature importance
└── fig_umap_errors.png            ← Figure 6: UMAP error geometry
```

---

## Running the experiments

### Option 1 — Google Colab (recommended)

Click the **Open in Colab** badge above, then select `Runtime → Run all`.

- Runtime: ~45–75 min on Colab CPU, ~25–40 min on T4 GPU
- Dataset downloads automatically via `ucimlrepo`
- All figures and tables are regenerated from scratch

### Option 2 — Local installation

```bash
git clone https://github.com/Sam422002/ctg-trustworthy-ml.git
cd ctg-trustworthy-ml

python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate

pip install -r requirements.txt
jupyter notebook CTG_Paper_Pipeline.ipynb
```

---

## Dataset

**UCI Cardiotocography Dataset** (Campos & Bernardes, 2010)
- 2,126 CTG recordings, labelled by three expert obstetricians
- 21 tabular features per record, 3 classes (Normal 77.8%, Suspect 13.9%, Pathological 8.3%)
- Auto-downloaded by the notebook via `ucimlrepo`
- Source: https://archive.ics.uci.edu/dataset/193/cardiotocography

---

## Headline numbers

| Metric | Value |
|--------|-------|
| **Best model** | LightGBM + Class Weighting |
| **Canonical test accuracy** | 94.60% |
| **Macro F1** | 0.8945 |
| **Pathological Recall @ default threshold (0.5)** | 91.43% |
| **Pathological Recall @ cost-optimal threshold (0.17)** | 94.29% |
| **Pathological Recall @ 15% FPR** | 97.14% |
| **Leakage inflation (P-LEAK vs P-CLEAN)** | +4.69 to +6.27 pp accuracy, up to +16.5 pp Path. Recall |
| **30-seed mean Macro F1 (LGBM+CW)** | 0.9190 ± 0.0182 |
| **Friedman χ² across 10 conditions** | 154.4 (p = 1.07 × 10⁻²⁸) |

---

## Why this matters

**For researchers** — When re-running published high-accuracy CTG papers, check whether resampling is applied before or after the train/test split. Pre-split resampling inflates accuracy by 5+ percentage points.

**For clinicians** — The default 0.5 probability threshold is not clinically optimal. This repo provides operating-point analysis that recovers up to 5.71 percentage points of Pathological sensitivity without retraining.

**For students** — A complete, honest reproduction of an ML benchmark with every methodological choice documented in the notebook.

---

## Paper

Submitted to **IEEE Access** (April 2026):

> Roy, S. (2026). *Beyond Accuracy: A Leakage-Audited, Calibration-Aware Benchmark for Fetal Health Classification from Cardiotocography with Threshold-Optimised Clinical Utility.* Submitted to IEEE Access.

### Citation (update after acceptance)

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

1. Stratified 80/20 train/test split (seed 42) — before any preprocessing
2. `StandardScaler` fit only on training data
3. SMOTE (when used) applied only to training folds inside CV
4. GridSearchCV with 5-fold stratified CV, optimising Macro F1
5. Final evaluation on untouched held-out test set

### Statistical tests
- **Friedman** rank test across 10 conditions over 30 seeds
- **Nemenyi** post-hoc with critical difference diagram
- **Paired McNemar** tests on the canonical split

### Calibration
- Expected Calibration Error (ECE) with 10 equal-width bins
- Brier score averaged across 3 classes
- Post-hoc via Platt scaling, isotonic regression, temperature scaling

### Threshold optimisation
- Cost-sensitive grid search minimising `c_FN · FN + c_FP · FP`
- Fixed-FPR operating points at 5%, 10%, 15%, 20%
- Vickers & Elkin (2006) Decision Curve Analysis

---

## License

MIT License for code. See [LICENSE](LICENSE).
UCI CTG dataset is licensed CC BY 4.0.

---

## Contact

**Soumyadeep Roy**
Post Graduate Department of Data Science
St. Xavier's College (Autonomous), Kolkata
📧 roysam422002@gmail.com

---

## Acknowledgments

- **UCI Machine Learning Repository** and **SisPorto team** (Ayres-de-Campos et al.) for the CTG dataset
- Maintainers of `scikit-learn`, `lightgbm`, `xgboost`, `catboost`, `imbalanced-learn`, `shap`, `umap-learn`
- Post Graduate Department of Data Science, St. Xavier's College (Autonomous), Kolkata
- AI-assisted scaffolding via Claude (Anthropic) for manuscript drafting and code structure. All experimental design, execution, results interpretation, and scientific conclusions are the author's sole responsibility.
