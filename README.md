# TRIAGE: Trustworthy Rank-graduation Index for Auditable Governance Evaluation

A reproducible, auditable **agentic pipeline for trustworthy prediction of next-year high
medical expenditure** on longitudinal health-panel data. TRIAGE extends the cross-sectional
*S.A.F.E agentic* rank-graduation paradigm to a **person-year panel** setting: features
observed in year *t* predict whether an individual falls in the **top decile of total
medical expenditure in year *t+1***, under a **prospective time split**. Each stage emits a
reusable artifact (datacard, model, evaluation report, system card), so the full
trustworthiness assessment is transparent and reproducible.

> **Name.** The composite governance score is reported in the manuscript as the **AGORA
> Score** (Auditable Governance Of Resource Allocation; four equally weighted rank-graduation
> dimensions). Some generated artifacts retain a `TRIAGE`/`safe_` filename from an earlier
> revision (e.g. `results/tables/safe_components.json`); the contents are identical to the
> AGORA components described in the paper.

> **Anonymized for peer review.** Author(s) and affiliation(s) are withheld. Manuscript
> under review (Taylor & Francis *Interact* layout, Chicago author-date), targeting the
> *International Public Management Journal*; it frames the pipeline as an ex-ante
> accountability instrument for predictive targeting in public administration (the relocation
> of administrative discretion, the auditability of automated public decisions, and the
> cost-as-need hazard). Do not de-anonymize in issues or commits.

## Framework

A sequential five-stage pipeline (notebooks `00`–`05`) plus five appendices
(`06`–`10`):

| Stage | Notebook | Output artifact |
|-------|----------|-----------------|
| **Data** | `00_data_preparation.ipynb` | person-year panel, `datacard.json`, `attrition.csv` |
| **(EDA)** | `01_eda.ipynb` | grayscale figures, `eda_summary.csv` |
| **Modelling** | `02_modelling.ipynb` | six candidates, 5-fold CV selection, fitted pipeline |
| **Evaluation** | `03_evaluation.ipynb` | RGA/AURGA, RGR, AURGE, fairness, calibration, `subgroup_outcomes.csv` |
| **Governance** | `04_governance.ipynb` | `system_card.json`, TRIAGE Score, weight sensitivity |
| **Chatbot** | `05_chatbot_agent.ipynb` | artifact-grounded reviewer QA (LLM) |

| Appendix | Notebook | What it adds |
|----------|----------|--------------|
| GroupKFold / death | `06_robustness_groupkfold.ipynb` | person-level `GroupKFold` vs the time split; death competing-risk sensitivity |
| Explainability | `07_explainability.ipynb` | grouped RGE over feature blocks; parsimonious top-k vs full AURGE |
| Recalibration | `08_recalibration.ipynb` | Platt & isotonic calibration; illustrative five-axis TRIAGE score |
| Continuous target | `09_continuous_cost.ipynb` | log-expenditure regression; RGA corroborates the binary ranking |
| Financial risk | `10_financial_risk.ipynb` | public/private split of the top decile; household catastrophic-expenditure (CHE) and impoverishment incidence by subgroup; `financial_risk.json` |

### Trustworthiness metrics

The **TRIAGE Score** is the equally weighted mean (0.25 each) of four rank-graduation
components, all in [0, 1]: **AURGA** (accuracy), **RGR** (robustness), **AURGE**
(explainability) and a **Fairness Aggregate** (SPD, EOD, AOD, DIR normalized to parity).
Rank-graduation quantities use the `safeaipackage` implementation of the RGA family; for
binary targets RGA equals ROC-AUC. Probability calibration (Brier, calibration
intercept/slope, Cramér–von Mises) is reported but **not** folded into the TRIAGE Score —
recalibration is an explicit extension axis (appendix `08`).

### Sensitive attributes

Fairness is evaluated across **Medical-Aid status, disability, sex, age ≥ 65, and low
income**. Because the target is cost, the pipeline foregrounds the *cost-as-need-proxy*
risk; the governance stage flags these subgroups for audit, and `subgroup_outcomes.csv`
separates the *observed* subgroup high-cost rate from the model's *predicted* positive rate.

## Headline results (held-out 2023→2024)

- Selected model **XGBoost** (5-fold CV ROC-AUC 0.799); test ROC-AUC 0.797.
- **AGORA Score 0.750** (TRIAGE Score in code) → governance verdict **REVIEW**; weakest dimension **AURGE** (0.666).
- GroupKFold-by-person AUC 0.799 ≈ time split → no person-level leakage.
- Over-prediction (calibration intercept −2.40) largely removed by Platt scaling
  (Brier 0.210 → 0.078).
- Binary and continuous (log-expenditure) framings rank individuals consistently
  (Spearman 0.80).

## Financial-risk appendix (household financial protection)

Notebook `10_financial_risk.ipynb` reads the high-cost target through a household
financial-risk and public-insurance-sustainability lens, for the emerging-markets /
public-finance presentation of the same data. It adds no dependency to the TRIAGE Score; it
emits `results/tables/financial_risk.json`.

- **Concentration.** The costliest decile accounts for **60.3%** of total next-year spending in
  the modelling panel (61.1% over all raw waves — the household-exposure sample used for the CHE
  measures below).
- **Public vs private split of the top decile.** Public insurance (NHI / Medical-Aid) absorbs
  **57.6%** of top-decile cost overall, rising to **87.3%** for Medical-Aid beneficiaries and
  **74.7%** for people with disabilities — the fiscal burden of the tail falls disproportionately
  on the public purse for the exposed groups.
- **Catastrophic health expenditure (CHE) and impoverishment, by subgroup.** Relative to income,
  the same groups remain the most exposed: CHE incidence (out-of-pocket above 10% of income) is
  **38.8%** for low-income and **34.8%** for disability households, versus **21.5%** overall;
  impoverishment incidence is roughly doubled. Strong public coverage and high residual household
  financial risk coexist for exactly the groups a targeting rule is most obliged to serve.

Definitions (see the notebook): out-of-pocket = facility co-payment (`MEXP3_1`) + pharmacy
(`MEXP4_1`); public = insurer share (`MEXP3_2`); capacity to pay = household income net of a 40%
subsistence allowance; CHE₁₀ = OOP above 10% of income, CHE₄₀ = OOP above 40% of capacity to pay;
impoverishment = a household pushed below 50% of median household income by its OOP. Income and OOP
are survey-reported, so the magnitudes are estimates.

## Repository layout

```
.
├── data/
│   ├── raw/          # KHP SAS files (NOT distributed; see Data access)
│   └── processed/    # generated panel + model artifacts (git-ignored)
├── notebooks/        # 00–10, one code cell each
│   └── appendix_reviewer/   # reviewer-requested robustness scripts (a1–a5)
├── results/
│   ├── figures/      # grayscale PNG + PDF, 600 dpi (fig01–fig20)
│   └── tables/       # datacard, metrics, system card, subgroup outcomes, financial_risk (JSON/CSV)
├── src/              # notebook generators
├── .env.example      # copy to .env at repo root (LLM agent only)
├── requirements.txt
└── README.md
```

## Data Availability Statement

The data that support the findings of this study are third-party microdata from the **Korea
Health Panel Study (KHP), 2nd wave (2019–2024)**, jointly operated by the Korea Institute for
Health and Social Affairs and the National Health Insurance Service. The microdata are available
from the official KHP portal (`https://www.khp.re.kr`) under its data-use agreement and were used
under licence for the current study; restrictions therefore apply and the authors cannot
redistribute them. The **analysis code that reproduces every quantity, figure and artifact** —
including the reviewer-requested robustness analyses — is openly available at
`https://anonymous.4open.science/r/HighCostHealth-Korea-FinancialRisk-D050` (anonymized for peer
review), and regenerates all results once a licensed user supplies the KHP microdata. The published
National Health Insurance aggregates used as an external benchmark are cited in the manuscript.

### Placing the microdata

After obtaining the data, place the extracted SAS files so that each year resolves as:

```
data/raw/sas2019/a_ind.sas7bdat , a_ms.sas7bdat , a_phi.sas7bdat , ...
data/raw/sas2020/b_ind.sas7bdat , ...
...
data/raw/sas2024/f_ind.sas7bdat , ...
```

## Reproducibility

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Notebooks 00–04 and appendices 06–10 need no API key:
for nb in 00_data_preparation 01_eda 02_modelling 03_evaluation 04_governance \
          06_robustness_groupkfold 07_explainability 08_recalibration 09_continuous_cost \
          10_financial_risk; do
  jupyter nbconvert --to notebook --execute --inplace notebooks/$nb.ipynb
done
```

The **LLM chatbot agent (`05`)** is optional and the only component requiring API access;
it produces no numbers the other notebooks depend on. Copy `.env.example` to `.env` at the
repository root (one level above `notebooks/`) and set `OPENAI_API_KEY` and `LLM_MODEL_*`.
Run it once (after `03`) only to reproduce the governance-chatbot transcript.

Conventions: notebooks hold a single code cell with English comments; all randomness is
seeded (`seed = 42`); figures are grayscale, caption-free, and saved as PNG + PDF at 600 dpi;
no absolute paths are printed.

## Reviewer-requested robustness analyses

The scripts in `notebooks/appendix_reviewer/` reproduce the additional robustness checks added
during peer review. They reuse `data/processed/model_artifacts.pkl` (written by `02_modelling`)
and the generated panel, and each writes its own table(s) under `results/tables/`. Run them from
inside `notebooks/` (paths resolve to the repository root) after `00`–`03`:

```bash
cd notebooks
python -I appendix_reviewer/a1_multiplicity_and_dimensions.py   # Bonferroni subgroup CIs
python -I appendix_reviewer/a2_measurement_consistency.py       # self-report vs public share
python -I appendix_reviewer/a3_external_benchmark.py            # KHP vs published NHIS totals
python -I appendix_reviewer/a4_dimension_independence.py        # bootstrap dimension correlations
python -I appendix_reviewer/a5_intersectional_and_needspecs.py  # Medical-Aid x disability; need specs
```

| Script | Addresses | Output table(s) |
|--------|-----------|-----------------|
| `a1_multiplicity_and_dimensions.py` | multiplicity of subgroup tests | `bonferroni_sensitivity.{csv,json}` |
| `a2_measurement_consistency.py` | differential self-report error (internal) | `measurement_consistency.json` |
| `a3_external_benchmark.py` | self-report vs published NHIS per-capita totals | `external_benchmark.{csv,json}` |
| `a4_dimension_independence.py` | whether the four AGORA dimensions are redundant | `dimension_independence.json` |
| `a5_intersectional_and_needspecs.py` | Medical-Aid × disability; alternative need specifications | `intersectional.csv`, `disability_need_specs.csv`, `round2_experiments.json` |

Key results these produce: every subgroup predicted-positive gap excludes zero under
Bonferroni-corrected 98.75% intervals; the four governance dimensions are near-uncorrelated across
bootstrap replicates (pairwise |r| ≤ 0.16); the self-reported per-capita total matches the published
2024 Health Insurance Statistical Yearbook to within ~6%; and the disability residual stays positive
across every need specification, sharpest at the Medical-Aid × disability intersection.

## Manuscript

The paper is written separately in LaTeX (Taylor & Francis *Interact* layout with the Chicago
author-date style), targeting the *International Public Management Journal*; it frames the pipeline
as an ex-ante accountability instrument for governing predictive fiscal targeting in public
administration. The `figures/fig*.pdf` files here are the camera-ready sources;
`results/tables/*.json` back the reported numbers.

## Design notes

- **Prospective split**: train on transitions 2019→2020 … 2022→2023; test on 2023→2024.
  A `GroupKFold`-by-person variant (appendix `06`) prevents person-level leakage.
- **Informative attrition**: dropout between *t* and *t+1* is reported by subgroup in the
  datacard; death during a transition censors cost and is treated as a competing risk.
- **Survey, not claims**: expenditures are self-reported panel values, not administrative
  claims — a stated limitation.

## License

Code released under the MIT License (see `LICENSE`). The KHP microdata are governed by their
own terms and are not covered by this license.

## Citation

Anonymous Author(s). *Auditing algorithmic discretion before it acts: an ex-ante accountability
instrument for predictive targeting in public administration.* Under review, 2026.
