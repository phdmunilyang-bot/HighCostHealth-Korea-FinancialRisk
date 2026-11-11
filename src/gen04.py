from make_notebooks import write_nb

CODE = r'''
# 04_governance.ipynb
# Governance Agent: read the upstream artifacts (datacard, evaluation metrics, SAFE
# components), aggregate the SAFE Score, compare it against governance thresholds,
# identify the weakest trustworthiness dimension, run a weight-sensitivity analysis,
# and publish a machine-readable System Card. No absolute paths are printed (anonymity).

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path.cwd().parent
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"; FIG.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 600, "font.size": 10,
                     "axes.edgecolor": "0.3", "image.cmap": "Greys"})
G_DARK, G_MID, G_LIGHT = "0.25", "0.5", "0.75"

def save_fig(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)

datacard = json.load(open(TAB / "datacard.json", encoding="utf-8"))
metrics = json.load(open(TAB / "evaluation_metrics.json", encoding="utf-8"))
comp = json.load(open(TAB / "safe_components.json", encoding="utf-8"))
cvres = pd.read_csv(TAB / "cv_results.csv")

AXES = ["AURGA", "RGR", "AURGE", "Fairness_Aggregate"]
THRESHOLDS = {"AURGA": 0.70, "RGR": 0.80, "AURGE": 0.80, "Fairness_Aggregate": 0.80, "SAFE_Score": 0.75}

safe = comp["SAFE_Score"]
weakest = min(AXES, key=lambda a: comp[a])
flags = {a: (comp[a] >= THRESHOLDS[a]) for a in AXES}
flags["SAFE_Score"] = safe >= THRESHOLDS["SAFE_Score"]

# ---- weight-sensitivity: give one axis weight 0.40, split 0.20 across the other three ----
def weighted(axis_up):
    w = {a: (0.40 if a == axis_up else 0.20) for a in AXES}
    return sum(w[a] * comp[a] for a in AXES)
sens = {a: weighted(a) for a in AXES}
sens["baseline_equal"] = safe
sens_vals = list(sens.values())

system_card = dict(
    title="SAFE System Card: next-year high-cost prediction (longitudinal health panel)",
    dataset=datacard["dataset"], unit=datacard["unit"],
    model=metrics["model"], n_test=metrics["n_test"],
    headline=dict(SAFE_Score=round(safe, 4), roc_auc=round(metrics["roc_auc"], 4),
                  passes_threshold=bool(flags["SAFE_Score"])),
    safe_components={a: round(comp[a], 4) for a in AXES},
    thresholds=THRESHOLDS, dimension_flags=flags,
    weakest_dimension=weakest,
    calibration=dict(brier=round(metrics["brier"], 4),
                     intercept=round(metrics["calibration_intercept"], 4),
                     slope=round(metrics["calibration_slope"], 4),
                     cvm_pvalue=metrics["cvm_pvalue"],
                     note="calibration is reported but NOT part of the SAFE Score; "
                          "recalibration (Platt/isotonic) is a documented extension point"),
    weight_sensitivity={k: round(v, 4) for k, v in sens.items()},
    weight_sensitivity_range=round(max(sens_vals) - min(sens_vals), 4),
    model_selection=cvres.round(6).to_dict(orient="records"),
    governance_recommendations=[
        f"Weakest dimension is {weakest}; prioritise mitigation there.",
        "Audit Medical-Aid and disability subgroups before any deployment "
        "(cost-as-need proxy risk).",
        "Report prospective-split and GroupKFold results together; treat attrition as "
        "informative (see datacard).",
    ])
json.dump(system_card, open(TAB / "system_card.json", "w"), ensure_ascii=False, indent=2)

# ---- Figures ----
fig, ax = plt.subplots(figsize=(5.0, 3.2))
vals = [comp[a] for a in AXES]
sns.barplot(x=AXES, y=vals, color=G_MID, edgecolor="0.3", ax=ax)
for i, a in enumerate(AXES):
    ax.plot([i - 0.4, i + 0.4], [THRESHOLDS[a]] * 2, color=G_DARK, linestyle="--", linewidth=1)
ax.set_ylim(0, 1.05); ax.set_ylabel("Score"); ax.set_xlabel("")
ax.set_xticks(range(len(AXES))); ax.set_xticklabels(["AURGA", "RGR", "AURGE", "Fairness"], rotation=0)
save_fig(fig, "fig11_safe_components")

fig, ax = plt.subplots(figsize=(5.2, 3.0))
order = ["baseline_equal"] + AXES
sns.barplot(x=[sens[k] for k in order], y=["equal", "AURGA^", "RGR^", "AURGE^", "Fair^"],
            color=G_MID, edgecolor="0.3", ax=ax)
ax.set_xlim(min(sens_vals) - 0.02, max(sens_vals) + 0.02)
ax.set_xlabel("SAFE Score under alternative weightings"); ax.set_ylabel("")
save_fig(fig, "fig12_weight_sensitivity")

print("SAFE Score: %.4f  (threshold %.2f -> %s)"
      % (safe, THRESHOLDS["SAFE_Score"], "PASS" if flags["SAFE_Score"] else "REVIEW"))
print("components:", {a: round(comp[a], 4) for a in AXES})
print("weakest dimension:", weakest)
print("weight-sensitivity range: %.4f" % (max(sens_vals) - min(sens_vals)))
print("system_card.json written")
'''

write_nb("04_governance.ipynb", CODE)
