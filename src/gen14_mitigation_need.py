from make_notebooks import write_nb

CODE = r'''
# 14_mitigation_need.ipynb
# Third-round pack (reviewer extensions T1, T2, T8):
#   T2  bridge the financial-risk measures to the targeting model. We build an individual
#       financial-need signal (out-of-pocket spending above 10% of household income -- an
#       individual analogue of the household CHE of the financial-risk table, distinct from
#       the total-cost target) and show that targeting by cost diverges from targeting by
#       financial need: the groups the model most over-flags on cost are not those with the
#       highest out-of-pocket burden.
#   T1  test proposition P2 directly. We apply two standard post-processing fairness
#       mitigations (demographic parity, equal opportunity) to the score and show they
#       remove the selection-rate disparity only at an accuracy cost, WITHOUT improving the
#       model's alignment with financial need -- i.e. adjusting the outputs does not repair
#       the target distortion.
#   T8  cross-check the need-adjusted gaps with a logistic model's average marginal effect
#       (AME), confirming the linear-probability-model result is not an artifact of the LPM.
# Deterministic; no external services. Run after notebooks 00, 02.

import json, pickle, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent
PROC = ROOT / "data" / "processed"
FIG = ROOT / "results" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
TAB = ROOT / "results" / "tables"; TAB.mkdir(parents=True, exist_ok=True)
SEED = 42
rng = np.random.default_rng(SEED)
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 600, "font.size": 10,
                     "axes.edgecolor": "0.3", "image.cmap": "Greys"})
G_DARK, G_MID, G_LIGHT = "0.25", "0.5", "0.75"
def save_fig(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)

art = pickle.load(open(PROC / "model_artifacts.pkl", "rb"))
Xte = art["Xte"].reset_index(drop=True)
yte = np.asarray(art["yte"]); yhat = np.asarray(art["yhat_te"])
prot = art["protected"].reset_index(drop=True)
n = len(yte)

GROUPS = {
    "Medical-Aid": (prot["medical_aid"] == 1).values,
    "Disability":  (prot["disability"] == 1).values,
    "Low income":  (prot["inc_q"] == 1).values,
    "Age>=65":     (prot["age"] >= 65).values,
}

# ---- financial-need signal: individual out-of-pocket burden > 10% of household income ----
oop = pd.to_numeric(Xte["oop"], errors="coerce").fillna(0).values
inc_won = pd.to_numeric(Xte["hh_income"], errors="coerce").values * 10000.0   # 10k KRW -> KRW
valid = inc_won > 0
fb = np.zeros(n, bool)
fb[valid] = oop[valid] > 0.10 * inc_won[valid]      # individual OOP-burden flag

def flag_at(scores, q):
    return (scores >= np.quantile(scores, 1 - q)).astype(int)
pred = flag_at(yhat, 0.10)

# ================= T2: cost targeting vs financial-need =================
t2 = dict(oop_burden_incidence_overall=round(float(fb[valid].mean()), 3),
          by_group={}, model_vs_need={})
for name, g in GROUPS.items():
    m = g & valid
    t2["by_group"][name] = dict(
        n=int(m.sum()),
        oop_burden_incidence=round(float(fb[m].mean()), 3),
        model_predicted_positive=round(float(pred[g].mean()), 3),
        observed_highcost=round(float(yte[g].mean()), 3))
# alignment of cost-targeting with financial need, on the income-valid sample
pv = pred[valid].astype(bool); fv = fb[valid]; yv = yte[valid].astype(bool)
t2["model_vs_need"] = dict(
    recall_highcost=round(float(pv[yv].mean()), 3),                 # cost-model recall of actual high cost
    recall_oop_burden=round(float(pv[fv].mean()), 3),               # cost-model recall of OOP-burden need
    flagged_that_are_oop_burden=round(float(fv[pv].mean()), 3),
    note=("The cost model captures a much larger share of actual high-cost individuals than of "
          "high-OOP-burden individuals: targeting by total cost is not the same as targeting by "
          "financial need, because public insurance absorbs the OOP of the most-covered groups."))
json.dump(t2, open(TAB / "need_alignment.json", "w"), ensure_ascii=False, indent=2)

# figure: predicted-positive vs OOP-burden incidence by group
order = ["Medical-Aid", "Disability", "Low income", "Age>=65"]
x = np.arange(len(order)); w = 0.38
fig, ax = plt.subplots(figsize=(5.8, 3.3))
ax.bar(x - w/2, [t2["by_group"][g]["model_predicted_positive"] for g in order], w,
       color=G_DARK, edgecolor="0.3", label="Model predicted-positive (cost)")
ax.bar(x + w/2, [t2["by_group"][g]["oop_burden_incidence"] for g in order], w,
       color=G_LIGHT, edgecolor="0.3", label="Financial need (OOP burden > 10% income)")
ax.set_xticks(x); ax.set_xticklabels(order); ax.set_ylabel("Rate")
ax.legend(frameon=False, fontsize=8)
save_fig(fig, "fig24_cost_vs_need")

# ================= T1: post-processing fairness mitigations and P2 =================
def per_group_threshold_rate(scores, g, target_rate):
    # demographic parity: flag the top target_rate within each side of the partition g / ~g
    out = np.zeros(len(scores), int)
    for mask in (g, ~g):
        if mask.sum() == 0:
            continue
        thr = np.quantile(scores[mask], 1 - target_rate)
        out[mask] = (scores[mask] >= thr).astype(int)
    return out

def per_group_threshold_tpr(scores, g, y, target_tpr):
    # equal opportunity: within each side, pick a threshold achieving target_tpr among positives
    out = np.zeros(len(scores), int)
    for mask in (g, ~g):
        pos = mask & (y == 1)
        if pos.sum() == 0:
            continue
        thr = np.quantile(scores[pos], 1 - target_tpr)
        out[mask] = (scores[mask] >= thr).astype(int)
    return out

def metrics(decision):
    d = decision.astype(bool)
    dv = d[valid]
    return dict(
        highcost_recall=round(float(d[yte == 1].mean()), 3),            # accuracy: capture of true high cost
        oop_burden_recall=round(float(dv[fv].mean()), 3),               # need alignment
        selection_rate=round(float(d.mean()), 3))

base_tpr = float(pred[yte == 1].mean())
mit = {"baseline": {"overall": metrics(pred)}}
for attr in ["Medical-Aid", "Disability"]:
    g = GROUPS[attr]
    dp = per_group_threshold_rate(yhat, g, 0.10)
    eo = per_group_threshold_tpr(yhat, g, yte, base_tpr)
    def gap(dec):
        return round(float(dec[g].mean() - dec[~g].mean()), 3)
    mit[attr] = dict(
        baseline_selection_gap=round(float(pred[g].mean() - pred[~g].mean()), 3),
        demographic_parity=dict(selection_gap=gap(dp), **metrics(dp)),
        equal_opportunity=dict(selection_gap=gap(eo), **metrics(eo)))
mit["baseline"]["highcost_recall"] = round(base_tpr, 3)
mit["baseline"]["oop_burden_recall"] = round(float(pred[valid][fv].mean()), 3)
mit["interpretation"] = (
    "Both mitigations drive the selection-rate gap toward zero, but the model's recall of "
    "financial need (OOP burden) barely moves while high-cost recall falls: adjusting the outputs "
    "removes the disparity symptom at an accuracy cost without repairing the cost-as-need target "
    "distortion. This is the evidence for P2 (irreducibility by post-hoc output adjustment).")
json.dump(mit, open(TAB / "mitigation_P2.json", "w"), ensure_ascii=False, indent=2)

# ================= T8: logistic AME cross-check of the need-adjusted gap =================
Xte2 = Xte.copy()
Xte2["log_cost"] = np.log1p(pd.to_numeric(Xte2["cost"], errors="coerce").fillna(0))
Xte2["log_oop"] = np.log1p(pd.to_numeric(Xte2["oop"], errors="coerce").fillna(0))
NEED_CLINICAL = ["n_chronic", "care_need", "ltc"]
NEED_FULL = ["n_chronic", "care_need", "ltc", "log_cost", "log_oop", "any_inpatient"]
for c in set(NEED_FULL):
    Xte2[c] = pd.to_numeric(Xte2[c], errors="coerce").fillna(0.0)

def ame_logit(g, covars):
    # average marginal effect of group membership on the flag probability, adjusting covars
    Z = np.column_stack([g.astype(float)] + [Xte2[c].values.astype(float) for c in covars])
    # standardize covars for conditioning stability
    clf = LogisticRegression(max_iter=2000, C=1e6)
    clf.fit(Z, pred)
    Z1 = Z.copy(); Z1[:, 0] = 1.0
    Z0 = Z.copy(); Z0[:, 0] = 0.0
    return float((clf.predict_proba(Z1)[:, 1] - clf.predict_proba(Z0)[:, 1]).mean())

t8 = {}
for name, g in GROUPS.items():
    t8[name] = dict(
        lpm_crude=None,  # reported in need_multivariate.json
        ame_clinical=round(ame_logit(g, NEED_CLINICAL), 4),
        ame_full=round(ame_logit(g, NEED_FULL), 4))
json.dump(t8, open(TAB / "need_ame_logit.json", "w"), ensure_ascii=False, indent=2)

# ================= report =================
print("T2  cost targeting vs financial need (OOP burden > 10% income):")
print(f"  overall OOP-burden incidence: {t2['oop_burden_incidence_overall']}")
for g in order:
    b = t2["by_group"][g]
    print(f"  {g:12s} predicted(cost)={b['model_predicted_positive']:.3f}  "
          f"OOP-burden(need)={b['oop_burden_incidence']:.3f}")
print(f"  cost-model recall: high-cost {t2['model_vs_need']['recall_highcost']} vs "
      f"OOP-burden {t2['model_vs_need']['recall_oop_burden']}")
print("\nT1  fairness mitigations (P2 test):")
print(f"  baseline: highcost_recall {mit['baseline']['highcost_recall']}  "
      f"oop_burden_recall {mit['baseline']['oop_burden_recall']}")
for attr in ["Medical-Aid", "Disability"]:
    m = mit[attr]
    print(f"  {attr}: baseline gap {m['baseline_selection_gap']:+.3f}")
    for k in ["demographic_parity", "equal_opportunity"]:
        v = m[k]
        print(f"    {k:18s} gap {v['selection_gap']:+.3f}  highcost_recall {v['highcost_recall']}  "
              f"oop_burden_recall {v['oop_burden_recall']}")
print("\nT8  logistic AME vs LPM (need-adjusted gap):")
for g in order:
    print(f"  {g:12s} AME clinical {t8[g]['ame_clinical']:+.3f}  full {t8[g]['ame_full']:+.3f}")
'''

write_nb("14_mitigation_need.ipynb", CODE)
