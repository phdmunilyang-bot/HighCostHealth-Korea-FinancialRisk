from make_notebooks import write_nb

CODE = r'''
# 03_evaluation.ipynb
# Evaluation Agent: compute the SAFE trustworthiness profile of the selected model on the
# held-out prospective test set (2023->2024): rank-graduation accuracy (RGA / AURGA),
# robustness (RGR), explainability (AURGE via greedy ablation), group fairness
# (SPD/EOD/AOD/DIR aggregate) across sensitive attributes, and probability calibration
# (Brier, calibration intercept/slope, Cramer-von Mises). Authoritative RGA/RGR/RGE/
# RGA-parity values from safeaipackage are reported alongside as a cross-check.
# No absolute paths are printed (anonymity).

import warnings, pickle, json, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import cramervonmises_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from safeaipackage.core import rga
from safeaipackage.check_explainability import compute_rge_values
from safeaipackage.check_robustness import compute_rgr_values
from safeaipackage.check_fairness import compute_rga_parity

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

trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz   # numpy>=2.0 renamed trapz -> trapezoid

art = pickle.load(open(PROC / "model_artifacts.pkl", "rb"))
model, FEAT = art["model"], art["FEAT"]
Xtr, ytr, Xte, yte, yhat = art["Xtr"], art["ytr"], art["Xte"], art["yte"], art["yhat_te"]
yte = np.asarray(yte); yhat = np.asarray(yhat)

# safeaipackage rejects NaN inputs; impute with TRAINING medians for the rank-metric calls
# (the fitted pipeline already imputes internally, so predictions are unchanged).
_med = Xtr.median(numeric_only=True)
Xtr_i = Xtr.fillna(_med)
Xte_i = Xte.fillna(_med)

# ---------- Accuracy: RGA and AURGA (progressive top-removal integral) ----------
rga_point = rga(list(yte), list(yhat))

def aurga(y, s, grid=np.linspace(0, 0.9, 19)):
    y = np.asarray(y); s = np.asarray(s)
    order = np.argsort(-s)            # descending predicted risk
    vals = []
    for x in grid:
        k = int(len(s) * x)
        keep = order[k:]              # remove top-x fraction
        if len(np.unique(y[keep])) < 2:
            vals.append(vals[-1] if vals else 0.5); continue
        vals.append(rga(list(y[keep]), list(s[keep])))
    return float(trapz(vals, grid) / (grid[-1] - grid[0])), grid, vals

AURGA, agrid, acurve = aurga(yte, yhat)

# ---------- Robustness: RGR (paper-style Gaussian score perturbation) ----------
def rgr_score(s, mult=0.5, reps=20):
    s = np.asarray(s, float); sd = s.std()
    out = []
    for _ in range(reps):
        sp = s + rng.normal(0, mult * sd, size=len(s))
        out.append(rga(list(s), list(sp)))   # concordance of original vs perturbed ranking
    return float(np.mean(out))

RGR = rgr_score(yhat)
rgr_feat = compute_rgr_values(Xte_i, list(yhat), model, FEAT, perturbation_percentage=0.05)  # cross-check

# ---------- Explainability: AURGE via greedy feature ablation ----------
rge_df = compute_rge_values(Xtr_i, Xte_i, list(yhat), model, FEAT).sort_values("RGE", ascending=False)
order = list(rge_df.index)
base = rga(list(yte), list(yhat))
Xp = Xte_i.copy(); curve = [base]
for var in order:
    Xp[var] = rng.permutation(Xp[var].values)
    yk = model.predict_proba(Xp)[:, 1]
    curve.append(rga(list(yte), list(yk)))
AURGE = float(trapz(curve, dx=1) / ((len(curve) - 1) * base))   # normalised area retained

# ---------- Fairness: SPD / EOD / AOD / DIR aggregate ----------
thr = np.quantile(yhat, 0.90)            # decision aligned with the top-decile target
pred = (yhat >= thr).astype(int)
prot = art["protected"].reset_index(drop=True)
groups = {
    "medical_aid": (prot["medical_aid"] == 1).values,
    "disability": (prot["disability"] == 1).values,
    "female": (prot["sex"] == 2).values,
    "age65": (prot["age"] >= 65).values,
    "income_low": (prot["inc_q"] == 1).values,
}

def fairness_row(unpriv):
    priv = ~unpriv
    def rate(m): return pred[m].mean() if m.sum() else np.nan
    def tpr(m): return pred[m & (yte == 1)].mean() if (m & (yte == 1)).sum() else np.nan
    def fpr(m): return pred[m & (yte == 0)].mean() if (m & (yte == 0)).sum() else np.nan
    spd = rate(unpriv) - rate(priv)
    eod = tpr(unpriv) - tpr(priv)
    aod = 0.5 * ((fpr(unpriv) - fpr(priv)) + (tpr(unpriv) - tpr(priv)))
    dir_ = rate(unpriv) / rate(priv) if rate(priv) else np.nan
    f = np.mean([1 - abs(spd), 1 - abs(eod), 1 - abs(aod), min(dir_, 1 / dir_)])
    return dict(SPD=spd, EOD=eod, AOD=aod, DIR=dir_, fairness=f)

fair = {g: fairness_row(m) for g, m in groups.items()}
fair_df = pd.DataFrame(fair).T
FAIRNESS_AGG = float(fair_df["fairness"].mean())

# ---------- Subgroup outcome table (observed vs predicted high-cost rates) ----------
# Lets the governance chatbot cite the ACTUAL subgroup outcome rate, not only fairness gaps.
sg_defs = {
    "overall": np.ones(len(yte), bool),
    "medical_aid": groups["medical_aid"], "nhi": ~groups["medical_aid"],
    "disability": groups["disability"], "non_disability": ~groups["disability"],
    "female": groups["female"], "male": ~groups["female"],
    "age65_plus": groups["age65"], "age_under65": ~groups["age65"],
    "income_low": groups["income_low"], "income_other": ~groups["income_low"],
}
sg_rows = []
for name, m in sg_defs.items():
    sg_rows.append(dict(subgroup=name, n=int(m.sum()),
                        observed_highcost_rate=round(float(yte[m].mean()), 4),
                        predicted_positive_rate=round(float(pred[m].mean()), 4),
                        mean_pred_prob=round(float(yhat[m].mean()), 4)))
subgroup_outcomes = pd.DataFrame(sg_rows)
subgroup_outcomes.to_csv(TAB / "subgroup_outcomes.csv", index=False)

# RGA-parity cross-check (safeaipackage)
def parity(var):
    try:
        s = compute_rga_parity(Xtr_i, Xte_i, list(yte), list(yhat), model, var)
        return float(re.findall(r"[-+]?\d*\.\d+|\d+", s)[-1])
    except Exception:
        return np.nan
rga_parity = {v: parity(v) for v in ["medical_aid", "disability", "sex"]}

# ---------- Calibration ----------
# Calibration slope and intercept from the joint model y ~ a + b*logit(p); ideal (a,b)=(0,1).
p = np.clip(yhat, 1e-6, 1 - 1e-6)
logit = np.log(p / (1 - p))
brier = brier_score_loss(yte, yhat)
cal = LogisticRegression(max_iter=1000, C=1e6).fit(logit.reshape(-1, 1), yte)  # near-unpenalized
slope = float(cal.coef_[0, 0]); intercept = float(cal.intercept_[0])
cvm = cramervonmises_2samp(yhat[yte == 1], yhat[yte == 0])

# ---------- SAFE Score (equal 0.25 weights, paper-faithful components) ----------
SAFE = 0.25 * AURGA + 0.25 * RGR + 0.25 * AURGE + 0.25 * FAIRNESS_AGG
components = dict(AURGA=AURGA, RGR=RGR, AURGE=AURGE, Fairness_Aggregate=FAIRNESS_AGG, SAFE_Score=SAFE)

metrics = dict(
    model=art["best_name"], n_test=int(len(yte)), roc_auc=float(roc_auc_score(yte, yhat)),
    RGA=float(rga_point), AURGA=AURGA, RGR=RGR, AURGE=AURGE,
    Fairness_Aggregate=FAIRNESS_AGG, SAFE_Score=SAFE,
    brier=float(brier), calibration_intercept=float(intercept), calibration_slope=float(slope),
    cvm_stat=float(cvm.statistic), cvm_pvalue=float(cvm.pvalue),
    rga_parity_crosscheck=rga_parity)
json.dump(metrics, open(TAB / "evaluation_metrics.json", "w"), ensure_ascii=False, indent=2)
json.dump(components, open(TAB / "safe_components.json", "w"), ensure_ascii=False, indent=2)
fair_df.round(4).to_csv(TAB / "fairness_table.csv")
rge_df.round(4).to_csv(TAB / "rge_importance.csv")

# ---------- Figures ----------
fig, ax = plt.subplots(figsize=(5.4, 4.2))
sns.barplot(x=rge_df["RGE"].values, y=[str(i) for i in rge_df.index], color=G_MID, edgecolor="0.3", ax=ax)
ax.set_xlabel("RGE (rank-graduation explainability)"); ax.set_ylabel("")
save_fig(fig, "fig07_rge_importance")

from sklearn.calibration import calibration_curve
frac, mean_pred = calibration_curve(yte, yhat, n_bins=10, strategy="quantile")
fig, ax = plt.subplots(figsize=(4.2, 4.0))
ax.plot(mean_pred, frac, marker="o", color=G_DARK)
ax.plot([0, 1], [0, 1], color=G_LIGHT, linestyle="--")
ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency")
ax.set_aspect("equal")
save_fig(fig, "fig08_calibration")

fig, ax = plt.subplots(figsize=(5.6, 3.4))
sns.barplot(x=fair_df.index, y=fair_df["fairness"].values, color=G_MID, edgecolor="0.3", ax=ax)
ax.axhline(1.0, color=G_LIGHT, linestyle="--"); ax.set_ylim(0, 1.05)
ax.set_ylabel("Fairness (1 = parity)"); ax.set_xlabel("")
ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
save_fig(fig, "fig09_fairness")

fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.plot(agrid, acurve, marker="o", color=G_DARK, label="AURGA curve (top-removal)")
ax.plot(np.linspace(0, 1, len(curve)), np.array(curve) / base, marker="s", color=G_MID,
        linestyle="--", label="AURGE curve (greedy ablation)")
ax.set_xlabel("Fraction removed"); ax.set_ylabel("Relative RGA")
ax.legend(frameon=False, fontsize=8)
save_fig(fig, "fig10_degradation")

# observed vs predicted high-cost rate by subgroup
sgo = subgroup_outcomes[subgroup_outcomes.subgroup != "overall"].set_index("subgroup")
fig, ax = plt.subplots(figsize=(6.0, 3.6))
xx = np.arange(len(sgo))
ax.bar(xx - 0.2, sgo["observed_highcost_rate"], width=0.4, color=G_DARK, label="Observed")
ax.bar(xx + 0.2, sgo["predicted_positive_rate"], width=0.4, color=G_LIGHT,
       edgecolor="0.3", label="Predicted positive")
ax.set_xticks(xx); ax.set_xticklabels(sgo.index, rotation=30, ha="right")
ax.set_ylabel("Rate at t+1"); ax.legend(frameon=False, fontsize=8)
save_fig(fig, "fig13_subgroup_outcomes")

print("selected model:", art["best_name"])
print("SAFE components:", {k: round(v, 4) for k, v in components.items()})
print("ROC-AUC %.4f  RGA %.4f  Brier %.4f  calib slope %.3f intercept %.3f  CvM p=%.2e"
      % (metrics["roc_auc"], rga_point, brier, slope, intercept, cvm.pvalue))
print("\nFairness by subgroup:")
print(fair_df.round(4).to_string())
print("\nRGA-parity cross-check:", {k: round(v, 4) for k, v in rga_parity.items() if v == v})
print("\nSubgroup outcomes (observed vs predicted high-cost rate):")
print(subgroup_outcomes.to_string(index=False))
'''

write_nb("03_evaluation.ipynb", CODE)
