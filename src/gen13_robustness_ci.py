from make_notebooks import write_nb

CODE = r'''
# 13_robustness_ci.ipynb
# Second-round robustness pack (reviewer extensions R1, R3, R4, M-c):
#   R1  bootstrap 95% CIs for the subgroup observed/predicted rates and gaps, and ROC-AUC.
#   R3  the observed-vs-predicted equity primitive recomputed for logistic regression and
#       random forest, to show the over-targeting is a property of the cost target rather
#       than of one model.
#   R4  a multivariable need adjustment of the predicted-positive gap (chronic-disease
#       count PLUS care need, long-term-care grade and prior utilisation), via a linear
#       probability model, to test whether the residual survives a richer need control.
#   M-c threshold sensitivity: the predicted-positive gap at the top 5%, 10% and 20%
#       operating points.
# Deterministic (fixed seeds); no external services. Run after notebooks 00, 02.

import json, pickle, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent
PROC = ROOT / "data" / "processed"
FIG = ROOT / "results" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
TAB = ROOT / "results" / "tables"; TAB.mkdir(parents=True, exist_ok=True)
SEED = 42
B = 2000
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

GROUPS = {
    "Medical-Aid": (prot["medical_aid"] == 1).values,
    "Disability":  (prot["disability"] == 1).values,
    "Low income":  (prot["inc_q"] == 1).values,
    "Age>=65":     (prot["age"] >= 65).values,
}

def ci(vals):
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)

# ================= R1: bootstrap CIs for subgroup rates / gaps and AUC =================
def flag_at(scores, q):
    return (scores >= np.quantile(scores, 1 - q)).astype(int)

pred = flag_at(yhat, 0.10)                      # top-decile decision
n = len(yte)
r1 = {}
for name, g in GROUPS.items():
    comp = ~g
    obs_g, pred_g = [], []
    gap_pc, gap_op = [], []                      # pred-vs-comparison, and pred-minus-observed
    for _ in range(B):
        idx = rng.integers(0, n, n)
        gg = g[idx]; cc = comp[idx]; yy = yte[idx]; pp = pred[idx]
        if gg.sum() == 0 or cc.sum() == 0:
            continue
        o = yy[gg].mean(); p = pp[gg].mean()
        obs_g.append(o); pred_g.append(p)
        gap_pc.append(p - pp[cc].mean())
        gap_op.append(p - o)
    r1[name] = dict(
        n=int(g.sum()), n_pos=int(yte[g].sum()),
        observed=round(float(yte[g].mean()), 4), observed_CI=ci(obs_g),
        predicted=round(float(pred[g].mean()), 4), predicted_CI=ci(pred_g),
        gap_pred_minus_observed=round(float(pred[g].mean() - yte[g].mean()), 4), gap_op_CI=ci(gap_op),
        gap_vs_comparison=round(float(pred[g].mean() - pred[~g].mean()), 4), gap_pc_CI=ci(gap_pc))

auc_bs = []
for _ in range(B):
    idx = rng.integers(0, n, n)
    if yte[idx].sum() in (0, len(idx)):
        continue
    auc_bs.append(roc_auc_score(yte[idx], yhat[idx]))
r1["_roc_auc"] = dict(point=round(float(roc_auc_score(yte, yhat)), 4), CI=ci(auc_bs))
json.dump(r1, open(TAB / "robustness_ci.json", "w"), ensure_ascii=False, indent=2)

# ================= R3: equity primitive across model classes =================
panel = pd.read_parquet(PROC / "khp_panel_highcost.parquet")
FEAT = art["FEAT"]
NUM = ["age", "hh_income", "n_chronic", "cost", "oop", "n_visit", "n_in", "n_er", "n_phi"]
CAT = ["health_ins", "marr", "edu", "inc_q", "sex"]
BIN = ["capital", "medical_aid", "disability", "care_need", "ltc", "cd_any", "any_inpatient", "has_phi"]
tr = panel[panel.split == "train"]; teo = panel[panel.split == "test"]
Xtr = tr[FEAT].reset_index(drop=True); ytr = tr["y_highcost"].to_numpy()
pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT),
    ("bin", "passthrough", BIN)])
def mkpipe(clf): return Pipeline([("pre", pre), ("clf", clf)])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

lr = GridSearchCV(mkpipe(LogisticRegression(max_iter=2000, class_weight="balanced")),
                  {"clf__C": [0.1, 1.0, 3.0]}, scoring="roc_auc", cv=cv, n_jobs=-1).fit(Xtr, ytr)
rf = GridSearchCV(mkpipe(RandomForestClassifier(random_state=SEED, class_weight="balanced")),
                  {"clf__n_estimators": [300], "clf__max_depth": [None, 12],
                   "clf__min_samples_leaf": [1, 5]}, scoring="roc_auc", cv=cv, n_jobs=-1).fit(Xtr, ytr)
preds = {"XGBoost": yhat,
         "LogReg": lr.predict_proba(Xte)[:, 1],
         "RandomForest": rf.predict_proba(Xte)[:, 1]}
by_model = {}
for mname, sc in preds.items():
    pf = flag_at(sc, 0.10)
    by_model[mname] = dict(auc=round(float(roc_auc_score(yte, sc)), 4))
    for gname, g in GROUPS.items():
        by_model[mname][gname] = dict(
            observed=round(float(yte[g].mean()), 4),
            predicted=round(float(pf[g].mean()), 4),
            gap_vs_comparison=round(float(pf[g].mean() - pf[~g].mean()), 4))
json.dump(by_model, open(TAB / "equity_by_model.json", "w"), ensure_ascii=False, indent=2)

# figure: predicted-vs-comparison gap by group, per model
order = ["Medical-Aid", "Disability", "Low income", "Age>=65"]
x = np.arange(len(order)); w = 0.26
fig, ax = plt.subplots(figsize=(6.0, 3.3))
for i, (mname, col) in enumerate(zip(["XGBoost", "LogReg", "RandomForest"], [G_DARK, G_MID, G_LIGHT])):
    ax.bar(x + (i - 1) * w, [by_model[mname][g]["gap_vs_comparison"] for g in order],
           w, color=col, edgecolor="0.3", label=mname)
ax.axhline(0, color="0.3", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(order)
ax.set_ylabel("Predicted-positive gap vs comparison"); ax.legend(frameon=False, fontsize=8)
save_fig(fig, "fig22_equity_by_model")

# ================= R4: multivariable need adjustment (linear probability model) =================
# crude gap = LPM  flag ~ group ; adjusted gap = LPM flag ~ group + need covariates.
# need covariates: chronic count, care need, LTC grade, prior cost/oop (log), inpatient.
Xte2 = Xte.copy()
Xte2["log_cost"] = np.log1p(pd.to_numeric(Xte2["cost"], errors="coerce").fillna(0))
Xte2["log_oop"] = np.log1p(pd.to_numeric(Xte2["oop"], errors="coerce").fillna(0))
# Two need specifications. CLINICAL uses morbidity/care-need proxies only. FULL adds prior
# utilisation (prior cost/oop/inpatient); because prior cost is itself an expenditure measure,
# controlling for it partly conditions on the very cost signal whose validity is in question,
# so FULL is a CONSERVATIVE (lower-bound) residual that may over-adjust the cost-as-need gap.
NEED_CLINICAL = ["n_chronic", "care_need", "ltc"]
NEED_FULL = ["n_chronic", "care_need", "ltc", "log_cost", "log_oop", "any_inpatient"]
for c in set(NEED_FULL):
    Xte2[c] = pd.to_numeric(Xte2[c], errors="coerce").fillna(0.0)

def lpm_gap(yv, gvec, covars, idx=None):
    if idx is None:
        idx = np.arange(len(yv))
    cols = [np.ones(len(idx)), gvec[idx].astype(float)]
    for c in covars:
        cols.append(Xte2[c].values[idx].astype(float))
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, yv[idx].astype(float), rcond=None)
    return float(beta[1])

def adj_with_ci(g, covars):
    point = lpm_gap(pred.astype(float), g, covars)
    bs = [lpm_gap(pred.astype(float), g, covars, rng.integers(0, n, n)) for _ in range(1000)]
    return round(point, 4), ci(bs)

r4 = {}
for gname, g in GROUPS.items():
    crude = lpm_gap(pred.astype(float), g, [])
    acl, acl_ci = adj_with_ci(g, NEED_CLINICAL)
    afu, afu_ci = adj_with_ci(g, NEED_FULL)
    r4[gname] = dict(
        crude_gap=round(crude, 4),
        clinical_adjusted_gap=acl, clinical_adjusted_CI=acl_ci,
        full_adjusted_gap=afu, full_adjusted_CI=afu_ci,
        share_explained_clinical=round(1 - acl / crude, 3) if crude else None,
        share_explained_full=round(1 - afu / crude, 3) if crude else None)
json.dump(r4, open(TAB / "need_multivariate.json", "w"), ensure_ascii=False, indent=2)

# ================= M-c: threshold sensitivity of the predicted-positive gap =================
thr_sens = {}
for q in [0.05, 0.10, 0.20]:
    pf = flag_at(yhat, q)
    thr_sens[f"top_{int(q*100)}pct"] = {
        gname: round(float(pf[g].mean() - pf[~g].mean()), 4) for gname, g in GROUPS.items()}
json.dump(thr_sens, open(TAB / "threshold_sensitivity.json", "w"), ensure_ascii=False, indent=2)

fig, ax = plt.subplots(figsize=(5.8, 3.3))
qs = [5, 10, 20]
for gname, col, mk in zip(order, [G_DARK, G_MID, "0.4", G_LIGHT], ["o", "s", "^", "D"]):
    ax.plot(qs, [thr_sens[f"top_{q}pct"][gname] for q in qs], marker=mk, color=col, label=gname)
ax.set_xlabel("Operating point (top % flagged)"); ax.set_ylabel("Predicted-positive gap vs comparison")
ax.set_xticks(qs); ax.legend(frameon=False, fontsize=8)
save_fig(fig, "fig23_threshold_sensitivity")

# ================= report =================
print("R1  subgroup gaps with 95% CI (pred-vs-comparison):")
for k, v in r1.items():
    if k.startswith("_"): continue
    print(f"  {k:12s} n={v['n']:5d} obs={v['observed']:.3f} pred={v['predicted']:.3f} "
          f"gap_vs_comp={v['gap_vs_comparison']:+.3f} CI{v['gap_pc_CI']}")
print(f"  ROC-AUC {r1['_roc_auc']['point']} CI{r1['_roc_auc']['CI']}")
print("\nR3  predicted-positive gap vs comparison, by model:")
for g in order:
    print(f"  {g:12s} " + "  ".join(f"{m}={by_model[m][g]['gap_vs_comparison']:+.3f}"
                                     for m in ["XGBoost", "LogReg", "RandomForest"]))
print("\nR4  need-adjusted predicted gap (crude -> clinical-adj [CI] -> full-adj [CI]):")
for g, v in r4.items():
    print(f"  {g:12s} crude {v['crude_gap']:+.3f} | clinical {v['clinical_adjusted_gap']:+.3f} "
          f"CI{v['clinical_adjusted_CI']} | full {v['full_adjusted_gap']:+.3f} CI{v['full_adjusted_CI']}")
print("\nM-c threshold sensitivity (gap vs comparison):")
print(pd.DataFrame(thr_sens).T.to_string())
'''

write_nb("13_robustness_ci.ipynb", CODE)
