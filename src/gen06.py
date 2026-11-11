from make_notebooks import write_nb

CODE = r'''
# 06_robustness_groupkfold.ipynb
# Robustness appendix: (a) GroupKFold-by-person cross-validation as a leakage-controlled
# complement to the prospective time split, and (b) a death-competing-risk sensitivity
# analysis that excludes transitions ending in death (cost censoring).
# No absolute paths are printed (anonymity).

import warnings, pickle, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from safeaipackage.core import rga

warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent
PROC = ROOT / "data" / "processed"
FIG = ROOT / "results" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
TAB = ROOT / "results" / "tables"; TAB.mkdir(parents=True, exist_ok=True)
SEED = 42
trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 600, "font.size": 10,
                     "axes.edgecolor": "0.3", "image.cmap": "Greys"})
G_DARK, G_MID, G_LIGHT = "0.25", "0.5", "0.75"

def save_fig(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)

def aurga(y, s, grid=np.linspace(0, 0.9, 19)):
    y = np.asarray(y); s = np.asarray(s); order = np.argsort(-s); vals = []
    for x in grid:
        keep = order[int(len(s) * x):]
        vals.append(rga(list(y[keep]), list(s[keep])) if len(np.unique(y[keep])) == 2
                    else (vals[-1] if vals else 0.5))
    return float(trapz(vals, grid) / (grid[-1] - grid[0]))

art = pickle.load(open(PROC / "model_artifacts.pkl", "rb"))
model, FEAT = art["model"], art["FEAT"]
panel = pd.read_parquet(PROC / "khp_panel_highcost.parquet")
X, y = panel[FEAT], panel["y_highcost"].to_numpy()
groups = panel["PIDWON"].to_numpy()

# ---- (a) GroupKFold-by-person out-of-fold predictions (no person in train & test) ----
gkf = GroupKFold(n_splits=5)
oof = cross_val_predict(clone(model), X, y, cv=gkf, groups=groups,
                        method="predict_proba", n_jobs=-1)[:, 1]
fold_auc = []
for _, te_idx in gkf.split(X, y, groups):
    fold_auc.append(roc_auc_score(y[te_idx], oof[te_idx]))
gkf_summary = dict(scheme="GroupKFold-by-PIDWON (pooled transitions)",
                   auc_mean=float(np.mean(fold_auc)), auc_sd=float(np.std(fold_auc)),
                   rga=float(rga(list(y), list(oof))), aurga=aurga(y, oof))

# time-split reference (from the main pipeline)
ts_auc = roc_auc_score(art["yte"], art["yhat_te"])
ts_rga = rga(list(np.asarray(art["yte"])), list(np.asarray(art["yhat_te"])))

# ---- (b) death competing-risk sensitivity on the time-split test set ----
te = panel[panel.split == "test"].reset_index(drop=True)
yhat_te = np.asarray(art["yhat_te"]); yte = te["y_highcost"].to_numpy()
alive = (te["death_next"] == 0).to_numpy()
death_sens = dict(
    n_test=int(len(te)), n_death=int((~alive).sum()),
    rga_all=float(rga(list(yte), list(yhat_te))),
    rga_excl_death=float(rga(list(yte[alive]), list(yhat_te[alive]))),
    auc_all=float(roc_auc_score(yte, yhat_te)),
    auc_excl_death=float(roc_auc_score(yte[alive], yhat_te[alive])))

out = dict(groupkfold=gkf_summary, fold_auc=[round(a, 4) for a in fold_auc],
           time_split=dict(auc=float(ts_auc), rga=float(ts_rga)),
           death_sensitivity=death_sens)
json.dump(out, open(TAB / "robustness_groupkfold.json", "w"), ensure_ascii=False, indent=2)

# ---- figures ----
fig, ax = plt.subplots(figsize=(5.0, 3.2))
labels = ["Time split", "GroupKFold"]
vals = [ts_rga, gkf_summary["rga"]]
sns.barplot(x=labels, y=vals, color=G_MID, edgecolor="0.3", ax=ax)
ax.set_ylim(0.5, 0.85); ax.set_ylabel("RGA (= AUC)"); ax.set_xlabel("")
save_fig(fig, "fig14_split_comparison")

fig, ax = plt.subplots(figsize=(4.6, 3.2))
sns.barplot(x=["All", "Excl. death"],
            y=[death_sens["rga_all"], death_sens["rga_excl_death"]],
            color=G_MID, edgecolor="0.3", ax=ax)
ax.set_ylim(0.5, 0.85); ax.set_ylabel("RGA (test, 2023->2024)"); ax.set_xlabel("")
save_fig(fig, "fig15_death_sensitivity")

print("GroupKFold-by-person: AUC %.4f (sd %.4f), RGA %.4f, AURGA %.4f"
      % (gkf_summary["auc_mean"], gkf_summary["auc_sd"], gkf_summary["rga"], gkf_summary["aurga"]))
print("Time-split reference:  AUC %.4f, RGA %.4f" % (ts_auc, ts_rga))
print("fold AUCs:", [round(a, 4) for a in fold_auc])
print("\nDeath competing-risk sensitivity (test set):")
print("  n_death=%d  RGA all=%.4f  RGA excl-death=%.4f  (delta %.4f)"
      % (death_sens["n_death"], death_sens["rga_all"], death_sens["rga_excl_death"],
         death_sens["rga_excl_death"] - death_sens["rga_all"]))
'''

write_nb("06_robustness_groupkfold.ipynb", CODE)
