from make_notebooks import write_nb

CODE = r'''
# 09_continuous_cost.ipynb
# Auxiliary continuous-target analysis: instead of the binary top-decile label, regress
# log(1 + next-year total expenditure) under the same prospective split and feature set.
# Rank-graduation accuracy (RGA) is defined for continuous responses, so the ranking
# quality is directly comparable, and we check that the regression ranking recovers the
# binary high-cost outcome. No absolute paths are printed (anonymity).

import warnings, pickle, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, roc_auc_score
from scipy.stats import spearmanr
from safeaipackage.core import rga

warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent
PROC = ROOT / "data" / "processed"
FIG = ROOT / "results" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
TAB = ROOT / "results" / "tables"; TAB.mkdir(parents=True, exist_ok=True)
SEED = 42

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 600, "font.size": 10,
                     "axes.edgecolor": "0.3", "image.cmap": "Greys"})
G_DARK, G_MID, G_LIGHT = "0.25", "0.5", "0.75"

def save_fig(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)

panel = pd.read_parquet(PROC / "khp_panel_highcost.parquet")
FEAT = ["age", "sex", "capital", "edu", "marr", "health_ins", "medical_aid", "hh_income",
        "inc_q", "disability", "care_need", "ltc", "cd_any", "n_chronic", "cost", "oop",
        "n_visit", "n_in", "n_er", "any_inpatient", "n_phi", "has_phi"]
NUM = ["age", "hh_income", "n_chronic", "cost", "oop", "n_visit", "n_in", "n_er", "n_phi"]
CAT = ["health_ins", "marr", "edu", "inc_q", "sex"]
BIN = ["capital", "medical_aid", "disability", "care_need", "ltc", "cd_any", "any_inpatient", "has_phi"]

tr, te = panel[panel.split == "train"], panel[panel.split == "test"]
Xtr, Xte = tr[FEAT].reset_index(drop=True), te[FEAT].reset_index(drop=True)
ytr = np.log1p(tr["cost_next"].clip(lower=0).to_numpy())
yte_cost = te["cost_next"].clip(lower=0).to_numpy()
yte_log = np.log1p(yte_cost)
y_bin = te["y_highcost"].to_numpy()

pre = ColumnTransformer([
    ("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), NUM),
    ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                      ("o", OneHotEncoder(handle_unknown="ignore"))]), CAT),
    ("bin", "passthrough", BIN)])

models = {
    "Ridge": Pipeline([("pre", pre), ("reg", Ridge(alpha=1.0))]),
    "XGBoostReg": Pipeline([("pre", pre), ("reg", XGBRegressor(
        random_state=SEED, tree_method="hist", n_estimators=500, max_depth=6, learning_rate=0.05))]),
}
rows = {}
preds = {}
for name, mdl in models.items():
    mdl.fit(Xtr, ytr)
    pred_log = mdl.predict(Xte)
    preds[name] = pred_log
    rows[name] = dict(
        r2_log=float(r2_score(yte_log, pred_log)),
        spearman=float(spearmanr(yte_cost, pred_log).correlation),
        rga_cost=float(rga(list(yte_cost), list(pred_log))),       # RGA on the continuous target
        auc_recover_binary=float(roc_auc_score(y_bin, pred_log)))  # does the ranking recover top-decile?
res = pd.DataFrame(rows).T
res.round(4).to_csv(TAB / "continuous_results.csv")

# cross-check vs the binary pipeline's ranking
art = pickle.load(open(PROC / "model_artifacts.pkl", "rb"))
yhat_bin = np.asarray(art["yhat_te"])
best_reg = "XGBoostReg"
cross = dict(binary_rga=float(rga(list(y_bin), list(yhat_bin))),
             continuous_rga_cost=rows[best_reg]["rga_cost"],
             continuous_recover_auc=rows[best_reg]["auc_recover_binary"],
             spearman_binaryscore_vs_regscore=float(spearmanr(yhat_bin, preds[best_reg]).correlation))
json.dump(dict(models=rows, cross_check=cross), open(TAB / "continuous_results.json", "w"),
          ensure_ascii=False, indent=2)

# observed vs predicted (log scale), grayscale density
fig, ax = plt.subplots(figsize=(4.4, 4.0))
ax.hexbin(yte_log, preds[best_reg], gridsize=40, cmap="Greys", bins="log", mincnt=1)
lims = [0, max(yte_log.max(), preds[best_reg].max())]
ax.plot(lims, lims, color=G_DARK, linestyle="--", linewidth=1)
ax.set_xlabel("Observed log(1+expenditure)"); ax.set_ylabel("Predicted")
ax.set_aspect("equal")
save_fig(fig, "fig19_continuous_fit")

fig, ax = plt.subplots(figsize=(5.0, 3.0))
sns.barplot(x=res.index, y=res["rga_cost"].values, color=G_MID, edgecolor="0.3", ax=ax)
ax.set_ylim(0.5, 0.85); ax.set_ylabel("RGA (continuous cost)"); ax.set_xlabel("")
save_fig(fig, "fig20_continuous_rga")

print("Continuous-target models (test 2023->2024):")
print(res.round(4).to_string())
print("\nCross-check vs binary pipeline:")
for k, v in cross.items():
    print("  %s: %.4f" % (k, v))
'''

write_nb("09_continuous_cost.ipynb", CODE)
