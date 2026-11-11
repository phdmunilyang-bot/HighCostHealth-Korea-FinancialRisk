from make_notebooks import write_nb

CODE = r'''
# 07_explainability.ipynb
# Explainability appendix (AURGE is the weakest SAFE dimension): (a) grouped rank-
# graduation explainability (RGE) over interpretable feature blocks, and (b) a
# parsimonious top-k model whose AURGE is compared with the full model to show that a
# smaller, more transparent feature set concentrates the explanation without losing rank
# accuracy. No absolute paths are printed (anonymity).

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
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from safeaipackage.core import rga
from safeaipackage.check_explainability import compute_rge_values

warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent
PROC = ROOT / "data" / "processed"
FIG = ROOT / "results" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
TAB = ROOT / "results" / "tables"; TAB.mkdir(parents=True, exist_ok=True)
SEED = 42
trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
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
model, FEAT = art["model"], art["FEAT"]
Xtr, ytr, Xte, yte, yhat = art["Xtr"], art["ytr"], art["Xte"], np.asarray(art["yte"]), np.asarray(art["yhat_te"])
_med = Xtr.median(numeric_only=True)
Xtr_i, Xte_i = Xtr.fillna(_med), Xte.fillna(_med)

def aurge(mdl, Xref, order):
    base = rga(list(yte), list(mdl.predict_proba(Xref)[:, 1]))
    Xp = Xref.copy(); curve = [base]
    for v in order:
        Xp[v] = rng.permutation(Xp[v].values)
        curve.append(rga(list(yte), list(mdl.predict_proba(Xp)[:, 1])))
    return float(trapz(curve, dx=1) / ((len(curve) - 1) * base)), base

# ---- (a) grouped RGE over interpretable feature blocks ----
GROUPS = {
    "demographic": ["age", "sex", "capital", "edu", "marr"],
    "socioeconomic": ["health_ins", "medical_aid", "hh_income", "inc_q"],
    "health_status": ["disability", "care_need", "ltc", "cd_any", "n_chronic"],
    "prior_utilisation": ["cost", "oop", "n_visit", "n_in", "n_er", "any_inpatient"],
    "private_insurance": ["n_phi", "has_phi"],
}
def group_rge(cols):
    df = compute_rge_values(Xtr_i, Xte_i, list(yhat), model, cols, group=True)
    return float(np.asarray(df).ravel()[0])
grge = {g: group_rge(cols) for g, cols in GROUPS.items()}
grge_df = pd.Series(grge).sort_values(ascending=False)

# ---- (b) parsimonious top-k model vs full model ----
rge_imp = pd.read_csv(TAB / "rge_importance.csv", index_col=0)["RGE"].sort_values(ascending=False)
order_full = list(rge_imp.index)
TOPK = 8
topk = list(rge_imp.head(TOPK).index)

NUM = ["age", "hh_income", "n_chronic", "cost", "oop", "n_visit", "n_in", "n_er", "n_phi"]
CAT = ["health_ins", "marr", "edu", "inc_q", "sex"]
BIN = ["capital", "medical_aid", "disability", "care_need", "ltc", "cd_any", "any_inpatient", "has_phi"]
def build_pipe(cols):
    num = [c for c in NUM if c in cols]; cat = [c for c in CAT if c in cols]; bn = [c for c in BIN if c in cols]
    pre = ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("o", OneHotEncoder(handle_unknown="ignore"))]), cat),
        ("bin", "passthrough", bn)])
    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    return Pipeline([("pre", pre), ("clf", XGBClassifier(random_state=SEED, eval_metric="logloss",
                     tree_method="hist", scale_pos_weight=spw, n_estimators=400, max_depth=6, learning_rate=0.1))])

red = build_pipe(topk).fit(Xtr[topk], ytr)
yhat_red = red.predict_proba(Xte[topk])[:, 1]
aurge_full, _ = aurge(model, Xte_i, order_full)
aurge_red, _ = aurge(red, Xte_i[topk], [c for c in order_full if c in topk])

summary = dict(
    grouped_rge={k: round(v, 4) for k, v in grge.items()},
    full_model=dict(n_features=len(FEAT), auc=float(roc_auc_score(yte, yhat)),
                    rga=float(rga(list(yte), list(yhat))), aurge=round(aurge_full, 4)),
    reduced_model=dict(n_features=TOPK, features=topk, auc=float(roc_auc_score(yte, yhat_red)),
                       rga=float(rga(list(yte), list(yhat_red))), aurge=round(aurge_red, 4)))
json.dump(summary, open(TAB / "explainability.json", "w"), ensure_ascii=False, indent=2)
grge_df.round(4).to_csv(TAB / "grouped_rge.csv")

fig, ax = plt.subplots(figsize=(5.4, 3.2))
sns.barplot(x=grge_df.values, y=grge_df.index, color=G_MID, edgecolor="0.3", ax=ax)
ax.set_xlabel("Grouped RGE"); ax.set_ylabel("")
save_fig(fig, "fig16_grouped_rge")

fig, ax = plt.subplots(figsize=(5.0, 3.2))
m = pd.DataFrame({"AUC": [summary["full_model"]["auc"], summary["reduced_model"]["auc"]],
                  "AURGE": [aurge_full, aurge_red]}, index=[f"Full ({len(FEAT)})", f"Top-{TOPK}"])
m.plot(kind="bar", ax=ax, color=[G_DARK, G_LIGHT], edgecolor="0.3", rot=0, width=0.7)
ax.set_ylim(0, 1.05); ax.set_ylabel("Score"); ax.legend(frameon=False, fontsize=8)
save_fig(fig, "fig17_full_vs_reduced")

print("Grouped RGE:"); print(grge_df.round(4).to_string())
print("\nFull model : AUC %.4f  RGA %.4f  AURGE %.4f  (%d features)"
      % (summary["full_model"]["auc"], summary["full_model"]["rga"], aurge_full, len(FEAT)))
print("Top-%d model: AUC %.4f  RGA %.4f  AURGE %.4f" %
      (TOPK, summary["reduced_model"]["auc"], summary["reduced_model"]["rga"], aurge_red))
print("top-k features:", topk)
'''

write_nb("07_explainability.ipynb", CODE)
