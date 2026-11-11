from make_notebooks import write_nb

CODE = r'''
# 08_recalibration.ipynb
# Calibration appendix: the selected model systematically over-predicts (large negative
# calibration intercept). Here we recalibrate the held-out predictions with Platt scaling
# and isotonic regression, compare calibration before/after, and illustrate how a
# calibration axis could enter an extended SAFE_5 score. No absolute paths are printed.

import warnings, pickle, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from scipy.stats import cramervonmises_2samp

warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent
PROC = ROOT / "data" / "processed"
FIG = ROOT / "results" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
TAB = ROOT / "results" / "tables"; TAB.mkdir(parents=True, exist_ok=True)
SEED = 42

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 600, "font.size": 10,
                     "axes.edgecolor": "0.3", "image.cmap": "Greys"})
G_DARK, G_MID, G_LIGHT = "0.2", "0.5", "0.72"

def save_fig(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)

art = pickle.load(open(PROC / "model_artifacts.pkl", "rb"))
model = art["model"]
Xtr, ytr, Xte, yte = art["Xtr"], art["ytr"], art["Xte"], np.asarray(art["yte"])
yhat_raw = np.asarray(art["yhat_te"])

# Recalibrate by wrapping a clone of the selected pipeline (internal 5-fold).
platt = CalibratedClassifierCV(clone(model), method="sigmoid", cv=5).fit(Xtr, ytr)
iso = CalibratedClassifierCV(clone(model), method="isotonic", cv=5).fit(Xtr, ytr)
preds = {"raw": yhat_raw,
         "platt": platt.predict_proba(Xte)[:, 1],
         "isotonic": iso.predict_proba(Xte)[:, 1]}

def cal_stats(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); logit = np.log(p / (1 - p))
    m = LogisticRegression(max_iter=1000, C=1e6).fit(logit.reshape(-1, 1), yte)
    brier = brier_score_loss(yte, p)
    cvm = cramervonmises_2samp(p[yte == 1], p[yte == 0])
    return dict(brier=float(brier), intercept=float(m.intercept_[0]),
                slope=float(m.coef_[0, 0]), cvm_stat=float(cvm.statistic))

stats = {k: cal_stats(v) for k, v in preds.items()}

# Illustrative calibration score in [0,1] from calibration-in-the-large and slope distance.
def cal_score(s, C=3.0):
    return float(max(0.0, 1 - (abs(s["intercept"]) + abs(s["slope"] - 1)) / C))
cal_scores = {k: round(cal_score(v), 4) for k, v in stats.items()}

# Extended SAFE_5 using the existing four components plus a calibration axis (equal 0.2).
comp = json.load(open(TAB / "safe_components.json", encoding="utf-8"))
four = [comp["AURGA"], comp["RGR"], comp["AURGE"], comp["Fairness_Aggregate"]]
safe5 = {k: round(float(np.mean(four + [cal_score(stats[k])])), 4) for k in preds}

out = dict(calibration=stats, calibration_score=cal_scores,
           safe4=round(comp["SAFE_Score"], 4), safe5=safe5,
           note="calibration score and SAFE_5 are illustrative (C=3); recalibration does "
                "not change rank-based AURGA/RGR/AURGE/Fairness, only the calibration axis")
json.dump(out, open(TAB / "calibration_recal.json", "w"), ensure_ascii=False, indent=2)

# reliability curves before/after
fig, ax = plt.subplots(figsize=(4.4, 4.2))
styles = {"raw": ("-", G_DARK, "o"), "platt": ("--", G_MID, "s"), "isotonic": (":", G_LIGHT, "^")}
for k, (ls, c, mk) in styles.items():
    frac, mp = calibration_curve(yte, preds[k], n_bins=10, strategy="quantile")
    ax.plot(mp, frac, ls, color=c, marker=mk, markersize=4, label=k)
ax.plot([0, 1], [0, 1], color="0.6", linewidth=0.8)
ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency")
ax.set_aspect("equal"); ax.legend(frameon=False, fontsize=8)
save_fig(fig, "fig18_recalibration")

print("Calibration before/after:")
print(pd.DataFrame(stats).T.round(4).to_string())
print("\ncalibration scores:", cal_scores)
print("SAFE_4 = %.4f ; SAFE_5 =" % comp["SAFE_Score"], safe5)
'''

write_nb("08_recalibration.ipynb", CODE)
