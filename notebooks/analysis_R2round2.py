# Round-2 new experiments on real data (reuses model_artifacts.pkl):
#  (1) Intersectional audit: Medical-Aid x disability subgroup observed/predicted/need-adj gap
#  (2) Alternative need specifications for the disability residual (care_need alone;
#      multi-clinical composite), to bound the specification dependence.
import json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

ROOT = Path.cwd().parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "results" / "tables"
rng = np.random.default_rng(42)

art = pickle.load(open(PROC / "model_artifacts.pkl", "rb"))
yte = np.asarray(art["yte"]).astype(int)
yhat = np.asarray(art["yhat_te"]).astype(float)
thr = np.quantile(yhat, 0.90)
flag = (yhat >= thr).astype(int)

# recover the full test-row feature frame (need columns for need adjustment)
panel = pd.read_csv(PROC / "khp_panel_highcost.csv")
te = panel[panel["split"] == "test"].reset_index(drop=True).copy()
assert len(te) == len(yte), (len(te), len(yte))
te["flag"] = flag
te["y"] = yte
te["age65"] = (te["age"] >= 65).astype(int)
te["inc_low"] = (te["inc_q"] == panel["inc_q"].min()).astype(int)

# ---------- (1) Intersectional: Medical-Aid x disability ----------
def rate_ci(mask, col, B=2000):
    v = te.loc[mask, col].values
    if len(v) == 0:
        return np.nan, (np.nan, np.nan)
    boot = [rng.choice(v, len(v), replace=True).mean() for _ in range(B)]
    return v.mean(), tuple(np.percentile(boot, [2.5, 97.5]))

cells = {
    "MedAid & Disab": (te["medical_aid"] == 1) & (te["disability"] == 1),
    "MedAid only":    (te["medical_aid"] == 1) & (te["disability"] == 0),
    "Disab only":     (te["medical_aid"] == 0) & (te["disability"] == 1),
    "Neither":        (te["medical_aid"] == 0) & (te["disability"] == 0),
}
rows = []
for name, m in cells.items():
    n = int(m.sum())
    obs, obs_ci = rate_ci(m, "y")
    pred, pred_ci = rate_ci(m, "flag")
    rows.append(dict(cell=name, n=n,
                     observed=round(float(obs), 3) if n else np.nan,
                     predicted=round(float(pred), 3) if n else np.nan,
                     pred_ci_lo=round(float(pred_ci[0]), 3) if n else np.nan,
                     pred_ci_hi=round(float(pred_ci[1]), 3) if n else np.nan,
                     gap_pred_minus_obs=round(float(pred - obs), 3) if n else np.nan))
inter = pd.DataFrame(rows)
inter.to_csv(OUT / "intersectional.csv", index=False)
print("=== Intersectional audit: Medical-Aid x disability (test set) ===")
print(inter.to_string(index=False))

# ---------- (2) Alternative need specifications for the disability residual ----------
# LPM of flag on disability indicator + need covariates; group coef = need-adjusted gap.
def lpm_gap(group_col, need_cols, B=2000):
    # Standardize NEED covariates (cost/oop are on a 1e8 scale and would otherwise
    # numerically dominate the fit); the group indicator stays 0/1 so its coefficient
    # is the need-adjusted gap in probability units.
    d = te.copy()
    keep = [group_col] + need_cols
    Xraw = d[keep].fillna(d[keep].median()).astype(float)
    Xstd = Xraw.copy()
    for c in need_cols:
        s = Xraw[c].std()
        if s > 0:
            Xstd[c] = (Xraw[c] - Xraw[c].mean()) / s
    X = Xstd.values
    gi = list(Xstd.columns).index(group_col)
    y = d["flag"].values
    coef = LinearRegression().fit(X, y).coef_[gi]
    boot = np.empty(B)
    idx = np.arange(len(d))
    for b in range(B):
        bi = rng.choice(idx, len(idx), replace=True)
        boot[b] = LinearRegression().fit(X[bi], y[bi]).coef_[gi]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return round(float(coef), 3), (round(float(lo), 3), round(float(hi), 3))

specs = {
    "clinical (n_chronic, care_need, ltc)": ["n_chronic", "care_need", "ltc"],
    "care_need alone":                      ["care_need"],
    "multi-clinical composite":             ["n_chronic", "care_need", "ltc", "any_inpatient"],
    "+ prior utilisation":                  ["n_chronic", "care_need", "ltc", "cost", "oop", "any_inpatient"],
}
spec_rows = []
for label, cols in specs.items():
    cols = [c for c in cols if c in te.columns]
    g, ci = lpm_gap("disability", cols)
    spec_rows.append(dict(specification=label, disability_residual_gap=g,
                          ci_lo=ci[0], ci_hi=ci[1], excl_zero=bool(ci[0] > 0 or ci[1] < 0)))
spec = pd.DataFrame(spec_rows)
spec.to_csv(OUT / "disability_need_specs.csv", index=False)
print("\n=== Disability residual gap under alternative need specifications ===")
print(spec.to_string(index=False))

json.dump({"intersectional": rows, "disability_specs": spec_rows},
          open(OUT / "round2_experiments.json", "w"), ensure_ascii=False, indent=2)
print("\nDONE")
