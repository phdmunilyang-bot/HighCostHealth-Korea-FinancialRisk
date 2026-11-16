from make_notebooks import write_nb

CODE = r'''
# 11_need_stratified.ipynb
# Need-conditioned equity audit (reviewer extension M4). The crude equity primitive
# separates a subgroup's OBSERVED high-cost rate from the model's PREDICTED positive
# rate. Here we ask WHY an exposed subgroup is over-flagged: is the over-targeting
# explained by the group's higher recorded morbidity (need composition), or does it
# persist AT EQUAL NEED (a residual, group-specific distortion)? We stratify on a
# recorded need proxy (chronic-disease count) and use direct standardization to split
# the crude gap into a need-composition component and a need-adjusted residual. This
# operationalizes the cost-as-need concern of Obermeyer et al. (2019) inside the audit.
# No external services; deterministic. Run after notebooks 00, 02, 03.

import json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

art = pickle.load(open(PROC / "model_artifacts.pkl", "rb"))
Xte = art["Xte"].reset_index(drop=True)
yte = np.asarray(art["yte"]); yhat = np.asarray(art["yhat_te"])
prot = art["protected"].reset_index(drop=True)

thr = np.quantile(yhat, 0.90)              # top-decile decision, aligned with the target
pred = (yhat >= thr).astype(int)

# ---- Need proxy: recorded chronic-disease count, binned into interpretable strata ----
nchr = pd.to_numeric(Xte["n_chronic"], errors="coerce").fillna(0).astype(int).values
def need_bin(v):
    if v <= 0: return "0"
    if v <= 2: return "1-2"
    if v <= 4: return "3-4"
    return "5+"
bins = np.array([need_bin(v) for v in nchr])
BIN_ORDER = ["0", "1-2", "3-4", "5+"]

# ---- sensitive groups (unprivileged mask), matching notebook 03 ----
GROUPS = {
    "Medical-Aid": (prot["medical_aid"] == 1).values,
    "Disability":  (prot["disability"] == 1).values,
    "Low income":  (prot["inc_q"] == 1).values,
    "Age>=65":     (prot["age"] >= 65).values,
}

def rates(mask):
    return float(yte[mask].mean()), float(pred[mask].mean()), int(mask.sum())

def standardized(gmask, ref_mask, outcome):
    # direct standardization: apply group g's within-stratum outcome rate to the
    # reference (privileged) group's stratum distribution -> g's rate "if it had the
    # comparison group's need profile". outcome is yte (observed) or pred (flag).
    ref_w, num = {}, 0.0
    denom = ref_mask.sum()
    for b in BIN_ORDER:
        rb = ref_mask & (bins == b)
        gb = gmask & (bins == b)
        if gb.sum() == 0 or rb.sum() == 0:
            # fall back to g's own stratum rate weighted by ref share (0 if ref empty)
            r = outcome[gb].mean() if gb.sum() else 0.0
        else:
            r = outcome[gb].mean()
        num += (rb.sum() / denom) * r
    return float(num)

rows, decomp = [], {}
for name, gmask in GROUPS.items():
    ref = ~gmask
    o_g, p_g, n_g = rates(gmask)
    o_r, p_r, n_r = rates(ref)
    crude_pred_gap = p_g - p_r               # crude over-flagging vs comparison
    crude_obs_gap  = o_g - o_r
    # need-adjusted (standardized to comparison group's need profile)
    p_g_std = standardized(gmask, ref, pred)
    o_g_std = standardized(gmask, ref, yte)
    adj_pred_gap = p_g_std - p_r             # residual flag gap at equal need
    adj_obs_gap  = o_g_std - o_r
    decomp[name] = dict(
        n_group=n_g, n_ref=n_r,
        observed_rate=round(o_g, 4), predicted_rate=round(p_g, 4),
        crude_pred_gap=round(crude_pred_gap, 4),
        need_adjusted_pred_gap=round(adj_pred_gap, 4),
        pred_gap_explained_by_need=round(crude_pred_gap - adj_pred_gap, 4),
        crude_obs_gap=round(crude_obs_gap, 4),
        need_adjusted_obs_gap=round(adj_obs_gap, 4),
        share_pred_gap_from_need=round(1 - adj_pred_gap / crude_pred_gap, 3) if crude_pred_gap else None)
    # per-stratum detail
    for b in BIN_ORDER:
        gb = gmask & (bins == b); rb = ref & (bins == b)
        rows.append(dict(group=name, need_bin=b,
                         n_group=int(gb.sum()),
                         obs_group=round(float(yte[gb].mean()), 4) if gb.sum() else None,
                         pred_group=round(float(pred[gb].mean()), 4) if gb.sum() else None,
                         obs_ref=round(float(yte[rb].mean()), 4) if rb.sum() else None,
                         pred_ref=round(float(pred[rb].mean()), 4) if rb.sum() else None))

strata_df = pd.DataFrame(rows)
strata_df.to_csv(TAB / "need_stratified.csv", index=False)

# overall need distribution by group (context)
need_dist = {}
for name, gmask in GROUPS.items():
    d = pd.Series(bins[gmask]).value_counts(normalize=True).reindex(BIN_ORDER).fillna(0)
    need_dist[name] = {b: round(float(d[b]), 3) for b in BIN_ORDER}
need_dist["Comparison (overall)"] = {b: round(float((bins == b).mean()), 3) for b in BIN_ORDER}

summary = dict(
    need_proxy="n_chronic (recorded chronic-disease count), binned {0,1-2,3-4,5+}",
    decision="top-decile score threshold (aligned with target)",
    decomposition=decomp,
    need_distribution=need_dist,
    interpretation=(
        "crude_pred_gap = model flag rate minus comparison group's; "
        "need_adjusted_pred_gap = the same gap after standardizing the group to the "
        "comparison group's chronic-disease-count distribution (residual over-flagging "
        "at equal recorded need); pred_gap_explained_by_need = crude minus adjusted."))
json.dump(summary, open(TAB / "need_stratified.json", "w"), ensure_ascii=False, indent=2)

# ---- Figure: crude vs need-adjusted predicted-positive gap by group ----
order = ["Medical-Aid", "Disability", "Low income", "Age>=65"]
crude = [decomp[g]["crude_pred_gap"] for g in order]
adj = [decomp[g]["need_adjusted_pred_gap"] for g in order]
x = np.arange(len(order)); w = 0.38
fig, ax = plt.subplots(figsize=(5.6, 3.2))
ax.bar(x - w/2, crude, w, color=G_DARK, edgecolor="0.3", label="Crude gap")
ax.bar(x + w/2, adj, w, color=G_LIGHT, edgecolor="0.3", label="Need-adjusted (equal morbidity)")
ax.axhline(0, color="0.3", linewidth=0.8)
ax.set_xticks(x); ax.set_xticklabels(order)
ax.set_ylabel("Predicted-positive gap vs comparison")
ax.legend(frameon=False, fontsize=8)
save_fig(fig, "fig21_need_stratified")

print("Need-adjusted equity decomposition (predicted-positive gap):")
print(pd.DataFrame(decomp).T[["crude_pred_gap", "need_adjusted_pred_gap",
      "pred_gap_explained_by_need", "share_pred_gap_from_need"]].to_string())
print("\nPer-stratum observed vs predicted (head):")
print(strata_df.to_string(index=False))
print("\nNeed distribution by group:")
print(pd.DataFrame(need_dist).T.to_string())
'''

write_nb("11_need_stratified.ipynb", CODE)
