# R2-major-1: differential measurement-error assessment.
# (A) Internal consistency: self-reported per-capita total cost and public (insurer) share
#     by feature-year, and by subgroup, to compare against published NHI aggregates and to
#     check whether any measurement bias is DIFFERENTIAL across the sensitive groups.
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.cwd().parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "results" / "tables"
p = pd.read_csv(PROC / "khp_panel_highcost.csv")

p["public"] = (p["cost"] - p["oop"]).clip(lower=0)
p["pub_share"] = np.where(p["cost"] > 0, p["public"] / p["cost"], np.nan)

# (A1) per-capita self-reported total by feature year (KRW)
by_year = p.groupby("feat_year").agg(
    n=("cost", "size"),
    mean_total_krw=("cost", "mean"),
    median_total_krw=("cost", "median"),
    mean_oop_krw=("oop", "mean"),
    mean_public_krw=("public", "mean"),
).reset_index()
by_year["public_share_pct"] = (by_year["mean_public_krw"] /
                               (by_year["mean_public_krw"] + by_year["mean_oop_krw"]) * 100)
by_year = by_year.round(0)
print("=== Self-reported per-capita annual cost by feature year (KRW) ===")
print(by_year.to_string(index=False))

# (A2) subgroup-level public share and per-capita — differential check
p["age65"] = (p["age"] >= 65).astype(int)
p["inc_low"] = (p["inc_q"] == p["inc_q"].min()).astype(int)
def sub(col, val):
    s = p[p[col] == val]
    return dict(n=len(s), mean_total=s["cost"].mean(), mean_oop=s["oop"].mean(),
                public_share=100 * s["public"].sum() / max(s["cost"].sum(), 1))
subs = {
    "overall": dict(n=len(p), mean_total=p["cost"].mean(), mean_oop=p["oop"].mean(),
                    public_share=100 * p["public"].sum() / p["cost"].sum()),
    "medical_aid": sub("medical_aid", 1),
    "nhi": sub("medical_aid", 0),
    "disability": sub("disability", 1),
    "age65": sub("age65", 1),
    "inc_low": sub("inc_low", 1),
}
subdf = pd.DataFrame(subs).T
subdf["mean_total"] = subdf["mean_total"].round(0)
subdf["mean_oop"] = subdf["mean_oop"].round(0)
subdf["public_share"] = subdf["public_share"].round(1)
print("\n=== Self-reported cost & public share by subgroup (consistency check) ===")
print(subdf.to_string())

# The differential-error question: is the OOP share (where self-report error would bite
# hardest) internally consistent with the administratively-determined coverage structure?
# Medical-Aid beneficiaries have near-zero statutory co-payments, so a LOW self-reported
# OOP share for them is the expected administrative pattern, not a reporting artifact.
out = {
    "by_year": by_year.to_dict(orient="records"),
    "by_subgroup": {k: {kk: (round(float(vv), 1) if isinstance(vv, (int, float, np.floating)) else vv)
                        for kk, vv in v.items()} for k, v in subs.items()},
    "pooled_mean_total_krw": round(float(p["cost"].mean()), 0),
    "pooled_public_share_pct": round(float(100 * p["public"].sum() / p["cost"].sum()), 1),
}
json.dump(out, open(OUT / "measurement_consistency.json", "w"), ensure_ascii=False, indent=2)
print("\npooled mean total (KRW):", out["pooled_mean_total_krw"],
      "| pooled public share %:", out["pooled_public_share_pct"])
print("DONE")
