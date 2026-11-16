from make_notebooks import write_nb

CODE = r'''
# 10_financial_risk.ipynb
# Financial-risk appendix (for the emerging-markets / public-finance framing):
# decompose the concentrated top-decile spending into its public (insurer-borne) and
# private (out-of-pocket) components, and estimate household catastrophic-health-expenditure
# (CHE) and impoverishment incidence by sensitive subgroup. These quantities support the
# household-financial-risk and public-insurance-sustainability reading of the high-cost target;
# they do not feed the TRIAGE Score. All paths relative; seed 42; no absolute paths printed.

import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import pyreadstat

warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
TAB = ROOT / "results" / "tables"; TAB.mkdir(parents=True, exist_ok=True)

YEARS = [(2019, "a"), (2020, "b"), (2021, "c"), (2022, "d"), (2023, "e"), (2024, "f")]
RNG_SEED = 42

def num(s):
    return pd.to_numeric(s, errors="coerce")

def clean_money(s):
    # mirror 00_data_preparation: negatives (-9 = don't-know) and implausible >1e9 -> 0
    s = num(s)
    return s.where((s >= 0) & (s < 1e9), np.nan).fillna(0.0)

# ---- Person-year public / private cost split from the medical-service (MS) file ----
def cost_split(y, p):
    df, _ = pyreadstat.read_sas7bdat(
        str(RAW / f"sas{y}" / f"{p}_ms.sas7bdat"),
        usecols=["PIDWON", "MEXP3_1", "MEXP3_2", "MEXP4_1"])
    df["PIDWON"] = df["PIDWON"].astype("Int64").astype(str)
    oop = clean_money(df["MEXP3_1"]) + clean_money(df["MEXP4_1"])   # facility co-pay + pharmacy
    pub = clean_money(df["MEXP3_2"])                                # insurer (NHI / Medical-Aid) share
    g = df.assign(oop=oop, pub=pub, tot=oop + pub).groupby("PIDWON").agg(
        oop=("oop", "sum"), pub=("pub", "sum"), tot=("tot", "sum"))
    g["year"] = y
    return g.reset_index()

# ---- Household attributes (income + subgroup flags) from the IND file ----
def hh_attr(y, p):
    df, _ = pyreadstat.read_sas7bdat(
        str(RAW / f"sas{y}" / f"{p}_ind.sas7bdat"),
        usecols=["PIDWON", "HHID", "HEALTH_INS", "DISA_YN", "H_INC_TOT", "BIRTH_Y"])
    df["PIDWON"] = df["PIDWON"].astype("Int64").astype(str)
    out = pd.DataFrame({"PIDWON": df["PIDWON"]})
    out["HHID"] = df["HHID"].astype("Int64").astype(str)
    out["year"] = y
    out["age"] = y - num(df["BIRTH_Y"])
    out["medical_aid"] = num(df["HEALTH_INS"]).isin([5, 6]).astype(int)
    out["disability"] = (num(df["DISA_YN"]) == 1).astype(int)
    out["hh_income_won"] = num(df["H_INC_TOT"]) * 10000.0           # KHP income in 10k KRW -> KRW
    return out

frames = []
for y, p in YEARS:
    frames.append(hh_attr(y, p).merge(cost_split(y, p)[["PIDWON", "oop", "pub", "tot"]],
                                      on="PIDWON", how="left"))
py = pd.concat(frames, ignore_index=True)
for c in ["oop", "pub", "tot"]:
    py[c] = py[c].fillna(0.0)
# low income = bottom income quintile within year (household income)
py["lowinc"] = py.groupby("year")["hh_income_won"].transform(
    lambda s: (s <= s.quantile(0.2)).astype(int))
py["age65"] = (py["age"] >= 65).astype(int)

# ---------- (A) Concentration and public/private split of the top decile ----------
cut = py["tot"].quantile(0.90)
top = py[py["tot"] >= cut]
conc = 100 * top["tot"].sum() / py["tot"].sum()

def split_block(mask):
    d = top if mask is None else top[mask]
    if d["tot"].sum() == 0:
        return (np.nan, np.nan)
    return (round(100 * d["pub"].sum() / d["tot"].sum(), 1),
            round(100 * d["oop"].sum() / d["tot"].sum(), 1))

groups = [("Overall", None), ("Medical-Aid", top["medical_aid"] == 1),
          ("Disability", top["disability"] == 1), ("Age>=65", top["age65"] == 1),
          ("Low income", top["lowinc"] == 1)]
split = {g: split_block(m) for g, m in groups}

# ---------- (B) Household CHE and impoverishment ----------
hh = py.groupby(["HHID", "year"], as_index=False).agg(
    hh_oop=("oop", "sum"), hh_inc=("hh_income_won", "first"),
    medical_aid=("medical_aid", "max"), disability=("disability", "max"),
    age65=("age65", "max"), lowinc=("lowinc", "max"))
hh = hh[hh["hh_inc"] > 0].copy()
hh["ctp"] = hh["hh_inc"] * 0.6                               # capacity to pay: income net of ~40% subsistence
hh["che10"] = (hh["hh_oop"] / hh["hh_inc"] > 0.10).astype(int)
hh["che40"] = (hh["hh_oop"] / hh["ctp"] > 0.40).astype(int)
pov = 0.5 * hh["hh_inc"].median()                           # relative poverty line = 50% median HH income
hh["impov"] = ((hh["hh_inc"] >= pov) & (hh["hh_inc"] - hh["hh_oop"] < pov)).astype(int)

hgroups = [("Overall", None), ("Medical-Aid", "medical_aid"), ("Disability", "disability"),
           ("Age>=65", "age65"), ("Low income", "lowinc")]
def che_block(key):
    d = hh if key is None else hh[hh[key] == 1]
    return dict(n=int(len(d)),
                che10=round(100 * d["che10"].mean(), 1),
                che40=round(100 * d["che40"].mean(), 1),
                impov=round(100 * d["impov"].mean(), 1))
che = {g: che_block(k) for g, k in hgroups}

# ---------- Emit artifact ----------
out = dict(
    n_person_years=int(len(py)), n_household_years=int(len(hh)),
    concentration_top_decile_pct=round(float(conc), 1),
    public_share_pct={g: split[g][0] for g, _ in groups},
    oop_share_pct={g: split[g][1] for g, _ in groups},
    che=che, seed=RNG_SEED,
    definitions=dict(
        oop="facility co-payment (MEXP3_1) + pharmacy (MEXP4_1)",
        public="insurer (NHI / Medical-Aid) share (MEXP3_2)",
        capacity_to_pay="household income net of a 40% subsistence allowance",
        che10="OOP > 10% of household income", che40="OOP > 40% of capacity to pay",
        impoverishment="household pushed below 50% of median household income by its OOP"),
    note="Survey-reported income and OOP; estimates for the public-finance / financial-risk framing. "
         "Not folded into the TRIAGE Score.")
with open(TAB / "financial_risk.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print("person-years:", len(py), "| household-years:", len(hh))
print("top-decile concentration: %.1f%%" % conc)
print("\npublic share of top-decile cost (%):",
      {g: split[g][0] for g, _ in groups})
print("\nCHE / impoverishment by subgroup:")
print(pd.DataFrame(che).T.to_string())
'''

write_nb("10_financial_risk.ipynb", CODE)
