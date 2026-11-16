from make_notebooks import write_nb

CODE = r'''
# 00_data_preparation.ipynb
# Data Agent: build the longitudinal person-year transition panel from KHP 2nd-wave
# (2019-2024), engineer features, define the binary high-cost target, and emit a
# machine-readable datacard plus an informative-attrition report.
# All paths are relative; no absolute paths are printed (anonymity).

import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import pyreadstat

warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"; PROC.mkdir(parents=True, exist_ok=True)
TAB = ROOT / "results" / "tables"; TAB.mkdir(parents=True, exist_ok=True)

YEARS = [(2019, "a"), (2020, "b"), (2021, "c"), (2022, "d"), (2023, "e"), (2024, "f")]
TRANS = [(2019, 2020), (2020, 2021), (2021, 2022), (2022, 2023), (2023, 2024)]
RNG_SEED = 42

def num(s):
    return pd.to_numeric(s, errors="coerce")

def clean_money(s):
    s = num(s)
    return s.where((s >= 0) & (s < 1e9), np.nan).fillna(0.0)

# ---- Medical-service (MS) cost aggregation to person-year ----
def cost_year(y, p):
    df, _ = pyreadstat.read_sas7bdat(
        str(RAW / f"sas{y}" / f"{p}_ms.sas7bdat"),
        usecols=["PIDWON", "MS1", "MEXP3_1", "MEXP3_2", "MEXP4_1"])
    df["PIDWON"] = df["PIDWON"].astype("Int64").astype(str)
    oop = clean_money(df["MEXP3_1"]) + clean_money(df["MEXP4_1"])      # out-of-pocket (facility + pharmacy)
    tot = oop + clean_money(df["MEXP3_2"])                             # + insurer share = total expenditure
    g = df.assign(oop=oop, tot=tot).groupby("PIDWON").agg(
        n_visit=("MS1", "size"), oop=("oop", "sum"), cost=("tot", "sum"),
        n_in=("MS1", lambda s: (s == 2).sum()), n_er=("MS1", lambda s: (s == 1).sum()))
    return g

# ---- Private health insurance (PHI): policies naming the person as an insured ----
def phi_counts(y, p):
    df, _ = pyreadstat.read_sas7bdat(
        str(RAW / f"sas{y}" / f"{p}_phi.sas7bdat"),
        usecols=[f"PHI_PID{i}" for i in range(1, 6)])
    long = pd.concat([df[c] for c in df.columns])
    long = num(long).dropna()
    long = long[long > 0].astype("Int64").astype(str)
    return long.value_counts()

# ---- Individual (IND) person-year feature frame ----
IND_BASE = ["PIDWON", "SEX", "BIRTH_Y", "REGION1", "EDU", "MARR", "HEALTH_INS",
            "H_INC_TOT", "DISA_YN", "CARE1", "CARE4", "DEATH_I_YN", "CD"]
CAPITAL = [11, 28, 41]  # Seoul, Incheon, Gyeonggi

def ind_year(y, p):
    path = str(RAW / f"sas{y}" / f"{p}_ind.sas7bdat")
    _, meta = pyreadstat.read_sas7bdat(path, metadataonly=True)
    cols = set(meta.column_names)
    cd1 = [c for c in meta.column_names if c.upper().startswith("CD1_")]
    use = [c for c in IND_BASE if c in cols] + cd1
    df, _ = pyreadstat.read_sas7bdat(path, usecols=use)
    df["PIDWON"] = df["PIDWON"].astype("Int64").astype(str)

    out = pd.DataFrame({"PIDWON": df["PIDWON"]})
    out["year"] = y
    out["age"] = y - num(df["BIRTH_Y"])
    out["sex"] = num(df["SEX"])                                   # 1 male, 2 female
    out["region"] = num(df["REGION1"])
    out["capital"] = out["region"].isin(CAPITAL).astype(int)
    out["edu"] = num(df.get("EDU"))
    out["marr"] = num(df.get("MARR"))
    hi = num(df["HEALTH_INS"])
    out["health_ins"] = hi
    out["medical_aid"] = hi.isin([5, 6]).astype(int)             # Medical Aid beneficiary
    inc = num(df["H_INC_TOT"])
    out["hh_income"] = inc
    out["inc_q"] = pd.qcut(inc.rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype("float")
    out["disability"] = (num(df["DISA_YN"]) == 1).astype(int)
    out["care_need"] = (num(df.get("CARE1")) == 1).astype(int)
    out["ltc"] = (num(df.get("CARE4")) == 1).astype(int)
    out["cd_any"] = (num(df["CD"]) == 1).astype(int)
    out["n_chronic"] = (df[cd1].apply(num) == 1).sum(axis=1)
    out["death"] = (num(df["DEATH_I_YN"]) == 1).astype(int)

    c = cost_year(y, p).reset_index()
    out = out.merge(c, on="PIDWON", how="left")
    for col in ["cost", "oop", "n_visit", "n_in", "n_er"]:
        out[col] = out[col].fillna(0.0)
    out["any_inpatient"] = (out["n_in"] > 0).astype(int)
    ph = phi_counts(y, p)
    out["n_phi"] = out["PIDWON"].map(ph).fillna(0).astype(int)
    out["has_phi"] = (out["n_phi"] > 0).astype(int)
    return out

frames = {y: ind_year(y, p) for y, p in YEARS}

FEAT = ["age", "sex", "capital", "edu", "marr", "health_ins", "medical_aid", "hh_income",
        "inc_q", "disability", "care_need", "ltc", "cd_any", "n_chronic", "cost", "oop",
        "n_visit", "n_in", "n_er", "any_inpatient", "n_phi", "has_phi"]

# ---- Build transition rows (feature year t -> outcome year t+1) ----
rows, attrition = [], []
for t, t1 in TRANS:
    ft = frames[t].set_index("PIDWON")
    f1 = frames[t1].set_index("PIDWON")
    in_t, in_t1 = set(ft.index), set(f1.index)
    elig = [pid for pid in ft.index if pid in in_t1]
    dropped = in_t - in_t1
    dft = ft.loc[list(in_t)]
    is_drop = pd.Series([pid in dropped for pid in dft.index], index=dft.index)

    def arate(mask):
        sub = is_drop[mask.values]
        return round(100 * sub.mean(), 1) if len(sub) else np.nan

    attrition.append(dict(
        transition=f"{t}->{t1}", n_t=len(in_t), dropped=len(dropped),
        rate=round(100 * len(dropped) / len(in_t), 1),
        rate_death=round(100 * dft["death"].mean(), 2),
        rate_medaid=arate(dft["medical_aid"] == 1),
        rate_nonmedaid=arate(dft["medical_aid"] == 0),
        rate_age65=arate(dft["age"] >= 65),
        rate_disab=arate(dft["disability"] == 1)))

    d = ft.loc[elig, FEAT].copy()
    d["cost_next"] = f1.loc[elig, "cost"].values
    d["death_next"] = f1.loc[elig, "death"].values
    thr = d["cost_next"].quantile(0.90)
    d["thr_p90"] = thr
    d["y_highcost"] = (d["cost_next"] >= thr).astype(int)
    d["feat_year"] = t; d["out_year"] = t1
    d["PIDWON"] = elig
    rows.append(d.reset_index(drop=True))

panel = pd.concat(rows, ignore_index=True)
panel["split"] = np.where(panel["feat_year"] == 2023, "test", "train")  # prospective split
order = ["PIDWON", "feat_year", "out_year", "split"] + FEAT + ["cost_next", "death_next", "thr_p90", "y_highcost"]
panel = panel[order]

panel.to_parquet(PROC / "khp_panel_highcost.parquet", index=False)
panel.to_csv(PROC / "khp_panel_highcost.csv", index=False)
pd.DataFrame(attrition).to_csv(TAB / "attrition.csv", index=False)

# ---- Datacard (SAFE-style artifact) ----
datacard = dict(
    dataset="Korea Health Panel Study, 2nd wave (2019-2024)",
    unit="person-year transition (features at t, outcome at t+1)",
    panel_key="PIDWON",
    n_rows=int(len(panel)),
    n_transitions=len(TRANS),
    target=dict(name="y_highcost", type="binary",
                definition="total medical expenditure at t+1 (facility OOP + statutory copay path "
                           "MEXP3_1 + insurer share MEXP3_2 + pharmacy MEXP4_1) in the top decile",
                prevalence=round(float(panel["y_highcost"].mean()), 4),
                threshold_rule="per-transition 90th percentile; negative/missing costs set to 0"),
    split=dict(scheme="prospective time split",
               train="2019->2020 ... 2022->2023", test="2023->2024",
               n_train=int((panel["split"] == "train").sum()),
               n_test=int((panel["split"] == "test").sum()),
               alternative="GroupKFold by PIDWON (prevents person-level leakage)"),
    features={c: str(panel[c].dtype) for c in FEAT},
    sensitive_attributes=["medical_aid", "disability", "sex", "age>=65", "inc_q(low)", "capital"],
    missingness={c: int(panel[c].isna().sum()) for c in FEAT if panel[c].isna().any()},
    attrition=attrition,
    seed=RNG_SEED,
    notes=[
        "2024 refresh sample (new PIDWONs) has no prior-year features; used as outcomes only.",
        "death_next=1 marks death during the transition -> cost censoring / competing risk "
        "(flagged for sensitivity analysis).",
        "Costs are self-reported survey values, not administrative claims.",
    ])
with open(TAB / "datacard.json", "w", encoding="utf-8") as f:
    json.dump(datacard, f, ensure_ascii=False, indent=2)

print("panel rows:", len(panel))
print(panel.groupby("feat_year").agg(n=("y_highcost", "size"),
      positives=("y_highcost", "sum"), rate=("y_highcost", "mean")).round(3).to_string())
print("split sizes:", panel["split"].value_counts().to_dict())
print("target prevalence:", round(float(panel["y_highcost"].mean()), 4))
print("attrition report:")
print(pd.DataFrame(attrition).to_string(index=False))
'''

write_nb("00_data_preparation.ipynb", CODE)
