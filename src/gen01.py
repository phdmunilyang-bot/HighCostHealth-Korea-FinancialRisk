from make_notebooks import write_nb

CODE = r'''
# 01_eda.ipynb
# Exploratory data analysis of the person-year transition panel.
# Figures are grayscale (seaborn), no titles/captions, saved as PNG and PDF at 600 dpi.
# No absolute paths are printed (anonymity).

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent
PROC = ROOT / "data" / "processed"
FIG = ROOT / "results" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
TAB = ROOT / "results" / "tables"; TAB.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 600, "font.size": 10,
    "axes.edgecolor": "0.3", "axes.linewidth": 0.8,
    "image.cmap": "Greys", "axes.prop_cycle": plt.cycler(color=["0.15", "0.45", "0.65", "0.80"]),
})
G_DARK, G_MID, G_LIGHT = "0.25", "0.5", "0.75"

def save_fig(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)

panel = pd.read_parquet(PROC / "khp_panel_highcost.parquet")

# Fig 1: distribution of next-year total expenditure (log1p), with the top-decile cutoff
fig, ax = plt.subplots(figsize=(5.2, 3.4))
vals = np.log1p(panel["cost_next"].clip(lower=0))
sns.histplot(vals, bins=60, color=G_MID, edgecolor="white", linewidth=0.3, ax=ax)
cut = np.log1p(panel["thr_p90"].median())
ax.axvline(cut, color=G_DARK, linestyle="--", linewidth=1.2)
ax.set_xlabel("log(1 + next-year total expenditure, KRW)")
ax.set_ylabel("Person-years")
save_fig(fig, "fig01_cost_distribution")

# Fig 2: target prevalence by feature year
pv = panel.groupby("feat_year")["y_highcost"].mean().reset_index()
fig, ax = plt.subplots(figsize=(5.0, 3.2))
sns.barplot(data=pv, x="feat_year", y="y_highcost", color=G_MID, edgecolor="0.3", ax=ax)
ax.set_xlabel("Feature year (t)"); ax.set_ylabel("High-cost rate at t+1")
ax.set_ylim(0, 0.15)
save_fig(fig, "fig02_target_prevalence")

# Fig 3: high-cost rate across sensitive subgroups
def rate(mask): return panel.loc[mask, "y_highcost"].mean()
groups = {
    "Medical Aid": rate(panel.medical_aid == 1), "NHI": rate(panel.medical_aid == 0),
    "Disabled": rate(panel.disability == 1), "Non-disabled": rate(panel.disability == 0),
    "Age>=65": rate(panel.age >= 65), "Age<65": rate(panel.age < 65),
    "Female": rate(panel.sex == 2), "Male": rate(panel.sex == 1),
}
gd = pd.DataFrame({"group": list(groups), "rate": list(groups.values())})
fig, ax = plt.subplots(figsize=(5.6, 3.4))
sns.barplot(data=gd, y="group", x="rate", color=G_MID, edgecolor="0.3", ax=ax)
ax.set_xlabel("High-cost rate at t+1"); ax.set_ylabel("")
save_fig(fig, "fig03_highcost_by_subgroup")

# Fig 4: attrition rate by transition (overall vs Medical Aid vs age>=65)
att = pd.read_csv(TAB / "attrition.csv")
fig, ax = plt.subplots(figsize=(5.6, 3.4))
x = np.arange(len(att))
ax.plot(x, att["rate"], marker="o", color=G_DARK, label="Overall")
ax.plot(x, att["rate_medaid"], marker="s", color=G_MID, linestyle="--", label="Medical Aid")
ax.plot(x, att["rate_age65"], marker="^", color=G_LIGHT, linestyle=":", label="Age>=65")
ax.set_xticks(x); ax.set_xticklabels(att["transition"], rotation=30, ha="right")
ax.set_ylabel("Attrition rate (%)"); ax.set_xlabel("")
ax.legend(frameon=False, fontsize=8)
save_fig(fig, "fig04_attrition")

# Fig 5: cost concentration (Lorenz curve of next-year expenditure)
c = np.sort(panel["cost_next"].clip(lower=0).values)
cum = np.cumsum(c) / c.sum()
p = np.linspace(0, 1, len(cum))
fig, ax = plt.subplots(figsize=(4.2, 4.0))
ax.plot(p, cum, color=G_DARK, linewidth=1.5)
ax.plot([0, 1], [0, 1], color=G_LIGHT, linestyle="--", linewidth=1.0)
ax.set_xlabel("Cumulative share of person-years")
ax.set_ylabel("Cumulative share of expenditure")
ax.set_aspect("equal")
save_fig(fig, "fig05_lorenz")

# Fig 6: correlation heatmap of numeric features (grayscale)
numf = ["age", "edu", "hh_income", "n_chronic", "cost", "oop", "n_visit",
        "n_in", "n_er", "n_phi", "y_highcost"]
corr = panel[numf].corr()
fig, ax = plt.subplots(figsize=(5.6, 4.8))
sns.heatmap(corr, cmap="Greys", vmin=-1, vmax=1, square=True, linewidths=0.4,
            linecolor="white", cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
save_fig(fig, "fig06_correlation")

# Compact EDA summary table
summ = panel.groupby("feat_year").agg(
    n=("y_highcost", "size"), highcost_rate=("y_highcost", "mean"),
    median_cost_next=("cost_next", "median"), mean_cost_next=("cost_next", "mean"),
    medaid_share=("medical_aid", "mean"), disab_share=("disability", "mean"),
    age65_share=("age", lambda s: (s >= 65).mean())).round(3)
summ.to_csv(TAB / "eda_summary.csv")
print("figures saved:", sorted(set(f.stem for f in FIG.glob("fig0*"))))
print(summ.to_string())
'''

write_nb("01_eda.ipynb", CODE)
