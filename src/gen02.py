from make_notebooks import write_nb

CODE = r'''
# 02_modelling.ipynb
# Modelling Agent: train six candidate models under a prospective split, select the
# model with the highest mean 5-fold stratified cross-validated ROC-AUC on the training
# window, and persist the fitted pipeline and test-set artifacts for evaluation.
# No absolute paths are printed (anonymity).

import warnings, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, VotingClassifier,
                              StackingClassifier)
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent
PROC = ROOT / "data" / "processed"
TAB = ROOT / "results" / "tables"; TAB.mkdir(parents=True, exist_ok=True)
SEED = 42

panel = pd.read_parquet(PROC / "khp_panel_highcost.parquet")

FEAT = ["age", "sex", "capital", "edu", "marr", "health_ins", "medical_aid", "hh_income",
        "inc_q", "disability", "care_need", "ltc", "cd_any", "n_chronic", "cost", "oop",
        "n_visit", "n_in", "n_er", "any_inpatient", "n_phi", "has_phi"]
NUM = ["age", "hh_income", "n_chronic", "cost", "oop", "n_visit", "n_in", "n_er", "n_phi"]
CAT = ["health_ins", "marr", "edu", "inc_q", "sex"]
BIN = ["capital", "medical_aid", "disability", "care_need", "ltc", "cd_any", "any_inpatient", "has_phi"]

tr = panel[panel.split == "train"]
te = panel[panel.split == "test"]
Xtr, ytr = tr[FEAT].reset_index(drop=True), tr["y_highcost"].to_numpy()
Xte, yte = te[FEAT].reset_index(drop=True), te["y_highcost"].to_numpy()

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT),
    ("bin", "passthrough", BIN),
])

def pipe(clf):
    return Pipeline([("pre", pre), ("clf", clf)])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# --- tuned base learners (modest grids to bound runtime) ---
lr_gs = GridSearchCV(pipe(LogisticRegression(max_iter=2000, class_weight="balanced")),
                     {"clf__C": [0.1, 1.0, 3.0]}, scoring="roc_auc", cv=cv, n_jobs=-1)
rf_gs = GridSearchCV(pipe(RandomForestClassifier(random_state=SEED, class_weight="balanced")),
                     {"clf__n_estimators": [300], "clf__max_depth": [None, 12],
                      "clf__min_samples_leaf": [1, 5]}, scoring="roc_auc", cv=cv, n_jobs=-1)
spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
xgb_gs = GridSearchCV(pipe(XGBClassifier(random_state=SEED, eval_metric="logloss",
                                         tree_method="hist", scale_pos_weight=spw)),
                      {"clf__n_estimators": [400], "clf__max_depth": [3, 6],
                       "clf__learning_rate": [0.05, 0.1]}, scoring="roc_auc", cv=cv, n_jobs=-1)
lr_gs.fit(Xtr, ytr); rf_gs.fit(Xtr, ytr); xgb_gs.fit(Xtr, ytr)
lr_best, rf_best, xgb_best = lr_gs.best_estimator_, rf_gs.best_estimator_, xgb_gs.best_estimator_

voting = pipe(VotingClassifier(estimators=[
    ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", C=lr_gs.best_params_["clf__C"])),
    ("rf", RandomForestClassifier(random_state=SEED, class_weight="balanced", n_estimators=300)),
    ("xgb", XGBClassifier(random_state=SEED, eval_metric="logloss", tree_method="hist",
                          scale_pos_weight=spw, n_estimators=400))], voting="soft"))
stacking = pipe(StackingClassifier(estimators=[
    ("rf", RandomForestClassifier(random_state=SEED, class_weight="balanced", n_estimators=300)),
    ("xgb", XGBClassifier(random_state=SEED, eval_metric="logloss", tree_method="hist",
                          scale_pos_weight=spw, n_estimators=400))],
    final_estimator=LogisticRegression(max_iter=2000), cv=cv))
baseline = pipe(DummyClassifier(strategy="stratified", random_state=SEED))

candidates = {
    "LogisticRegression": lr_best, "RandomForest": rf_best, "XGBoost": xgb_best,
    "VotingEnsemble": voting, "StackingEnsemble": stacking, "RandomBaseline": baseline,
}

# --- model selection by mean 5-fold stratified CV ROC-AUC on the training window ---
rows = []
for name, est in candidates.items():
    sc = cross_val_score(est, Xtr, ytr, scoring="roc_auc", cv=cv, n_jobs=-1)
    rows.append(dict(model=name, cv_auc_mean=sc.mean(), cv_auc_sd=sc.std()))
cvres = pd.DataFrame(rows).sort_values("cv_auc_mean", ascending=False).reset_index(drop=True)
cvres.to_csv(TAB / "cv_results.csv", index=False)

best_name = cvres.iloc[0]["model"]
best = candidates[best_name].fit(Xtr, ytr)
yhat_te = best.predict_proba(Xte)[:, 1]
test_auc = roc_auc_score(yte, yhat_te)

with open(PROC / "model_artifacts.pkl", "wb") as f:
    pickle.dump(dict(best_name=best_name, model=best, FEAT=FEAT,
                     Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, yhat_te=yhat_te,
                     protected=te[["medical_aid", "disability", "sex", "age", "inc_q"]].reset_index(drop=True)), f)

print("5-fold CV ROC-AUC (training window):")
print(cvres.round(6).to_string(index=False))
print("\nselected model:", best_name)
print("held-out test (2023->2024) ROC-AUC: %.6f" % test_auc)
'''

write_nb("02_modelling.ipynb", CODE)
