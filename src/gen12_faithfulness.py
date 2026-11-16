from make_notebooks import write_nb

CODE = r'''
# 12_faithfulness_benchmark.ipynb
# Grounding-faithfulness benchmark for the retrieval-grounded reviewer interface
# (reviewer extension M3). Notebook 05 wires the interface to an external LLM; the
# claim the governance argument rests on is narrower and LLM-independent: the account
# is drawn ONLY from recorded artifacts, and questions whose answers are not recorded
# are declined. This notebook tests exactly that substrate, deterministically and with
# no external service: (a) every in-scope reviewer query's gold answer is verified to
# be a verbatim figure in an emitted artifact (groundedness), and (b) every out-of-scope
# query is verified to have NO supporting figure in the artifact context, so a faithful
# interface must decline it (decline-correctness). It does NOT test an LLM's generation
# quality or whether a non-technical supervisor is helped -- those remain the user-study
# limitation. Run after notebooks 00-04, 10.

import json, re
from pathlib import Path
import pandas as pd

ROOT = Path.cwd().parent
TAB = ROOT / "results" / "tables"

# ---- Load the SAME artifacts the interface (notebook 05) is grounded on ----
def _load(name):
    p = TAB / name
    if p.suffix == ".json":
        return json.dumps(json.load(open(p, encoding="utf-8")), ensure_ascii=False)
    return p.read_text(encoding="utf-8")

ARTIFACT_FILES = ["datacard.json", "system_card.json", "evaluation_metrics.json",
                  "safe_components.json", "fairness_table.csv", "subgroup_outcomes.csv",
                  "attrition.csv", "cv_results.csv", "need_stratified.json",
                  "calibration_recal.json", "robustness_groupkfold.json"]
ARTIFACTS = {f: _load(f) for f in ARTIFACT_FILES if (TAB / f).exists()}
CONTEXT = "\n\n".join(f"### {k}\n{v}" for k, v in ARTIFACTS.items())

def grounded(token, source):
    # a gold figure is grounded iff it appears verbatim in its named source artifact
    return (token in ARTIFACTS.get(source, "")), source

def present_anywhere(tokens):
    return any(t in CONTEXT for t in tokens)

# ---- Benchmark: in-scope queries (answer must be a recorded figure) ----
IN_SCOPE = [
    ("What is the AGORA/SAFE governance score?",            "0.7508", "system_card.json"),
    ("Which governance dimension is weakest?",              "AURGE",  "system_card.json"),
    ("What is the held-out ROC-AUC?",                       "0.7974", "evaluation_metrics.json"),
    ("What is the Fairness Aggregate?",                     "0.7023", "system_card.json"),
    ("What is the robustness (RGR) score?",                 "0.9505", "safe_components.json"),
    ("What is the Medical-Aid observed high-cost rate?",    "0.1916", "subgroup_outcomes.csv"),
    ("What is the Medical-Aid predicted positive rate?",    "0.2731", "subgroup_outcomes.csv"),
    ("What is the disability predicted positive rate?",     "0.32",   "subgroup_outcomes.csv"),
    ("What is the calibration intercept?",                  "-2.40",  "evaluation_metrics.json"),
    ("How many test rows are there?",                       "11571",  "evaluation_metrics.json"),
    ("Which model was selected?",                           "XGBoost","system_card.json"),
    ("What is the target prevalence?",                      "0.1001", "datacard.json"),
    ("What is the person-level GroupKFold ROC-AUC?",        "0.7985", "robustness_groupkfold.json"),
    ("What is the Brier score before recalibration?",       "0.2101", "evaluation_metrics.json"),
]

# ---- Benchmark: out-of-scope queries (no recorded answer -> must decline) ----
# Each carries tokens a hallucinating system might fabricate; we verify they are ABSENT.
OUT_SCOPE = [
    ("What is the model's ROC-AUC on 2025 data?",               ["2025"]),
    ("What is the name of the highest-cost patient?",           ["patient_name", "resident_registration", "date_of_birth"]),
    ("What treatment should a flagged patient receive?",        ["treatment", "prescribe"]),
    ("What is the model's F1 score?",                           ["f1", "F1"]),
    ("What is the cost per QALY of the targeting program?",     ["QALY", "qaly"]),
    ("What causes a patient to become high-cost?",              ["causal", "causes"]),
]

rows = []
for q, gold, src in IN_SCOPE:
    ok, _ = grounded(gold, src)
    rows.append(dict(query=q, type="in_scope", gold=gold, source=src,
                     grounded=bool(ok), outcome="answer" if ok else "FAIL_missing"))
for q, toks in OUT_SCOPE:
    absent = not present_anywhere(toks)
    rows.append(dict(query=q, type="out_of_scope", gold="(none)", source="-",
                     grounded=bool(absent), outcome="decline" if absent else "FAIL_leak"))

bench = pd.DataFrame(rows)
bench.to_csv(TAB / "faithfulness_benchmark.csv", index=False)

n_in = (bench.type == "in_scope").sum()
n_out = (bench.type == "out_of_scope").sum()
grounded_rate = float(bench.loc[bench.type == "in_scope", "grounded"].mean())
decline_rate = float(bench.loc[bench.type == "out_of_scope", "grounded"].mean())

summary = dict(
    n_in_scope=int(n_in), n_out_of_scope=int(n_out),
    groundedness_in_scope=round(grounded_rate, 3),      # share whose gold figure is in an artifact
    decline_correctness_out_of_scope=round(decline_rate, 3),  # share correctly unanswerable
    n_artifacts=len(ARTIFACTS),
    note=("Tests the auditable substrate only: every in-scope gold figure is verified "
          "verbatim in an emitted artifact, and every out-of-scope query is verified to "
          "have no supporting figure. LLM generation faithfulness and supervisor utility "
          "are out of scope (user study)."))
json.dump(summary, open(TAB / "faithfulness_benchmark.json", "w"), ensure_ascii=False, indent=2)

print("Grounding-faithfulness benchmark")
print("  in-scope queries:      %d   groundedness: %.0f%%" % (n_in, 100 * grounded_rate))
print("  out-of-scope queries:  %d   correct declines: %.0f%%" % (n_out, 100 * decline_rate))
print("  artifacts grounded on: %d" % len(ARTIFACTS))
fails = bench[bench.outcome.str.startswith("FAIL")]
print("  failures:", "none" if fails.empty else fails["query"].tolist())
print()
print(bench.to_string(index=False))
'''

write_nb("12_faithfulness_benchmark.ipynb", CODE)
