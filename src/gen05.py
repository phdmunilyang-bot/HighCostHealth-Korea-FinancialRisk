from make_notebooks import write_nb

CODE = r'''
# 05_chatbot_agent.ipynb
# Chatbot Agent (LLM): an artifact-grounded conversational interface for human reviewers.
# It answers ONLY from the artifacts produced by the upstream agents (datacard, system
# card, evaluation metrics, fairness and attrition tables). It never retrains a model or
# recomputes a metric, which keeps answers auditable and suppresses hallucination.
#
# This is the ONLY notebook that calls an external LLM API. It requires a `.env` file at
# the repository root (one level above notebooks/) with OPENAI_API_KEY and LLM_MODEL_*.
# Run it locally after notebooks 00-04. No absolute paths are printed (anonymity).

import os, json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path.cwd().parent
TAB = ROOT / "results" / "tables"
load_dotenv(ROOT / ".env")

MODEL = os.getenv("LLM_MODEL_STRONG", "gpt-5.4")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---- Retrieval context: load every governance artifact as grounding ----
def _load(name):
    p = TAB / name
    if p.suffix == ".json":
        return json.dumps(json.load(open(p, encoding="utf-8")), ensure_ascii=False)
    return p.read_text(encoding="utf-8")

ARTIFACTS = {
    "datacard.json": _load("datacard.json"),
    "system_card.json": _load("system_card.json"),
    "evaluation_metrics.json": _load("evaluation_metrics.json"),
    "safe_components.json": _load("safe_components.json"),
    "fairness_table.csv": _load("fairness_table.csv"),
    "subgroup_outcomes.csv": _load("subgroup_outcomes.csv"),
    "attrition.csv": _load("attrition.csv"),
    "cv_results.csv": _load("cv_results.csv"),
}
CONTEXT = "\n\n".join(f"### {k}\n{v}" for k, v in ARTIFACTS.items())

SYSTEM = (
    "You are the governance assistant for a SAFE-agentic health-expenditure model. "
    "Answer the reviewer's question using ONLY the artifacts provided below. "
    "Quote the exact figures (SAFE components, fairness metrics, calibration, attrition). "
    "If the answer is not in the artifacts, say so explicitly and do not speculate. "
    "Never claim to retrain the model or recompute a metric.\n\n"
    f"=== ARTIFACTS ===\n{CONTEXT}"
)

def ask(question: str, model: str = MODEL) -> str:
    resp = client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": question}])
    return resp.choices[0].message.content

# ---- Example reviewer queries (uncomment to run locally with a valid API key) ----
EXAMPLES = [
    "What is the SAFE Score and which dimension is weakest?",
    "How does the high-cost rate differ for Medical-Aid beneficiaries, and is that a fairness concern?",
    "Is the model well calibrated, and is calibration part of the SAFE Score?",
    "Which transition shows the highest attrition and for which subgroup?",
]
# for q in EXAMPLES:
#     print("Q:", q)
#     print("A:", ask(q), "\n")
print("Chatbot agent ready. Provide .env with OPENAI_API_KEY, then call ask(<question>).")
'''

write_nb("05_chatbot_agent.ipynb", CODE)
