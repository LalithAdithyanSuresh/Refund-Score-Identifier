"""
create_classification_notebook.py
───────────────────────────────────
Builds classification_pipeline.ipynb using nbformat.

The notebook covers:
  Block [3] Feature Extraction  (Sentence-BERT + extras)
  Block [4] Classification      (load model.pkl → APPROVE / REJECT)

Input  : validation_results.csv  (filter decision == VALID)
         model.pkl
Output : predictions.csv
"""

import nbformat as nbf

def make_md(source): return nbf.v4.new_markdown_cell(source)
def make_code(source): return nbf.v4.new_code_cell(source)

# ═══════════════════ CELL SOURCES (raw strings) ════════════════════════════

CELL_IMPORTS = r'''import re
import sys
import numpy as np
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──
VALIDATION_CSV  = "validation_results.csv"
MODEL_PKL       = "model.pkl"
PREDICTIONS_CSV = "predictions.csv"

# ── Sentence-BERT model ──
MODEL_NAME = "all-MiniLM-L6-v2"
print("Loading Sentence-BERT model...")
bert_model = SentenceTransformer(MODEL_NAME)
print("Model loaded.\n")

# ── Embedding cache ──
_emb_cache: dict = {}

def _embed(text: str) -> np.ndarray:
    """Return cached Sentence-BERT embedding."""
    if text not in _emb_cache:
        _emb_cache[text] = bert_model.encode(text, show_progress_bar=False)
    return _emb_cache[text]

def _max_sim(text_emb: np.ndarray, refs: list) -> tuple:
    ref_embs = np.array([_embed(r) for r in refs])
    sims = cosine_similarity(text_emb.reshape(1, -1), ref_embs)[0]
    idx = int(np.argmax(sims))
    return float(sims[idx]), idx

print("Configuration ready.")
'''

CELL_REFERENCES = r'''# ── Reference phrases (Sentence-BERT used for ALL similarity tasks) ──

ISSUE_TYPE_REFS = [
    "screen damage cracked display broken screen",   # 0
    "battery draining not charging battery problem",  # 1
    "performance slow lagging freezing",              # 2
    "connectivity disconnecting wifi bluetooth",      # 3
    "hardware failure dead not working broken",       # 4
]

ISSUE_TYPE_NAMES = ["screen", "battery", "performance", "connectivity", "hardware"]

SEVERITY_HIGH_REFS = [
    "completely broken and unusable",
    "dead on arrival will not turn on",
    "cracked screen device destroyed",
    "not working at all hardware failure",
    "completely non functional",
]

SEVERITY_MEDIUM_REFS = [
    "disconnecting frequently from bluetooth",
    "overheating during normal use",
    "randomly restarting several times a day",
    "battery drains very fast",
    "camera blurry all the time",
]

SEVERITY_LOW_REFS = [
    "slightly slow minor delay",
    "small cosmetic scratch on the body",
    "very minor issue barely noticeable",
    "slight audio delay not a big problem",
    "minor creak in chassis",
]

URGENCY_KEYWORDS = {"asap", "urgent", "urgently", "immediately",
                    "right away", "as soon as possible"}

# Pre-warm cache
print("Pre-warming Sentence-BERT cache with reference phrases...")
all_refs = (ISSUE_TYPE_REFS + SEVERITY_HIGH_REFS
            + SEVERITY_MEDIUM_REFS + SEVERITY_LOW_REFS)
for ref in all_refs:
    _embed(ref)
print(f"Cache warmed — {len(_emb_cache)} entries.\n")
'''

CELL_PREPROCESS = r'''def preprocess(text: str) -> str:
    """
    Minimal normalisation:
    - Lowercase
    - Replace 5+ digit numbers with <ID>
    - Normalise repeated characters (3+ → 2)
    - Strip non-essential symbols
    - Collapse whitespace
    """
    text = text.lower()
    text = re.sub(r"\b\d{5,}\b", "<ID>", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"[^\w\s!?.,]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ── Quick test ──
tests = [
    "The iPhone 14 screen is CRACKED. All caps urgency!",
    "order #1234567890 is bad",
    "soooo frustrating",
]
for t in tests:
    print(f"  {t!r}")
    print(f"  → {preprocess(t)!r}\n")
'''

CELL_EXTRACT_FEATURES = r'''def extract_features(text: str) -> dict:
    """
    Extract all five features from complaint text using Sentence-BERT.

    Feature vector layout (388 dims):
      [0:384]  Sentence-BERT embedding
      [384]    severity  (1=Low, 2=Medium, 3=High)
      [385]    sentiment (-1.0 to +1.0, TextBlob polarity)
      [386]    urgency   (0 or 1, keyword-based)
      [387]    issue_type (0-4, Sentence-BERT cosine similarity)
    """
    cleaned = preprocess(text)
    emb = _embed(cleaned)

    # Issue Type — Sentence-BERT cosine similarity
    _, issue_idx = _max_sim(emb, ISSUE_TYPE_REFS)

    # Severity — Sentence-BERT cosine similarity to 3-tier references
    high_score,   _ = _max_sim(emb, SEVERITY_HIGH_REFS)
    medium_score, _ = _max_sim(emb, SEVERITY_MEDIUM_REFS)
    low_score,    _ = _max_sim(emb, SEVERITY_LOW_REFS)

    scores = {3: high_score, 2: medium_score, 1: low_score}
    severity = max(scores, key=scores.get)

    # Sentiment — TextBlob polarity
    sentiment = round(TextBlob(cleaned).sentiment.polarity, 4)

    # Urgency — keyword match
    words = set(cleaned.split())
    urgency = int(bool(words & URGENCY_KEYWORDS))

    # Combined feature vector
    feature_vector = (
        list(emb.astype(float))
        + [float(severity)]
        + [float(sentiment)]
        + [float(urgency)]
        + [float(issue_idx)]
    )

    return {
        "cleaned_text":   cleaned,
        "embedding":      emb,
        "issue_type":     issue_idx,
        "issue_name":     ISSUE_TYPE_NAMES[issue_idx],
        "severity":       severity,
        "sentiment":      sentiment,
        "urgency":        urgency,
        "feature_vector": feature_vector,
    }

print("extract_features() function defined.")

# ── Smoke test ──
test_text = "The iPhone 14 screen is completely cracked and won't turn on. Need help ASAP."
r = extract_features(test_text)
print(f"\nSmoke test:")
print(f"  Text      : {test_text}")
print(f"  Issue     : {r['issue_name']}  (idx={r['issue_type']})")
print(f"  Severity  : {r['severity']}  ({'High' if r['severity']==3 else 'Medium' if r['severity']==2 else 'Low'})")
print(f"  Sentiment : {r['sentiment']}")
print(f"  Urgency   : {r['urgency']}")
print(f"  Vec dims  : {len(r['feature_vector'])}")
'''

CELL_CLASSIFY_FN = r'''def classify(text: str, model) -> str:
    """
    Run inference for a single complaint.
    Returns "APPROVE" or "REJECT".
    """
    feats = extract_features(text)
    vec   = np.array(feats["feature_vector"]).reshape(1, -1)
    pred  = int(model.predict(vec)[0])
    return "APPROVE" if pred == 1 else "REJECT"

print("classify() function defined.")
'''

CELL_LOAD = r'''# ── Load validation results (VALID only) ──
print(f"Loading '{VALIDATION_CSV}'...")
df_all = pd.read_csv(VALIDATION_CSV)
df = df_all[df_all["decision"] == "VALID"].copy().reset_index(drop=True)
print(f"Total rows    : {len(df_all)}")
print(f"VALID rows    : {len(df)}"  )
print(df[["complaint_id", "complaint_text"]].head(5))
'''

CELL_LOAD_MODEL = r'''# ── Load trained model ──
print(f"\nLoading model from '{MODEL_PKL}'...")
clf = joblib.load(MODEL_PKL)
print(f"Model loaded  : {type(clf).__name__}")
'''

CELL_INFERENCE = r'''# ── Run inference on all VALID complaints ──
print(f"\nRunning feature extraction + classification on {len(df)} complaints...\n")

results = []
for i, (_, row) in enumerate(df.iterrows(), 1):
    feats = extract_features(row["complaint_text"])
    pred  = classify(row["complaint_text"], clf)

    results.append({
        "complaint_id":  row["complaint_id"],
        "complaint_text": row["complaint_text"],
        "cleaned_text":  feats["cleaned_text"],
        "issue_type":    feats["issue_name"],
        "severity":      feats["severity"],
        "severity_label": ("High" if feats["severity"] == 3
                           else "Medium" if feats["severity"] == 2 else "Low"),
        "sentiment":     feats["sentiment"],
        "urgency":       feats["urgency"],
        "prediction":    pred,
    })

    if i % 30 == 0 or i == len(df):
        print(f"  Processed {i}/{len(df)}...")

preds_df = pd.DataFrame(results)
print(f"\nDone. {len(preds_df)} predictions generated.")
'''

CELL_SUMMARY = r'''# ── Summary ──
total    = len(preds_df)
approved = (preds_df["prediction"] == "APPROVE").sum()
rejected = (preds_df["prediction"] == "REJECT").sum()

print("=" * 60)
print("        CLASSIFICATION RESULTS SUMMARY")
print("=" * 60)
print(f"  Total VALID complaints : {total}")
print(f"  APPROVE                : {approved}  ({approved/total*100:.1f}%)")
print(f"  REJECT                 : {rejected}  ({rejected/total*100:.1f}%)")
print()
print("Severity breakdown (VALID complaints):")
sev_map = {3: "High", 2: "Medium", 1: "Low"}
for sev_val, sev_name in sev_map.items():
    n = (preds_df["severity"] == sev_val).sum()
    app = ((preds_df["severity"] == sev_val) & (preds_df["prediction"] == "APPROVE")).sum()
    rej = n - app
    print(f"  {sev_name:6s} (sev={sev_val}) : {n:3d} total | APPROVE={app} | REJECT={rej}")

print()
print("Issue type breakdown:")
for itype, count in preds_df["issue_type"].value_counts().items():
    app = ((preds_df["issue_type"] == itype) & (preds_df["prediction"] == "APPROVE")).sum()
    print(f"  {itype:15s}: {count:3d} complaints  |  APPROVE={app}")
'''

CELL_SAVE = r'''# ── Save predictions.csv ──
preds_df.to_csv(PREDICTIONS_CSV, index=False)
print(f"\nSaved '{PREDICTIONS_CSV}' with {len(preds_df)} rows.\n")

# ── Sample APPROVE ──
print("=" * 60)
print("  SAMPLE APPROVED COMPLAINTS")
print("=" * 60)
approved_samples = preds_df[preds_df["prediction"] == "APPROVE"].head(6)
for _, r in approved_samples.iterrows():
    print(f"\n  ID={r['complaint_id']:3d} | Severity={r['severity_label']:6s} | Issue={r['issue_type']}")
    print(f"  Text: {r['complaint_text'][:95]}")

# ── Sample REJECT ──
print("\n" + "=" * 60)
print("  SAMPLE REJECTED COMPLAINTS")
print("=" * 60)
rejected_samples = preds_df[preds_df["prediction"] == "REJECT"].head(6)
for _, r in rejected_samples.iterrows():
    print(f"\n  ID={r['complaint_id']:3d} | Severity={r['severity_label']:6s} | Issue={r['issue_type']}")
    print(f"  Text: {r['complaint_text'][:95]}")
'''

CELL_TABLE = r'''# ── Styled results table ──
preds_df.style.map(
    lambda v: "background-color:#d4edda" if v == "APPROVE" else
              "background-color:#f8d7da" if v == "REJECT"  else "",
    subset=["prediction"]
).map(
    lambda v: "background-color:#fff3cd" if v == "High" else
              "background-color:#d1ecf1" if v == "Medium" else "",
    subset=["severity_label"]
)
'''


# ═══════════════════ BUILD NOTEBOOK ════════════════════════════════════════

def build_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }

    cells = []

    cells.append(make_md("""# AI Refund Approval — Block [3] & [4]: Feature Extraction + Classification

This notebook applies **Block [3] BERT Feature Extraction** and **Block [4] Classification**
on all complaints that passed the primitive validation pipeline.

**Input**:  `validation_results.csv`  (VALID decisions only)  +  `model.pkl`
**Output**: `predictions.csv`  (APPROVE / REJECT per complaint)

```
VALID Complaints
      ↓
[3] Feature Extraction
    ├── Sentence-BERT embedding (384-dim, all-MiniLM-L6-v2)
    ├── Issue Type  (Sentence-BERT cosine similarity → 5 categories)
    ├── Severity    (Sentence-BERT cosine similarity → High/Med/Low)
    ├── Sentiment   (TextBlob polarity)
    └── Urgency     (keyword detection)
      ↓
[4] XGBoost Classifier
    └── APPROVE (high severity) / REJECT (medium or low severity)
```
"""))

    cells.append(make_md("## 1. Imports & Configuration"))
    cells.append(make_code(CELL_IMPORTS))

    cells.append(make_md("""## 2. Reference Phrases (Sentence-BERT)

All semantic similarity checks use the **same Sentence-BERT model** (`all-MiniLM-L6-v2`).
- **Issue type**: cosine similarity vs 5 category phrases
- **Severity**: cosine similarity vs High / Medium / Low tier phrases
"""))
    cells.append(make_code(CELL_REFERENCES))

    cells.append(make_md("""## 3. `preprocess()` — Text Normalisation

Minimal cleaning to match the pre-processing used in Block [1] / [2].
"""))
    cells.append(make_code(CELL_PREPROCESS))

    cells.append(make_md("""## 4. `extract_features()` — Feature Extraction (Block [3])

Produces a **388-dimensional feature vector** per complaint:
| Dims | Feature | Method |
|------|---------|--------|
| 0–383 | BERT embedding | Sentence-BERT `all-MiniLM-L6-v2` |
| 384 | Severity (1/2/3) | Sentence-BERT cosine similarity |
| 385 | Sentiment | TextBlob polarity |
| 386 | Urgency | Keyword match |
| 387 | Issue type (0-4) | Sentence-BERT cosine similarity |
"""))
    cells.append(make_code(CELL_EXTRACT_FEATURES))

    cells.append(make_md("""## 5. `classify()` — Inference Function (Block [4])

Wraps the trained XGBoost model to output `APPROVE` or `REJECT` for any text.
"""))
    cells.append(make_code(CELL_CLASSIFY_FN))

    cells.append(make_md("## 6. Load Validated Complaints"))
    cells.append(make_code(CELL_LOAD))

    cells.append(make_md("## 7. Load Trained Model (`model.pkl`)"))
    cells.append(make_code(CELL_LOAD_MODEL))

    cells.append(make_md("## 8. Run Inference on All VALID Complaints"))
    cells.append(make_code(CELL_INFERENCE))

    cells.append(make_md("## 9. Results Summary"))
    cells.append(make_code(CELL_SUMMARY))

    cells.append(make_md("## 10. Save `predictions.csv` + Sample Output"))
    cells.append(make_code(CELL_SAVE))

    cells.append(make_md("## 11. Full Predictions Table (Styled)"))
    cells.append(make_code(CELL_TABLE))

    nb.cells = cells
    return nb


if __name__ == "__main__":
    nb = build_notebook()
    fname = "classification_pipeline.ipynb"
    with open(fname, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook saved to '{fname}'")
