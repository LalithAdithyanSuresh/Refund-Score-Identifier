"""
feature_extractor.py
─────────────────────
Modular feature extraction using Sentence-BERT (all-MiniLM-L6-v2).

All semantic similarity tasks (BERT embedding, issue-type classification,
severity estimation) use the SAME Sentence-BERT model for consistency.

Feature vector layout (388 dims total):
  [0:384]  → Sentence-BERT embedding (384-dim)
  [384]    → severity score  (1=Low, 2=Medium, 3=High)
  [385]    → sentiment score (-1.0 to +1.0)
  [386]    → urgency score   (0 or 1)
  [387]    → issue_type      (0=screen, 1=battery, 2=performance,
                               3=connectivity, 4=hardware)
"""

import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob

# ── Initialise Sentence-BERT (loaded once, shared) ──────────────────────────
_MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


# ── Embedding cache (text → ndarray) ────────────────────────────────────────
_emb_cache: dict[str, np.ndarray] = {}


def _embed(text: str) -> np.ndarray:
    """Return (cached) Sentence-BERT embedding for a string."""
    if text not in _emb_cache:
        _emb_cache[text] = _get_model().encode(text, show_progress_bar=False)
    return _emb_cache[text]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0])


def _max_sim(text_emb: np.ndarray, refs: list[str]) -> tuple[float, int]:
    """Return (max_cosine_score, best_ref_index) against a list of references."""
    ref_embs = np.array([_embed(r) for r in refs])
    sims = cosine_similarity(text_emb.reshape(1, -1), ref_embs)[0]
    idx = int(np.argmax(sims))
    return float(sims[idx]), idx


# ── Reference phrases (Sentence-BERT-encoded for similarity) ─────────────────

ISSUE_TYPE_REFS = [
    "screen damage cracked display broken screen",   # 0
    "battery draining not charging battery problem",  # 1
    "performance slow lagging freezing",              # 2
    "connectivity disconnecting wifi bluetooth",      # 3
    "hardware failure dead not working broken",       # 4
]

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


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

def preprocess(text: str) -> str:
    """
    Minimal text normalisation:
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


def extract_features(text: str) -> dict:
    """
    Extract all five features from a (raw or cleaned) complaint string.

    Returns a dict with keys:
        embedding     : list[float]  (384 values)
        severity      : int          (1 / 2 / 3)
        sentiment     : float        (-1.0 to +1.0)
        urgency       : int          (0 or 1)
        issue_type    : int          (0-4)
        feature_vector: list[float]  (388-dim flat vector)
    """
    cleaned = preprocess(text)
    emb = _embed(cleaned)                          # 384-dim Sentence-BERT vec

    # ── Issue Type (via Sentence-BERT cosine similarity) ──────────────────
    _, issue_idx = _max_sim(emb, ISSUE_TYPE_REFS)

    # ── Severity (via Sentence-BERT cosine similarity) ────────────────────
    high_score,   _ = _max_sim(emb, SEVERITY_HIGH_REFS)
    medium_score, _ = _max_sim(emb, SEVERITY_MEDIUM_REFS)
    low_score,    _ = _max_sim(emb, SEVERITY_LOW_REFS)

    scores = {3: high_score, 2: medium_score, 1: low_score}
    severity = max(scores, key=scores.get)         # 3=High, 2=Med, 1=Low

    # ── Sentiment (TextBlob polarity) ─────────────────────────────────────
    sentiment = round(TextBlob(cleaned).sentiment.polarity, 4)

    # ── Urgency (keyword match) ───────────────────────────────────────────
    words = set(cleaned.split())
    urgency = int(bool(words & URGENCY_KEYWORDS))

    # ── Combine into flat feature vector ─────────────────────────────────
    feature_vector = (
        list(emb.astype(float))           # 384 dims
        + [float(severity)]               # 1 dim
        + [float(sentiment)]              # 1 dim
        + [float(urgency)]                # 1 dim
        + [float(issue_idx)]              # 1 dim
    )                                     # total: 388 dims

    return {
        "embedding":      list(emb.astype(float)),
        "severity":       severity,
        "sentiment":      sentiment,
        "urgency":        urgency,
        "issue_type":     issue_idx,
        "feature_vector": feature_vector,
    }


def classify(text: str, model) -> str:
    """
    Run inference for a single complaint text.
    model: a trained sklearn/XGBoost classifier with .predict()

    Returns: "APPROVE" or "REJECT"
    """
    features = extract_features(text)
    vec = np.array(features["feature_vector"]).reshape(1, -1)
    pred = int(model.predict(vec)[0])
    return "APPROVE" if pred == 1 else "REJECT"


# ── Cache warming helper ──────────────────────────────────────────────────────

def warm_cache():
    """Pre-encode all reference phrases so first inference is fast."""
    all_refs = (ISSUE_TYPE_REFS + SEVERITY_HIGH_REFS
                + SEVERITY_MEDIUM_REFS + SEVERITY_LOW_REFS)
    for ref in all_refs:
        _embed(ref)
    print(f"Sentence-BERT cache warmed ({len(_emb_cache)} entries).")


if __name__ == "__main__":
    # Quick smoke-test
    warm_cache()
    sample = "The iPhone 14 screen is completely cracked and won't turn on."
    result = extract_features(sample)
    print(f"\nSample: {sample!r}")
    print(f"  Issue type : {result['issue_type']}  "
          f"({ISSUE_TYPE_REFS[result['issue_type']].split()[0]})")
    print(f"  Severity   : {result['severity']}  "
          f"({'High' if result['severity']==3 else 'Medium' if result['severity']==2 else 'Low'})")
    print(f"  Sentiment  : {result['sentiment']}")
    print(f"  Urgency    : {result['urgency']}")
    print(f"  Vector dim : {len(result['feature_vector'])}")
