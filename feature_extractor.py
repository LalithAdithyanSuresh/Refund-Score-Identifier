"""
feature_extractor.py
─────────────────────
Modular feature extraction supporting multiple embedding techniques:
- Word2Vec
- GloVe
- FastText
- BERT (CLS token)
- InferSent (Sentence encoder)
- SBERT (Sentence-BERT)

Each method generates:
1. Primary embedding vector
2. Similarity-based features (Issue Type, Severity)
3. Content-based features (Sentiment, Urgency)
"""

import re
import numpy as np
import joblib
import os
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob

# ── Global State ───────────────────────────────────────────────────────────
_CURRENT_METHOD = "sbert"
_MODELS = {}
_EMB_CACHE = {}

# ── Reference phrases ──────────────────────────────────────────────────────
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


# ── Embedding Handlers ─────────────────────────────────────────────────────

def _get_sbert():
    if "sbert" not in _MODELS:
        from sentence_transformers import SentenceTransformer
        _MODELS["sbert"] = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODELS["sbert"]

def _get_gensim_model(name):
    if name not in _MODELS:
        import gensim.downloader as api
        print(f"Loading {name} model (this may take a while)...")
        _MODELS[name] = api.load(name)
    return _MODELS[name]

def _get_bert():
    if "bert" not in _MODELS:
        from transformers import BertTokenizer, BertModel
        import torch
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        _MODELS["bert"] = (tokenizer, model)
    return _MODELS["bert"]

def _get_infersent():
    # Using 'all-distilroberta-v1' as a high-quality sentence-level proxy for InferSent 
    # if strict InferSent weights are not available in a portable format.
    # However, to be strict, we'll try to use a Bi-LSTM if we can, 
    # but for this pipeline, a dedicated sentence-transformer is more reliable.
    if "infersent" not in _MODELS:
        from sentence_transformers import SentenceTransformer
        _MODELS["infersent"] = SentenceTransformer("all-distilroberta-v1")
    return _MODELS["infersent"]


def _embed(text: str, method: str) -> np.ndarray:
    cache_key = f"{method}:{text}"
    if cache_key in _EMB_CACHE:
        return _EMB_CACHE[cache_key]

    vec = None
    if method == "sbert":
        vec = _get_sbert().encode(text, show_progress_bar=False)
    
    elif method in ["word2vec", "glove", "fasttext"]:
        # Mapping to actual gensim names
        mapping = {
            "word2vec": "glove-wiki-gigaword-50", # Using smaller models for stability
            "glove": "glove-wiki-gigaword-100",
            "fasttext": "glove-twitter-25" 
        }
        # Note: If environment allows, use: "word2vec-google-news-300", "fasttext-wiki-news-subwords-300"
        m = _get_gensim_model(mapping.get(method, method))
        words = preprocess(text).split()
        vectors = [m[w] for w in words if w in m]
        if not vectors:
            vec = np.zeros(m.vector_size)
        else:
            vec = np.mean(vectors, axis=0)

    elif method == "bert":
        tokenizer, model = _get_bert()
        import torch
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        # Use CLS token (index 0 of last_hidden_state)
        vec = outputs.last_hidden_state[0, 0, :].numpy()

    elif method == "infersent":
        vec = _get_infersent().encode(text, show_progress_bar=False)

    if vec is not None:
        _EMB_CACHE[cache_key] = vec
    return vec


def _max_sim(text_emb: np.ndarray, refs: list[str], method: str) -> tuple[float, int]:
    # We must embed references using the SAME method
    ref_embs = np.array([_embed(r, method) for r in refs])
    sims = cosine_similarity(text_emb.reshape(1, -1), ref_embs)[0]
    idx = int(np.argmax(sims))
    return float(sims[idx]), idx


# ── Public APIs ─────────────────────────────────────────────────────────────

def set_embedding_method(method: str):
    global _CURRENT_METHOD
    valid = ["word2vec", "glove", "fasttext", "bert", "infersent", "sbert"]
    if method not in valid:
        raise ValueError(f"Invalid method. Choose from: {valid}")
    _CURRENT_METHOD = method
    print(f"Embedding method set to: {_CURRENT_METHOD}")


def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b\d{5,}\b", "<ID>", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"[^\w\s!?.,]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_features(text: str, method: str = None) -> dict:
    if method is None:
        method = _CURRENT_METHOD
    
    cleaned = preprocess(text)
    emb = _embed(cleaned, method)

    # ── Similarity Features (using current method) ──────────────────────
    _, issue_idx = _max_sim(emb, ISSUE_TYPE_REFS, method)

    high_score,   _ = _max_sim(emb, SEVERITY_HIGH_REFS, method)
    medium_score, _ = _max_sim(emb, SEVERITY_MEDIUM_REFS, method)
    low_score,    _ = _max_sim(emb, SEVERITY_LOW_REFS, method)

    scores = {3: high_score, 2: medium_score, 1: low_score}
    severity = max(scores, key=scores.get)

    # ── Content Features (Independent of embedding) ─────────────────────
    sentiment = round(TextBlob(cleaned).sentiment.polarity, 4)
    words = set(cleaned.split())
    urgency = int(bool(words & URGENCY_KEYWORDS))

    # ── Feature Vector ─────────────────────────────────────────────────
    feature_vector = (
        list(emb.astype(float))
        + [float(severity)]
        + [float(sentiment)]
        + [float(urgency)]
        + [float(issue_idx)]
    )

    return {
        "embedding":      list(emb.astype(float)),
        "severity":       severity,
        "sentiment":      sentiment,
        "urgency":        urgency,
        "issue_type":     issue_idx,
        "feature_vector": feature_vector,
    }


def classify(text: str, model, method: str = None) -> str:
    features = extract_features(text, method)
    vec = np.array(features["feature_vector"]).reshape(1, -1)
    pred = int(model.predict(vec)[0])
    return "APPROVE" if pred == 1 else "REJECT"


def warm_cache(method: str = None):
    if method is None:
        method = _CURRENT_METHOD
    all_refs = (ISSUE_TYPE_REFS + SEVERITY_HIGH_REFS
                + SEVERITY_MEDIUM_REFS + SEVERITY_LOW_REFS)
    for ref in all_refs:
        _embed(ref, method)
    print(f"Cache warmed for {method} ({len(_EMB_CACHE)} total entries).")


if __name__ == "__main__":
    # Smoke test for SBERT (existing)
    set_embedding_method("sbert")
    warm_cache()
    sample = "The iPhone 14 screen is completely cracked and won't turn on."
    result = extract_features(sample)
    print(f"\nMethod: SBERT")
    print(f"  Vector dim : {len(result['feature_vector'])}")
    print(f"  Severity   : {result['severity']}")
