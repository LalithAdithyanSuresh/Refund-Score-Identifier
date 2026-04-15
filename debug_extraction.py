import feature_extractor as fe
import numpy as np

methods = ["word2vec", "glove", "fasttext", "bert", "infersent", "sbert"]
sample = "My phone screen is broken and I need a refund."

for m in methods:
    print(f"Testing {m}...")
    try:
        fe.set_embedding_method(m)
        features = fe.extract_features(sample)
        print(f"  Success: Feature vector length {len(features['feature_vector'])}")
    except Exception as e:
        print(f"  FAILED {m}: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
