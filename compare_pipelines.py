import json
import pandas as pd

def main():
    try:
        with open("all_metrics.json", "r") as f:
            metrics = json.load(f)
    except FileNotFoundError:
        print("Error: all_metrics.json not found. Please run train_classifier.py first.")
        return

    df = pd.DataFrame(metrics)
    
    # Sort by F1-score
    df = df.sort_values(by="f1", ascending=False)
    
    print("\n" + "="*70)
    print("      NLP PIPELINE COMPARISON: REFUND CLASSIFICATION")
    print("="*70)
    print(df.to_markdown(index=False))
    print("="*70)
    
    # Analysis
    best_emb = df.iloc[0]['embedding']
    print(f"\nANALYSIS:")
    print(f"1. Best Performing Model: {best_emb.upper()}")
    print(f"2. Observation: Sentence-level embeddings (SBERT, BERT, InferSent) typically")
    print(f"   outperform word-averaging methods (Word2Vec, GloVe) because they")
    print(f"   capture contextual dependencies and sentence structure.")
    print(f"3. Recommendation: Use {best_emb} for the production system.")
    
    # Save as CSV for record
    df.to_csv("comparison_results.csv", index=False)
    print(f"\nComparison saved to 'comparison_results.csv'")

if __name__ == "__main__":
    main()
