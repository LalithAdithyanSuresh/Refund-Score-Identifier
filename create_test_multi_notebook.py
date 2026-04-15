"""
create_test_multi_notebook.py
──────────────────────────────
Builds `test_multi.ipynb` with improved reporting:
1. Detailed pass/fail logs for every stage.
2. Text-based SHAP reasoning (replaces visual plots).
3. Systematic comparison of 6 embeddings on "tricky" cases.
"""

import nbformat as nbf

def make_md(source): return nbf.v4.new_markdown_cell(source)
def make_code(source): return nbf.v4.new_code_cell(source)

CELL_IMPORTS = r'''import sqlite3
import re
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import shap
import warnings
from feature_extractor import extract_features, set_embedding_method, preprocess, ISSUE_TYPE_REFS
warnings.filterwarnings('ignore')

print("All libraries loaded successfully.")
'''

CELL_SETUP = r'''# ── Configuration & Global State ──

DB_PATH = 'refund_system_final.db'
AVAILABLE_METHODS = ["word2vec", "glove", "fasttext", "bert", "infersent", "sbert"]

# State variables
current_method = "sbert"
current_clf = None

def load_pipeline(method):
    global current_method, current_clf
    if method not in AVAILABLE_METHODS:
        print(f"Error: {method} is not a valid method.")
        return
    
    set_embedding_method(method)
    
    model_path = f"model_{method}.pkl"
    try:
        current_clf = joblib.load(model_path)
        current_method = method
    except FileNotFoundError:
        print(f"Error: Model file {model_path} not found.")

def get_shap_reasoning(X, clf, method):
    """Generates textual reasoning based on SHAP values."""
    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(X)
    
    # SHAP returns a list for multiclass, or array for binary. 
    # For XGBoost binary, it's usually the shift in log-odds.
    if isinstance(shap_vals, list):
        vals = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
    else:
        vals = shap_vals[0]

    dim = X.shape[1]
    emb_dim = dim - 4
    feature_names = [f"Semantic Pattern (Dim {i})" for i in range(emb_dim)] + ["Severity", "Sentiment", "Urgency", "Issue_Type"]
    
    # Get top 3 features by absolute impact
    top_indices = np.argsort(np.abs(vals))[-3:][::-1]
    
    reasons = []
    for idx in top_indices:
        val = vals[idx]
        name = feature_names[idx]
        direction = "Positive (towards Approval)" if val > 0 else "Negative (towards Rejection)"
        reasons.append(f"- {name}: {direction} impact.")
    
    return "\n".join(reasons)

# Default load
load_pipeline("sbert")
'''

CELL_PRIMITIVE_FUNCTIONS = r'''# ── Database & Primitive Checks ──

def validate_db(customer_id, order_id, date_str):
    print(f" [Step 1/3] Database Validation...", end="")
    conn = sqlite3.connect(DB_PATH)
    try:
        cust = pd.read_sql(f"SELECT * FROM customers WHERE customer_id={customer_id}", conn)
        order = pd.read_sql(f"SELECT * FROM orders WHERE order_id={order_id}", conn)
        if cust.empty or order.empty:
            print(" [FAIL]")
            return False, "Customer or Order not found."
            
        pur_date = datetime.strptime(order.iloc[0]['purchase_date'], '%Y-%m-%d')
        cmp_date = datetime.strptime(date_str, '%Y-%m-%d')
        days = (cmp_date - pur_date).days
        
        pid = int(order.iloc[0]['product_id'])
        prod = pd.read_sql(f"SELECT * FROM products WHERE product_id={pid}", conn)
        
        if days > prod.iloc[0]['refund_window_days']:
            print(" [FAIL]")
            return False, f"Outside refund window ({days} days)."
        
        print(" [PASS]")
        return True, prod.iloc[0]['product_name']
    finally:
        conn.close()

def check_text_primitives(text):
    print(f" [Step 2/3] Text Primitive Checks...", end="")
    from feature_extractor import _max_sim, _embed, SEVERITY_HIGH_REFS # Shared refs if needed
    
    # Simplified COM/Vague check for demonstration
    # In practice, these use similarity thresholds
    cleaned = preprocess(text)
    if any(kw in cleaned for kw in ["changed my mind", "prefer another"]):
        print(" [FAIL]")
        return False, "Change of mind detected."
    if len(cleaned.split()) < 3:
        print(" [FAIL]")
        return False, "Vague complaint."
    
    print(" [PASS]")
    return True, "Valid Context"
'''

CELL_PIPELINE_RUNNER = r'''# ── Enhanced Pipeline Runner ──

def run_test(complaint_text, customer_id, order_id, date, method=None, silent=False):
    if method: load_pipeline(method)
    
    if not silent:
        print("=" * 60)
        print(f" 🔍 TESTING PIPELINE: {current_method.upper()}")
        print(f" INPUT: {complaint_text}")
        print("-" * 60)
    
    # 1. DB
    db_ok, db_res = validate_db(customer_id, order_id, date)
    if not db_ok:
        print(f" ❌ FINAL STATUS: REJECTED")
        print(f" REASON: {db_res}")
        return "REJECT"

    # 2. Primitive
    txt_ok, txt_res = check_text_primitives(complaint_text)
    if not txt_ok:
        print(f" ❌ FINAL STATUS: REJECTED")
        print(f" REASON: {txt_res}")
        return "REJECT"

    # 3. AI
    print(f" [Step 3/3] AI Classification...", end="")
    feats = extract_features(complaint_text)
    X = np.array(feats["feature_vector"]).reshape(1, -1)
    pred = current_clf.predict(X)[0]
    decision = "APPROVE" if pred == 1 else "REJECT"
    print(f" [{decision}]")
    
    if not silent:
        print("\n 📝 SHAP REASONING (Text-based):")
        print(get_shap_reasoning(X, current_clf, current_method))
        print("-" * 60)
        print(f" 🏁 FINAL DECISION: {decision}")
        print("=" * 60)
        
    return decision
'''

CELL_COMPARE_TRICKY = r'''# ── Cross-Embedding Comparison for Tricky Cases ──

def compare_tricky_cases():
    cases = [
        ("The battery is working.", "The battery is not working."),
        ("I want to return the phone.", "The phone has good return on battery."),
        ("The screen is fine, but the battery is broken.", "The battery is fine, but the screen is broken.")
    ]
    
    results = []
    for pair in cases:
        for text in pair:
            row = {"Complaint": text}
            for method in AVAILABLE_METHODS:
                # Use Silent mode for compact table
                decision = run_test(text, 19, 19, "2026-04-16", method=method, silent=True)
                row[method] = decision
            results.append(row)
    
    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("      CROSS-EMBEDDING TRICKY CASE COMPARISON")
    print("="*80)
    display(df)
    print("="*80)

# Run the comparison
compare_tricky_cases()
'''

def build_nb():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        make_md("# Improved Multi-Embedding Refund System Test\nDetailed logs, Text-based SHAP, and Cross-Embedding Tricky Case evaluation."),
        make_code(CELL_IMPORTS),
        make_code(CELL_SETUP),
        make_code(CELL_PRIMITIVE_FUNCTIONS),
        make_code(CELL_PIPELINE_RUNNER),
        make_md("## 🧪 Cross-Embedding Comparison\nRunning the three tricky cases across all models to see which one handles context correctly."),
        make_code(CELL_COMPARE_TRICKY)
    ]
    with open("test_multi.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Updated test_multi.ipynb created.")

if __name__ == "__main__":
    build_nb()
