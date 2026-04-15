"""
create_test_notebook.py
───────────────────────
Builds `test.ipynb` which encapsulates the complete End-to-End Pipeline:
1. Primitive Checks (DB, Time, Brand/Color, Change-of-mind, Vague)
2. Feature Extraction (Sentence-BERT + severity/sentiment/urgency)
3. XGBoost Classification
4. SHAP Explanation
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
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import shap
import warnings
warnings.filterwarnings('ignore')

print("All libraries loaded successfully (including SHAP).")
'''

CELL_SETUP = r'''# ── Configuration & Models ──

DB_PATH   = 'refund_system_final.db'
MODEL_PKL = 'model.pkl'

print("Loading Sentence-BERT model (all-MiniLM-L6-v2)...")
bert_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
print("Model loaded.")

print("Loading XGBoost Classifier...")
classifier = joblib.load(MODEL_PKL)
print("Classifier loaded.")

# Initialize SHAP explainer
explainer = shap.TreeExplainer(classifier)

# Global caches
_emb_cache = {}

def get_embedding(text):
    if text not in _emb_cache:
        _emb_cache[text] = bert_model.encode(text, show_progress_bar=False)
    return _emb_cache[text]

def cosine_sim(text_a, text_b):
    emba = get_embedding(text_a).reshape(1, -1)
    embb = get_embedding(text_b).reshape(1, -1)
    return float(cosine_similarity(emba, embb)[0][0])
'''

CELL_PRIMITIVE_CONFIG = r'''# ── Primitive Check Thresholds & References ──

# Change of mind references
CHANGE_OF_MIND_THRESHOLD = 0.50
CHANGE_OF_MIND_REFS = [
    "i changed my mind about this product", "i prefer another product or brand",
    "i dont like this product anymore", "want a different color or model",
    "i want to return this i dont need it", "decided to go with a different brand",
    "i prefer a different model", "dont need this anymore changed my mind",
    "i want to exchange for something else", "not the product i wanted prefer another",
    "dont like the color want to exchange", "prefer something else want to return"
]
CHANGE_OF_MIND_KEYWORDS = [
    'changed my mind', 'change my mind', 'prefer another', 'prefer a different', 
    'dont need', 'dont like', 'do not like', 'do not need', 'dont want',
    'want to exchange', 'want to return', 'prefer something else', 
    'decided to go with', 'dont like the color', 'prefer a different model',
    'just dont like it', 'fine but i just dont like'
]

# Vague complaint references
VAGUE_THRESHOLD = 0.45
VAGUE_REFS = [
    "bad product need refund", "not good want refund", "refund please",
    "i want my money back", "worst product", "not satisfied refund needed",
    "terrible just give refund", "very unhappy want refund",
    "just want a refund for this", "need my money back", "waste of money",
    "unhappy with this purchase refund needed", "need a refund for this product",
    "very unhappy with purchase refund", "not happy refund",
]
VAGUE_KEYWORDS = [
    'very unhappy with this purchase', 'unhappy with this purchase',
    'worst purchase ever', 'waste of money',
]

# Shared defect keywords (bypasses change-of-mind and vague checks)
DEFECT_KEYWORDS = [
    'broken', 'cracked', 'dead', 'not working', 'defect', 'overheat',
    'disconnect', 'restart', 'lag', 'freeze', 'shut down', 'shutting',
    'battery drain', 'blurry', 'distort', 'stuck', 'unresponsive',
    'buzzing', 'noise', 'pixel', 'charging', 'wrking', 'brokn',
    'stoppd', 'nt wrking', 'ded', 'dosnt', 'stopped working',
    'not turn', 'wont turn', 'hardware failure', 'stopped functioning',
    'broke', 'malfunction', 'does not power', 'does not respond',
    'does not work', 'wont work', 'cannot use', 'screen cracked',
    'dead pixel', 'overheating', 'disconnecting', 'restarting',
    'lagging', 'freezing'
]

# Pre-computation to save time on references
def max_similarity(cleaned_text, reference_list):
    emb_text = get_embedding(cleaned_text)
    ref_embs = np.array([get_embedding(ref) for ref in reference_list])
    sims = cosine_similarity(emb_text.reshape(1, -1), ref_embs)[0]
    idx = int(np.argmax(sims))
    return float(sims[idx]), idx
'''

CELL_PRIMITIVE_FUNCTIONS = r'''# ── Primitive Check Functions ──

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\b\d{5,}\b', '<ID>', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r"[^\w\s!?.,]", " ", text)
    return re.sub(r'\s+', ' ', text).strip()

def has_defect_keyword(text_lower):
    # Fixed to use \b word boundaries to prevent substring matching bugs
    return any(re.search(r'\b' + re.escape(kw) + r'\b', text_lower) for kw in DEFECT_KEYWORDS)

def check_change_of_mind(cleaned_text):
    text_lower = cleaned_text.lower()
    score, idx = max_similarity(cleaned_text, CHANGE_OF_MIND_REFS)
    match = CHANGE_OF_MIND_REFS[idx]
    if score >= CHANGE_OF_MIND_THRESHOLD:
        return True, f"Change of mind detected (similarity vs '{match}')"
    for kw in CHANGE_OF_MIND_KEYWORDS:
        if kw in text_lower and not has_defect_keyword(text_lower):
            return True, f"Change of mind detected (keyword: '{kw}')"
    return False, ""

def check_vague_complaint(cleaned_text):
    text_lower = cleaned_text.lower()
    has_defect = has_defect_keyword(text_lower)
    
    score, idx = max_similarity(cleaned_text, VAGUE_REFS)
    match = VAGUE_REFS[idx]
    if score >= VAGUE_THRESHOLD and not has_defect:
        return True, f"Vague complaint detected (similarity vs '{match}')"
    if not has_defect:
        for kw in VAGUE_KEYWORDS:
            if kw in text_lower:
                return True, f"Vague complaint detected (keyword: '{kw}')"
    return False, ""

def validate_db(customer_id, order_id, complaint_date_str):
    conn = sqlite3.connect(DB_PATH)
    cust = pd.read_sql(f"SELECT * FROM customers WHERE customer_id={customer_id}", conn)
    order = pd.read_sql(f"SELECT * FROM orders WHERE order_id={order_id}", conn)
    
    if cust.empty:
        conn.close()
        return False, None, "Customer ID not found."
    if order.empty:
        conn.close()
        return False, None, "Order ID not found."
    if int(order.iloc[0]['customer_id']) != int(customer_id):
        conn.close()
        return False, None, "Order does not belong to this customer."
        
    pur_date = datetime.strptime(order.iloc[0]['purchase_date'], '%Y-%m-%d')
    cmp_date = datetime.strptime(complaint_date_str, '%Y-%m-%d')
    days_diff = (cmp_date - pur_date).days
    
    pid = int(order.iloc[0]['product_id'])
    prod = pd.read_sql(f"SELECT * FROM products WHERE product_id={pid}", conn)
    conn.close()
    
    if days_diff > prod.iloc[0]['refund_window_days']:
        return False, None, f"Outside refund window: {days_diff} days (allowed: {prod.iloc[0]['refund_window_days']})"
        
    return True, prod.iloc[0], "Valid DB"
'''

CELL_FEATURE_CONFIG = r'''# ── Feature Extraction Configurations ──

ISSUE_TYPE_REFS = [
    "screen damage cracked display broken screen",   # 0
    "battery draining not charging battery problem",  # 1
    "performance slow lagging freezing",              # 2
    "connectivity disconnecting wifi bluetooth",      # 3
    "hardware failure dead not working broken",       # 4
]
ISSUE_NAMES = ["Screen", "Battery", "Performance", "Connectivity", "Hardware"]

SEVERITY_HIGH_REFS   = ["completely broken and unusable", "dead on arrival", "cracked screen device destroyed", "hardware failure"]
SEVERITY_MEDIUM_REFS = ["disconnecting frequently", "overheating during use", "restarting several times", "battery drains fast"]
SEVERITY_LOW_REFS    = ["slightly slow minor delay", "small scratch", "minor issue barely noticeable", "slight audio delay"]

URGENCY_KEYWORDS = {"asap", "urgent", "urgently", "immediately", "right away"}
'''

CELL_FEATURE_EXTRACTION = r'''# ── Block 3: Feature Extraction ──

def extract_features(cleaned_text):
    emb = get_embedding(cleaned_text)

    # Issue
    _, issue_idx = max_similarity(cleaned_text, ISSUE_TYPE_REFS)
    
    # Severity
    h_score, _ = max_similarity(cleaned_text, SEVERITY_HIGH_REFS)
    m_score, _ = max_similarity(cleaned_text, SEVERITY_MEDIUM_REFS)
    l_score, _ = max_similarity(cleaned_text, SEVERITY_LOW_REFS)
    scores = {3: h_score, 2: m_score, 1: l_score}
    severity = max(scores, key=scores.get)
    
    # Sentiment & Urgency
    sentiment = round(TextBlob(cleaned_text).sentiment.polarity, 4)
    words = set(cleaned_text.split())
    urgency = int(bool(words & URGENCY_KEYWORDS))
    
    feature_vector = (
        list(emb.astype(float)) + 
        [float(severity), float(sentiment), float(urgency), float(issue_idx)]
    )
    
    return np.array(feature_vector).reshape(1, -1), {
        "issue": ISSUE_NAMES[issue_idx],
        "severity": severity,
        "sentiment": sentiment,
        "urgency": urgency
    }
'''

CELL_MAIN_WRAPPER = r'''# ── 🚨 COMPLETE PIPELINE RUNNER 🚨 ──

def run_test_complaint(complaint_text, customer_id, order_id, complaint_date):
    print("="*60)
    print(" 🛠️ REFUND VALIDATION PIPELINE")
    print("="*60)
    print(f"INPUT TEXT: {complaint_text}")
    print(f"ORDER ID  : {order_id} | CUST ID: {customer_id}")
    print("-"*60)
    
    # --- 1. DB Checks ---
    db_ok, product, reason = validate_db(customer_id, order_id, complaint_date)
    if not db_ok:
        print(f"❌ REJECTED [Block 1: Database Check]")
        print(f"   Reason: {reason}")
        return
    print(f"✅ DB Match: Ordered a {product['brand']} {product['product_name']} ({product['color']})")
    
    cleaned = preprocess_text(complaint_text)
    
    # --- 2. NLP Context Checks ---
    # Quick simple brand mismatch check using the product context
    brand_found = any(b in cleaned.lower() for b in ['apple', 'samsung', 'logitech', 'sony', 'asus', 'dell', 'razer', 'jbl', 'anker', 'corsair', 'bose', 'oneplus'])
    if brand_found and product['brand'].lower() not in cleaned.lower():
        print(f"❌ REJECTED [Block 2: Context]")
        print(f"   Reason: Brand mismatch in text for {product['brand']}.")
        return
        
    # --- 3. Primitive Checks ---
    is_com, com_reason = check_change_of_mind(cleaned)
    if is_com:
        print(f"❌ REJECTED [Block 2: Text Primitive]")
        print(f"   Reason: {com_reason}")
        return
        
    is_vague, vague_reason = check_vague_complaint(cleaned)
    if is_vague:
        print(f"❌ REJECTED [Block 2: Text Primitive]")
        print(f"   Reason: {vague_reason}")
        return

    print("✅ Primitive NLP Checks Passed -> Moving to AI Classification!")
    print("-"*60)
    
    # --- 4. Feature Extraction ---
    feature_vec, meta = extract_features(cleaned)
    print("📊 Block 3: Feature Extraction Matrix")
    print(f"   Issue Type : {meta['issue']} (Sev Level: {meta['severity']})")
    print(f"   Sentiment  : {meta['sentiment']}")
    print(f"   Urgency    : {'Yes' if meta['urgency']==1 else 'No'}")
    
    # --- 5. Classification ---
    pred = classifier.predict(feature_vec)[0]
    decision_text = "🟢 APPROVE" if pred == 1 else "🔴 REJECT"
    print("\n🔮 Block 4: Final XGBoost Inference")
    print(f"   DECISION   : {decision_text}")
    print("="*60)
    
    # --- 6. SHAP Explanation ---
    print("\n🤖 Block 6: SHAP Explanation (Why?)")
    # Generate SHAP values for the single instance
    shap_vals = explainer(feature_vec)
    
    # Create feature names dynamically to match the 388 dim array
    feature_names = [f"BERT_{i}" for i in range(384)] + ["Severity (1-3)", "Sentiment", "Urgency", "Issue_Type_ID"]
    shap_vals.feature_names = feature_names
    
    display(shap.plots.waterfall(shap_vals[0], show=True))
'''

CELL_TEST_CASES = r'''# ── TEST YOUR COMPLAINTS BELOW ──
# Modify this function call with the parameters provided to test your pipeline!

test_complaint_text = "Got the Apple iPhone 14 and the screen was cracked when I opened the package."
test_customer_id = 19
test_order_id = 19
test_complaint_date = "2026-04-16" # Within the 15-day refund window for order 19 (purchased 2026-04-05)

run_test_complaint(
    complaint_text=test_complaint_text, 
    customer_id=test_customer_id, 
    order_id=test_order_id, 
    complaint_date=test_complaint_date
)
'''


def build_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        make_md("# NLP Refund Approval Pipeline — Interactive Testing\nTest the full workflow here from DB verification to XGBoost + SHAP Explainability."),
        make_code(CELL_IMPORTS),
        make_code(CELL_SETUP),
        make_code(CELL_PRIMITIVE_CONFIG),
        make_code(CELL_PRIMITIVE_FUNCTIONS),
        make_code(CELL_FEATURE_CONFIG),
        make_code(CELL_FEATURE_EXTRACTION),
        make_code(CELL_MAIN_WRAPPER),
        make_md("--- \n### 🧪 TRY IT YOURSELF"),
        make_code(CELL_TEST_CASES)
    ]
    return nb

if __name__ == "__main__":
    nb = build_notebook()
    with open("test.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Notebook 'test.ipynb' created successfully!")
