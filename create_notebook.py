"""
Creates the refund_validation_pipeline.ipynb notebook using nbformat.
The notebook implements Block [1] Database Validation and Block [2] Text Validation.
"""
import nbformat as nbf

def make_md(source):
    return nbf.v4.new_markdown_cell(source)

def make_code(source):
    return nbf.v4.new_code_cell(source)

# ═══════════════════ CELL SOURCE CODE (as plain strings) ═══════════════════

CELL_IMPORTS = r'''import sqlite3
import re
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──
DB_PATH = 'refund_system_final.db'
OUTPUT_CSV = 'validation_results.csv'

# Thresholds (tuned for all-MiniLM-L6-v2)
CHANGE_OF_MIND_THRESHOLD = 0.50
VAGUE_THRESHOLD = 0.45

# Reference texts for change-of-mind detection
CHANGE_OF_MIND_REFS = [
    "i changed my mind about this product",
    "i prefer another product or brand",
    "i dont like this product anymore",
    "want a different color or model",
    "i want to return this i dont need it",
    "decided to go with a different brand",
    "i prefer a different model",
    "dont need this anymore changed my mind",
    "i want to exchange for something else",
    "not the product i wanted prefer another",
    "dont like the color want to exchange",
    "prefer something else want to return",
]

# Keywords that strongly indicate change-of-mind (used as fallback)
CHANGE_OF_MIND_KEYWORDS = [
    'changed my mind', 'change my mind', 'prefer another',
    'prefer a different', 'dont need', 'dont like',
    'do not like', 'do not need', 'dont want',
    'want to exchange', 'want to return',
    'prefer something else', 'decided to go with',
    'dont like the color', 'prefer a different model',
    'just dont like it', 'fine but i just dont like',
]

# Reference texts for vague complaint detection
VAGUE_REFS = [
    "bad product need refund",
    "not good want refund",
    "refund please",
    "i want my money back",
    "worst product",
    "not satisfied refund needed",
    "terrible just give refund",
    "very unhappy want refund",
    "just want a refund for this",
    "need my money back",
    "waste of money",
    "unhappy with this purchase refund needed",
    "need a refund for this product",
    "very unhappy with purchase refund",
    "not happy refund",
]

# Keywords that strongly indicate vague complaints (no specific issue)
VAGUE_KEYWORDS = [
    'very unhappy with this purchase',
    'unhappy with this purchase',
    'worst purchase ever',
    'waste of money',
]

print("Configuration loaded successfully.")
'''

CELL_LOAD_DB = r'''def load_database(db_path):
    """Load all tables from the SQLite database into DataFrames."""
    conn = sqlite3.connect(db_path)
    customers = pd.read_sql("SELECT * FROM customers", conn)
    products = pd.read_sql("SELECT * FROM products", conn)
    orders = pd.read_sql("SELECT * FROM orders", conn)
    complaints = pd.read_sql("SELECT * FROM complaints", conn)
    conn.close()
    return customers, products, orders, complaints

customers_df, products_df, orders_df, complaints_df = load_database(DB_PATH)

print(f"Customers : {len(customers_df)}")
print(f"Products  : {len(products_df)}")
print(f"Orders    : {len(orders_df)}")
print(f"Complaints: {len(complaints_df)}")
print()
print("Sample complaints:")
complaints_df[['complaint_id', 'complaint_text']].head(5)
'''

CELL_PREPROCESS = r'''def preprocess_text(text):
    """Apply minimal text preprocessing."""
    # Lowercase
    text = text.lower()
    # Replace long numbers (5+ digits) with <ID>
    text = re.sub(r'\b\d{5,}\b', '<ID>', text)
    # Normalize repeated characters (3+ of same char -> 2)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    # Remove irrelevant symbols, keep letters, digits, spaces, and ! ? . ,
    text = re.sub(r"[^\w\s!?.,]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Test preprocessing
test_cases = [
    "My LPTOP is NOT WRKING!!!",
    "scren brokn pls halpppp",
    "Order #12345678 is terrible",
    "sooooo frustrating, overheating baaadly",
]
for t in test_cases:
    print(f"  {t!r:50s} -> {preprocess_text(t)!r}")
'''

CELL_BERT = r'''# Load BERT model
print("Loading sentence-transformer model (all-MiniLM-L6-v2)...")
bert_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.\n")

# Embedding cache for efficiency
_emb_cache = {}

def get_embedding(text):
    """Get BERT embedding with caching."""
    if text not in _emb_cache:
        _emb_cache[text] = bert_model.encode(text, show_progress_bar=False)
    return _emb_cache[text]

def cosine_sim(text_a, text_b):
    """Compute cosine similarity between two texts."""
    ea = get_embedding(text_a).reshape(1, -1)
    eb = get_embedding(text_b).reshape(1, -1)
    return float(cosine_similarity(ea, eb)[0][0])

def max_similarity(text, references):
    """Return (max_score, best_match) against a list of reference texts."""
    text_emb = get_embedding(text).reshape(1, -1)
    ref_embs = np.array([get_embedding(r) for r in references])
    sims = cosine_similarity(text_emb, ref_embs)[0]
    idx = int(np.argmax(sims))
    return float(sims[idx]), references[idx]

# Pre-cache reference embeddings
print("Pre-caching reference embeddings...")
for ref in CHANGE_OF_MIND_REFS + VAGUE_REFS:
    get_embedding(ref)
print(f"Cached {len(_emb_cache)} embeddings.")
'''

CELL_DB_VALIDATION = r'''# Pre-compute lookup structures
ALL_BRANDS = sorted(products_df['brand'].unique().tolist(), key=len, reverse=True)
ALL_COLORS = sorted(products_df['color'].unique().tolist(), key=len, reverse=True)

def validate_customer(cust_id, customers_df):
    """Check if customer exists in the database."""
    if cust_id not in customers_df['customer_id'].values:
        return False, f"Customer ID {cust_id} not found in database"
    return True, ""

def validate_order(order_id, cust_id, orders_df):
    """Check if order exists and belongs to the given customer."""
    order_row = orders_df[orders_df['order_id'] == order_id]
    if order_row.empty:
        return False, f"Order ID {order_id} not found in database"
    if order_row.iloc[0]['customer_id'] != cust_id:
        return False, f"Order {order_id} does not belong to customer {cust_id}"
    return True, ""

def get_product_for_order(order_id, orders_df, products_df):
    """Retrieve the product record linked to an order."""
    order_row = orders_df[orders_df['order_id'] == order_id]
    if order_row.empty:
        return None
    pid = order_row.iloc[0]['product_id']
    prod_row = products_df[products_df['product_id'] == pid]
    return prod_row.iloc[0] if not prod_row.empty else None

def find_mentioned_items(text, items):
    """Find which items from a list appear as whole words in the text."""
    found = []
    text_lower = text.lower()
    for item in items:
        item_lower = item.lower()
        if ' ' in item_lower:
            if item_lower in text_lower:
                found.append(item)
        else:
            if re.search(r'\b' + re.escape(item_lower) + r'\b', text_lower):
                found.append(item)
    return found

def validate_product_context(cleaned_text, product):
    """
    Validate complaint is about the correct product.
    1. Check if complaint mentions a brand different from the product's brand.
    2. Check if complaint mentions a color different from the product's color.
    Returns: (is_valid, reason, similarity_score)
    """
    product_context = f"{product['brand']} {product['product_name']} {product['color']}"
    similarity = cosine_sim(cleaned_text, product_context)

    actual_brand = product['brand'].lower()
    actual_color = product['color'].lower()

    # Brand mismatch check
    mentioned_brands = find_mentioned_items(cleaned_text, ALL_BRANDS)
    for mb in mentioned_brands:
        if mb.lower() != actual_brand:
            return False, (
                f"Brand mismatch: complaint mentions '{mb}' but ordered product brand is '{product['brand']}'"
            ), similarity

    # Color mismatch check
    mentioned_colors = find_mentioned_items(cleaned_text, ALL_COLORS)
    for mc in mentioned_colors:
        mc_lower = mc.lower()
        if mc_lower == actual_color or mc_lower in actual_color or actual_color in mc_lower:
            continue
        mismatch_patterns = [
            r'ordered.*\b' + re.escape(mc_lower) + r'\b',
            r'wanted.*\b' + re.escape(mc_lower) + r'\b',
            r'\b' + re.escape(mc_lower) + r'\b.*(?:but|however|instead)',
            r'(?:wrong|different).*\b' + re.escape(mc_lower) + r'\b',
            r'\b' + re.escape(mc_lower) + r'\b.*(?:version|variant|model|one)',
        ]
        for pat in mismatch_patterns:
            if re.search(pat, cleaned_text.lower()):
                return False, (
                    f"Color mismatch: complaint claims '{mc}' but ordered product color is '{product['color']}'"
                ), similarity
        if product['color'].lower() not in cleaned_text.lower():
            return False, (
                f"Color mismatch: complaint mentions '{mc}' but ordered product color is '{product['color']}'"
            ), similarity

    return True, "", similarity

def validate_return_window(complaint_date_str, order_id, orders_df, products_df):
    """Check if the complaint is within the product's refund window."""
    order_row = orders_df[orders_df['order_id'] == order_id]
    if order_row.empty:
        return False, "Order not found"
    prod = products_df[products_df['product_id'] == order_row.iloc[0]['product_id']]
    if prod.empty:
        return False, "Product not found"

    purchase_date = pd.to_datetime(order_row.iloc[0]['purchase_date'])
    complaint_date = pd.to_datetime(complaint_date_str)
    days_elapsed = (complaint_date - purchase_date).days
    window = int(prod.iloc[0]['refund_window_days'])

    if days_elapsed > window:
        return False, (
            f"Outside refund window: {days_elapsed} days since purchase "
            f"(allowed: {window} days)"
        )
    return True, ""

def run_db_validation(row, cleaned_text, customers_df, orders_df, products_df):
    """Run all database validation checks for a single complaint."""
    valid, reason = validate_customer(row['customer_id'], customers_df)
    if not valid:
        return False, reason

    valid, reason = validate_order(row['order_id'], row['customer_id'], orders_df)
    if not valid:
        return False, reason

    product = get_product_for_order(row['order_id'], orders_df, products_df)
    if product is None:
        return False, "Product not found for the linked order"
    valid, reason, sim = validate_product_context(cleaned_text, product)
    if not valid:
        return False, reason

    valid, reason = validate_return_window(
        row['complaint_date'], row['order_id'], orders_df, products_df
    )
    if not valid:
        return False, reason

    return True, "Database validation passed"

print("Database validation functions defined.")
'''

CELL_TEXT_VALIDATION = r'''def detect_change_of_mind(cleaned_text):
    """
    Detect if the complaint expresses a change-of-mind.
    Two-pronged approach:
    1. BERT similarity against reference phrases
    2. Keyword-based fallback for explicit change-of-mind language
    """
    text_lower = cleaned_text.lower()

    # Method 1: BERT similarity
    score, best_match = max_similarity(cleaned_text, CHANGE_OF_MIND_REFS)
    if score >= CHANGE_OF_MIND_THRESHOLD:
        return True, (
            f"Change of mind detected (similarity={score:.3f}, "
            f"matched: '{best_match}')"
        )

    # Method 2: Keyword-based fallback
    for kw in CHANGE_OF_MIND_KEYWORDS:
        if kw in text_lower:
            defect_indicators = [
                'broken', 'cracked', 'dead', 'not working', 'defect', 'overheat',
                'disconnect', 'restart', 'lag', 'freeze', 'shut down', 'battery',
                'blurry', 'distort', 'stuck', 'unresponsive', 'buzzing', 'noise',
                'pixel', 'stopped working', 'malfunction', 'error',
            ]
            has_defect = any(re.search(r'\b' + re.escape(d) + r'\b', text_lower) for d in defect_indicators)
            if not has_defect:
                return True, (
                    f"Change of mind detected (keyword: '{kw}')"
                )

    return False, ""

def detect_vague_complaint(cleaned_text):
    """
    Detect if the complaint is too vague - no specific product issue mentioned.
    Two-pronged: BERT similarity + keyword fallback.
    """
    text_lower = cleaned_text.lower()

    # Defect keywords - if present, complaint is NOT vague
    defect_keywords = [
        'broken', 'cracked', 'dead', 'not working', 'defect', 'overheat',
        'disconnect', 'restart', 'lag', 'freeze', 'shut down', 'shutting',
        'battery drain', 'blurry', 'distort', 'stuck', 'unresponsive',
        'buzzing', 'noise', 'pixel', 'charging', 'wrking', 'brokn',
        'stoppd', 'nt wrking', 'ded', 'dosnt', 'stopped working',
        'not turn', 'wont turn', 'hardware failure', 'stopped functioning',
        'broke', 'malfunction', 'does not power', 'does not respond',
        'does not work', 'wont work', 'cannot use', 'screen cracked',
        'dead pixel', 'overheating', 'disconnecting', 'restarting',
        'lagging', 'freezing',
    ]
    has_defect = any(re.search(r'\b' + re.escape(kw) + r'\b', text_lower) for kw in defect_keywords)

    # Method 1: BERT similarity
    score, best_match = max_similarity(cleaned_text, VAGUE_REFS)
    if score >= VAGUE_THRESHOLD and not has_defect:
        return True, (
            f"Vague complaint detected (score={score:.3f}, "
            f"matched: '{best_match}')"
        )

    # Method 2: Keyword-based fallback
    if not has_defect:
        for kw in VAGUE_KEYWORDS:
            if kw in text_lower:
                return True, (
                    f"Vague complaint detected (keyword: '{kw}')"
                )

    return False, ""

def run_text_validation(cleaned_text):
    """Run all text-based validation checks."""
    is_com, reason = detect_change_of_mind(cleaned_text)
    if is_com:
        return False, reason

    is_vague, reason = detect_vague_complaint(cleaned_text)
    if is_vague:
        return False, reason

    return True, "Text validation passed"

print("Text validation functions defined.")
'''

CELL_PIPELINE = r'''def run_pipeline(complaints_df, customers_df, orders_df, products_df):
    """Execute the complete validation pipeline on all complaints."""
    results = []
    total = len(complaints_df)

    print(f"Processing {total} complaints...\n")
    for idx, row in complaints_df.iterrows():
        cleaned = preprocess_text(row['complaint_text'])

        # Block 1: Database Validation
        db_ok, db_reason = run_db_validation(
            row, cleaned, customers_df, orders_df, products_df
        )
        if not db_ok:
            results.append({
                'complaint_id': row['complaint_id'],
                'complaint_text': row['complaint_text'],
                'cleaned_text': cleaned,
                'decision': 'REJECT',
                'reason': db_reason,
            })
            continue

        # Block 2: Text Validation
        txt_ok, txt_reason = run_text_validation(cleaned)
        if not txt_ok:
            results.append({
                'complaint_id': row['complaint_id'],
                'complaint_text': row['complaint_text'],
                'cleaned_text': cleaned,
                'decision': 'REJECT',
                'reason': txt_reason,
            })
            continue

        results.append({
            'complaint_id': row['complaint_id'],
            'complaint_text': row['complaint_text'],
            'cleaned_text': cleaned,
            'decision': 'VALID',
            'reason': 'All validations passed - complaint is valid for further processing',
        })

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{total}...")

    print(f"\nDone! Processed {len(results)} complaints.")
    return pd.DataFrame(results)

# Execute
results_df = run_pipeline(complaints_df, customers_df, orders_df, products_df)
'''

CELL_SUMMARY = r'''print("=" * 65)
print("           VALIDATION RESULTS SUMMARY")
print("=" * 65)

total = len(results_df)
valid_count = len(results_df[results_df['decision'] == 'VALID'])
reject_count = len(results_df[results_df['decision'] == 'REJECT'])

print(f"\nTotal complaints : {total}")
print(f"VALID            : {valid_count}  ({valid_count/total*100:.1f}%)")
print(f"REJECT           : {reject_count}  ({reject_count/total*100:.1f}%)")

print(f"\n{'---'*22}")
print("Rejection Reason Breakdown:")
print('---'*22)
rejected = results_df[results_df['decision'] == 'REJECT']

def categorize_reason(reason):
    if 'Brand mismatch' in reason:
        return 'Brand Mismatch (DB)'
    elif 'Color mismatch' in reason:
        return 'Color Mismatch (DB)'
    elif 'Outside refund window' in reason:
        return 'Outside Refund Window'
    elif 'Change of mind' in reason:
        return 'Change of Mind'
    elif 'Vague complaint' in reason:
        return 'Vague Complaint'
    elif 'Customer ID' in reason:
        return 'Invalid Customer'
    elif 'Order' in reason:
        return 'Invalid Order'
    else:
        return reason

rejected_cats = rejected['reason'].apply(categorize_reason).value_counts()
for cat, count in rejected_cats.items():
    print(f"  {cat:35s} : {count}")
'''

CELL_SAVE = r'''# Save to CSV
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"Results saved to '{OUTPUT_CSV}'\n")

# Sample VALID complaints
print("=" * 65)
print("  SAMPLE VALID COMPLAINTS")
print("=" * 65)
valid_samples = results_df[results_df['decision'] == 'VALID'].sample(
    n=min(8, valid_count), random_state=42
)
for _, r in valid_samples.iterrows():
    print(f"\n  ID    : {r['complaint_id']}")
    print(f"  Text  : {r['complaint_text'][:100]}")
    print(f"  Clean : {r['cleaned_text'][:100]}")
    print(f"  Result: {r['decision']} --- {r['reason']}")

# Sample REJECTED complaints
print("\n" + "=" * 65)
print("  SAMPLE REJECTED COMPLAINTS")
print("=" * 65)
reject_samples = results_df[results_df['decision'] == 'REJECT'].sample(
    n=min(8, reject_count), random_state=42
)
for _, r in reject_samples.iterrows():
    print(f"\n  ID    : {r['complaint_id']}")
    print(f"  Text  : {r['complaint_text'][:100]}")
    print(f"  Clean : {r['cleaned_text'][:100]}")
    print(f"  Result: {r['decision']} --- {r['reason']}")
'''

CELL_TABLE = r'''results_df.style.map(
    lambda v: 'background-color: #d4edda' if v == 'VALID' else (
        'background-color: #f8d7da' if v == 'REJECT' else ''
    ), subset=['decision']
)
'''


# ═══════════════════ BUILD NOTEBOOK ═══════════════════

def build_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }

    cells = []

    cells.append(make_md("""# AI-Based Refund Approval System - Primitive Validation Pipeline

This notebook implements **Block [1] Database Validation** and **Block [2] Text Validation**.

**Pipeline:**
```
Complaint Text -> [1] DB Validation -> [2] Text Validation -> Decision (VALID / REJECT)
```
Only complaints that pass both stages are marked **VALID** for downstream BERT classification + SHAP.
"""))

    cells.append(make_md("## 1. Imports & Configuration"))
    cells.append(make_code(CELL_IMPORTS))

    cells.append(make_md("## 2. Load Database"))
    cells.append(make_code(CELL_LOAD_DB))

    cells.append(make_md("""## 3. Text Preprocessing

Minimal preprocessing:
- Lowercase
- Normalize repeated characters (3+ -> 2)
- Remove irrelevant symbols (keep `!`, `?`, `.`)
- Replace long numbers (5+ digits) with `<ID>`
- **NO** spelling correction, stopword removal, stemming, or lemmatization
"""))
    cells.append(make_code(CELL_PREPROCESS))

    cells.append(make_md("""## 4. BERT Semantic Similarity Engine

Using `all-MiniLM-L6-v2` from sentence-transformers for embedding generation.
All embeddings are cached to avoid redundant computation.
"""))
    cells.append(make_code(CELL_BERT))

    cells.append(make_md("""## 5. Block [1] - Database Validation

For each complaint:
1. **Customer existence** - `customer_id` must be in the customers table
2. **Order existence** - `order_id` must exist and belong to the same customer
3. **Product context** - complaint should be about the product in the linked order
   - Build context: `brand + product_name + color`
   - Check for brand / color mismatches using keyword extraction
4. **Return window** - `complaint_date - purchase_date <= refund_window_days`
"""))
    cells.append(make_code(CELL_DB_VALIDATION))

    cells.append(make_md("""## 6. Block [2] - Text Validation

Semantic similarity-based checks:
1. **Change-of-Mind Detection** - reject if complaint is semantically similar to preference/change-of-mind phrases
2. **Vague Complaint Detection** - reject if complaint has no specific reason (just asks for refund without explaining the issue)
"""))
    cells.append(make_code(CELL_TEXT_VALIDATION))

    cells.append(make_md("""## 7. Run the Full Validation Pipeline

```
IF   DB validation passed
AND  NOT change-of-mind
AND  NOT vague
-> VALID
ELSE -> REJECT
```
"""))
    cells.append(make_code(CELL_PIPELINE))

    cells.append(make_md("## 8. Results Summary"))
    cells.append(make_code(CELL_SUMMARY))

    cells.append(make_md("## 9. Save Results & Display Samples"))
    cells.append(make_code(CELL_SAVE))

    cells.append(make_md("## 10. Full Results Table"))
    cells.append(make_code(CELL_TABLE))

    nb.cells = cells
    return nb


if __name__ == "__main__":
    nb = build_notebook()
    fname = "refund_validation_pipeline.ipynb"
    with open(fname, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook saved to {fname}")
