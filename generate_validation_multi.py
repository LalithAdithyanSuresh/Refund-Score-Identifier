"""
generate_validation_multi.py
──────────────────────────────
Generates a validation dataset of 100 complaints with:
- Complaint_Text
- Customer_ID
- Order_ID
- Complaint_Date
- Ground_Truth
- Predictions from 6 models: Word2Vec, Fasttext, Glove, Bert, InferSent, SBert

Aligned with refund_system_final.db.
"""

import sqlite3
import random
import pandas as pd
import joblib
import os
import numpy as np
from datetime import datetime, timedelta
import feature_extractor as fe

# ── Configuration ──────────────────────────────────────────────────────────
DB_PATH = "refund_system_final.db"
OUTPUT_CSV = "validation_db_100.csv"
random.seed(99)

MODELS_CONFIG = {
    "Word2Vec": "model_word2vec.pkl",
    "Fasttext": "model_fasttext.pkl",
    "Glove": "model_glove.pkl",
    "Bert": "model_bert.pkl",
    "InferSent": "model_infersent.pkl",
    "SBert": "model_sbert.pkl"
}

# ── Database Loading ───────────────────────────────────────────────────────
def load_db_data():
    conn = sqlite3.connect(DB_PATH)
    customers = pd.read_sql_query("SELECT * FROM customers", conn)
    products = pd.read_sql_query("SELECT * FROM products", conn)
    orders = pd.read_sql_query("SELECT * FROM orders", conn)
    conn.close()
    return customers, products, orders

# ── Complaint Templates (Simplified from generate_database.py) ──────────────
def get_templates(product):
    n, b, cat, col = product["product_name"], product["brand"], product["category"], product["color"]
    
    valid_templates = [
        f"My {b} {n} is completely broken. Screen is cracked and it won't turn on.",
        f"The {n} arrived dead on arrival. Tried charging but no response.",
        f"My {col} {n} has a hardware defect. The charging port is extremely loose.",
        f"The {b} {n} display has several dead pixels in the center.",
        f"Battery on my {n} drains in less than an hour. Seems like a faulty unit.",
        f"The {n} keeps restarting randomly throughout the day. Very unreliable.",
        f"WiFi connection keeps dropping on my {b} {n} every few minutes.",
        f"The {n} overheats significantly within 10 minutes of use.",
        f"The speaker on my {b} {n} produces distorted sound at all volumes.",
        f"Touchscreen on my {n} is unresponsive in certain areas."
    ]
    
    reject_templates = [
        f"I want to return the {b} {n}. I changed my mind and want a different model.",
        f"Changed my mind about the {n}. I prefer the other color instead.",
        f"I don't need the {n} anymore. Decided to go with a laptop instead.",
        f"I bought this {b} {n} but I don't like the feel of it. Refund please.",
        f"Just not satisfied with the {n}. I want my money back.",
        f"I ordered a Red {n} but I actually want the {col} one. Oh wait, this is {col}? I want Red then.", # Mismatch logic
        f"Bad product. Give me a refund now.",
        f"This is the worst {b} {n} ever. Not what I expected. Refund please.",
        f"The {n} is fine but I found a better price elsewhere.",
        f"I ordered this {n} by mistake. Please cancel and refund."
    ]
    
    return valid_templates, reject_templates

# ── Generation Logic ───────────────────────────────────────────────────────
def generate_100_samples(products, orders):
    samples = []
    
    # 60 Valid (APPROVE), 40 Invalid (REJECT)
    for i in range(100):
        is_valid = i < 60
        label = "APPROVE" if is_valid else "REJECT"
        
        # Pick a random order
        order = orders.sample(1).iloc[0]
        product = products[products['product_id'] == order['product_id']].iloc[0]
        
        valid_t, reject_t = get_templates(product)
        text = random.choice(valid_t if is_valid else reject_t)
        
        # Calculate complaint date (within 1-10 days of purchase)
        purchase_date = datetime.strptime(order['purchase_date'], "%Y-%m-%d")
        complaint_date = purchase_date + timedelta(days=random.randint(1, 10))
        
        # For REJECT, some might be "outside window"
        if not is_valid and random.random() < 0.3:
            # Force outside window
            complaint_date = purchase_date + timedelta(days=product['refund_window_days'] + random.randint(5, 20))
            text = f"The {product['product_name']} stopped working, but I am reporting it late. Hope you can still refund."

        samples.append({
            "Complaint_Text": text,
            "Customer_ID": int(order['customer_id']),
            "Order_ID": int(order['order_id']),
            "Complaint_Date": complaint_date.strftime("%Y-%m-%d"),
            "what label should be given to this samples(ground truth)": label
        })
        
    return pd.DataFrame(samples)

# ── Prediction Logic ───────────────────────────────────────────────────────
def run_predictions(df):
    results_df = df.copy()
    
    for model_name, pkl_file in MODELS_CONFIG.items():
        print(f"Running predictions for {model_name}...")
        
        # Set method in feature_extractor
        method_key = model_name.lower()
        fe.set_embedding_method(method_key)
        
        # Load model
        if not os.path.exists(pkl_file):
            print(f"Warning: Model file {pkl_file} not found. Skipping.")
            results_df[model_name] = "N/A"
            continue
            
        model = joblib.load(pkl_file)
        
        preds = []
        for text in df['Complaint_Text']:
            try:
                pred = fe.classify(text, model, method_key)
                preds.append(pred)
            except Exception as e:
                print(f"Error classifying with {model_name}: {e}")
                preds.append("ERROR")
        
        results_df[model_name] = preds
        
        # Clean up memory if possible (del model, del feature vectors if stored)
        del model
        
    return results_df

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("Loading database data...")
    customers, products, orders = load_db_data()
    
    print("Generating 100 complaint samples...")
    df = generate_100_samples(products, orders)
    
    print("Starting prediction phase (this may take a few minutes)...")
    final_df = run_predictions(df)
    
    # Save to CSV
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSuccess! Validation dataset saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
