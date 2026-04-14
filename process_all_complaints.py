import pandas as pd
import re
import os

def clean_and_prepare_data(complaints_path, orders_path):
    # Load data
    df = pd.read_csv(complaints_path)
    orders_df = pd.read_csv(orders_path)
    
    # Cleaning: Remove rows where complaint_date is the literal string 'complaint_date'
    df = df[df['complaint_date'] != 'complaint_date'].copy()
    
    # Feature Engineering
    df['complaint_date'] = pd.to_datetime(df['complaint_date'], errors='coerce')
    df = df.sort_values(by=['customer_id', 'complaint_date']).reset_index(drop=True)
    
    df['prior_complaints'] = df.groupby('customer_id').cumcount()
    order_counts = orders_df.groupby('customer_id').size().to_dict()
    df['total_orders'] = df['customer_id'].map(order_counts).fillna(0)
    
    df['complaint_ratio'] = (df['prior_complaints'] + 1) / df['total_orders'].apply(lambda x: 1 if x == 0 else x)
    
    return df

# --- METHOD 1: POS Tagging / Basic Rule-Based Logic (from rule_based_extraction.ipynb) ---

def compute_fraud_features_pos(row):
    text = str(row['complaint_text']).lower()
    score = 6
    reason = "Valid defect"
    classification = "Legitimate"
    refund = "Yes"
    
    # Extraction Logic
    if 'box' in text and any(w in text for w in ['random', 'broken', 'scratches', 'tampered', 'older', 'different']):
        score, classification, refund, reason = 95, 'Fraud', 'No', 'Switcheroo suspected'
    elif 'ordered' in text and ('got' in text or 'looks' in text or 'why' in text):
        score, classification, refund, reason = 93, 'Fraud', 'No', 'Color contradiction'
    elif 'used' in text and ('already' in text or 'returning' in text or 'few' in text or 'event' in text or 'now' in text):
        score, classification, refund, reason = 85, 'Fraud', 'No', 'Usage abuse'
    elif any(phrase in text for phrase in ['wrong item', 'wrong model', 'different from', 'not matching', 'match description']): 
        score, classification, refund, reason = 90, 'Fraud', 'No', 'Unverified mismatch'
    elif any(phrase in text for phrase in ['changed my mind', 'dont like', "don't like", 'prefer', 'not my style', 'expected better', 'not what i expected', 'feels cheap', 'not worth']):
        score, classification, refund, reason = 18, 'Fraud', 'No', 'Change of mind'
    else:
        words = text.split()
        if text.strip() in ['ok', 'bad product', 'not good', 'worst', 'bad', 'meh'] or (len(words) <= 3 and 'dead' not in text and 'not working' not in text):
            score, classification, refund, reason = 22, 'Fraud', 'No', 'Vague complaint'

    # Behavioral Logic
    history_penalty = 0
    prior_complaints = row['prior_complaints']
    ratio = row['complaint_ratio']
    
    if prior_complaints >= 3:
        history_penalty += 15
        classification = "Fraud"
        refund = "No"
        reason = "Frequent Abuser"
        
    if ratio > 0.5 and row['total_orders'] > 2:
        history_penalty += 10
        if classification == "Legitimate":
            classification = "Fraud"
            refund = "No"
            reason = "High Ratio Offender"

    if ratio <= 0.1 and prior_complaints == 0:
        history_penalty -= 5  
        
    score += history_penalty
    score = max(0, min(100, score))
    
    if score >= 80:
        classification = "Fraud"
        refund = "No"
        
    return pd.Series([score, classification, refund, reason])

# --- METHOD 2: Dependency Parsing Logic (from rule_based_evaluation.ipynb) ---

def heuristic_dependency_parse(text, window=3):
    words = str(text).lower().split()
    dependencies = []
    targets = {'box', 'screen', 'item', 'model', 'audio', 'device', 'touch'}
    modifiers = {'broken', 'scratches', 'tampered', 'older', 'different', 'wrong', 'low', 'randomly', 'dead', 'rebooting'}
    
    for i, word in enumerate(words):
        if word in targets:
            start = max(0, i - window)
            end = min(len(words), i + window + 1)
            for j in range(start, end):
                if words[j] in modifiers:
                    if j > 0 and words[j-1] in ['not', 'no', 'never']:\
                        dependencies.append((word, 'not_' + words[j]))
                    elif i != j:
                        dependencies.append((word, words[j]))
    return dependencies

def compute_fraud_features_dep(row):
    text = str(row['complaint_text']).lower()
    score = 6
    classification = "Legitimate"
    refund = "Yes"
    reason = "Valid defect"
    
    deps = heuristic_dependency_parse(text)
    
    # Dependency-specific triggers
    if ('box', 'tampered') in deps or ('box', 'different') in deps:
        score, classification, refund, reason = 95, 'Fraud', 'No', 'Switcheroo suspected'
    elif ('item', 'wrong') in deps or ('model', 'wrong') in deps:
        score, classification, refund, reason = 90, 'Fraud', 'No', 'Unverified mismatch'
    elif any((t, 'broken') in deps for t in ['screen', 'device']):
        score, classification, refund, reason = 85, 'Fraud', 'No', 'Physical damage suspected'
    elif any(phrase in text for phrase in ['changed my mind', 'dont like', "don't like", 'prefer', 'not my style']):
        score, classification, refund, reason = 18, 'Fraud', 'No', 'Change of mind'
        
    # Behavioral Logic (same as POS version)
    prior_complaints = row['prior_complaints']
    ratio = row['complaint_ratio']
    
    if prior_complaints >= 3:
        score += 15
        classification = "Fraud"
        refund = "No"
        reason = "Frequent Abuser"
        
    if ratio > 0.5 and row['total_orders'] > 2:
        score += 10
        if classification == "Legitimate":
            classification = "Fraud"
            refund = "No"
            reason = "High Ratio Offender"
            
    if ratio <= 0.1 and prior_complaints == 0:
        score -= 5  

    score = max(0, min(100, score))
    if score >= 80:
        classification = "Fraud"
        refund = "No"
        
    return pd.Series([score, classification, refund, reason])

def main():
    print("Loading and preparing data...")
    df = clean_and_prepare_data('Complaints.csv', 'Orders.csv')
    
    print("Processing with POS Tagging logic...")
    df_pos = df.copy()
    df_pos[['fraud_score', 'fraud_classification', 'refund_applicable', 'fraud_reason']] = \
        df_pos.apply(compute_fraud_features_pos, axis=1)
    df_pos.to_csv('Complaints_POS_Processed.csv', index=False)
    print(f"Saved {len(df_pos)} rows to Complaints_POS_Processed.csv")
    
    print("Processing with Dependency Parsing logic...")
    df_dep = df.copy()
    df_dep[['fraud_score', 'fraud_classification', 'refund_applicable', 'fraud_reason']] = \
        df_dep.apply(compute_fraud_features_dep, axis=1)
    df_dep.to_csv('Complaints_Dependency_Parsing_Processed.csv', index=False)
    print(f"Saved {len(df_dep)} rows to Complaints_Dependency_Parsing_Processed.csv")
    
    print("Processing complete.")

if __name__ == "__main__":
    main()
