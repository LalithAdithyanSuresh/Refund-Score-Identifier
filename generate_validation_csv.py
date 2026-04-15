import sqlite3
import pandas as pd
import os

# Connect to the database
DB_PATH = "refund_system_final.db"
OUTPUT_CSV = "validation_dataset.csv"

def get_label(cid):
    """
    Determine the 'Correct Label' (APPROVE=1, REJECT=0) based on the ranges 
    defined in generate_database.py.
    
    Ranges from generate_database.py:
    1-45:   High Severity -> 1 (APPROVE)
    46-80:  Medium Severity -> 0 (REJECT)
    81-95:  Low Severity -> 0 (REJECT)
    96-120: Normal Valid -> 1 (APPROVE)
    121-135: Urgency -> 1 (APPROVE)
    136-150: Typos/Noise -> 1 (APPROVE)
    151-160: Sarcasm -> 1 (APPROVE)
    161-185: Change of Mind -> 0 (REJECT)
    186-200: Vague -> 0 (REJECT)
    201-208: DB Mismatch Color -> 0 (REJECT)
    209-213: DB Mismatch Brand -> 0 (REJECT)
    214-225: Outside Window -> 0 (REJECT)
    """
    if 1 <= cid <= 45: return 1
    if 46 <= cid <= 80: return 0
    if 81 <= cid <= 95: return 0
    if 96 <= cid <= 120: return 1
    if 121 <= cid <= 135: return 1
    if 136 <= cid <= 150: return 1
    if 151 <= cid <= 160: return 1
    if 161 <= cid <= 185: return 0
    if 186 <= cid <= 200: return 0
    if 201 <= cid <= 208: return 0
    if 209 <= cid <= 213: return 0
    if 214 <= cid <= 225: return 0
    return 0

def main():
    if not os.path.exists(DB_PATH):
        print(f"Error: {DB_PATH} not found.")
        return

    conn = sqlite3.connect(DB_PATH)
    query = "SELECT complaint_id, complaint_text, customer_id, order_id, complaint_date FROM complaints"
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Assign labels
    df['Correct label that should be assigned for this row'] = df['complaint_id'].apply(get_label)

    # Rename columns to match user's request
    df = df.rename(columns={
        'complaint_text': 'test_complaint_text',
        'customer_id': 'test_customer_id',
        'order_id': 'test_order_id',
        'complaint_date': 'test_complaint_date'
    })

    # Drop the temporary complaint_id used for labeling
    df = df.drop(columns=['complaint_id'])

    # Pick 100 rows - a balanced mix from different ranges
    # We take every ~2nd or 3rd row to get a spread
    # Or just take top 100 if the order is already spread, but here it's sequential by category.
    # To get exactly 100 rows with a good mix:
    indices = []
    # High: 15 rows
    indices.extend(range(0, 15))
    # Med: 15 rows
    indices.extend(range(45, 60))
    # Low: 5 rows
    indices.extend(range(80, 85))
    # Normal Valid: 15 rows
    indices.extend(range(95, 110))
    # Urgency: 10 rows
    indices.extend(range(120, 130))
    # Typos: 10 rows
    indices.extend(range(135, 145))
    # Change of Mind: 10 rows
    indices.extend(range(160, 170))
    # Vague: 10 rows
    indices.extend(range(185, 195))
    # Misc defects (Mismatch/Window): 10 rows
    indices.extend(range(200, 210))
    
    # Ensure exactly 100 if we have enough
    selected_df = df.iloc[indices[:100]]

    # Save to CSV
    selected_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Successfully saved 100 rows to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
