import sqlite3
import csv

conn = sqlite3.connect('refund_system_final.db')
c = conn.cursor()

c.execute('''
    SELECT c.complaint_id, c.customer_id, c.order_id, c.complaint_date, o.purchase_date, p.product_name, c.complaint_text 
    FROM complaints c 
    JOIN orders o ON c.order_id = o.order_id 
    JOIN products p ON o.product_id = p.product_id 
    LIMIT 20
''')

rows = c.fetchall()

with open('db_test_dump.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['complaint_id', 'customer_id', 'order_id', 'complaint_date', 'purchase_date', 'product_name', 'complaint_text'])
    writer.writerows(rows)

print("Export complete.")
