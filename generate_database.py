"""
Generate the refund system SQLite database with realistic electronic products data.
Ensures strict data alignment: each complaint -> valid order -> valid product -> valid customer.
"""
import sqlite3
import random
from datetime import datetime, timedelta

random.seed(42)

DB_PATH = "refund_system_final.db"

# ─────────────────────────── STATIC DATA ───────────────────────────

CUSTOMERS = [
    (1, "Rahul Sharma", "rahul.sharma@email.com", "9876543210"),
    (2, "Priya Patel", "priya.patel@email.com", "9876543211"),
    (3, "Amit Kumar", "amit.kumar@email.com", "9876543212"),
    (4, "Sneha Reddy", "sneha.reddy@email.com", "9876543213"),
    (5, "Vikram Singh", "vikram.singh@email.com", "9876543214"),
    (6, "Ananya Gupta", "ananya.gupta@email.com", "9876543215"),
    (7, "Rohan Joshi", "rohan.joshi@email.com", "9876543216"),
    (8, "Kavita Nair", "kavita.nair@email.com", "9876543217"),
    (9, "Arjun Menon", "arjun.menon@email.com", "9876543218"),
    (10, "Divya Iyer", "divya.iyer@email.com", "9876543219"),
    (11, "Siddharth Das", "siddharth.das@email.com", "9876543220"),
    (12, "Meera Verma", "meera.verma@email.com", "9876543221"),
    (13, "Karthik Rao", "karthik.rao@email.com", "9876543222"),
    (14, "Pooja Agarwal", "pooja.agarwal@email.com", "9876543223"),
    (15, "Nikhil Bhatt", "nikhil.bhatt@email.com", "9876543224"),
    (16, "Ritu Saxena", "ritu.saxena@email.com", "9876543225"),
    (17, "Aditya Chopra", "aditya.chopra@email.com", "9876543226"),
    (18, "Swati Mishra", "swati.mishra@email.com", "9876543227"),
    (19, "Manish Tiwari", "manish.tiwari@email.com", "9876543228"),
    (20, "Neha Kapoor", "neha.kapoor@email.com", "9876543229"),
    (21, "Rajesh Pandey", "rajesh.pandey@email.com", "9876543230"),
    (22, "Ankita Jain", "ankita.jain@email.com", "9876543231"),
    (23, "Suresh Nambiar", "suresh.nambiar@email.com", "9876543232"),
    (24, "Tanvi Shah", "tanvi.shah@email.com", "9876543233"),
    (25, "Gaurav Malhotra", "gaurav.malhotra@email.com", "9876543234"),
    (26, "Ishita Bose", "ishita.bose@email.com", "9876543235"),
    (27, "Deepak Chauhan", "deepak.chauhan@email.com", "9876543236"),
    (28, "Sonal Deshmukh", "sonal.deshmukh@email.com", "9876543237"),
    (29, "Varun Mehta", "varun.mehta@email.com", "9876543238"),
    (30, "Pallavi Kulkarni", "pallavi.kulkarni@email.com", "9876543239"),
    (31, "Harsh Trivedi", "harsh.trivedi@email.com", "9876543240"),
    (32, "Lakshmi Pillai", "lakshmi.pillai@email.com", "9876543241"),
    (33, "Sameer Dutta", "sameer.dutta@email.com", "9876543242"),
    (34, "Nidhi Srivastava", "nidhi.srivastava@email.com", "9876543243"),
    (35, "Akash Thakur", "akash.thakur@email.com", "9876543244"),
]

# (product_id, product_name, brand, category, color, price, refund_window_days)
PRODUCTS = [
    (1, "Inspiron 15 Laptop", "Dell", "Laptop", "Silver", 54999, 30),
    (2, "Galaxy S23", "Samsung", "Smartphone", "Black", 74999, 15),
    (3, "iPhone 14", "Apple", "Smartphone", "Blue", 79999, 15),
    (4, "WH-1000XM5 Headphones", "Sony", "Headphones", "Black", 29999, 30),
    (5, "MacBook Air M2", "Apple", "Laptop", "Space Gray", 114999, 30),
    (6, "Galaxy Tab S9", "Samsung", "Tablet", "Graphite", 74999, 15),
    (7, "MX Master 3S Mouse", "Logitech", "Mouse", "White", 8999, 30),
    (8, "ROG Strix G15 Laptop", "Asus", "Laptop", "Black", 89999, 30),
    (9, "Tune 760NC Headphones", "JBL", "Headphones", "Blue", 4999, 20),
    (10, "Galaxy Watch 5", "Samsung", "Smartwatch", "Silver", 27999, 15),
    (11, "iPad Air", "Apple", "Tablet", "Pink", 59999, 15),
    (12, "Viper Mini Mouse", "Razer", "Mouse", "Black", 3999, 30),
    (13, "K70 RGB Keyboard", "Corsair", "Keyboard", "Black", 12999, 30),
    (14, "UltraSharp 27 Monitor", "Dell", "Monitor", "Black", 34999, 30),
    (15, "Flip 6 Speaker", "JBL", "Speaker", "Red", 12999, 20),
    (16, "Nord CE 3", "OnePlus", "Smartphone", "Green", 26999, 15),
    (17, "AirPods Pro 2", "Apple", "Earbuds", "White", 24999, 15),
    (18, "PowerCore 20000", "Anker", "Power Bank", "Black", 3999, 30),
    (19, "QuietComfort 45", "Bose", "Headphones", "White", 25999, 30),
    (20, "Nitro 5 Laptop", "Acer", "Laptop", "Black", 64999, 30),
]

# ─────────────────────── COMPLAINT TEMPLATES ───────────────────────

def _high_severity(p):
    """Generate high-severity complaint texts (broken/dead/cracked/not working)."""
    n, b, cat, col = p["product_name"], p["brand"], p["category"], p["color"]
    return [
        f"My {b} {n} is completely broken. It fell once and now the screen is cracked beyond repair.",
        f"The {n} I received is dead on arrival. It does not power up no matter what I try.",
        f"My {col} {n} has a cracked display. Totally unusable now.",
        f"The {b} {n} stopped working after 2 days. Completely dead, not charging.",
        f"Received the {n} and it is not working at all. Tried everything, the device is dead.",
        f"The screen on my {b} {n} is cracked out of the box. Please process refund.",
        f"My {n} broke within one week of purchase. It won't turn on anymore.",
        f"The {b} {n} arrived broken. The power button does not respond at all.",
        f"My {n} is completely dead. No lights, no response, nothing works.",
        f"Got the {b} {n} and the screen was cracked when I opened the package.",
    ]

def _medium_severity(p):
    """Generate medium-severity complaint texts (disconnecting/overheating/restarting/lagging)."""
    n, b = p["product_name"], p["brand"]
    return [
        f"My {b} {n} keeps disconnecting from WiFi every few minutes. Very frustrating.",
        f"The {n} is overheating badly during normal use. It gets too hot to hold.",
        f"My {b} {n} keeps restarting on its own randomly throughout the day.",
        f"The {n} is lagging terribly. Apps take forever to load and it freezes often.",
        f"My {b} {n} disconnects from bluetooth constantly. Cannot use it properly.",
        f"The {n} overheats within 10 minutes of usage. I am worried it might be dangerous.",
        f"My {n} restarts by itself at least 5 times a day. It is very unreliable.",
        f"The {b} {n} is extremely laggy. Every operation takes several seconds to respond.",
        f"WiFi keeps dropping on my {b} {n}. Have to reconnect every 15 minutes.",
        f"The {n} overheats and then shuts down automatically. This is not acceptable.",
    ]

def _low_severity(p):
    n, b = p["product_name"], p["brand"]
    return [
        f"My {b} {n} is slightly slow compared to what was advertised.",
        f"The {n} has a minor issue with the speaker. Volume is slightly lower than expected.",
        f"My {n} has a minor scratch on the surface but otherwise works fine. Still want a replacement.",
        f"The {b} {n} is a bit slower than I expected for this price range.",
    ]

def _normal_valid(p):
    n, b, col = p["product_name"], p["brand"], p["color"]
    return [
        f"My {b} {n} stopped working after just 3 days of normal use.",
        f"The {n} display has dead pixels in the center of the screen.",
        f"Battery on my {b} {n} drains completely in 2 hours even on standby.",
        f"The {col} {n} has a hardware defect. The charging port is loose.",
        f"My {n} makes a buzzing noise whenever I turn it on.",
        f"The {b} {n} randomly shuts down when battery is at 40 percent.",
        f"Touchscreen on my {n} is unresponsive in the bottom half.",
        f"The {b} {n} arrived with a defective camera module. Photos are all blurry.",
        f"My {n} speaker produces distorted sound at any volume level.",
        f"The {b} {n} has a manufacturing defect. One of the buttons is stuck.",
    ]

def _urgency(p):
    n, b = p["product_name"], p["brand"]
    return [
        f"URGENT! My {b} {n} is not working at all. I need ASAP refund please!",
        f"Urgent issue with my {n}! It completely stopped functioning. Need immediate refund!",
        f"ASAP refund needed! The {b} {n} broke on the first day of use!",
        f"Please process my refund URGENTLY. The {n} has a critical hardware failure!",
    ]

def _typos_noise(p):
    n, b, cat = p["product_name"], p["brand"], p["category"]
    cat_l = cat.lower()
    b_l = b.lower()[:4] if len(b) > 4 else b.lower()[:3]
    return [
        f"{cat_l} nt wrking at all. pls halp",
        f"scren brokn on my {b_l} {cat_l}. ned refnd",
        f"my {cat_l} iz ded. dosnt turn on anymor",
        f"da {b_l} {cat_l} stoppd wrking aftr 2 dayz",
        f"batry drains 2 fast on {cat_l}!! plz refnd asap!!!",
    ]

def _sarcasm(p):
    n, b = p["product_name"], p["brand"]
    return [
        f"Great {b} {n}, absolutely wonderful! Stopped working on day one. Brilliant quality!",
        f"Amazing {n}! The screen cracked all by itself. Best product ever!",
        f"Wow what a fantastic {b} {n}. Overheats in 5 minutes. Truly impressive engineering!",
        f"Love how my {n} randomly shuts down. Really adds excitement to my day!",
    ]

def _change_of_mind(p):
    n, b, col = p["product_name"], p["brand"], p["color"]
    return [
        f"I want to return the {b} {n}. I changed my mind, I prefer a different brand.",
        f"I dont like the {col} color of the {n}. Want to exchange for a different color.",
        f"Changed my mind about the {b} {n}. I prefer another model instead.",
        f"I dont need the {n} anymore. I decided to go with a different product entirely.",
        f"Want to return the {b} {n}. I prefer a different model that my friend recommended.",
        f"The {n} is fine but I just dont like it. Prefer something else.",
    ]

def _vague(p):
    n, b = p["product_name"], p["brand"]
    return [
        f"Bad product. Need refund.",
        f"Not satisfied with the {n}. Refund please.",
        f"Worst purchase ever. I want my money back.",
        f"The {b} {n} is terrible. Just give me a refund.",
        f"Not good. Want refund for the {n}.",
        f"I need a refund for this product now.",
        f"Very unhappy with this purchase. Refund needed.",
    ]

def _db_mismatch_color(p, wrong_color):
    """Complaint mentions wrong color — mismatch with DB."""
    n, b, col = p["product_name"], p["brand"], p["color"]
    return [
        f"I ordered a {wrong_color} {n} but received a {col} one instead. This is not what I ordered!",
        f"Wrong color! I specifically ordered the {wrong_color} {b} {n} but got {col}.",
        f"I wanted the {wrong_color} version of the {n}. Why did I receive {col}?",
    ]

def _db_mismatch_brand(p, wrong_brand):
    """Complaint mentions wrong brand — mismatch with DB."""
    n, b = p["product_name"], p["brand"]
    return [
        f"I ordered a {wrong_brand} product but received a {b} {n} instead!",
        f"This is wrong! I ordered {wrong_brand} but got {b} {n}. Please fix this.",
    ]


# ─────────────────────── ORDER GENERATION ──────────────────────────

BASE_DATE = datetime(2026, 4, 13)  # "today"

def generate_orders(num_orders=260):
    """Generate orders with controlled purchase dates."""
    orders = []
    for oid in range(1, num_orders + 1):
        cid = random.choice([c[0] for c in CUSTOMERS])
        pid = random.choice([p[0] for p in PRODUCTS])
        # Most orders within last 20 days, some older
        if oid <= 220:
            days_ago = random.randint(1, 20)
        elif oid <= 245:
            days_ago = random.randint(10, 35)
        else:
            days_ago = random.randint(35, 70)  # outside window for many products
        purchase_date = (BASE_DATE - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        orders.append((oid, cid, pid, purchase_date))
    return orders


# ─────────────────────── COMPLAINT GENERATION ──────────────────────

def get_product_dict(pid):
    for p in PRODUCTS:
        if p[0] == pid:
            return {
                "product_id": p[0], "product_name": p[1], "brand": p[2],
                "category": p[3], "color": p[4], "price": p[5],
                "refund_window_days": p[6],
            }
    return None

ALL_COLORS = list(set(p[4] for p in PRODUCTS))
ALL_BRANDS = list(set(p[2] for p in PRODUCTS))

def pick_wrong_color(actual_color):
    options = [c for c in ALL_COLORS if c.lower() != actual_color.lower()]
    return random.choice(options)

def pick_wrong_brand(actual_brand):
    options = [b for b in ALL_BRANDS if b.lower() != actual_brand.lower()]
    return random.choice(options)


def generate_complaints(orders):
    """
    Generate 220+ complaints with proper alignment and category distribution.
    Ensures more VALID than REJECTED.
    """
    complaints = []
    cid_counter = 1
    used_orders = set()

    def pick_order(prefer_recent=True, prefer_old=False):
        """Pick an unused order (or reuse if needed)."""
        available = [o for o in orders if o[0] not in used_orders]
        if not available:
            available = orders
        if prefer_old:
            # Pick orders with old dates
            old = [o for o in available if (BASE_DATE - datetime.strptime(o[3], "%Y-%m-%d")).days > 30]
            if old:
                chosen = random.choice(old)
                used_orders.add(chosen[0])
                return chosen
        if prefer_recent:
            recent = [o for o in available if (BASE_DATE - datetime.strptime(o[3], "%Y-%m-%d")).days <= 20]
            if recent:
                chosen = random.choice(recent)
                used_orders.add(chosen[0])
                return chosen
        chosen = random.choice(available)
        used_orders.add(chosen[0])
        return chosen

    def add_complaint(order, text, days_after_purchase=None):
        nonlocal cid_counter
        purchase_date = datetime.strptime(order[3], "%Y-%m-%d")
        if days_after_purchase is not None:
            complaint_date = purchase_date + timedelta(days=days_after_purchase)
        else:
            complaint_date = purchase_date + timedelta(days=random.randint(1, 12))
        complaints.append((
            cid_counter, order[1], order[0],
            text, complaint_date.strftime("%Y-%m-%d")
        ))
        cid_counter += 1

    # ── HIGH SEVERITY (45 complaints) → VALID ──
    for _ in range(45):
        order = pick_order()
        prod = get_product_dict(order[2])
        templates = _high_severity(prod)
        add_complaint(order, random.choice(templates))

    # ── MEDIUM SEVERITY (35 complaints) → VALID ──
    for _ in range(35):
        order = pick_order()
        prod = get_product_dict(order[2])
        templates = _medium_severity(prod)
        add_complaint(order, random.choice(templates))

    # ── LOW SEVERITY (15 complaints) → VALID ──
    for _ in range(15):
        order = pick_order()
        prod = get_product_dict(order[2])
        templates = _low_severity(prod)
        add_complaint(order, random.choice(templates))

    # ── NORMAL VALID (25 complaints) → VALID ──
    for _ in range(25):
        order = pick_order()
        prod = get_product_dict(order[2])
        templates = _normal_valid(prod)
        add_complaint(order, random.choice(templates))

    # ── URGENCY (15 complaints) → VALID (they mention real issues) ──
    for _ in range(15):
        order = pick_order()
        prod = get_product_dict(order[2])
        templates = _urgency(prod)
        add_complaint(order, random.choice(templates))

    # ── TYPOS / NOISE (15 complaints) → VALID ──
    for _ in range(15):
        order = pick_order()
        prod = get_product_dict(order[2])
        templates = _typos_noise(prod)
        add_complaint(order, random.choice(templates))

    # ── SARCASM (10 complaints) → VALID ──
    for _ in range(10):
        order = pick_order()
        prod = get_product_dict(order[2])
        templates = _sarcasm(prod)
        add_complaint(order, random.choice(templates))

    # ── CHANGE OF MIND (25 complaints) → REJECT ──
    for _ in range(25):
        order = pick_order()
        prod = get_product_dict(order[2])
        templates = _change_of_mind(prod)
        add_complaint(order, random.choice(templates))

    # ── VAGUE (15 complaints) → REJECT ──
    for _ in range(15):
        order = pick_order()
        prod = get_product_dict(order[2])
        templates = _vague(prod)
        add_complaint(order, random.choice(templates))

    # ── DB MISMATCH - COLOR (8 complaints) → REJECT ──
    for _ in range(8):
        order = pick_order()
        prod = get_product_dict(order[2])
        wrong_col = pick_wrong_color(prod["color"])
        templates = _db_mismatch_color(prod, wrong_col)
        add_complaint(order, random.choice(templates))

    # ── DB MISMATCH - BRAND (5 complaints) → REJECT ──
    for _ in range(5):
        order = pick_order()
        prod = get_product_dict(order[2])
        wrong_brand = pick_wrong_brand(prod["brand"])
        templates = _db_mismatch_brand(prod, wrong_brand)
        add_complaint(order, random.choice(templates))

    # ── OUTSIDE RETURN WINDOW (12 complaints) → REJECT ──
    for _ in range(12):
        order = pick_order(prefer_recent=False, prefer_old=True)
        prod = get_product_dict(order[2])
        # Force complaint date well outside window
        templates = _high_severity(prod) + _medium_severity(prod)
        add_complaint(order, random.choice(templates),
                      days_after_purchase=prod["refund_window_days"] + random.randint(3, 15))

    return complaints


# ─────────────────────── DATABASE CREATION ─────────────────────────

def create_database():
    import os
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            phone TEXT NOT NULL
        );
        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY,
            product_name TEXT NOT NULL,
            brand TEXT NOT NULL,
            category TEXT NOT NULL,
            color TEXT NOT NULL,
            price REAL NOT NULL,
            refund_window_days INTEGER NOT NULL
        );
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            purchase_date TEXT NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
            FOREIGN KEY (product_id) REFERENCES products(product_id)
        );
        CREATE TABLE complaints (
            complaint_id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            order_id INTEGER NOT NULL,
            complaint_text TEXT NOT NULL,
            complaint_date TEXT NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
            FOREIGN KEY (order_id) REFERENCES orders(order_id)
        );
    """)

    cur.executemany("INSERT INTO customers VALUES (?,?,?,?)", CUSTOMERS)
    cur.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?)", PRODUCTS)

    orders = generate_orders(260)
    cur.executemany("INSERT INTO orders VALUES (?,?,?,?)", orders)

    complaints = generate_complaints(orders)
    cur.executemany("INSERT INTO complaints VALUES (?,?,?,?,?)", complaints)

    conn.commit()

    # Verification
    for table in ["customers", "products", "orders", "complaints"]:
        count = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {count} rows")

    conn.close()
    print(f"\nDatabase saved to {DB_PATH}")


if __name__ == "__main__":
    print("Creating refund system database...")
    create_database()
    print("Done!")
