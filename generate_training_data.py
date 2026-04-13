"""
generate_training_data.py
─────────────────────────
Creates a balanced synthetic training dataset (450 samples) for the
refund-approval classifier.

Label mapping
─────────────
  APPROVE (1) → High severity   (product physically unusable / broken)
  REJECT  (0) → Medium severity (unreliable but usable)
  REJECT  (0) → Low severity    (minor annoyance)

NOTE: NO vague / change-of-mind text is included here.
      Those cases were already filtered at Block [1] / [2].
"""

import csv
import random
import os

random.seed(42)

OUTPUT_PATH = "training_data.csv"

# ── 1. Raw sentence pools (factual defect complaints only) ─────────────────

# HIGH severity → label = 1 (APPROVE)
HIGH_TEMPLATES = [
    # Cracked / physical damage
    "The {product} screen was cracked when I opened the package.",
    "The {product} has a massive crack across the display. Completely unusable.",
    "Received a {product} with a shattered screen. Cannot use it at all.",
    "The {product} display cracked on arrival. The device is totally broken.",
    "My {product} screen cracked after one small drop. Everything is unreadable.",
    "The {product} arrived with a broken screen. Cannot power it on.",
    "Screen is badly cracked on my new {product}. No way to operate it.",
    "The {product} casing is cracked and the internal components are exposed.",
    "Dead on arrival: the {product} screen is cracked and shows nothing.",
    "The {product} was delivered cracked and completely non-functional.",

    # Dead / won't power on
    "My {product} is completely dead. It won't turn on no matter what I try.",
    "The {product} stopped working after 2 days. Completely dead, not charging.",
    "The {product} won't power on at all. No lights, no response.",
    "My {product} went completely dead overnight. Nothing happens when I press power.",
    "The {product} is dead on arrival. Pressing the power button does nothing.",
    "My {product} is completely unresponsive. It is dead and will not start.",
    "Bought a {product} and it is totally dead. Cannot turn it on at all.",
    "The {product} died completely after 3 days of use. No charging, no response.",

    # Broken / not working
    "The {product} broke within one week of purchase. It won't turn on anymore.",
    "My {product} is completely broken. Nothing works and it won't start.",
    "The {product} arrived broken. The power button does not respond at all.",
    "My {product} is completely broken. The screen is black and there is no power.",
    "The {product} stopped working entirely. I cannot use it for anything now.",
    "The {product} is not working at all. It is completely non-functional.",
    "My {product} spontaneously stopped working. Nothing turns on.",
    "The {product} is completely dead and cannot be repaired. Hardware failure.",
    "My {product} hardware failed completely. It no longer turns on or responds.",
    "The {product} has a severe hardware failure. It is entirely non-functional.",
    "The {product} fell apart within days. It is unusable now.",
    "Severe damage to my {product}. The screen is shattered and won't turn on.",
    "The {product} is physically destroyed. Could not use it even one day.",
    "My {product} arrived and would not boot. Completely dead unit.",
    "The {product} has catastrophic damage. Nothing works at all.",
]

# MEDIUM severity → label = 0 (REJECT)
MEDIUM_TEMPLATES = [
    # Overheating
    "The {product} overheats during normal use. It gets too hot to hold.",
    "My {product} is overheating badly. It burns my hand during use.",
    "The {product} runs very hot all the time. Overheating is a serious problem.",
    "The {product} overheats and then shuts down automatically. Not acceptable.",
    "My {product} overheats frequently. It is warm even when idle.",
    "The {product} overheats after 20 minutes of use. I have to put it down.",

    # Disconnecting / connectivity
    "The {product} disconnects from Bluetooth constantly. Cannot use it properly.",
    "My {product} keeps losing its Bluetooth connection every few minutes.",
    "The {product} drops Bluetooth intermittently. I have to reconnect every time.",
    "WiFi keeps dropping on my {product}. Have to reconnect every 15 minutes.",
    "The {product} disconnects from the network at least 5 times per hour.",
    "My {product} Bluetooth is unstable and disconnects without warning.",
    "The {product} keeps disconnecting from WiFi unexpectedly. Very frustrating.",
    "Bluetooth disconnects every few minutes on my {product}. Unusable for calls.",

    # Restarting / freezing but recoverable
    "The {product} restarts randomly several times a day.",
    "My {product} keeps restarting by itself. It does this multiple times daily.",
    "The {product} randomly restarts during use. Losing unsaved work constantly.",
    "My {product} freezes and then restarts unexpectedly. This is happening daily.",
    "The {product} reboots on its own without warning at least twice a day.",
    "My {product} is continuously restarting. Very difficult to use.",
    "The {product} freezes intermittently and then performs a forced reboot.",

    # Battery issues (degraded but device still turns on)
    "The {product} battery drains very fast. It barely lasts 2 hours.",
    "My {product} battery life is extremely poor. Dies within 90 minutes.",
    "The {product} charges very slowly and drains battery very quickly.",
    "Battery on my {product} depletes in 2 hours despite a full charge.",
    "The {product} battery health is very bad. Charging does not hold.",

    # Other medium issues
    "The {product} camera is blurry all the time. Every photo is out of focus.",
    "My {product} camera produces distorted images. Cannot use the camera at all.",
    "The {product} display has dead pixels scattered across the screen.",
    "There is a loud buzzing noise from my {product} at all times.",
    "The {product} speaker produces a constant buzzing sound when playing audio.",
]

# LOW severity → label = 0 (REJECT)
LOW_TEMPLATES = [
    # Slowness / lag
    "My {product} is slightly slow when loading apps. Minor performance issue.",
    "The {product} has a slight lag when switching between apps.",
    "There is a minor delay when typing on my {product}.",
    "The {product} takes a second or two longer to load than expected.",
    "My {product} is a bit slow at startup. Not a big issue but noticeable.",
    "Very minor lag on the {product} when opening apps. Otherwise fine.",
    "The {product} response time is slightly slower than advertised.",
    "My {product} has occasional minor slow-downs that are barely noticeable.",

    # Minor cosmetic issues
    "The {product} has a very small scratch on the back. Barely visible.",
    "There is a slight scuff on the corner of my {product}. Minor cosmetic issue.",
    "The {product} box was slightly dented, but the product itself is fine.",
    "My {product} has a minor blemish on the casing. Very small.",
    "Small cosmetic scratch on the {product} body. Functionality is perfect.",
    "The {product} has a tiny mark on the screen protector. Nothing major.",

    # Minor audio / display quirks
    "Slight audio delay on my {product} when watching videos. Minor issue.",
    "The {product} sometimes has a very minor echo on calls. Barely noticeable.",
    "The {product} display brightness flickers very briefly occasionally.",
    "There is a very slight colour tint on one corner of the {product} screen.",
    "The {product} occasionally shows a brief flicker on startup. Very rare.",
    "Minor audio imbalance on the {product}. One side is slightly louder.",

    # Touch / button minor issues
    "The {product} home button is slightly stiff but still clicks properly.",
    "My {product} touch screen has a very slight delay in one small area.",
    "The {product} volume button is a tiny bit loose. Everything works normally.",
    "The {product} charging port is slightly tight. Charges fine once connected.",
    "There is a minor creak in the {product} chassis. No functional issue.",
]

PRODUCTS = [
    "iPhone 14",
    "Samsung Galaxy S23",
    "OnePlus Nord CE 3",
    "iPad Air",
    "Galaxy Tab S9",
    "Dell Inspiron 15 Laptop",
    "Asus ROG Strix G15 Laptop",
    "Dell UltraSharp 27 Monitor",
    "JBL Tune 760NC Headphones",
    "Bose QuietComfort 45",
    "Sony WH-1000XM5 Headphones",
    "Logitech MX Master 3S Mouse",
    "Razer Viper Mini Mouse",
    "Corsair K70 RGB Keyboard",
    "JBL Flip 6 Speaker",
    "Anker PowerCore 20000",
    "Samsung Galaxy Watch 5",
    "Apple AirPods Pro 2",
]


def fill(template, product):
    return template.replace("{product}", product)


def generate_samples(templates, label, n):
    """Generate n samples from the given templates with random products."""
    samples = []
    while len(samples) < n:
        tmpl = random.choice(templates)
        prod = random.choice(PRODUCTS)
        text = fill(tmpl, prod)
        samples.append({"complaint_text": text, "label": label,
                         "severity": {1: "HIGH", 0: "MEDIUM_OR_LOW"}[label]})
    return samples


def main():
    # Aim for balanced: ~150 high, ~150 medium, ~150 low = 450 total
    high_samples   = generate_samples(HIGH_TEMPLATES,   label=1, n=150)
    medium_samples = generate_samples(MEDIUM_TEMPLATES, label=0, n=150)
    low_samples    = generate_samples(LOW_TEMPLATES,    label=0, n=150)

    all_samples = high_samples + medium_samples + low_samples
    random.shuffle(all_samples)

    fieldnames = ["complaint_text", "label", "severity"]
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_samples)

    print(f"Saved {len(all_samples)} samples to '{OUTPUT_PATH}'")
    label_counts = {1: 0, 0: 0}
    for s in all_samples:
        label_counts[s["label"]] += 1
    print(f"  APPROVE (1 / HIGH)  : {label_counts[1]}")
    print(f"  REJECT  (0 / MED+LO): {label_counts[0]}")


if __name__ == "__main__":
    main()
