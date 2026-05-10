import pandas as pd
import random

# ─────────────────────────────────────────
# Fix B: Expand Training Dataset
# Adds ~300 new examples covering:
#  - Indian clothing (kurti, saree, salwar)
#  - Single-word product queries
#  - Size letters (S/M/L/XL)
#  - Hinglish / informal queries
# ─────────────────────────────────────────

NEW_EXAMPLES = {

    "product_inquiry": [
        "kurti", "kurti set", "show me kurtis", "do you have kurtis",
        "saree", "any sarees available", "salwar suit", "show salwar",
        "leggings", "any leggings", "show me tops", "palazzo pants",
        "ethnic wear", "casual wear", "western wear", "kurta for men",
        "any new arrivals", "latest collection", "trending clothes",
        "show me dresses", "party wear", "office wear", "cotton shirt",
        "formal pants", "casual t shirt", "printed shirt", "plain shirt",
        "what clothes do you have", "what do you sell", "your collection",
    ],

    "size_inquiry": [
        "M", "L", "XL", "S", "XXL", "XS",
        "size M available", "do you have L size", "XL size hai kya",
        "size chart", "show size chart", "what sizes do you have",
        "medium size", "large size", "small size", "extra large",
        "size 38", "size 40", "size 32", "size 34", "size 36",
        "plus size", "do you have plus size", "size for kurti",
        "shirt size M", "jeans size 32", "M size shirt",
        "L size available", "size XL hai", "kya XL milega",
    ],

    "availability_check": [
        "kurti available", "is kurti available", "kurti hai kya",
        "saree stock", "do you have saree", "salwar available",
        "shirt stock", "is shirt in stock", "hoodie available",
        "black jeans", "do you have black color", "red kurti",
        "blue shirt available", "white top", "green dress",
        "is this available", "stock hai kya", "milega kya",
        "available hai", "in stock", "out of stock",
        "cotton kurti available", "printed saree stock",
    ],

    "price_inquiry": [
        "kurti price", "kurti ka rate", "saree cost", "how much kurti",
        "shirt price", "price of shirt", "jeans kitne ka",
        "rate batao", "price batao", "cost kya hai",
        "kitna paisa", "how much does it cost", "what is cost",
        "cheap clothes", "affordable", "budget friendly option",
        "price range", "under 500", "under 1000", "under 2000",
        "discount price", "sale price", "offer price",
        "kurti kitne ki hai", "saree ka price", "pants price",
    ],

    "delivery_query": [
        "delivery kitne din", "how many days delivery",
        "express delivery", "same day delivery", "fast delivery",
        "delivery to chennai", "deliver to mumbai", "deliver karte ho",
        "shipping charge", "delivery fee", "free shipping",
        "delivery charges kya hai", "kab milega", "when will i get",
        "track order", "delivery time", "estimated delivery",
        "deliver to my city", "pan india delivery",
    ],

    "order_status": [
        "my order", "order status", "where is my package",
        "order track karo", "order nahi aaya", "order delayed",
        "order cancel", "cancel my order", "return order",
        "refund status", "exchange order", "order number check",
        "mera order kahan hai", "package status", "shipment status",
        "order confirm hua", "order placed", "order dispatched",
    ],

    "greeting": [
        "hi bro", "hey", "hlo", "hii", "helloo", "good morning",
        "good evening", "good afternoon", "namaste", "namaskar",
        "sup", "yo", "heyy", "hiiii", "hai", "hola",
    ],

    "goodbye": [
        "ok thanks", "thank you bro", "ok bye", "tata", "cya",
        "will come back", "ok will check", "noted thanks",
        "ok done", "thanks for help", "ok got it bye",
        "shukriya", "dhanyawad", "ok thank you",
    ],

    "material_inquiry": [
        "fabric", "what fabric", "cotton hai kya", "polyester",
        "pure cotton", "material kya hai", "fabric quality",
        "soft fabric", "comfortable material", "breathable fabric",
        "kurti material", "shirt fabric", "washing instructions",
        "how to wash", "dry clean only", "machine wash",
    ],
}

RESPONSES_MAP = {
    "product_inquiry":     "we've got a great range! what specifically are you looking for?",
    "size_inquiry":        "yes we have that size! which product?",
    "availability_check":  "let me check! which product and color?",
    "price_inquiry":       "jeans are rs. 999 to rs. 2499. great quality for the price!",
    "delivery_query":      "we use trusted couriers delhivery bluedart dtdc",
    "order_status":        "please share your order id and ill look it up",
    "greeting":            "hello welcome to our store how can i help you",
    "goodbye":             "bye come back anytime",
    "material_inquiry":    "premium quality fabric comfortable and durable",
}

def expand():
    # Load original
    df_orig = pd.read_csv("llm_level_dataset.csv")
    print(f"Original dataset: {len(df_orig)} rows")

    # Build new rows
    rows = []
    for intent, examples in NEW_EXAMPLES.items():
        for ex in examples:
            rows.append({
                "input":    ex,
                "intent":   intent,
                "response": RESPONSES_MAP[intent],
                "title":    intent.replace("_", " ").title()
            })

    df_new = pd.DataFrame(rows)
    print(f"New examples added: {len(df_new)} rows")

    # Merge and shuffle
    df_combined = pd.concat([df_orig, df_new], ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

    df_combined.to_csv("llm_level_dataset.csv", index=False)
    print(f"Final dataset: {len(df_combined)} rows → saved to llm_level_dataset.csv")
    print("\nNow run these in order:")
    print("  python preprocess.py")
    print("  python vectorizer.py")
    print("  python train.py")
    print("  python evaluate.py")

if __name__ == "__main__":
    expand()