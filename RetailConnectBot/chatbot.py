from response_mapper import ResponseMapper

# ─────────────────────────────────────────────────────────────────
# chatbot.py — Advanced Hybrid Chatbot
#
# Architecture:
#   spaCy NER       → smart entity extraction
#   MLP (yours)     → primary intent classification
#   Sentence-BERT   → semantic fallback for low confidence
#   Session memory  → multi-turn context
#   Dialogue mgr    → conversation flow
# ─────────────────────────────────────────────────────────────────

# ── spaCy entity extractor ────────────────────────────────────────
_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
            print("spaCy loaded.")
        except Exception:
            print("spaCy not available. Using fallback extractor.")
            _nlp = False
    return _nlp

# Retail-specific keyword lists (spaCy supplements these)
PRODUCTS = ["kurti","saree","shirt","jeans","hoodie","dress","top","kurta",
            "trouser","pant","jacket","leggings","skirt","lehenga","salwar"]
SIZES    = ["xs","s","m","l","xl","xxl","small","medium","large",
            "28","30","32","34","36","38","40","42"]
COLORS   = ["black","white","blue","red","green","yellow","pink",
            "orange","purple","grey","gray","brown","navy","maroon"]

FILLER_WORDS = {"yes","no","okay","ok","sure","fine","alright","yep","nope",
                "yeah","got it","thanks","thank you","no change","noted"}


def new_session():
    return {
        "intent":   None,
        "product":  None,
        "size":     None,
        "color":    None,
        "order_id": None,
        "stage":    None,
        "address":  None,
    }


def extract_entities(text: str) -> dict:
    """
    spaCy-powered entity extraction with keyword fallback.
    spaCy handles: plurals, typos, informal text
    Fallback handles: retail-specific terms spaCy misses
    """
    import re
    t     = text.lower()
    words = t.split()

    product = color = size = order_id = None

    # ── Try spaCy first ──
    nlp = get_nlp()
    if nlp:
        doc = nlp(text)
        for ent in doc.ents:
            val = ent.text.lower()
            if ent.label_ in ("PRODUCT", "ORG", "WORK_OF_ART"):
                matched = next((p for p in PRODUCTS if p in val), None)
                if matched: product = matched
            if ent.label_ == "ORDINAL" or re.match(r'ord\w*\d+|\d{5,}', val):
                order_id = val.upper()

    # ── Keyword fallback for retail terms spaCy doesn't know ──
    if not product:
        # handle plurals: "shirts" → "shirt"
        for p in PRODUCTS:
            if p in t or p+'s' in t:
                product = p; break

    # size: exact word match (avoid "small" matching "smallest")
    size = next((s for s in SIZES if s in words), None)

    # color: substring match
    color = next((c for c in COLORS if c in t), None)

    # order ID: pattern match
    if not order_id:
        m = re.search(r'\b(ord\w*\d+|\d{5,})\b', t)
        if m: order_id = m.group(0).upper()

    return {"product": product, "size": size,
            "color": color, "order_id": order_id}


def update_slots(session: dict, text: str):
    for k, v in extract_entities(text).items():
        if v: session[k] = v


# ── Dialogue Manager ──────────────────────────────────────────────
def dialogue(intent: str, session: dict, mapper, raw: str = "") -> str:
    s = session
    t = raw.lower().strip()

    # ── Active stage: bot is waiting for info ──
    if s["stage"] == "ask_product":
        if s["product"]:
            s["stage"] = None
            return dialogue(s["intent"], s, mapper, raw)
        return "Which product? (e.g. shirt, kurti, jeans, hoodie...)"

    if s["stage"] == "ask_size":
        if s["size"]:
            s["stage"] = None
            return dialogue(s["intent"], s, mapper, raw)
        return "What size? (XS / S / M / L / XL / XXL)"

    if s["stage"] == "ask_address":
        if t and t not in FILLER_WORDS:
            s["address"] = raw
            s["stage"]   = None
            c = f"{s['color']} " if s.get("color") else ""
            return (f"Order confirmed! {c}{s['product']} size {s['size']} "
                    f"will be delivered to {s['address']}. "
                    f"You'll receive a confirmation shortly!")
        return "Please share your delivery address."

    if s["stage"] == "ask_order_id":
        if s["order_id"]:
            s["stage"] = None
            return dialogue(s["intent"], s, mapper, raw)
        return "Please share your Order ID (e.g. ORD12345)"

    # ── "yes/sure" after availability = wants to order ──
    if t in {"yes","yeah","yep","sure","ok","okay"} and s["intent"] == "availability_check":
        intent = "place_order"
        s["intent"] = intent

    # ── Intent handlers ──────────────────────────────────────────
    c_str = f"{s['color']} " if s.get("color") else ""

    if intent == "availability_check":
        if not s["product"]: s["stage"] = "ask_product"; return "Which product are you checking?"
        if not s["size"]:    s["stage"] = "ask_size";    return f"What size for {s['product']}?"
        return f"Yes! {c_str}{s['product']} in size {s['size']} is available. Want to order?"

    elif intent == "price_inquiry":
        if not s["product"]: s["stage"] = "ask_product"; return "Which product's price do you want?"
        return mapper.get_response_by_intent("price_inquiry", s)["response"]

    elif intent == "size_inquiry":
        if not s["product"]: s["stage"] = "ask_product"; return "Which product?"
        if not s["size"]:    s["stage"] = "ask_size";    return f"{s['product'].capitalize()} comes in XS, S, M, L, XL, XXL. Which size do you need?"
        return f"{s['product'].capitalize()} size {s['size']} is available!"

    elif intent == "color_inquiry":
        if not s["product"]: s["stage"] = "ask_product"; return "Which product?"
        if not s["color"]:   return f"{s['product'].capitalize()} comes in black, white, blue, red, pink & more. Which color?"
        return f"{s['color'].capitalize()} {s['product']} is available! What size do you need?"

    elif intent == "place_order":
        if not s["product"]: s["stage"] = "ask_product"; return "What would you like to order?"
        if not s["size"]:    s["stage"] = "ask_size";    return f"What size {s['product']} do you need?"
        s["stage"] = "ask_address"
        return f"Great choice! {c_str}{s['product']} size {s['size']} — please share your delivery address."

    elif intent in ("order_status","order_modification","order_cancellation",
                    "return_request","exchange_request"):
        if not s["order_id"]:
            s["stage"] = "ask_order_id"
            return "Please share your Order ID (e.g. ORD12345) and I'll look it up right away."
        return mapper.get_response_by_intent(intent, s)["response"]

    else:
        return mapper.get_response_by_intent(intent, s)["response"]


# ── Main Loop ─────────────────────────────────────────────────────
def run():
    print("=" * 55)
    print("  🛍️  STYLEBOT — Hybrid NLP Chatbot")
    print("  MLP + spaCy + Sentence-BERT")
    print("  type 'quit' to exit")
    print("=" * 55)

    mapper  = ResponseMapper()
    session = new_session()
    CONF    = 0.45
    print()

    while True:
        raw = input("You: ").strip()
        if not raw: continue
        if raw.lower() in ["quit", "exit"]:
            print("Bot: Thanks for visiting! Have a great day 👋")
            break

        # 1. Extract entities via spaCy
        update_slots(session, raw)

        # 2. Classify intent
        if raw.lower() in FILLER_WORDS and session["intent"]:
            intent = session["intent"]
            source = "filler→context"
            conf   = 1.0
        else:
            intent, conf, source = mapper.classify(raw)

            # Low conf + active stage = user answering bot's question
            if conf < CONF and session["stage"]:
                intent = session["intent"]
                source = f"stage({session['stage']})"
            elif conf >= CONF:
                session["intent"] = intent

        # 3. Dialogue manager
        response = dialogue(intent, session, mapper, raw=raw)

        print(f"Bot: {response}")
        print(f"     [{source} | intent:{intent} | conf:{conf*100:.1f}% | "
              f"product:{session['product']} size:{session['size']} "
              f"color:{session['color']} stage:{session['stage']}]")
        print()

        if intent == "goodbye": break


if __name__ == "__main__":
    run()