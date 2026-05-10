import numpy as np
import json
import pickle
import re
from model import MLP

# ─────────────────────────────────────────────────────────────────
# response_mapper.py — Hybrid: MLP + Sentence-BERT fallback
#
# Flow:
#   1. MLP (TF-IDF)       → primary intent + confidence
#   2. If conf < threshold → Sentence-BERT semantic similarity
#   3. Best of both        → final intent
# ─────────────────────────────────────────────────────────────────

PRIMARY_THRESHOLD  = 0.45   # MLP confidence cutoff
SBERT_THRESHOLD    = 0.40   # minimum SBERT similarity to trust

# ── Example sentences per intent for SBERT matching ──────────────
# More examples = better semantic matching
INTENT_EXAMPLES = {
    "greeting": [
        "hi", "hello", "hey", "good morning", "good evening",
        "what's up", "hii", "helo", "sup"
    ],
    "goodbye": [
        "bye", "goodbye", "see you", "take care", "cya", "ok bye",
        "thanks bye", "that's all"
    ],
    "product_browsing": [
        "what do you have", "show me products", "what can I buy",
        "what's available", "browse your collection", "what do you sell",
        "show me everything", "what collections do you have"
    ],
    "price_inquiry": [
        "how much does it cost", "what is the price", "what's the rate",
        "how much for this", "price please", "cost of shirt",
        "how much is kurti", "what's the cost", "rate batao"
    ],
    "availability_check": [
        "is it available", "do you have this", "is it in stock",
        "can I get this", "is this there", "milega kya",
        "do you have blue shirt", "is M size available"
    ],
    "size_inquiry": [
        "what sizes do you have", "which size should I take",
        "size chart please", "available sizes", "size M hai kya",
        "do you have XL", "size guide", "which size fits me"
    ],
    "color_inquiry": [
        "what colors are available", "do you have red", "any other colors",
        "which color is good", "color options", "blue hai kya",
        "I want black one", "show me white"
    ],
    "place_order": [
        "I want to order", "I'll take it", "place my order", "buy this",
        "add to cart", "I want to buy", "yes I want it", "order karna hai",
        "book this for me", "I'll take the blue one"
    ],
    "order_status": [
        "where is my order", "track my order", "order status",
        "when will it arrive", "delivery update", "mera order kahan hai",
        "has it shipped", "order tracking"
    ],
    "order_cancellation": [
        "cancel my order", "I want to cancel", "don't want it anymore",
        "cancel karo", "please cancel", "order cancel kar do"
    ],
    "return_request": [
        "I want to return", "return this item", "how to return",
        "return policy", "wapas karna hai", "return please"
    ],
    "delivery_query": [
        "how long will delivery take", "delivery time", "when will I get it",
        "free delivery", "shipping charges", "how many days",
        "delivery to kerala", "ship to my address"
    ],
    "complaint": [
        "wrong item received", "damaged product", "quality is bad",
        "I have a complaint", "not satisfied", "wrong size delivered",
        "it's torn", "poor quality"
    ],
    "payment_methods": [
        "how can I pay", "payment options", "do you accept UPI",
        "can I pay cash", "COD available", "online payment",
        "GPay accepted", "payment methods"
    ],
}


def _p(slots):   return slots.get("product", "that item")
def _s(slots):   return slots.get("size", "your size")
def _c(slots):   return slots.get("color", "")
def _pcs(slots):
    parts = []
    if slots.get("color"):   parts.append(slots["color"])
    if slots.get("product"): parts.append(slots["product"])
    base = " ".join(parts) if parts else "that item"
    if slots.get("size"):    base += f" in {slots['size']}"
    return base

RESPONSES = {
    "greeting": [
        lambda s: "Hey! Welcome to our store. How can I help you today?",
        lambda s: "Hi there! Looking for something specific?",
        lambda s: "Hello! Ask me about products, prices, sizes or your orders.",
    ],
    "goodbye": [
        lambda s: "Bye! Come back anytime.",
        lambda s: "Thanks for visiting! Have a great day.",
        lambda s: "See you soon!",
    ],
    "product_browsing": [
        lambda s: "We have kurtis, sarees, shirts, jeans, hoodies, dresses and more — Rs.299 to Rs.2999. What interests you?",
        lambda s: "We carry Indian and Western wear for men, women and kids. What are you looking for?",
    ],
    "new_arrivals": [
        lambda s: f"New stock every week! Fresh {_p(s)} designs just arrived." if s.get("product") else "New arrivals this week across kurtis, dresses, shirts and more!",
        lambda s: "Latest collection just dropped — Indian and Western both. Anything specific?",
    ],
    "category_inquiry": [
        lambda s: f"Yes we carry {_p(s)}. What size and color?" if s.get("product") else "We have ethnic wear (kurti, saree, lehenga) and western wear (shirts, jeans, tops, dresses). What do you need?",
    ],
    "price_inquiry": [
        lambda s: f"The {_pcs(s)} is around Rs.{np.random.choice([499,699,899,999,1199,1499])}. Our range is Rs.299–Rs.2999." if s.get("product") else "Prices range from Rs.299 to Rs.2999. Which product are you asking about?",
        lambda s: f"{_p(s).capitalize()} starts from Rs.499. Which style do you want?" if s.get("product") else "What product are you asking about? I can give a specific price.",
    ],
    "discount_inquiry": [
        lambda s: f"We have offers on selected {_p(s)}s. Check our website for current deals." if s.get("product") else "We run seasonal sales. Any specific product?",
    ],
    "bulk_price_inquiry": [
        lambda s: f"Bulk discounts on {_p(s)} for orders above 5 pieces." if s.get("product") else "Wholesale rates for orders above 5 pieces. Which product and quantity?",
    ],
    "size_inquiry": [
        lambda s: f"{_pcs(s)} is available. Want to place an order?" if s.get("product") and s.get("size") else (f"We have {_p(s)} in XS to XXL. Which size?" if s.get("product") else "Which product and size are you checking?"),
    ],
    "size_guide_request": [
        lambda s: f"For {_p(s)}, share your measurements and I'll suggest a size." if s.get("product") else "Share chest, waist and hip measurements and I'll help pick your size.",
    ],
    "fit_inquiry": [
        lambda s: f"The {_p(s)} comes in slim and regular fit. Which do you prefer?" if s.get("product") else "We have slim, regular and relaxed fit. Which product?",
    ],
    "material_inquiry": [
        lambda s: f"The {_p(s)} is premium cotton — soft, breathable and durable." if s.get("product") else "We use cotton, silk, georgette and linen. Which product?",
    ],
    "color_inquiry": [
        lambda s: f"Yes {_p(s)} is available in {_c(s)}. Which size?" if s.get("product") and s.get("color") else (f"The {_p(s)} comes in many colors. Which shade?" if s.get("product") else "Which product and color are you looking for?"),
    ],
    "care_instructions": [
        lambda s: f"For {_p(s)}: machine wash cold, no bleach, tumble dry low." if s.get("product") else "Generally: cold wash, gentle cycle, no bleach. Which product?",
    ],
    "availability_check": [
        lambda s: f"Yes! {_pcs(s)} is in stock. Want to order?" if s.get("product") and s.get("size") else (f"{_pcs(s)} is available. Which size?" if s.get("product") else "Which product, color and size?"),
    ],
    "restock_inquiry": [
        lambda s: f"{_pcs(s)} should restock in 3–5 days." if s.get("product") else "We restock weekly. Which product are you waiting for?",
    ],
    "place_order": [
        lambda s: f"To order {_pcs(s)}, confirm your size and delivery address." if s.get("product") else "Which product, size and color would you like to order?",
    ],
    "order_status": [
        lambda s: f"Order {s['order_id']} should arrive in 3–7 days." if s.get("order_id") else "Share your Order ID and I'll check the status.",
    ],
    "order_modification": [
        lambda s: f"I can modify order {s['order_id']} within 24 hours. What needs to change?" if s.get("order_id") else "Share your Order ID and what you'd like to change.",
    ],
    "order_cancellation": [
        lambda s: f"Cancelling order {s['order_id']} now. Refund processed if payment was made." if s.get("order_id") else "Share your Order ID and I'll cancel it immediately.",
    ],
    "return_request": [
        lambda s: f"Returns accepted within 10 days. Initiating return for {s['order_id']}." if s.get("order_id") else "Returns within 10 days of delivery. Share your Order ID.",
    ],
    "exchange_request": [
        lambda s: f"Exchange for {s['order_id']} accepted. Which size or color do you want?" if s.get("order_id") else "Share your Order ID and preferred size/color.",
    ],
    "delivery_query": [
        lambda s: "Free delivery in Tamil Nadu. Other states vary. Arrives in 3–7 working days.",
        lambda s: "We deliver pan-India. Free shipping in Tamil Nadu. Standard 3–7 days.",
    ],
    "payment_methods": [
        lambda s: "We accept UPI, credit/debit cards, net banking and cash on delivery.",
        lambda s: "GPay, PhonePe, Paytm, cards and COD all accepted!",
    ],
    "complaint": [
        lambda s: f"Sorry about order {s['order_id']}. Share a photo and we'll resolve immediately." if s.get("order_id") else "Sorry to hear that! Share your Order ID and issue and we'll fix it right away.",
    ],
}

FALLBACK = [
    "Sorry, I didn't quite get that. Could you rephrase?",
    "Hmm, not sure about that. Try asking about price, size, delivery or your order.",
    "Could you be more specific? I can help with products, orders, delivery and returns.",
]

def render(template, slots: dict) -> str:
    if callable(template):
        return template(slots)
    return template

def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return re.sub(r'\s+', ' ', text)


class ResponseMapper:
    def __init__(self):
        # ── Load MLP ──
        with open("tfidf_vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)
        with open("label_map.json") as f:
            self.label_map = json.load(f)

        input_size = len(self.vectorizer.vocabulary_)
        self.model = MLP(input_size, 128, 64, len(self.label_map), lr=0.01)
        self.model.load("model_weights.npz")

        # ── Load Sentence-BERT ──
        self._sbert = None
        self._example_embeddings = None
        self._load_sbert()

        print("ResponseMapper ready (MLP + Sentence-BERT).")

    def _load_sbert(self):
        try:
            from sentence_transformers import SentenceTransformer
            print("Loading Sentence-BERT... (first time takes ~30s)")
            self._sbert = SentenceTransformer("all-MiniLM-L6-v2")
            self._build_example_embeddings()
            print("Sentence-BERT ready.")
        except ImportError:
            print("sentence-transformers not installed. Run: pip install sentence-transformers")
            print("Falling back to MLP only.")

    def _build_example_embeddings(self):
        """Pre-compute embeddings for all intent examples."""
        self._example_embeddings = {}
        for intent, examples in INTENT_EXAMPLES.items():
            embs = self._sbert.encode(examples, convert_to_numpy=True)
            self._example_embeddings[intent] = embs
        print(f"Precomputed embeddings for {len(INTENT_EXAMPLES)} intents.")

    def _cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

    def _sbert_classify(self, text: str) -> tuple[str, float]:
        """Find best intent via semantic similarity to examples."""
        if self._sbert is None or self._example_embeddings is None:
            return "unknown", 0.0

        query_emb = self._sbert.encode([text], convert_to_numpy=True)[0]

        best_intent = "unknown"
        best_score  = 0.0

        for intent, embs in self._example_embeddings.items():
            # max similarity across all examples for this intent
            scores = [self._cosine_similarity(query_emb, e) for e in embs]
            score  = float(max(scores))
            if score > best_score:
                best_score  = score
                best_intent = intent

        return best_intent, best_score

    def classify(self, user_input: str) -> tuple[str, float, str]:
        """
        Returns (intent, confidence, source)
        source = 'mlp' | 'sbert' | 'mlp+sbert'
        """
        cleaned = clean_text(user_input)

        # 1. MLP classification
        vector = self.vectorizer.transform([cleaned]).toarray()
        probs  = self.model.predict_proba(vector)[0]
        mlp_idx    = int(np.argmax(probs))
        mlp_conf   = float(probs[mlp_idx])
        mlp_intent = self.label_map[str(mlp_idx)]

        # 2. If MLP is confident → use it directly
        if mlp_conf >= PRIMARY_THRESHOLD:
            return mlp_intent, mlp_conf, "mlp"

        # 3. MLP not confident → try Sentence-BERT
        sbert_intent, sbert_score = self._sbert_classify(user_input)

        if sbert_score >= SBERT_THRESHOLD:
            # SBERT wins
            return sbert_intent, sbert_score, "sbert"

        # 4. Both uncertain → blend: pick whichever scored higher
        if mlp_conf >= sbert_score:
            return mlp_intent, mlp_conf, "mlp(low)"
        else:
            return sbert_intent, sbert_score, "sbert(low)"

    def get_response_by_intent(self, intent: str, slots: dict | None = None) -> dict:
        """Return a slot-filled response for a known intent."""
        slots     = slots or {}
        templates = RESPONSES.get(intent, FALLBACK)
        template  = templates[np.random.randint(len(templates))]
        response  = render(template, slots)
        return {"intent": intent, "confidence": 0.0, "response": response}