from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from chatbot import new_session, update_slots, dialogue
from response_mapper import ResponseMapper
import os, re, random

app    = Flask(__name__, static_folder="static")
CORS(app)
mapper = ResponseMapper()
sessions: dict = {}

SIZE_LETTERS    = {"xs","s","m","l","xl","xxl","28","30","32","34","36","38","40","42"}
CONFIRM_PHRASES = {"lets confirm","confirm order","confirm it","place it",
                   "go ahead","proceed","book it","yes place","yes confirm","yes order"}
GREETING_WORDS  = {"hi","hello","hey","hlo","hii","sup","yo","howdy"}
GOODBYE_WORDS   = {"bye","goodbye","cya","tata","see you","take care"}
THANKS_WORDS    = {"thanks","thank you","thx","ty","thak","thnak","thbak","thanku","thank"}
PRODUCTS        = {"shirt","shirts","kurti","kurtis","saree","jeans","hoodie","dress",
                   "top","kurta","trouser","pant","jacket","leggings","skirt","lehenga",
                   "salwar","shawl","blazer","ethnic","western","clothes","outfit"}
ACKS            = {"okay","ok","sure","fine","alright","got it","noted","k","cool","yep","yeah"}

UNKNOWN_RESPONSES = [
    "I did not get that. Ask about products, prices, sizes or orders.",
    "Could you rephrase? I can help with availability, pricing, delivery and more.",
    "Not sure about that. Try asking about a product, size or your order."
]
THANKS_RESPONSES = [
    "You are welcome! Anything else I can help with?",
    "Happy to help! Let me know if you need anything else.",
    "Glad I could help! Feel free to ask anytime."
]
ACK_RESPONSES = [
    "Great! Is there anything else I can help you with?",
    "Got it! Let me know if you need anything else.",
    "Sure! Anything else?"
]

def hard_rule(msg: str, session: dict):
    t     = msg.lower().strip()
    words = set(t.split())

    # Greeting
    if words & GREETING_WORDS:
        return "greeting", 1.0

    # Thanks
    if any(w in t for w in THANKS_WORDS):
        return "thanks", 1.0

    # Goodbye
    if words & GOODBYE_WORDS:
        return "goodbye", 1.0

    # Confirm order
    if any(p in t for p in CONFIRM_PHRASES):
        return "place_order", 1.0

    # "size M" or "M size" or just "M" — any size mention
    size_match = re.search(r'\b(xs|s|m|l|xl|xxl|\d{2})\b', t)
    if size_match:
        # If bot asked for size OR message contains word "size"
        if session.get("stage") == "ask_size" or "size" in t:
            return session.get("intent", "size_inquiry"), 1.0
        # Single size letter when in order/availability flow
        if t in SIZE_LETTERS and session.get("intent") in ("place_order","availability_check","size_inquiry"):
            return session["intent"], 1.0
        if t in SIZE_LETTERS:
            return "size_inquiry", 1.0

    # Acknowledgement
    if t in ACKS:
        stage = session.get("stage")
        if stage and stage != "order_done":
            return session.get("intent","unknown"), 1.0
        return "ack", 1.0

    # Single product word
    if t in PRODUCTS or (len(words) <= 2 and words & PRODUCTS):
        if session.get("stage") == "ask_product":
            return session.get("intent","product_browsing"), 1.0
        return "product_browsing", 0.9

    return None, None


@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data       = request.json
    user_msg   = data.get("message","").strip()
    session_id = data.get("session_id","default")

    if not user_msg:
        return jsonify({"response":"Please say something!","intent":"unknown","confidence":0})

    if session_id not in sessions:
        sessions[session_id] = new_session()
    session = sessions[session_id]

    update_slots(session, user_msg)

    # 1. Hard rules
    intent, conf = hard_rule(user_msg, session)

    # 2. ML fallback
    if intent is None:
        intent, conf, _ = mapper.classify(user_msg)
        conf_pct = conf * 100
        if conf_pct < 45.0:
            intent = session["intent"] if session.get("stage") else "unknown"
            conf   = 0.23
        else:
            session["intent"] = intent
    else:
        if intent not in ("unknown","ack","thanks","goodbye"):
            session["intent"] = intent

    # 3. Build response
    if intent == "unknown":
        response = random.choice(UNKNOWN_RESPONSES)
    elif intent == "thanks":
        response = random.choice(THANKS_RESPONSES)
    elif intent == "ack":
        response = random.choice(ACK_RESPONSES)
    else:
        response = dialogue(intent, session, mapper, raw=user_msg)

    return jsonify({
        "response":   response,
        "intent":     intent,
        "confidence": round(conf * 100 if conf <= 1.0 else conf, 1),
        "slots": {
            "product":  session["product"],
            "size":     session["size"],
            "color":    session["color"],
            "order_id": session["order_id"],
        }
    })

@app.route("/reset", methods=["POST"])
def reset():
    data = request.json
    sessions[data.get("session_id","default")] = new_session()
    return jsonify({"status":"reset"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",5000)), debug=False)