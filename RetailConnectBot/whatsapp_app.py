from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from response_mapper import ResponseMapper
from chatbot import new_session, update_slots, dialogue
import re, random

app    = Flask(__name__)
mapper = ResponseMapper()
user_sessions: dict = {}
CONF_THRESHOLD = 45.0

SIZE_LETTERS    = {"xs","s","m","l","xl","xxl","28","30","32","34","36","38","40","42"}
CONFIRM_PHRASES = {"lets confirm","confirm order","place it","go ahead","proceed","book it"}
GREETING_WORDS  = {"hi","hello","hey","hlo","hii","sup","yo"}
THANKS_WORDS    = {"thanks","thank you","thx","ty","thanku","thbak"}
ACKS            = {"okay","ok","sure","fine","alright","k","cool","yep","yeah"}

def hard_rule(msg, session):
    t     = msg.lower().strip()
    words = set(t.split())
    if words & GREETING_WORDS:              return "greeting", 1.0
    if any(w in t for w in THANKS_WORDS):   return "thanks", 1.0
    if any(p in t for p in CONFIRM_PHRASES):return "place_order", 1.0
    size_match = re.search(r'\b(xs|s|m|l|xl|xxl|\d{2})\b', t)
    if size_match:
        if session.get("stage") == "ask_size" or "size" in t:
            return session.get("intent","size_inquiry"), 1.0
        if t in SIZE_LETTERS and session.get("intent") in ("place_order","availability_check","size_inquiry"):
            return session["intent"], 1.0
        if t in SIZE_LETTERS:
            return "size_inquiry", 1.0
    if t in ACKS:
        if session.get("stage") and session["stage"] != "order_done":
            return session.get("intent","unknown"), 1.0
        return "ack", 1.0
    return None, None

@app.route("/webhook", methods=["POST"])
def webhook():
    user_msg = request.form.get("Body", "").strip()
    sender   = request.form.get("From", "unknown")

    if sender not in user_sessions:
        user_sessions[sender] = new_session()
    session = user_sessions[sender]

    update_slots(session, user_msg)

    intent, conf = hard_rule(user_msg, session)

    if intent is None:
        intent, conf, _ = mapper.classify(user_msg)
        if conf * 100 < CONF_THRESHOLD:
            intent = session["intent"] if session.get("stage") else "unknown"
        else:
            session["intent"] = intent
    else:
        if intent not in ("unknown","ack","thanks"):
            session["intent"] = intent

    if intent == "unknown":
        response = "I did not get that. Ask about products, prices, sizes or your orders."
    elif intent == "thanks":
        response = "You are welcome! Anything else I can help with?"
    elif intent == "ack":
        response = "Got it! Anything else I can help with?"
    else:
        response = dialogue(intent, session, mapper, raw=user_msg)

    resp = MessagingResponse()
    resp.message(response)
    return str(resp)

if __name__ == "__main__":
    app.run(port=5000, debug=False)