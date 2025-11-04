import os, pandas as pd, joblib, json, logging,random
from datetime import datetime
from database.db import insert_request, insert_qna, insert_response

# -------------------- ASSETS --------------------
ASSETS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))
ANSWER_KEY = os.path.join(ASSETS, "answer_key.json")
MBTI_WIKI = os.path.join(ASSETS, "mbti_wiki.json")
PRED_KEY = os.path.join(ASSETS, "pred_key.json")
QNA_PATH = os.path.join(ASSETS, "qna1.json")

MODEL_PATH = "model/multioutput_model.pkl"
ENC_PATH = "model/label_encoders.pkl"

# -------------------- LOAD STATIC DATA --------------------
model = joblib.load(MODEL_PATH)
enc = joblib.load(ENC_PATH)
with open(QNA_PATH, "r", encoding="utf-8") as f:
    QNA_DATA = json.load(f)

FEATURES = [
    "Who are you purchasing for?", "Gender", "Relation", "Profession", "Occasion",
    "Purpose", "Typical Day", "Weekend Preference", "Work Dress", "Social Dress",
    "Trip Preference", "Waiting Line", "Artwork Response", "Mother Response", "Last Minute Plans"
]

# -------------------- HELPERS --------------------
def get_answer_mapping():
    """Reads answer_key.json and returns normalized mapping."""
    with open(ANSWER_KEY, "r", encoding="utf-8") as f:
        answer_map = json.load(f)

    if isinstance(answer_map, list):
        return {
            str(item.get("answer") or item.get("Answer", "")).strip():
            str(item.get("option") or item.get("Option", "")).strip()
            for item in answer_map
        }
    return answer_map


# -------------------- MAIN FUNCTION --------------------
def predict_quiz(payload):
    meta, answers = payload["meta"], payload["answers"]

    # 1️⃣ FLOW DETECTION
    first_answer = next((a for a in answers if a.get("id") == "Q1"), None)
    purchase_val = first_answer.get("selectedOption", {}).get("value", "Self").strip().lower() if first_answer else "self"
    flow_type = "Self" if purchase_val == "self" else "Others"
    logging.info(f"🧭 Flow detected: {flow_type}")

    # 2️⃣ NORMALIZE ANSWERS
    mapping = get_answer_mapping()
    row = {f: "-1" for f in FEATURES}

    from pathlib import Path
    ASSETS = Path(__file__).resolve().parents[1] / "assets"
    QMAP_PATH = ASSETS / "question_mapping.json"
    with open(QMAP_PATH, "r", encoding="utf-8") as f:
        qmap_all = json.load(f)
    qmap = qmap_all.get(flow_type, {})

    QUESTION_ALIAS = {
        "How would you describe your typical day?": "Typical Day",
        "How would you describe her typical day?": "Typical Day",
        "How would you describe his typical day?": "Typical Day",
        "How do you prefer to spend your weekends?": "Weekend Preference",
        "How does she prefer to spend her weekends?": "Weekend Preference",
        "How does he prefer to spend his weekends?": "Weekend Preference",
        "How do you typically dress for work?": "Work Dress",
        "How does she typically dress for work?": "Work Dress",
        "How does he typically dress for work?": "Work Dress",
        "How do you typically dress for social events?": "Social Dress",
        "How does she typically dress for social events?": "Social Dress",
        "How does he typically dress for social events?": "Social Dress",
        "On a trip, how do you like to organize things?": "Trip Preference",
        "On a trip, how does she like to organize things?": "Trip Preference",
        "On a trip, how does he like to organize things?": "Trip Preference",
        "You are waiting in a long line": "Waiting Line",
        "She is waiting in a long line": "Waiting Line",
        "He is waiting in a long line": "Waiting Line",
        "Whenever you see a piece of artwork": "Artwork Response",
        "Whenever she sees a piece of artwork": "Artwork Response",
        "Whenever he sees a piece of artwork": "Artwork Response",
        "Your mother is not feeling well & you have an important meeting to attend": "Mother Response",
        "She has an important meeting to attend but her mother is unwell": "Mother Response",
        "He has an important meeting to attend but his mother is unwell": "Mother Response",
        "Last minute plans": "Last Minute Plans",
        "How does she react to last minute plans?": "Last Minute Plans",
        "How does he react to last minute plans?": "Last Minute Plans"
    }
    for a in answers:
        qid = a.get("id")
        raw_text = a.get("text") or ""
        selected = a.get("selectedOption", {}).get("value", "")

        # 1️⃣ Prefer canonical mapping from question_mapping.json by ID
        question = qmap.get(qid)    

        # 2️⃣ Fallback: try alias on the raw text (legacy safety)
        if not question and raw_text:
            question = QUESTION_ALIAS.get(raw_text, raw_text)

        # 3️⃣ If we still don't have a usable question, skip
        if question not in row:
            logging.debug(
                f"Skipping answer: id={qid}, raw_text='{raw_text}', resolved='{question}'"
            )
            continue

        # 4️⃣ Normalize answer text for lookup in answer_key.json
        ans = (selected or "").strip()
        if not ans:
            row[question] = "-1"
            continue

        # Try direct match
        mapped = mapping.get(ans)

        # Try normalized version (& → and) if not found
        if mapped is None:
            norm_ans = ans.replace("&", "and")
            mapped = mapping.get(norm_ans)

        # Final fallback: if still not found, use the raw text itself
        row[question] = str(mapped if mapped is not None else ans or "-1")

    #FOR TESTING : QUESTIONS AND ANSWER MAPPING
    # print("\n===============================")
    # print("🧭 FLOW TYPE:", flow_type)
    # print("🔍 RAW QUESTION → ANSWER MAPPING (before encoding):")
    # for k, v in row.items():
    #     print(f"{k:30} : {v}")

    # print("-------------------------------")
    # print("🧾 ANSWER KEY MAPPING EXAMPLE (from answer_key.json):")
    # sample_items = list(mapping.items())[:10]
    # for ans_text, code in sample_items:
    #     print(f"  '{ans_text}' → {code}")

    # print("===============================\n")


    # 3️⃣ ENCODING
    X = pd.DataFrame([row])
    for c in X.columns:
        le = enc[c]
        X[c] = X[c].apply(lambda v: le.transform([v])[0] if v in le.classes_ else le.transform(["-1"])[0])

    # 4️⃣ PREDICTION
    try:
        pred = model.predict(X)[0]
        probas = model.predict_proba(X)
    except Exception as e:
        logging.error(f"❌ Prediction failed, using random fallback: {e}")
        pred = [random.randint(1, 16), random.randint(1, 12)]
        probas = None

    # --- Top-2 ring styles ---
    if isinstance(probas, list) and len(probas) > 1:
        ring_probs = probas[1][0]
        sorted_idx = ring_probs.argsort()[::-1]

        # Pick top 2 *unique* indices
        top2_idx = []
        for idx in sorted_idx:
            if idx not in top2_idx:
                top2_idx.append(idx)
            if len(top2_idx) == 2:
                break
    else:
        top2_idx = [int(pred[1]), int(pred[1])]

    ring_label_encoder = enc.get("purchased_ring_style_key")
    if ring_label_encoder:
        top2_ring_keys = ring_label_encoder.inverse_transform(top2_idx)
    else:
        top2_ring_keys = [int(pred[1]), int(pred[1])]

    # Ensure two numeric ring keys are always defined
    predicted_ring_key = int(top2_ring_keys[0])
    second_key = int(top2_ring_keys[1] if len(top2_ring_keys) > 1 else top2_ring_keys[0])



    # 5️⃣ DECODE MBTI USING WIKI FILE (NUMERIC → TEXT)
    with open(MBTI_WIKI, "r", encoding="utf-8") as f:
        wiki = json.load(f)

    predicted_mbti_key = int(pred[0])
    match = next((item for item in wiki if item["key_val"] == predicted_mbti_key), None)

    if not match:
        match = random.choice(wiki)  # fallback random MBTI
        logging.warning(f"⚠️ Unknown MBTI key {predicted_mbti_key}, using random fallback {match['MBTI Personality']}")

    mbti_code = match["MBTI Personality"]
    info = {
        "Personality": match["Personality"],
        "Description": match["Description"],
        "Known Personalities (Male)": match["Known Personalities (Male)"],
        "Known Personalities (Female)": match["Known Personalities (Female)"]
    }

    # 6️⃣ RING STYLE LOOKUP
    with open(PRED_KEY, "r", encoding="utf-8") as f:
        pred_map = json.load(f)
    key_to_style = {int(item["Key_val"]): item for item in pred_map}

    ring1 = key_to_style.get(predicted_ring_key, {
        "Purchased Ring Style": "Unknown Style", "img_link": None, "link": None
    })
    ring2 = key_to_style.get(second_key, {
        "Purchased Ring Style": "Unknown Style", "img_link": None, "link": None
    })

    # 🚫 Prevent duplicate ring styles
    if ring2["Purchased Ring Style"] == ring1["Purchased Ring Style"]:
        for k, v in key_to_style.items():
            if v["Purchased Ring Style"] != ring1["Purchased Ring Style"]:
                ring2 = v
                break

    # 7️⃣ TRIP + BYOR LOGIC
    trip_question_texts = {
        "on a trip, how do you like to organize things?",
        "on a trip, how does she like to organize things?",
        "on a trip, how does he like to organize things?",
        "trip",
        "trip preference"
    }

    trip_scratch = False
    for a in answers:
        qtext = (a.get("text") or "").strip().lower()
        if any(q in qtext for q in trip_question_texts):
            sel = a.get("selectedOption", {})
            val = str(sel.get("value", "")).strip().lower()
            # Broaden the matching to all natural variants
            if any(keyword in val for keyword in [
                "from scratch", "plan everything", "plans everything", "organize myself",
                "he plans everything", "you plan everything"
            ]):
                trip_scratch = True
            break

    byor_tokens = {"solitaire", "halo", "cluster", "wedding set"}
    def is_byor_style(name): return any(tok in (name or "").lower() for tok in byor_tokens)
    ring1_name, ring2_name = ring1["Purchased Ring Style"].lower(), ring2["Purchased Ring Style"].lower()

    section = "BYOR" if trip_scratch and (is_byor_style(ring1_name) and is_byor_style(ring2_name)) else "STANDARD"

    # 8️⃣ FINAL RESPONSE
    response = {
        "request_id": meta["request_id"],
        "mbti_personality_type": info["Personality"],
        "mbti_personality_code": mbti_code,
        "description": info["Description"],
        "similar_personality_1": info["Known Personalities (Male)"],
        "similar_personality_2": info["Known Personalities (Female)"],
        "purchased_ring_style": None,
        "collection_1": {
            "name": ring1["Purchased Ring Style"],
            "image": ring1["img_link"],
            "link": ring1["link"]
        },
        "collection_2": {
            "name": ring2["Purchased Ring Style"],
            "image": ring2["img_link"],
            "link": ring2["link"]
        },
        "section": section
    }
    try:
        insert_qna(meta["request_id"], answers)
        insert_request(meta)
        insert_response(response)
    except Exception as e:
        logging.error(f"❌ DB insert failed: {e}")
    return response
