import os, pandas as pd, joblib, json, logging,random
from datetime import datetime
from database.db import insert_request, insert_qna, insert_response
import numpy as np

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
    print("\n===============================")
    print("🧭 FLOW TYPE:", flow_type)
    print("🔍 RAW QUESTION → ANSWER MAPPING (before encoding):")
    for k, v in row.items():
        print(f"{k:30} : {v}")

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
    pred, probas = None, None  # <-- initialize early to avoid "possibly unbound" warning

    try:
        pred = model.predict(X)[0]
        probas = model.predict_proba(X)

        # --- Top-2 ring styles (using model.classes_[1] directly) ---
        if probas is not None:
            ring_probs = probas[1][0] if len(np.shape(probas[1])) > 1 else probas[1]

            sorted_idx = np.argsort(ring_probs)[::-1]
            top2_idx = list(dict.fromkeys(sorted_idx[:2]))
            if len(top2_idx) < 2:
                top2_idx.append(top2_idx[0])

            ring_classes = model.classes_[1]  # numeric 1–12 labels
            predicted_ring_key = int(ring_classes[top2_idx[0]])
            second_key = int(ring_classes[top2_idx[1]]) if len(top2_idx) > 1 else predicted_ring_key
        else:
            predicted_ring_key = int(pred[1])
            second_key = predicted_ring_key

    except Exception as e:
        logging.error(f"❌ Prediction failed, using random fallback: {e}")
        pred = [random.randint(1, 16), random.randint(1, 12)]
        predicted_ring_key = random.randint(1, 12)
        second_key = random.randint(1, 12)

    # 🚫 Ensure two different ring keys
    if second_key == predicted_ring_key:
        all_keys = list(model.classes_[1])
        for alt_key in all_keys:
            if alt_key != predicted_ring_key:
                second_key = int(alt_key)
                break


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

    # If prediction key missing, fallback to random valid key
    def safe_get_ring(key):
        if key in key_to_style:
            return key_to_style[key]
        else:
            random_key = random.choice(list(key_to_style.keys()))
            logging.warning(f"⚠️ Ring key {key} not found. Using random fallback {random_key}.")
            return key_to_style[random_key]

    # Get ring1 and ring2 safely
    ring1 = safe_get_ring(predicted_ring_key)
    ring2 = safe_get_ring(second_key)


    # 7️⃣ TRIP + BYOR LOGIC (key-based + mapping aware)
    trip_qid = "Q10" if flow_type == "Self" else "Q11"
    trip_question = qmap.get(trip_qid, "Trip Preference").lower()

    trip_scratch = False
    for a in answers:
        qid = a.get("id", "").strip().upper()
        qtext = (a.get("text") or "").strip().lower()
        if qid == trip_qid or trip_question in qtext:
            sel = a.get("selectedOption", {})
            val = str(sel.get("value", "")).strip().lower()
            if any(keyword in val for keyword in [
                "from scratch", "plan everything", "plans everything", "organize myself",
                "he plans everything", "you plan everything"
            ]):
                trip_scratch = True
            break

    # 2️⃣ Define numeric keys for BYOR ring styles (from pred_key.json)
    BYOR_KEYS = {2, 7, 9, 11}  # SOLITAIRE=2, CLUSTER=7, HALO=9, WEDDING SET=11

    # 3️⃣ Check if both rings are BYOR type using numeric keys
    is_ring1_byor = predicted_ring_key in BYOR_KEYS
    is_ring2_byor = second_key in BYOR_KEYS

    # 4️⃣ Final logic for BYOR / STANDARD section
    section = "BYOR" if trip_scratch and is_ring1_byor and is_ring2_byor else "STANDARD"


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
