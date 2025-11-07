import os, pandas as pd, joblib, json
from sqlalchemy import create_engine, text
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sqlalchemy import create_engine, text

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np, json, os, pandas as pd
from sqlalchemy import create_engine, text

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np, json, os, pandas as pd
from sqlalchemy import create_engine, text

pd.set_option('future.no_silent_downcasting', True)

# -------------------- DB CONFIG --------------------
DB_PARAMS = {
    'dbname': 'projectai-ringsandi-pict',
    'user': 'postgres',
    'password': 'Apollo11',
    'host': 'projectai-pict-postgresql.postgres.database.azure.com',
    'port': '5432'
}

# -------------------- ASSETS --------------------
ASSETS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))
ANSWER_KEY_JSON = os.path.join(ASSETS, "answer_key.json")
PRED_KEY_JSON = os.path.join(ASSETS, "pred_key.json")

# -------------------- SQL QUERY (Optimized) --------------------
SQL_QUERY = """
WITH qna_pivot AS (
  SELECT
    request_id,
    MAX(CASE WHEN question ILIKE '%Who are you purchasing for%' THEN answer END) AS "Who are you purchasing for?",
    MAX(CASE WHEN question ILIKE '%Gender%' THEN answer END) AS "Gender",
    MAX(CASE WHEN question ILIKE '%Relation%' THEN answer END) AS "Relation",
    MAX(CASE WHEN question ILIKE '%Profession%' THEN answer END) AS "Profession",
    MAX(CASE WHEN question ILIKE '%Occasion%' THEN answer END) AS "Occasion",
    MAX(CASE WHEN question ILIKE '%Purpose%' THEN answer END) AS "Purpose",
    MAX(CASE WHEN question ILIKE '%typical day%' THEN answer END) AS "Typical Day",
    MAX(CASE WHEN question ILIKE '%weekends%' THEN answer END) AS "Weekend Preference",
    MAX(CASE WHEN question ILIKE '%dress for work%' THEN answer END) AS "Work Dress",
    MAX(CASE WHEN question ILIKE '%dress for social%' THEN answer END) AS "Social Dress",
    MAX(CASE WHEN question ILIKE '%trip%' THEN answer END) AS "Trip Preference",
    MAX(CASE WHEN question ILIKE '%waiting in a long line%' THEN answer END) AS "Waiting Line",
    MAX(CASE WHEN question ILIKE '%artwork%' THEN answer END) AS "Artwork Response",
    MAX(CASE WHEN question ILIKE '%mother%' THEN answer END) AS "Mother Response",
    MAX(CASE WHEN question ILIKE '%Last minute%' THEN answer END) AS "Last Minute Plans"
  FROM public.requests_qna
  GROUP BY request_id
)
SELECT
  COALESCE(q."Who are you purchasing for?", '-1') AS "Who are you purchasing for?",
  COALESCE(q."Gender", '-1') AS "Gender",
  COALESCE(q."Relation", '-1') AS "Relation",
  COALESCE(q."Profession", '-1') AS "Profession",
  COALESCE(q."Occasion", '-1') AS "Occasion",
  COALESCE(q."Purpose", '-1') AS "Purpose",
  COALESCE(q."Typical Day", '-1') AS "Typical Day",
  COALESCE(q."Weekend Preference", '-1') AS "Weekend Preference",
  COALESCE(q."Work Dress", '-1') AS "Work Dress",
  COALESCE(q."Social Dress", '-1') AS "Social Dress",
  COALESCE(q."Trip Preference", '-1') AS "Trip Preference",
  COALESCE(q."Waiting Line", '-1') AS "Waiting Line",
  COALESCE(q."Artwork Response", '-1') AS "Artwork Response",
  COALESCE(q."Mother Response", '-1') AS "Mother Response",
  COALESCE(q."Last Minute Plans", '-1') AS "Last Minute Plans",
  COALESCE(r.mbti_personality_code, '-1') AS mbti_personality_code,
  COALESCE(r.collection_1, '-1') AS collection_1,
  COALESCE(r.collection_2, '-1') AS collection_2,
  COALESCE(r.purchased_ring_style, '-1') AS purchased_ring_style
FROM public.responses r
LEFT JOIN qna_pivot q ON r.response_id = q.request_id
WHERE r.status='PURCHASED';
"""

# -------------------- HELPERS --------------------
def normalize_answers(df):
    """Normalize textual answers using answer_key.json"""
    with open(ANSWER_KEY_JSON, "r", encoding="utf-8") as f:
        answer_map = json.load(f)

    if isinstance(answer_map, list):
        mapping = {
            str(item.get("answer") or item.get("Answer", "")).strip(): 
            str(item.get("option") or item.get("Option", "")).strip()
            for item in answer_map
        }

        for col in df.columns:
            df[col] = df[col].replace(mapping)
    elif isinstance(answer_map, dict):
        for question, mapping in answer_map.items():
            if question in df.columns:
                df[question] = df[question].replace(mapping)
    return df


def fetch_data():
    """Fetch training data from DB"""
    db_uri = f"postgresql+psycopg2://{DB_PARAMS['user']}:{DB_PARAMS['password']}@{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['dbname']}"
    engine = create_engine(db_uri)
    with engine.connect() as conn:
        df = pd.read_sql_query(text(SQL_QUERY), conn)
    return df


# -------------------- TRAIN MODEL --------------------
def train_model():
    df = fetch_data()
    if df.empty:
        print("❌ No purchased data found. Training skipped.")
        return

    # ---------------- Normalize Answers ----------------
    df = normalize_answers(df)

    # ---------------- Load Style & MBTI Mappings ----------------
    with open(PRED_KEY_JSON, "r", encoding="utf-8") as f:
        pred_map = json.load(f)
    style_to_key = {
        item["Purchased Ring Style"].strip().title(): int(item["Key_val"])
        for item in pred_map
    }

    # --- Load MBTI Key Mapping ---
    MBTI_KEY_JSON = os.path.join(ASSETS, "mbti_key.json")
    with open(MBTI_KEY_JSON, "r", encoding="utf-8") as f:
        mbti_map = {item["mbti"].strip().upper(): int(item["key_val"]) for item in json.load(f)}

    # ---------------- Clean & Normalize Labels ----------------
    # Normalize inconsistent style capitalization
    df["purchased_ring_style"] = (
        df["purchased_ring_style"]
        .astype(str)
        .str.strip()
        .str.title()
        .replace({
            "Solitaire Ring Style": "Solitaire Ring Style",
            "SOLITAIRE RING STYLE": "Solitaire Ring Style",
            "Cluster Ring Style": "Cluster Ring Style",
            "CLUSTER RING STYLE": "Cluster Ring Style",
            "Band Ring Style": "Band Ring Style",
            "BAND RING STYLE": "Band Ring Style",
        })
    )

    # Map ring style → numeric key
    df["purchased_ring_style_key"] = df["purchased_ring_style"].map(style_to_key)

    # Map MBTI → numeric key from mbti_key.json
    df["mbti_personality_code"] = (
        df["mbti_personality_code"]
        .astype(str)
        .str.strip()
        .str.upper()
        .map(mbti_map)
    )

    # Replace missing mappings
    if df["purchased_ring_style_key"].isna().any():
        print("⚠️ Missing style mappings:", df[df["purchased_ring_style_key"].isna()]["purchased_ring_style"].unique())
        df["purchased_ring_style_key"] = df["purchased_ring_style_key"].fillna(-1).astype(int)

    if df["mbti_personality_code"].isna().any():
        print("⚠️ Missing MBTI mappings:", df[df["mbti_personality_code"].isna()])
        df["mbti_personality_code"] = df["mbti_personality_code"].fillna(-1).astype(int)

    # ---------------- Prepare Features ----------------
    features = [
        "Who are you purchasing for?", "Gender", "Relation", "Profession", "Occasion",
        "Purpose", "Typical Day", "Weekend Preference", "Work Dress", "Social Dress",
        "Trip Preference", "Waiting Line", "Artwork Response", "Mother Response", "Last Minute Plans"
    ]
    X = df[features].astype(str)

    # ---------------- Encode Categorical Features ----------------
    label_encoders = {}
    for c in X.columns:
        le = LabelEncoder()
        unique_vals = list(X[c].unique()) + ["-1"]
        le.fit(unique_vals)
        X[c] = le.transform(X[c])
        label_encoders[c] = le

    # ---------------- Define Targets ----------------
    y_mbti = np.array(df["mbti_personality_code"].astype(int))
    y_ring = np.array(df["purchased_ring_style_key"].astype(int))
    y = np.column_stack((y_mbti, y_ring))

    # ---------------- Train Model ----------------
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score, f1_score


    # IF YOU WANT TO SEE HOW THE TRAINING DATA LOOK LIKE 
    # print("\n🔍 --- TRAINING DATA PREVIEW ---")
    # print("X shape:", X.shape)
    # print("y_ring shape:", y_ring.shape if 'y_ring' in locals() else 'N/A')
    # print("y_mbti shape:", y_mbti.shape if 'y_mbti' in locals() else 'N/A')

    # # Show first few rows of features + labels
    # df_preview = X.copy()
    # df_preview['ring_style'] = y_ring
    # df_preview['mbti'] = y_mbti
    # print(df_preview.head(20))


    # Handle missing or unseen classes (<1 sample)
    class_counts = pd.Series(y[:, 1]).value_counts()
    all_style_ids = [int(item["Key_val"]) for item in pred_map]
    missing_classes = [cls for cls in all_style_ids if cls not in class_counts.index]

    if len(missing_classes) > 0:
        print(f"⚠️ Found ring styles with <1 sample (missing in DB): {missing_classes}")
        y[:, 1] = np.where(np.isin(y[:, 1], missing_classes), -1, y[:, 1])

    y = np.nan_to_num(y, nan=-1)

    # Split (no stratify)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        max_depth=15
    )
    model.fit(X_train, y_train)

    # ---------------- Validation Report ----------------
    val_pred = model.predict(X_val)
    ring_acc = accuracy_score(y_val[:, 1], val_pred[:, 1])
    ring_f1 = f1_score(y_val[:, 1], val_pred[:, 1], average="weighted", zero_division=0)
    mbti_acc = accuracy_score(y_val[:, 0], val_pred[:, 0])
    mbti_f1 = f1_score(y_val[:, 0], val_pred[:, 0], average="weighted", zero_division=0)

    print(f"📊 Validation Metrics — Ring Style: acc={ring_acc:.3f}, f1={ring_f1:.3f}")
    print(f"📊 Validation Metrics — MBTI: acc={mbti_acc:.3f}, f1={mbti_f1:.3f}")

    # ---------------- Save Model & Encoders ----------------
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/multioutput_model.pkl")
    joblib.dump(label_encoders, "model/label_encoders.pkl")

    # Export mappings for reference
    df[["purchased_ring_style", "purchased_ring_style_key"]].drop_duplicates().to_csv(
        "model/training_ring_map.csv", index=False
    )

    print("✅ Model trained successfully using mbti_key.json + balanced RandomForest + COALESCE SQL cleanup.")


def extract_all_question_ids(obj):
    ids = set()
    if isinstance(obj, dict):
        if "id" in obj:
            ids.add(obj["id"])
        for v in obj.values():
            ids |= extract_all_question_ids(v)
    elif isinstance(obj, list):
        for item in obj:
            ids |= extract_all_question_ids(item)
    return ids




def evaluate_model():
    """Evaluate the trained RandomForest model on current DB data and print metrics (console only)."""
    print("\n🔍 Evaluating model performance...")

    # --- Load model and encoders ---
    model_path = "model/multioutput_model.pkl"
    enc_path = "model/label_encoders.pkl"
    if not os.path.exists(model_path) or not os.path.exists(enc_path):
        print("❌ Trained model or encoders not found. Train first!")
        return

    model = joblib.load(model_path)
    enc = joblib.load(enc_path)

    # --- Load data from DB ---
    db_uri = f"postgresql+psycopg2://{DB_PARAMS['user']}:{DB_PARAMS['password']}@{DB_PARAMS['host']}:{DB_PARAMS['port']}/{DB_PARAMS['dbname']}"
    engine = create_engine(db_uri)
    with engine.connect() as conn:
        df = pd.read_sql_query(text(SQL_QUERY), conn)

    if df.empty:
        print("❌ No purchased data found for evaluation.")
        return

    # --- Load mappings ---
    MBTI_KEY_JSON = os.path.join(ASSETS, "mbti_key.json")
    PRED_KEY_JSON_PATH = os.path.join(ASSETS, "pred_key.json")

    with open(MBTI_KEY_JSON, "r", encoding="utf-8") as f:
        mbti_map = {int(item["key_val"]): item["mbti"].upper() for item in json.load(f)}

    with open(PRED_KEY_JSON_PATH, "r", encoding="utf-8") as f:
        pred_map = json.load(f)
    style_to_key = {item["Purchased Ring Style"].strip().upper(): int(item["Key_val"]) for item in pred_map}

    # --- Encode features using saved encoders ---
    FEATURES = [
        "Who are you purchasing for?", "Gender", "Relation", "Profession", "Occasion",
        "Purpose", "Typical Day", "Weekend Preference", "Work Dress", "Social Dress",
        "Trip Preference", "Waiting Line", "Artwork Response", "Mother Response", "Last Minute Plans"
    ]
    X = df[FEATURES].astype(str)
    for c in X.columns:
        if c in enc:
            le = enc[c]
            X[c] = X[c].apply(lambda v: le.transform([v])[0] if v in le.classes_ else le.transform(["-1"])[0])

    # --- Prepare Y (MBTI + Ring Style) ---
    if "mbti_personality_code" in enc:
        mbti_encoder = enc["mbti_personality_code"]
        y_mbti = mbti_encoder.transform(df["mbti_personality_code"].astype(str))
    else:
        y_mbti = df["mbti_personality_code"].fillna(-1).astype(str)
        # Convert textual MBTI (like ENFP) to key numbers for numeric comparison
        y_mbti = y_mbti.map({v: k for k, v in mbti_map.items()}).fillna(-1).astype(int)
        print("⚠️ Using numeric MBTI codes (no encoder found).")

    y_ring = df["purchased_ring_style"].astype(str).str.strip().str.upper().map(style_to_key).fillna(-1).astype(int)

    # --- Predict ---
    y_pred = model.predict(X)
    y_pred_mbti = y_pred[:, 0]
    y_pred_ring = y_pred[:, 1]

    # --- Decode numeric MBTI back to readable codes ---
    def decode_mbti(arr):
        return [mbti_map.get(int(x), f"Unknown({x})") for x in arr]

    y_mbti_labels = decode_mbti(y_mbti)
    y_pred_mbti_labels = decode_mbti(y_pred_mbti)

    # --- Compute metrics safely ---
    def safe_metric(fn, y_true, y_pred, avg="weighted"):
        try:
            return fn(y_true, y_pred, average=avg, zero_division=0)
        except Exception:
            return np.nan

    results = {
        "MBTI": {
            "Accuracy": accuracy_score(y_mbti, y_pred_mbti),
            "F1": safe_metric(f1_score, y_mbti, y_pred_mbti),
            "Precision": safe_metric(precision_score, y_mbti, y_pred_mbti),
            "Recall": safe_metric(recall_score, y_mbti, y_pred_mbti),
        },
        "Ring Style": {
            "Accuracy": accuracy_score(y_ring, y_pred_ring),
            "F1": safe_metric(f1_score, y_ring, y_pred_ring),
            "Precision": safe_metric(precision_score, y_ring, y_pred_ring),
            "Recall": safe_metric(recall_score, y_ring, y_pred_ring),
        }
    }

    # --- Print readable report ---
    print("\n📊 --- MODEL PERFORMANCE REPORT ---")
    for section, metrics in results.items():
        print(f"\n[{section}]")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

    print("\n--- Detailed MBTI Report (Readable) ---")
    print(classification_report(y_mbti_labels, y_pred_mbti_labels, zero_division=0))

    print("\n--- Detailed Ring Style Report ---")
    print(classification_report(y_ring, y_pred_ring, zero_division=0))

    print("\n✅ Evaluation complete (console only, no file saved).")
    print("Label Distribution (y_ring):")
    print(df["purchased_ring_style"].value_counts())
    print("\nLabel Distribution (y_mbti):")
    print(df["mbti_personality_code"].value_counts())

    return results


if __name__ == "__main__":
    train_model()
