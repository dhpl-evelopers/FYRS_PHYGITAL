import psycopg2, os
from datetime import datetime
import logging
from typing import Optional, Dict, Any
import json


def _conn():
    cfg = {
        "host": os.getenv("PGHOST", "projectai-pict-postgresql.postgres.database.azure.com"),
        "port": int(os.getenv("PGPORT", "5432")),
        "dbname": os.getenv("PGDATABASE", "projectai-ringsandi-pict"),
        "user": os.getenv("PGUSER", "postgres"),
        "password": os.getenv("PGPASSWORD", "Apollo11"),
        "sslmode": os.getenv("PGSSLMODE", "require"),
    }
    return psycopg2.connect(**cfg)


def insert_request(meta):
    with _conn() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO public.requests (request_id, name, email, birth_date, phonenumber, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            meta["request_id"],
            meta.get("name", ""),
            meta.get("email", ""),
            meta.get("birth_date", ""),
            meta.get("phonenumber", ""),
            datetime.utcnow()
        ))

def insert_qna(request_id, answers):
    """Insert raw question + selected answer from POST request (exactly as sent)."""
    with _conn() as conn, conn.cursor() as cur:
        for idx, a in enumerate(answers, start=1):
            question = a.get("text") or a.get("question") or "Unknown Question"
            selected_option = a.get("selectedOption", {})
            answer_val = ""

            # Handle both object and string cases
            if isinstance(selected_option, dict):
                answer_val = selected_option.get("value", "")
            elif isinstance(selected_option, str):
                answer_val = selected_option

            option_index = selected_option.get("index", idx) if isinstance(selected_option, dict) else idx

            try:
                cur.execute("""
                    INSERT INTO public.requests_qna (request_id, question, answer, ord)
                    VALUES (%s, %s, %s, %s)
                """, (request_id, question, answer_val, option_index))
            except Exception as e:
                logging.error(f"❌ Failed to insert QnA row {idx}: {e}")

        conn.commit()


def insert_response(data):
    with _conn() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO public.responses (
                response_id, mbti_personality_type, mbti_personality_code, description,
                similar_personality_1, similar_personality_2, collection_1, collection_2,
                section, status, timestamp, purchased_ring_style
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,'PENDING',%s,%s)
            ON CONFLICT (response_id) DO UPDATE SET
                mbti_personality_type = EXCLUDED.mbti_personality_type,
                mbti_personality_code = EXCLUDED.mbti_personality_code,
                description = EXCLUDED.description,
                similar_personality_1 = EXCLUDED.similar_personality_1,
                similar_personality_2 = EXCLUDED.similar_personality_2,
                collection_1 = EXCLUDED.collection_1,
                collection_2 = EXCLUDED.collection_2,
                section = EXCLUDED.section,
                status = 'PENDING',
                timestamp = EXCLUDED.timestamp,
                purchased_ring_style = EXCLUDED.purchased_ring_style;
        """, (
            data["request_id"],
            data["mbti_personality_type"],
            data["mbti_personality_code"],
            data["description"],
            data["similar_personality_1"],
            data["similar_personality_2"],
            data["collection_1"]["name"],
            data["collection_2"]["name"],
            data["section"],
            datetime.utcnow(),
            data["purchased_ring_style"]
        ))



def fetch_latest_user_full(phone_number: str) -> Optional[Dict[str, Any]]:
    """Fetch the latest submission (request + qna + response) for a user by phone number, including ring style image + link."""
    raw_number = phone_number.strip()
    normalized = raw_number.replace("+91", "").replace(" ", "")

    with _conn() as cn, cn.cursor() as cur:
        # Count visits
        cur.execute("""
            SELECT COUNT(*)
            FROM public.requests
            WHERE REPLACE(REPLACE(phonenumber, ' ', ''), '+91', '') LIKE %s;
        """, (f"%{normalized[-10:]}%",))
        visit_count = cur.fetchone()[0]

        # Fetch latest request info
        if raw_number.startswith("+91"):
            cur.execute("""
                SELECT request_id, timestamp, birth_date, email, name, phonenumber
                FROM public.requests
                WHERE phonenumber LIKE %s
                ORDER BY timestamp DESC
                LIMIT 1;
            """, (f"%{raw_number[-10:]}%",))
        else:
            cur.execute("""
                SELECT request_id, timestamp, birth_date, email, name, phonenumber
                FROM public.requests
                WHERE REPLACE(REPLACE(phonenumber, ' ', ''), '+91', '') LIKE %s
                ORDER BY timestamp DESC
                LIMIT 1;
            """, (f"%{normalized[-10:]}%",))

        req_row = cur.fetchone()
        if not req_row:
            return None

        colnames = [d.name for d in cur.description]
        request_data = dict(zip(colnames, req_row))
        request_id = request_data["request_id"]

        # Fetch Q&A
        cur.execute("""
            SELECT question, answer
            FROM public.requests_qna
            WHERE request_id = %s
            ORDER BY ord ASC;
        """, (request_id,))
        qna_rows = cur.fetchall()
        qna_data = [{"question": q, "answer": a} for q, a in qna_rows] if qna_rows else []

        # Fetch response
        cur.execute("""
            SELECT *
            FROM public.responses
            WHERE response_id = %s;
        """, (request_id,))
        resp_row = cur.fetchone()
        resp_data = None
        if resp_row:
            resp_cols = [d.name for d in cur.description]
            resp_data = dict(zip(resp_cols, resp_row))

        # 🧠 Add image and link from pred_key.json if ring style found
        if resp_data and resp_data.get("collection_1"):
            try:
                assets_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "pred_key.json"))
                with open(assets_path, "r", encoding="utf-8") as f:
                    pred_data = json.load(f)

                def find_style_info(style_name: str):
                    for item in pred_data:
                        if style_name.strip().lower() == item["Purchased Ring Style"].strip().lower():
                            return {
                                "name": item["Purchased Ring Style"],
                                "image": item.get("img_link"),
                                "link": item.get("link")
                            }
                    return None

                style1 = find_style_info(resp_data["collection_1"]) if resp_data.get("collection_1") else None
                style2 = find_style_info(resp_data["collection_2"]) if resp_data.get("collection_2") else None

                if style1:
                    resp_data["collection_1_details"] = style1
                if style2:
                    resp_data["collection_2_details"] = style2
            except Exception as e:
                print(f"⚠️ Could not load pred_key.json or match ring style: {e}")

        # Final combined result
        return {
            "request": request_data,
            "qna": qna_data,
            "response": resp_data,
            "visit_count": visit_count,
            "last_visit_time": request_data["timestamp"]
        }
