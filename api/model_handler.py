# api/model_handler.py
from model.predict import predict_mbti_and_rings
from database.db import insert_request, insert_qna, insert_response
import uuid
import datetime


def handle_quiz_submission(data):
    request_id = data['meta'].get("request_id") or str(uuid.uuid4())
    timestamp = datetime.datetime.now().isoformat()

    # Store in DB
    insert_request(request_id, data['meta'], timestamp)
    insert_qna(request_id, data['answers'])

    # Run prediction
    result = predict_mbti_and_rings(data)

    # Save response
    insert_response(result, timestamp)

    return result
