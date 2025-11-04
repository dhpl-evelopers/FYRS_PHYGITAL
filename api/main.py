from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn, uuid, datetime, json, joblib, os, logging
from model.predict import predict_quiz
from model.train import train_model, extract_all_question_ids  
from database.db import fetch_latest_user_full
from fastapi.responses import JSONResponse

app = FastAPI()

# ------------------ CORS setup ------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)

# ------------------ Startup & Shutdown ------------------
@app.on_event("startup")
async def startup_event():
    logging.info("🚀 FastAPI API started.")
    check_schema_sync()

@app.on_event("shutdown")
async def shutdown_event():
    logging.info("🛑 API shutting down.")

# ------------------ Request Model ------------------
class QuizRequest(BaseModel):
    meta: dict
    answers: list

# ------------------ Routes ------------------
@app.get("/")
async def root():
    return {"status": "ok", "message": "RINGS & I AI API Running ✅"}

@app.post("/train")
async def retrain_model():
    try:
        train_model()
        return {"status": "success", "message": "Model retrained and saved."}
    except Exception as e:
        logging.exception("Training failed")
        return {"status": "error", "message": str(e)}

@app.post("/predict")
async def predict_endpoint(payload: QuizRequest):
    try:
        data = payload.dict()
        if "request_id" not in data["meta"]:
            data["meta"]["request_id"] = str(uuid.uuid4())
        result = predict_quiz(data)
        return {"status": "success", "data": result}
    except Exception as e:
        logging.exception("Prediction failed")
        return {"status": "error", "message": str(e)}

# ------------------ Schema Sync ------------------
def check_schema_sync():
    MODEL_ENCODER_PATH = "model/label_encoders.pkl"
    if not os.path.exists(MODEL_ENCODER_PATH):
        print("🆕 Model not found — skipping schema check.")
        return

    enc = joblib.load(MODEL_ENCODER_PATH)
    trained_features = {k for k in enc.keys() if k not in {"mbti_personality_code"}}

    from model.predict import FEATURES  # reuse the same list
    current_features = set(FEATURES)

    if trained_features != current_features:
        print("⚠️ Feature set changed since last training — please retrain the model.")
    else:
        print("✅ Model features match current FEATURES.")


@app.get("/getLatestUserFull")
async def get_latest_user_full(phone_number: str):
    """Return latest full data (request + qna + response) for given phone number."""
    data = fetch_latest_user_full(phone_number)
    if not data:
        return JSONResponse({"error": "No record found for this user"}, status_code=404)
    return JSONResponse(data)



# ------------------ Run ------------------
if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
