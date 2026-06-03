import os
import re
import pandas as pd
import numpy as np

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# SYSTEM SETTINGS (IMPORTANT)
# -----------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ensure folders exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

app = FastAPI()

# -----------------------------
# TEXT CLEANING
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9 ]', '', text)
    return text

# -----------------------------
# LOAD TRAINING DATA
# -----------------------------
train_df = pd.read_csv("training.csv")
train_df = train_df.dropna(subset=["Code"])
train_df["Description"] = train_df["Description"].fillna("").apply(clean_text)
train_df["Code"] = train_df["Code"].astype(int)

# -----------------------------
# ML MODEL (LIGHTWEIGHT BASE)
# -----------------------------
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=5000,
        min_df=2,
        max_df=0.9
    )),
    ("clf", LinearSVC())
])

model.fit(train_df["Description"], train_df["Code"])
print("✅ ML model ready")

# -----------------------------
# BERT (LAZY LOAD)
# -----------------------------
bert_model = None
train_embeddings = None

def get_bert():
    global bert_model, train_embeddings

    if bert_model is None:
        print("⏳ Loading BERT model...")

        from sentence_transformers import SentenceTransformer

        bert_model = SentenceTransformer("all-MiniLM-L6-v2")

        train_embeddings = bert_model.encode(
            train_df["Description"].tolist(),
            show_progress_bar=False
        )

        print("✅ BERT model loaded")

    return bert_model, train_embeddings

# -----------------------------
# RULE ENGINE (BOOST ACCURACY)
# -----------------------------
def apply_rules(text):
    text = text.lower()

    if "tax" in text:
        return 500, 0.99
    if "salary" in text:
        return 300, 0.99
    if "refund" in text:
        return 200, 0.95

    return None, None

# -----------------------------
# HYBRID PREDICTION
# -----------------------------
def predict_smart(text):

    text_clean = clean_text(text)

    # ✅ Rule-based prediction
    code, conf = apply_rules(text_clean)
    if code:
        return code, conf

    # ✅ ML prediction (fast)
    ml_code = model.predict([text_clean])[0]

    # ✅ Skip BERT for short/simple text
    if len(text_clean) < 5:
        return int(ml_code), 0.6

    # ✅ BERT fallback (only if needed)
    try:
        bert_model, train_embeddings = get_bert()

        text_embedding = bert_model.encode([text_clean])
        similarities = cosine_similarity(text_embedding, train_embeddings)

        best_idx = np.argmax(similarities)
        score = similarities[0][best_idx]

        if score > 0.75:
            bert_code = train_df.iloc[best_idx]["Code"]
            return int(bert_code), float(score)

    except Exception as e:
        print("⚠️ BERT error:", e)

    return int(ml_code), 0.7

# -----------------------------
# UPLOAD API
# -----------------------------
@app.post("/upload")
async def upload(file: UploadFile = File(...)):

    file_id = file.filename.replace(" ", "_")
    input_path = f"uploads/{file_id}"
    output_path = f"outputs/output_{file_id}"

    # Save file
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # Read Excel
    df = pd.read_excel(input_path)

    # Normalize columns
    df.columns = df.columns.str.strip().str.lower()

    # Detect description column
    desc_col = None
    for col in df.columns:
        if "desc" in col or "text" in col:
            desc_col = col
            break

    if desc_col is None:
        desc_col = df.columns[0]

    df[desc_col] = df[desc_col].fillna("")

    # Predict
    codes = []
    confidences = []

    for text in df[desc_col]:
        code, conf = predict_smart(text)
        codes.append(code)
        confidences.append(conf)

    df["PredictedCode"] = codes
    df["Confidence"] = confidences

    # Save output
    df.to_excel(output_path, index=False)

    return {
        "download_url": f"/download/output_{file_id}"
    }

# -----------------------------
# DOWNLOAD API
# -----------------------------
@app.get("/download/{file_name}")
def download(file_name: str):
    file_path = f"outputs/{file_name}"
    return FileResponse(file_path, filename=file_name)

# -----------------------------
# SERVE UI
# -----------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return FileResponse("static/index.html")
