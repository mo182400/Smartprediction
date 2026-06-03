import pandas as pd
import re
import numpy as np
import os

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Ensure folders exist (important for Render)
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# -----------------------------
# Text cleaning
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9 ]', '', text)
    return text

# -----------------------------
# Load training data
# -----------------------------
train_df = pd.read_csv("training.csv")
train_df = train_df.dropna(subset=["Code"])
train_df["Description"] = train_df["Description"].fillna("").apply(clean_text)
train_df["Code"] = train_df["Code"].astype(int)

# -----------------------------
# ML MODEL (fast baseline)
# -----------------------------
ml_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words='english',
        ngram_range=(1,2),
        max_features=5000
    )),
    ("clf", LinearSVC())
])

ml_model.fit(train_df["Description"], train_df["Code"])

print("✅ ML model ready")

# -----------------------------
# BERT MODEL (semantic AI)
# -----------------------------
bert_model = SentenceTransformer("all-MiniLM-L6-v2")

train_embeddings = bert_model.encode(
    train_df["Description"].tolist(),
    show_progress_bar=False
)

print("✅ BERT model ready")

# -----------------------------
# HYBRID PREDICTION FUNCTION
# -----------------------------
def predict_smart(text):

    text_clean = clean_text(text)

    # 🔹 Rule-based layer (fast + accurate)
    if "tax" in text_clean:
        return 500, 0.99
    if "salary" in text_clean:
        return 300, 0.99

    # 🔹 ML prediction
    ml_code = ml_model.predict([text_clean])[0]

    # 🔹 BERT similarity
    text_embedding = bert_model.encode([text_clean])
    similarities = cosine_similarity(text_embedding, train_embeddings)

    best_idx = np.argmax(similarities)
    bert_score = similarities[0][best_idx]
    bert_code = train_df.iloc[best_idx]["Code"]

    # 🔹 Decision logic
    if bert_score > 0.75:
        return int(bert_code), float(bert_score)

    return int(ml_code), float(bert_score)

# -----------------------------
# API: Upload
# -----------------------------
@app.post("/upload")
async def upload(file: UploadFile = File(...)):

    file_id = file.filename.replace(" ", "_")

    input_path = f"uploads/{file_id}"
    output_path = f"outputs/output_{file_id}"

    # Save file
    with open(input_path, "wb") as f:
        f.write(await file.read())

    # Process file
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

    # Save result
    df.to_excel(output_path, index=False)

    return {
        "download_url": f"/download/{output_file_id(file_id)}"
    }

def output_file_id(file_id):
    return f"output_{file_id}"

# -----------------------------
# Download endpoint
# -----------------------------
@app.get("/download/{file_name}")
def download(file_name: str):
    file_path = f"outputs/{file_name}"
    return FileResponse(file_path, filename=file_name)

# -----------------------------
# Serve UI
# -----------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return FileResponse("static/index.html")
